"""
FSDP2 GSM8K training scaffold (3 training GPUs + 1 VLLM GPU).

Assumptions:
  - GPU0 runs VLLM serving Qwen2.5-7B-Instruct.
  - GPUs 1-3 are reserved for training (set CUDA_VISIBLE_DEVICES=1,2,3).
  - Launch with torchrun: torchrun --nproc_per_node=3 examples/fsdp2_training/train_gsm8k_fsdp2.py

This is a skeleton to illustrate FSDP2 wrapping + Ludic trainer wiring.
Tune batch sizes, steps, and sampling to your hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import queue
import sys
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.distributed import fsdp
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset  # type: ignore

from environments.gsm8k import GSM8KEnv
from ludic.agents.base_agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.distributed.adapters import create_vllm_publisher
from ludic.inference.vllm_client import VLLMChatClient
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.parsers import boxed_parser
from ludic.training.algorithm import RLAlgorithm
from ludic.training.batching.rollout_engine import RolloutEngine
from ludic.training.batching.synced_batching import RolloutBatchSource
from ludic.training.batching.intra_batch_control import GRPORequestStrategy
from ludic.training.credit_assignment import GroupNormalizedReturn
from ludic.training.loss import ReinforceLoss
from ludic.training.trainer import Trainer
from ludic.training.config import TrainerConfig
from ludic.training.checkpoint import CheckpointConfig
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.training.stats import Reducer
from ludic.training.loggers import RichLiveLogger


class _NoopPublisher:
    def publish(self, state_dict, version=None) -> None:  # type: ignore[no-untyped-def]
        return


async def run_eval(
    *,
    samples: List[Dict[str, Any]],
    client: VLLMChatClient,
    model: str,
    system_prompt: str | None,
    concurrency: int,
    max_tokens: int,
    temperature: float,
) -> float:
    """
    Simple eval loop: run each sample once and report accuracy (% correct).
    """
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(sample: Dict[str, Any]) -> bool:
        async with sem:
            env = GSM8KEnv(sample=sample, system_prompt=system_prompt)
            protocol = SingleAgentSyncProtocol(
                agent=Agent(
                    client=client,
                    model=model,
                    ctx=FullDialog(),
                    parser=boxed_parser,
                )
            )
            rollouts = await protocol.run(
                env=env,
                max_steps=1,
                sampling_args={"temperature": temperature, "max_tokens": max_tokens},
            )
            info = rollouts[0].steps[-1].info
            return bool(info.get("correct"))

    results = await asyncio.gather(*[_run_one(s) for s in samples])
    correct = sum(1 for r in results if r)
    return 100.0 * correct / len(samples) if samples else 0.0


def configure_logging(*, rank: int, level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format=f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    # Keep very chatty libraries quieter by default.
    for noisy in ("urllib3", "aiohttp", "httpx", "openai", "datasets", "transformers"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))


def init_dist(*, local_rank: int) -> int:
    if dist.is_initialized():
        return dist.get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    else:
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
        )
    return dist.get_rank()


def shard_samples(samples: List[Dict[str, Any]], rank: int, world_size: int) -> List[Dict[str, Any]]:
    return [s for i, s in enumerate(samples) if i % world_size == rank]


def load_gsm8k(split: str, limit: int | None) -> List[Dict[str, Any]]:
    ds = load_dataset("gsm8k", "main", split=split)
    samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        samples.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "id": row.get("id", idx),
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def build_requests_fn(
    samples_q: queue.Queue,
    batch_size: int,
    sampling_args: Dict[str, Any],
    *,
    group_size: int,
) -> callable:
    def _fn() -> List[RolloutRequest]:
        reqs: List[RolloutRequest] = []
        for _ in range(batch_size):
            if samples_q.empty():
                break
            idx, sample = samples_q.get()
            reqs.append(
                RolloutRequest(
                    env=EnvSpec(kind="gsm8k", kwargs={"sample": sample}),
                    protocol=ProtocolSpec(kind="single_agent", kwargs={}),
                    num_episodes=1,
                    seed=int(idx),
                    sampling_args=sampling_args,
                )
            )
        if not reqs:
            return []
        strategy = GRPORequestStrategy(group_size=group_size)
        return strategy.expand(reqs)

    return _fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    # With 3 training ranks, 11 -> ~33 total in-flight rollouts hitting vLLM,
    # which tends to saturate a --max-num-seqs 32 server without overdoing it.
    parser.add_argument("--concurrency", type=int, default=11)
    parser.add_argument("--train-temperature", type=float, default=1.0)
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--rollout-log", type=str, default="fsdp2_gsm8k_rollouts.jsonl")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_gsm8k_fsdp2")
    parser.add_argument("--eval-every", type=int, default=10, help="Eval every N train steps (0 disables).")
    parser.add_argument("--eval-before-start", action="store_true", default=False, help="Run eval once at step 0.")
    parser.add_argument("--eval-limit", type=int, default=100, help="Number of test samples for eval (0 disables).")
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-max-tokens", type=int, default=512)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--logger", choices=["rich", "print", "none"], default="rich")
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = init_dist(local_rank=env_local_rank)
    world_size = dist.get_world_size()
    configure_logging(rank=rank, level=args.log_level)

    device = torch.device(f"cuda:{env_local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Avoid multiple ranks writing to the same file.
    rollout_log_path = args.rollout_log.replace(".jsonl", f".rank{rank}.jsonl")

    if rank == 0:
        os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    dist.barrier()
    logging.getLogger(__name__).info(
        "Initialized distributed training (world_size=%s, device=%s).", world_size, device
    )

    # Data
    all_train_samples = load_gsm8k(args.split, args.limit)
    train_samples = shard_samples(all_train_samples, rank, world_size)
    if not train_samples:
        raise SystemExit(f"Rank {rank}: no samples after sharding.")

    eval_samples: List[Dict[str, Any]] = []
    if rank == 0 and args.eval_limit and args.eval_limit > 0:
        eval_samples = load_gsm8k("test", args.eval_limit)

    samples_q: queue.Queue = queue.Queue()
    for idx, s in enumerate(train_samples):
        samples_q.put((idx, s))

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    mp_policy = fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # Load on CPU then fully_shard to training devices
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    # Shard transformer blocks first (recommended) then shard the root model.
    blocks = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers  # type: ignore[attr-defined]
    elif hasattr(model, "layers"):
        blocks = model.layers  # type: ignore[attr-defined]
    if blocks is not None:
        for layer in blocks:
            fsdp.fully_shard(layer, mp_policy=mp_policy)
    fsdp.fully_shard(model, mp_policy=mp_policy)

    # Shared client for inference
    # Only rank0 should enable NCCL-based weight updates into vLLM.
    client = VLLMChatClient(
        host=args.vllm_host,
        port=args.vllm_port,
        enable_weight_updates=(rank == 0),
        device=str(device),
    )
    publisher = create_vllm_publisher(client) if rank == 0 else _NoopPublisher()

    env_registry = {"gsm8k": lambda sample: GSM8KEnv(sample=sample, system_prompt=args.system_prompt)}

    def protocol_factory():
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=args.model,
                ctx=FullDialog(),
                parser=boxed_parser,
            )
        )

    protocol_registry = {"single_agent": protocol_factory}

    algo = RLAlgorithm(
        name="grpo",
        credit_assigner=GroupNormalizedReturn(normalize_adv=True),
        loss=ReinforceLoss(length_normalize=True),
    )

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=rollout_log_path,
    )
    sampling_args = {
        "temperature": args.train_temperature,
        "max_tokens": 512,
        "extras": {"extra_body": {"return_token_ids": True}},
    }
    requests_fn = build_requests_fn(samples_q, args.batch_size, sampling_args, group_size=args.group_size)
    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=algo.credit_assigner,
        requests_fn=requests_fn,
        max_steps=1,
        concurrency=args.concurrency,
        retokenize=False,
    )

    cfg = TrainerConfig(
        model_device=str(device),
        grad_accum_steps=2,
        max_grad_norm=0.5,
        pad_token_id=tokenizer.pad_token_id,
    )
    checkpoint_cfg = CheckpointConfig(
        output_dir=args.checkpoint_dir,
        every_n_steps=10,
        max_to_keep=2,
        save_optimizer=True,
    )
    reducers = {
        "correct_rate": Reducer(
            kind="count_true",
            source="correct",
            normalize_by="rollouts",
        ),
        "parse_err_rate": Reducer(
            kind="count_true",
            source="parse_error",
            normalize_by="samples",
        ),
    }
    train_logger = None
    if rank == 0:
        if args.logger == "none":
            train_logger = None
        elif args.logger == "print" or not sys.stdout.isatty():
            from ludic.training.loggers import PrintLogger

            train_logger = PrintLogger(
                prefix="[trainer]",
                keys=[
                    "loss",
                    "avg_total_reward",
                    "correct_rate",
                    "parse_err_rate",
                    "num_rollouts",
                    "num_samples",
                ],
                precision=4,
            )
        else:
            train_logger = RichLiveLogger(
                keys=[
                    "loss",
                    "avg_total_reward",
                    "correct_rate",
                    "parse_err_rate",
                    "num_rollouts",
                    "num_samples",
                ],
                spark_key="avg_total_reward",
                history=100,
                precision=4,
            )

    trainer = Trainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        publisher=publisher,
        cfg=cfg,
        enable_gradient_checkpointing=True,
        checkpoint_config=checkpoint_cfg,
        train_logger=train_logger,
        reducers=reducers,
    )

    async def train_loop():
        if args.eval_before_start and eval_samples:
            if rank == 0:
                acc = await run_eval(
                    samples=eval_samples,
                    client=client,
                    model=args.model,
                    system_prompt=args.system_prompt,
                    concurrency=args.eval_concurrency,
                    max_tokens=args.eval_max_tokens,
                    temperature=args.eval_temperature,
                )
                print(f"[eval @ step 0] accuracy={acc:.2f}% on {len(eval_samples)} samples", flush=True)
            if dist.is_initialized():
                dist.barrier()

        for _ in range(args.train_steps):
            # Stop all ranks if any rank runs out of samples to avoid deadlocks.
            local_done = 1 if samples_q.empty() else 0
            if dist.is_initialized():
                done = torch.tensor(local_done, device=device)
                dist.all_reduce(done, op=dist.ReduceOp.MAX)
                if int(done.item()) != 0:
                    break
            else:
                if local_done:
                    break

            stats = await trainer.train_step()
            if rank == 0:
                step = int(stats["train_step"])
                print(
                    f"[rank0 step {step}] loss={stats.get('loss'):.4f} reward={stats.get('avg_total_reward'):.4f}",
                    flush=True,
                )

            if args.eval_every and args.eval_every > 0 and eval_samples:
                step = int(stats["train_step"])
                if step % args.eval_every == 0:
                    if dist.is_initialized():
                        dist.barrier()
                    if rank == 0:
                        acc = await run_eval(
                            samples=eval_samples,
                            client=client,
                            model=args.model,
                            system_prompt=args.system_prompt,
                            concurrency=args.eval_concurrency,
                            max_tokens=args.eval_max_tokens,
                            temperature=args.eval_temperature,
                        )
                        print(
                            f"[eval @ step {step}] accuracy={acc:.2f}% on {len(eval_samples)} samples",
                            flush=True,
                        )
                    if dist.is_initialized():
                        dist.barrier()

    asyncio.run(train_loop())

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
