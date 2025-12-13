"""
Minimal GSM8K training scaffold using DatasetQAEnv/GSM8KEnv and the built-in Trainer.

This wires together:
  - HF datasets for GSM8K samples
  - single-sample QA envs (GSM8KEnv)
  - SingleAgentSyncProtocol with a shared VLLMChatClient
  - RolloutBatchSource + MonteCarloReturn credit
  - Trainer with REINFORCE loss

This is a skeleton; adjust hyperparameters, batching, and model loading to your setup.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import queue
from typing import List, Dict, Any

import torch
from datasets import load_dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def build_requests_fn(
    samples_q: queue.Queue,
    batch_size: int,
    sampling_args: Dict[str, Any],
    *,
    group_size: int,
):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2, help="Rollout requests per batch source call")
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=8, help="GRPO group size per prompt")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Optional system prompt for GSM8K env; set to '' to use the model default.",
    )
    parser.add_argument("--train-temperature", type=float, default=1.0, help="Sampling temperature for training rollouts.")
    parser.add_argument("--eval-every", type=int, default=10, help="Eval every N train steps.")
    parser.add_argument("--eval-before-start", action="store_true", default=True, help="Run eval once before training begins.")
    parser.add_argument("--eval-limit", type=int, default=500, help="Number of test samples for eval.")
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument("--eval-temperature", type=float, default=0.0, help="Sampling temperature for eval passes.")
    parser.add_argument("--rollout-log", type=str, default="gsm8k_train_rollouts.jsonl")
    args = parser.parse_args()

    rollout_log_path = os.path.abspath(args.rollout_log)
    os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
    # Touch the file so tailing works even before the first rollout is written
    open(rollout_log_path, "a", encoding="utf-8").close()

    # Data
    train_samples = load_gsm8k(args.split, args.limit)
    if not train_samples:
        raise SystemExit("No GSM8K samples loaded.")
    eval_samples = load_gsm8k("test", args.eval_limit) if args.eval_limit else []

    samples_q: queue.Queue = queue.Queue()
    for idx, s in enumerate(train_samples):
        samples_q.put((idx, s))

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Shared client for inference
    client = VLLMChatClient(host=args.host, port=args.port, enable_weight_updates=True)
    publisher = create_vllm_publisher(client)

    # Registries
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

    # Algorithm first (GRPO-style)
    algo = RLAlgorithm(
        name="grpo",
        credit_assigner=GroupNormalizedReturn(normalize_adv=True),
        loss=ReinforceLoss(length_normalize=True),
    )

    # Engine + batch source
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

    # Trainer
    cfg = TrainerConfig(
        model_device="cuda" if torch.cuda.is_available() else "cpu",
        grad_accum_steps=4,
        max_grad_norm=0.5,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Checkpoint every 25 steps into ./checkpoints_gsm8k
    checkpoint_cfg = CheckpointConfig(
        output_dir="checkpoints_gsm8k",
        every_n_steps=25,
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
        "total_completion_tokens": Reducer(
            kind="sum",
            source="completion_length",
        ),
    }

    # Choose your logger: RichLiveLogger (with ASCII chart + metrics panel).
    train_logger = RichLiveLogger(
        keys=[
            "loss",
            "avg_total_reward",
            "correct_rate",
            "parse_err_rate",
            "avg_completion_length",
            "total_completion_tokens",
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
        checkpoint_config=checkpoint_cfg,
        train_logger=train_logger,
        reducers=reducers,
    )

    async def train_loop():
        train_step = 0
        if args.eval_before_start and eval_samples:
            acc = await run_eval(
                samples=eval_samples,
                client=client,
                model=args.model,
                system_prompt=args.system_prompt,
                concurrency=args.eval_concurrency,
                max_tokens=512,
                temperature=args.eval_temperature,
            )
            print(f"[eval @ step 0] accuracy={acc:.2f}% on {len(eval_samples)} samples")

        for _ in range(args.train_steps):
            if samples_q.empty():
                print("No more samples; stopping training loop.")
                break
            stats = await trainer.train_step()
            train_step = int(stats["train_step"])
            if args.eval_every > 0 and train_step % args.eval_every == 0 and eval_samples:
                acc = await run_eval(
                    samples=eval_samples,
                    client=client,
                    model=args.model,
                    system_prompt=args.system_prompt,
                    concurrency=args.eval_concurrency,
                    max_tokens=512,
                    temperature=args.eval_temperature,
                )
                print(f"[eval @ step {train_step}] accuracy={acc:.2f}% on {len(eval_samples)} samples")

    asyncio.run(train_loop())


if __name__ == "__main__":
    main()
