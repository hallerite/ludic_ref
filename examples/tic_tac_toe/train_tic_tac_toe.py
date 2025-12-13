"""
Minimal Tic-Tac-Toe training scaffold.

This wires together:
  - TicTacToeEnv single-agent episodes
  - SingleAgentSyncProtocol with a shared VLLMChatClient
  - RolloutBatchSource + GroupNormalizedReturn credit
  - Trainer with REINFORCE loss
  - Optional periodic eval of win rate
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from environments.tic_tac_toe import TicTacToeEnv
from ludic.agents.base_agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.distributed.adapters import create_vllm_publisher
from ludic.inference.vllm_client import VLLMChatClient
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.parsers import compose_parsers, cot_prefix_parser, xml_move_parser
from ludic.training.algorithm import RLAlgorithm
from ludic.training.batching.rollout_engine import RolloutEngine
from ludic.training.batching.synced_batching import RolloutBatchSource
from ludic.training.credit_assignment import GroupNormalizedReturn
from ludic.training.loss import ReinforceLoss
from ludic.training.trainer import Trainer
from ludic.training.config import TrainerConfig
from ludic.training.checkpoint import CheckpointConfig
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.training.stats import Reducer
from ludic.training.loggers import RichLiveLogger

# Compose parsers to strip optional <think>...</think> and then require <move>...</move>.
TICTACTOE_PARSER = compose_parsers(cot_prefix_parser, xml_move_parser)

async def run_eval(
    *,
    seeds: List[int],
    client: VLLMChatClient,
    model: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    max_steps: int,
) -> float:
    """
    Run a batch of Tic-Tac-Toe episodes and report win rate (%).
    """
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(seed: int) -> bool:
        async with sem:
            env = TicTacToeEnv(agent_starts=True)
            base_prompt = env.suggested_sysprompt or ""
            sys_prompt = (
                base_prompt
                + "\n\nThink through the board in <think>...</think> and output your move as a single XML tag, e.g., <move>A1</move>."
            )
            protocol = SingleAgentSyncProtocol(
                agent=Agent(
                    client=client,
                    model=model,
                    ctx=FullDialog(),
                    parser=TICTACTOE_PARSER,
                ),
                prompt=sys_prompt,
            )
            rollouts = await protocol.run(
                env=env,
                max_steps=max_steps,
                seed=seed,
                sampling_args={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "extras": {"extra_body": {"return_token_ids": True}},
                },
            )
            info = rollouts[0].steps[-1].info
            return info.get("result") == "win"

    results = await asyncio.gather(*[_run_one(s) for s in seeds])
    wins = sum(1 for r in results if r)
    return 100.0 * wins / len(seeds) if seeds else 0.0


def build_requests_fn(
    rng: torch.Generator,
    batch_size: int,
    sampling_args: Dict[str, Any],
):
    def _fn() -> List[RolloutRequest]:
        reqs: List[RolloutRequest] = []
        for _ in range(batch_size):
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
            reqs.append(
                RolloutRequest(
                    env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
                    protocol=ProtocolSpec(kind="single_agent", kwargs={}),
                    num_episodes=1,
                    seed=int(seed),
                    sampling_args=sampling_args,
                )
            )
        return reqs

    return _fn


def main():
    parser = argparse.ArgumentParser(description="Train a model on Tic-Tac-Toe using Ludic + vLLM.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for sampling episode seeds.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4, help="Rollout requests per batch source call.")
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--max-steps-per-episode", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (RL-friendly defaults from LoRA best-practice guides).")
    parser.add_argument(
        "--lora-alpha-mult",
        type=float,
        default=2.0,
        help="Multiplier applied to rank to set lora_alpha (alpha = rank * mult).",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout probability.")
    parser.add_argument("--train-temperature", type=float, default=1.0, help="Sampling temperature for training rollouts.")
    parser.add_argument("--eval-every", type=int, default=10, help="Eval every N train steps.")
    parser.add_argument("--eval-before-start", action="store_true", default=True, help="Run eval once before training begins.")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of episodes for eval.")
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument("--eval-temperature", type=float, default=0.6, help="Sampling temperature for eval passes.")
    parser.add_argument("--rollout-log", type=str, default="tictactoe_train_rollouts.jsonl")
    args = parser.parse_args()

    rollout_log_path = os.path.abspath(args.rollout_log)
    os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
    open(rollout_log_path, "a", encoding="utf-8").close()

    # Seeds for deterministic episode resets
    rng = torch.Generator()
    rng.manual_seed(args.seed if args.seed is not None else 0)

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Apply a lightweight LoRA adapter to train only a small subset of params.
    # Apply LoRA to all linear projections (per “LoRA Without Regret” guidance: all-linear > attention-only).
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=int(args.lora_rank * args.lora_alpha_mult),
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(base_model, lora_config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.print_trainable_parameters()

    # Shared client for inference
    client = VLLMChatClient(host=args.host, port=args.port, enable_weight_updates=True)
    publisher = create_vllm_publisher(client)

    # Registries
    env_registry = {"tictactoe": lambda agent_starts=True: TicTacToeEnv(agent_starts=agent_starts)}

    def protocol_factory():
        # Extend the env's suggested system prompt with explicit CoT + XML move instructions.
        base_prompt = TicTacToeEnv().suggested_sysprompt or ""
        prompt = (
            base_prompt
            + "\n\nThink through the board in <think>...</think> and output your move as a single XML tag, e.g., <move>A1</move>."
        )
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=args.model,
                ctx=FullDialog(),
                parser=TICTACTOE_PARSER,
            ),
            prompt=prompt,
        )

    protocol_registry = {"single_agent": protocol_factory}

    # Algorithm
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
        "max_tokens": 250,
        "extras": {"extra_body": {"return_token_ids": True}},
    }
    requests_fn = build_requests_fn(rng, args.batch_size, sampling_args)
    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=algo.credit_assigner,
        requests_fn=requests_fn,
        max_steps=args.max_steps_per_episode,
        concurrency=args.concurrency,
        retokenize=False,
    )

    # Trainer
    cfg = TrainerConfig(
        model_device="cuda" if torch.cuda.is_available() else "cpu",
        grad_accum_steps=4,
        max_grad_norm=0.5,
        pad_token_id=tokenizer.pad_token_id,
        lr=5e-5,
    )
    checkpoint_cfg = CheckpointConfig(
        output_dir="checkpoints_tictactoe",
        every_n_steps=25,
        max_to_keep=2,
        save_optimizer=True,
    )
    reducers = {
        "win_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "win",
            normalize_by="rollouts",
        ),
        "loss_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "loss",
            normalize_by="rollouts",
        ),
        "draw_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "draw",
            normalize_by="rollouts",
        ),
        "illegal_rate": Reducer(
            kind="count_true",
            source="illegal_move",
            normalize_by="samples",
        ),
        "total_completion_tokens": Reducer(
            kind="sum",
            source="completion_length",
        ),
    }

    train_logger = RichLiveLogger(
        keys=[
            "loss",
            "avg_total_reward",
            "win_rate",
            "loss_rate",
            "draw_rate",
            "illegal_rate",
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
        eval_seeds = list(range(args.eval_episodes))
        if args.eval_before_start and eval_seeds:
            acc = await run_eval(
                seeds=eval_seeds,
                client=client,
                model=args.model,
                concurrency=args.eval_concurrency,
                max_tokens=250,
                temperature=args.eval_temperature,
                max_steps=args.max_steps_per_episode,
            )
            print(f"[eval @ step 0] win_rate={acc:.2f}% on {len(eval_seeds)} episodes")

        for _ in range(args.train_steps):
            stats = await trainer.train_step()
            train_step = int(stats["train_step"])
            if args.eval_every > 0 and train_step % args.eval_every == 0 and eval_seeds:
                acc = await run_eval(
                    seeds=eval_seeds,
                    client=client,
                    model=args.model,
                    concurrency=args.eval_concurrency,
                    max_tokens=250,
                    temperature=args.eval_temperature,
                    max_steps=args.max_steps_per_episode,
                )
                print(f"[eval @ step {train_step}] win_rate={acc:.2f}% on {len(eval_seeds)} episodes")

    asyncio.run(train_loop())


if __name__ == "__main__":
    main()
