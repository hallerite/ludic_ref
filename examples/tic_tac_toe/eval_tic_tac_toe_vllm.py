"""
Eval a vLLM-served model on Tic-Tac-Toe: optionally start a vLLM server,
run concurrent episodes via SingleAgentSyncProtocol, and report win/draw/loss
rates plus reducer metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from typing import List, Dict

import requests

from ludic.agents.base_agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.parsers import compose_parsers, cot_prefix_parser, xml_move_parser
from ludic.types import SamplingArgs
from environments.tic_tac_toe import TicTacToeEnv
from ludic.training.stats import Reducer, apply_reducers_to_records

TICTACTOE_PARSER = compose_parsers(cot_prefix_parser, xml_move_parser)


def start_vllm_server(model: str, host: str, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    cmd = [
        sys.executable,
        "-m",
        "ludic.inference.vllm_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu_memory_utilization",
        "0.7",
        "--enforce-eager",
    ]

    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def wait_for_health(host: str, port: int, proc: subprocess.Popen, timeout_s: float = 180.0) -> None:
    health_url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e

        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"vLLM server exited early with code {proc.returncode}\n"
                f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        time.sleep(2.0)

    proc.terminate()
    stdout, stderr = proc.communicate(timeout=10)
    raise RuntimeError(
        f"vLLM server failed to become healthy at {health_url}\n"
        f"Last error: {last_err}\n"
        f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )


async def eval_episodes(
    *,
    seeds: List[int],
    model: str,
    host: str,
    port: int,
    temperature: float,
    max_tokens: int,
    timeout_s: float | None,
    max_steps: int,
    concurrency: int = 1,
) -> List[dict]:
    reducers: Dict[str, Reducer] = {
        "win_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "win", normalize_by="rollouts"),
        "loss_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "loss", normalize_by="rollouts"),
        "draw_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "draw", normalize_by="rollouts"),
        "illegal_rate": Reducer(kind="count_true", source="illegal_move", normalize_by="samples"),
        "parse_err_rate": Reducer(kind="count_true", source="parse_error", normalize_by="samples"),
        "avg_completion_length": Reducer(kind="mean", source="completion_length"),
        "total_completion_tokens": Reducer(kind="sum", source="completion_length"),
    }

    client = VLLMChatClient(
        host=host,
        port=port,
        enable_weight_updates=False,
    )

    sargs: SamplingArgs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extras": {"extra_body": {"return_token_ids": True}},
    }

    total = 0
    wins = 0
    losses = 0
    draws = 0
    parse_errors = 0
    records: List[dict] = []

    idx = 0
    while idx < len(seeds):
        batch_seeds = seeds[idx : idx + concurrency]
        batch_size = len(batch_seeds)
        tasks = []
        for seed in batch_seeds:
            env = TicTacToeEnv(agent_starts=True)
            base_prompt = env.suggested_sysprompt or ""
            sys_prompt = (
                base_prompt
                + "\n\nThink through the board in <think>...</think> and output your move as a single XML tag, e.g., <move>A1</move>."
            )
            tasks.append(
                SingleAgentSyncProtocol(
                    agent=Agent(
                        client=client,
                        model=model,
                        ctx=FullDialog(),
                        parser=TICTACTOE_PARSER,
                    ),
                    prompt=sys_prompt,
                ).run(
                    env=env,
                    max_steps=max_steps,
                    seed=seed,
                    sampling_args=sargs,
                    timeout_s=timeout_s,
                )
            )

        batch_rollouts = await asyncio.gather(*tasks)
        idx += batch_size
        for seed, rollouts in zip(batch_seeds, batch_rollouts):
            step = rollouts[0].steps[-1]
            info = step.info

            total += 1
            result = info.get("result")
            if result == "win":
                wins += 1
            elif result == "loss":
                losses += 1
            elif result == "draw":
                draws += 1
            if info.get("parse_error") or step.truncated:
                parse_errors += 1

            completion_ids = info.get("completion_token_ids") or []

            records.append(
                {
                    "seed": seed,
                    "first_obs": rollouts[0].steps[0].prev_obs,
                    "raw_action": step.action,
                    "parsed_action": info.get("parsed_action"),
                    "result": result,
                    "illegal_move": info.get("illegal_move"),
                    "parse_error": info.get("parse_error"),
                    "reward": step.reward,
                    "truncated": step.truncated,
                    "terminated": step.terminated,
                    "completion_length": len(completion_ids) if isinstance(completion_ids, list) else 0,
                }
            )

        win_rate = 100 * wins / total
        print(f"[{total}/{len(seeds)}] win_rate={win_rate:.2f}% parse_errors={parse_errors}")

    win_rate = 100 * wins / total if total else 0.0
    loss_rate = 100 * losses / total if total else 0.0
    draw_rate = 100 * draws / total if total else 0.0
    print("---- TicTacToe Evaluation ----")
    print(f"Total episodes : {total}")
    print(f"Wins           : {wins} ({win_rate:.2f}%)")
    print(f"Losses         : {losses} ({loss_rate:.2f}%)")
    print(f"Draws          : {draws} ({draw_rate:.2f}%)")
    print(f"Parse errors   : {parse_errors}")
    reducer_stats = apply_reducers_to_records(records, reducers)
    if reducer_stats:
        print("Reducer stats :")
        for k, v in reducer_stats.items():
            print(f"  {k}: {v}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a vLLM server (optional) and evaluate a model on Tic-Tac-Toe.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="If set, launch a local vLLM server for the chosen model before eval.",
    )
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to run.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--timeout-s", type=float, default=None, help="Per-call timeout.")
    parser.add_argument(
        "--out",
        type=str,
        default="tictactoe_eval.jsonl",
        help="Path to write rollout results as JSONL.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of parallel episodes to run.",
    )
    parser.add_argument("--max-steps", type=int, default=5, help="Max steps per episode.")

    args = parser.parse_args()

    seeds = list(range(args.episodes))
    print(f"Prepared {len(seeds)} Tic-Tac-Toe episodes (seeded).")
    print(f"Evaluating model '{args.model}' via vLLM at {args.host}:{args.port}")

    proc = None
    if args.start_server:
        print("Starting local vLLM server...")
        proc = start_vllm_server(args.model, args.host, args.port)
        try:
            wait_for_health(args.host, args.port, proc)
            print("vLLM server is healthy.")
        except Exception:
            proc.kill()
            raise

    try:
        records = asyncio.run(
            eval_episodes(
                seeds=seeds,
                model=args.model,
                host=args.host,
                port=args.port,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
                max_steps=args.max_steps,
                concurrency=args.concurrency,
            )
        )
        if records:
            import json
            out_path = args.out
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Wrote {len(records)} records to {out_path}")
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    main()
