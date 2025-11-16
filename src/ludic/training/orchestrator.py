from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    List,
    Optional,
)

from ludic.agent import Agent
from ludic.context.base import ContextStrategy
from ludic.context.full_dialog import FullDialog
from ludic.env import Env
from ludic.interaction import run_episode
from ludic.types import Rollout, SamplingArgs, Step

from ludic.training.types import (
    RolloutRequest,
    RolloutPolicy,
    WeightingStrategy,
    SAWItem,
    SAWBatch,
    RolloutStepKey,
    TokenizeFn,
    StateFromStepFn,
)


# ---------------------------------------------------------------------------
# Factory aliases
# ---------------------------------------------------------------------------

# Build a fresh Env each episode
EnvFactory = Callable[..., Env]
CtxFactory = Callable[[], ContextStrategy]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    episodes: int
    max_steps: int = 64
    sampling_args: Optional[SamplingArgs] = None
    concurrency: int = 8
    timeout_s: Optional[float] = None
    system_prompt: Optional[str] = None
    jsonl_path: Optional[str] = None  # if set, append each rollout as JSONL


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    Dumb, reliable orchestrator:

      - spawns N episodes
      - runs them with an asyncio.Semaphore
      - returns List[Rollout]
      - optionally writes each rollout to JSONL

    Extended variant:

      - can use a RolloutPolicy to control per-episode env/ctx/sampling/meta
      - can build State–Action–Weight batches via `generate_batch`, using:
          * a WeightingStrategy for credit assignment
          * model token IDs from Step.info when available
          * a fallback tokenizer otherwise
    """

    def __init__(
        self,
        env_factory: EnvFactory,
        agent: Agent,
        *,
        cfg: OrchestratorConfig,
        ctx_factory: Optional[CtxFactory] = None,
        rollout_policy: Optional[RolloutPolicy] = None,
    ) -> None:
        self.env_factory = env_factory
        self.agent = agent
        self.cfg = cfg
        self.ctx_factory = ctx_factory or (lambda: FullDialog())
        self.rollout_policy = rollout_policy

        if self.cfg.jsonl_path:
            Path(os.path.dirname(self.cfg.jsonl_path) or ".").mkdir(
                parents=True, exist_ok=True
            )

    # ---- internal helpers ------------------------------------------------

    def _default_request(self, idx: int) -> RolloutRequest:
        """
        Backwards-compatible per-episode config if no RolloutPolicy is provided.
        """
        env = self.env_factory()
        ctx = self.ctx_factory()
        sampling_args: SamplingArgs = self.cfg.sampling_args or {}
        return RolloutRequest(
            env=env,
            ctx=ctx,
            sampling_args=sampling_args,
            system_prompt=self.cfg.system_prompt,
            meta={"episode_idx": idx},
        )

    async def _run_one(self, idx: int, sem: asyncio.Semaphore) -> Rollout:
        async with sem:
            if self.rollout_policy is not None:
                req = self.rollout_policy.make_rollout(idx)
            else:
                req = self._default_request(idx)

            rollout = await run_episode(
                env=req.env,
                agent=self.agent,
                max_steps=self.cfg.max_steps,
                sampling_args=req.sampling_args,
                ctx=req.ctx,
                system_prompt=req.system_prompt,
                timeout_s=self.cfg.timeout_s,
            )

            # Attach orchestrator + policy metadata
            rollout.meta["orchestrator_cfg"] = {
                "max_steps": self.cfg.max_steps,
                "timeout_s": self.cfg.timeout_s,
            }
            rollout.meta.update(req.meta)

            if self.cfg.jsonl_path:
                self._append_jsonl(rollout)

            return rollout

    def _append_jsonl(self, rollout: Rollout) -> None:
        assert self.cfg.jsonl_path is not None
        payload = {
            "id": rollout.id,
            "meta": rollout.meta,
            "steps": [
                {
                    "index": s.index,
                    "prev_obs": s.prev_obs,
                    "action": s.action,
                    "next_obs": s.next_obs,
                    "reward": s.reward,
                    "truncated": s.truncated,
                    "terminated": s.terminated,
                    "info": s.info,
                    "ts_ns": s.ts_ns,
                }
                for s in rollout.steps
            ],
            "total_reward": rollout.total_reward,
            "length": rollout.length,
            "duration_ns": rollout.duration_ns,
        }
        with open(self.cfg.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---- rollout generation ----------------------------------------------

    async def generate(self) -> List[Rollout]:
        """
        Run `cfg.episodes` episodes and return the resulting rollouts.
        """
        sem = asyncio.Semaphore(max(1, self.cfg.concurrency))
        tasks = [self._run_one(i, sem) for i in range(self.cfg.episodes)]
        return await asyncio.gather(*tasks)

    def generate_sync(self) -> List[Rollout]:
        """
        Synchronous wrapper around `generate()`.

        Intended for scripts/CLIs without an existing event loop.
        If you're in a notebook or async app, call `await generate()` instead.
        """
        return asyncio.run(self.generate())

    # ---- SAW batch generation --------------------------------------------

    async def generate_batch(
        self,
        *,
        weighting: WeightingStrategy,
        tokenize: TokenizeFn,
        state_from_step: Optional[StateFromStepFn] = None,
        use_model_token_ids: bool = True,
        retokenize: bool = False,
    ) -> SAWBatch:
        """
        High-level entrypoint for RL-style training:

        - runs episodes (via `generate`)
        - computes weights per (rollout, step) via WeightingStrategy
        - builds a State–Action–Weight batch, including:
            * tokenized input_ids (state + action)
            * attention_mask
            * action_mask (1 on action tokens, 0 elsewhere)
            * scalar weight per item
            * batch-level metadata (in SAWBatch.meta)

        Tokenization strategy:

        - If `use_model_token_ids=True`, this looks for stored model token IDs:
                step.info["prompt_token_ids"]
                step.info["token_ids"]      # TODO: naming inconsistent; fix later
          These must be populated by the Agent or run_episode.
          If present, they are always used.

        - If model token IDs are missing:

                * If `retokenize=True`, we fall back to the provided `tokenize(text)`
                  function for both state and action.

                * If `retokenize=False`, we raise an error.
                  This avoids silent misalignment between model tokenization and
                  post-hoc text tokenization.

        `state_from_step` default:
        - If not provided, the “state” is the observation before the action:
                state_text = step.prev_obs
        """

        rollouts = await self.generate()
        weights = weighting.compute(rollouts)

        if state_from_step is None:
            def default_state_from_step(r: Rollout, i: int, step: Step) -> str:
                return step.prev_obs
            state_fn: StateFromStepFn = default_state_from_step
        else:
            state_fn = state_from_step

        items: List[SAWItem] = []

        for r in rollouts:
            for i, step in enumerate(r.steps):
                key: RolloutStepKey = (r.id, step.index)

                # ---- every step must have an explicit weight ----
                try:
                    w_raw = weights[key]
                except KeyError as exc:
                    raise KeyError(
                        f"WeightingStrategy did not provide a weight for "
                        f"(rollout_id={r.id!r}, step_index={step.index}). "
                        "All steps must be covered."
                    ) from exc

                w = float(w_raw)

                info = step.info or {}

                # Try model token IDs
                prompt_ids = info.get("prompt_token_ids")
                # TODO: naming here is inconsistent; "token_ids" are the completion ids.
                completion_ids = info.get("token_ids")

                has_model_ids = (
                    isinstance(prompt_ids, list)
                    and isinstance(completion_ids, list)
                    and all(isinstance(t, int) for t in prompt_ids)
                    and all(isinstance(t, int) for t in completion_ids)
                )

                if use_model_token_ids and has_model_ids:
                    # Path A: model token IDs
                    input_ids = list(prompt_ids) + list(completion_ids)
                    attention_mask = [1] * len(input_ids)
                    action_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

                    items.append(
                        SAWItem(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            weight=w,
                            meta={
                                "rollout_id": r.id,
                                "step_index": step.index,
                                "reward": step.reward,
                                "total_reward": r.total_reward,
                                **(r.meta),
                            },
                        )
                    )
                    continue

                # --- Missing model IDs ---
                if not retokenize and use_model_token_ids:
                    raise ValueError(
                        f"Missing model token IDs for rollout {r.id}, step {step.index}, "
                        "but retokenize=False. "
                        "Either enable retokenize=True or fix your Agent/run_episode "
                        "to store 'prompt_token_ids' and 'token_ids' in Step.info."
                    )

                # Path B: retokenize using text
                state_text = state_fn(r, i, step)
                action_text = step.action

                state_ids = tokenize(state_text)
                action_ids = tokenize(action_text)

                input_ids = state_ids + action_ids
                attention_mask = [1] * len(input_ids)
                action_mask = [0] * len(state_ids) + [1] * len(action_ids)

                items.append(
                    SAWItem(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        weight=w,
                        meta={
                            "rollout_id": r.id,
                            "step_index": step.index,
                            "reward": step.reward,
                            "total_reward": r.total_reward,
                            **(r.meta),
                        },
                    )
                )

        # ---- Build batch-level metadata -----------------------------------
        # TODO: Add more metrics for logging
        meta = {
            "episodes": len(rollouts),
            "total_items": len(items),
            "avg_total_reward": (
                float(sum(r.total_reward for r in rollouts) / len(rollouts))
                if rollouts else 0.0
            ),
        }

        return SAWBatch(items=items, meta=meta)
