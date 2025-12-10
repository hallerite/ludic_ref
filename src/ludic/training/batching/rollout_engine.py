from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ludic.envs.env import LudicEnv
from ludic.interaction.base import InteractionProtocol
from ludic.types import Rollout, SamplingArgs

from ludic.training.types import (
    CreditAssigner,
    SAWItem,
    SAWBatch,
    RolloutStepKey,
    TokenizeFn,
    RolloutRequest,
    ProtocolSpec,
    EnvSpec,
)

# ---------------------------------------------------------------------------
# Factory aliases
# ---------------------------------------------------------------------------

EnvFactory = Callable[..., LudicEnv]
ProtocolFactory = Callable[..., InteractionProtocol]

EnvRegistry = Dict[str, EnvFactory]
ProtocolRegistry = Dict[str, ProtocolFactory]


class RolloutEngine:
    """
    Stateless rollout executor.
    
    Responsibilities:
      1. Instantiating Envs and Protocols from Requests.
      2. Executing Episodes (generate_rollouts).
      3. Collating Training Data (generate_batch).
    """

    def __init__(
        self,
        *,
        env_registry: EnvRegistry,
        protocol_registry: ProtocolRegistry,
        jsonl_path: Optional[str] = None,
    ) -> None:
        self.env_registry = dict(env_registry)
        self.protocol_registry = dict(protocol_registry)
        self.jsonl_path = jsonl_path

        if self.jsonl_path:
            Path(os.path.dirname(self.jsonl_path) or ".").mkdir(
                parents=True, exist_ok=True
            )

    # ---- registry helpers ------------------------------------------------
    def _build_env(self, spec: EnvSpec) -> LudicEnv:
        """Instantiate an Env from an EnvSpec via the env_registry."""
        try:
            factory = self.env_registry[spec.kind]
        except KeyError as exc:
            raise KeyError(f"Unknown env kind: {spec.kind!r}") from exc
        return factory(**spec.kwargs)

    def _build_protocol(self, spec: ProtocolSpec) -> InteractionProtocol:
        """Instantiate an InteractionProtocol from a ProtocolSpec via the registry."""
        try:
            factory = self.protocol_registry[spec.kind]
        except KeyError as exc:
            raise KeyError(f"Unknown protocol kind: {spec.kind!r}") from exc
        return factory(**spec.kwargs)

    # ---- internal helpers ------------------------------------------------

    async def _run_one_request(
        self,
        request: RolloutRequest,
        episode_idx: int,
        sem: asyncio.Semaphore,
        *,
        max_steps: int,
        timeout_s: Optional[float],
    ) -> List[Rollout]:
        """
        Run a single rollout for a given RolloutRequest.
        """
        async with sem:
            # 1. Create a fresh, independent protocol worker (and its agent)
            protocol = self._build_protocol(request.protocol)

            # 2. Create a fresh env
            env = self._build_env(request.env)

            sargs: SamplingArgs = request.sampling_args or {}

            # 3. Determine the seed to use for env.reset()
            run_seed = request.seed if request.seed is not None else episode_idx
            is_forced_seed = request.seed is not None

            # 4. Run the episode using the fresh protocol and env
            rollouts = await protocol.run(
                env=env,
                max_steps=max_steps,
                seed=run_seed,
                sampling_args=sargs,
                timeout_s=timeout_s,
            )

            # 5. Log metadata for ALL returned rollouts
            for r in rollouts:
                r.meta.setdefault("episode_idx", episode_idx)
                
                # We flatten the request metadata into the rollout metadata
                # so keys like 'policy_version' are accessible at the top level.
                if request.meta:
                    r.meta.update(request.meta)

                r.meta.setdefault("request_meta", {})
                r.meta["request_meta"].update(request.meta)
                r.meta.setdefault("engine", {})
                r.meta["engine"].update(
                    {
                        "max_steps": max_steps,
                        "timeout_s": timeout_s,
                        "env_kind": request.env.kind,
                        "protocol_kind": request.protocol.kind,
                        "used_seed": run_seed,
                        "forced_seed": is_forced_seed,
                    }
                )

                if self.jsonl_path:
                    self._append_jsonl(r)

            return rollouts

    def _append_jsonl(self, rollout: Rollout) -> None:
        assert self.jsonl_path is not None
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
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---- rollout generation ----------------------------------------------

    async def generate_rollouts(
        self,
        *,
        requests: List[RolloutRequest],
        max_steps: int,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
    ) -> List[Rollout]:
        """
        Run all rollouts described by `requests` and return them.
        """
        if not requests:
            return []

        sem = asyncio.Semaphore(max(1, concurrency))
        tasks: List[asyncio.Task[List[Rollout]]] = []

        global_idx = 0
        for req in requests:
            for _ in range(req.num_episodes):
                tasks.append(
                    asyncio.create_task(
                        self._run_one_request(
                            request=req,
                            episode_idx=global_idx,
                            sem=sem,
                            max_steps=max_steps,
                            timeout_s=timeout_s,
                        )
                    )
                )
                global_idx += 1

        results = await asyncio.gather(*tasks)
        # Flatten the list of lists (one list per episode -> single flat list of rollouts)
        return [r for sublist in results for r in sublist]

    # ---- SAW batch generation --------------------------------------------

    async def generate_batch(
        self,
        *,
        requests: List[RolloutRequest],
        max_steps: int,
        credit_assigner: CreditAssigner,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
        retokenize: bool = False,
        tokenize: Optional[TokenizeFn] = None,
    ) -> SAWBatch:
        """
        High-level entrypoint for RL-style training:
        
        1. Generates rollouts.
        2. Computes credit (advantages/rewards).
        3. Collates into SAWItems (handling tokenization/masking).
        
        Tokenization strategy:
        - If Step.info contains `prompt_token_ids` and `completion_token_ids`,
          those are used *unless* retokenize=True.
        - Otherwise, if retokenize=True, use provided tokenizer.
        - Else raise an error.
        """
        assert (not retokenize) or tokenize, (
            "Either use a chat client that populates token IDs, "
            "or pass a tokenizer if retokenize=True."
        )

        rollouts = await self.generate_rollouts(
            requests=requests,
            max_steps=max_steps,
            timeout_s=timeout_s,
            concurrency=concurrency,
        )
        weights = credit_assigner.compute(rollouts)

        items: List[SAWItem] = []

        for r in rollouts:
            for step in r.steps:
                key: RolloutStepKey = (r.id, step.index)

                try:
                    w_raw = weights[key]
                except KeyError as exc:
                    raise KeyError(
                        f"CreditAssigner did not provide a weight for "
                        f"(rollout_id={r.id!r}, step_index={step.index}). "
                        "All steps must be covered."
                    ) from exc

                w = float(w_raw)
                info = step.info or {}

                prompt_ids = info.get("prompt_token_ids")
                completion_ids = info.get("completion_token_ids")

                has_model_ids = (
                    isinstance(prompt_ids, list)
                    and isinstance(completion_ids, list)
                    and all(isinstance(t, int) for t in prompt_ids)
                    and all(isinstance(t, int) for t in completion_ids)
                )

                # Use model IDs only if they exist AND retokenize is False
                if has_model_ids and not retokenize:
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
                                "prev_obs": step.prev_obs,
                                "action": step.action,
                                "total_reward": r.total_reward,
                                **(r.meta),  # Rollout-level meta
                                **(step.info),  # Step-level info
                            },
                        )
                    )
                    continue

                if not retokenize:
                    raise ValueError(
                        f"Missing model token IDs for rollout {r.id}, step {step.index}, "
                        "and retokenize=False. Either enable retokenize=True or fix your "
                        "Agent/run_episode to store 'prompt_token_ids' and "
                        "'completion_token_ids' in Step.info."
                    )

                # Retokenize path
                state_text = step.prev_obs
                action_text = step.action

                state_ids = tokenize(state_text)  # type: ignore[arg-type]
                action_ids = tokenize(action_text)  # type: ignore[arg-type]

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
                            "prev_obs": step.prev_obs,
                            "action": step.action,
                            "total_reward": r.total_reward,
                            **(r.meta),  # Rollout-level meta
                            **(step.info),  # Step-level info
                        },
                    )
                )

        # ---- Build batch-level metadata -----------------------------------
        # Note: batch_size now reflects total number of *agent trajectories*, not global episodes.
        meta = {
            "batch_size": len(rollouts),
            "total_items": len(items),
            "avg_total_reward": (
                float(sum(r.total_reward for r in rollouts) / len(rollouts))
                if rollouts else 0.0
            ),
        }

        return SAWBatch(items=items, meta=meta)