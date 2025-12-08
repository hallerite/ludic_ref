from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from dataclasses import replace
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
    Stateless rollout executor:

      - Is configured with a ProtocolRegistry and an EnvRegistry.
      - For each rollout, it:
        1. Spawns a task.
        2. Calls the ProtocolRegistry to create a *new* protocol/agent worker
           based on the RolloutRequest.
        3. Calls the EnvRegistry to create a *new* env.
        4. Runs the episode and returns the rollout.
      - Optionally writes each rollout to JSONL

    Extended variant:

      - can build State–Action–Weight batches via `generate_batch`, using:
          * a CreditAssigner for credit assignment
          * model token IDs from Step.info when available
          * a fallback tokenizer otherwise

    Higher-level policies (curriculum, branching from snapshots, replay, etc.)
    should live in a BatchSource-like abstraction that decides which
    RolloutRequests to run. The RolloutEngine stays a dumb executor.
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

        episode_idx is a global index across all requests; purely for logging.
        This function is run concurrently in its own task.
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
            # Returns a LIST of rollouts (one per managed agent trace)
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

            # 6. The protocol and env go out of scope and are garbage collected
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

        Each RolloutRequest may specify a different env / protocol / sampling config
        and a different num_episodes, supporting heterogeneous batches.
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

        - runs all requested rollouts
        - computes weights via CreditAssigner
        - builds a State–Action–Weight batch

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


# ---------------------------------------------------------------------------
# Default BatchSource: on-policy rollouts via RolloutEngine
# ---------------------------------------------------------------------------


class RolloutBatchSource:
    """
    Default BatchSource that uses a RolloutEngine to generate on-policy
    rollouts each time `next_batch()` is called.

    - Delegates entirely to RolloutEngine.generate_batch(...)
    - For exotic behaviors (branching from snapshots, curricula, replay),
      write your own BatchSource instead of modifying Trainer.

    This class is intentionally a thin "policy" over:

        - which RolloutRequests to run this step
        - and how to configure max_steps / timeouts / tokenization
    """

    def __init__(
        self,
        *,
        orchestrator: RolloutEngine,
        credit_assigner: CreditAssigner,
        requests_fn: Callable[[], List[RolloutRequest]],
        max_steps: int,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
        retokenize: bool = False,
        tokenize: Optional[TokenizeFn] = None,
    ) -> None:
        if retokenize and tokenize is None:
            raise ValueError(
                "RolloutBatchSource: retokenize=True requires a tokenize() function."
            )

        self._engine = orchestrator
        self._credit_assigner = credit_assigner
        self._requests_fn = requests_fn
        self._max_steps = max_steps
        self._timeout_s = timeout_s
        self._concurrency = concurrency
        self._retokenize = retokenize
        self._tokenize = tokenize

    async def next_batch(self) -> SAWBatch:
        """
        Produce one SAWBatch by running fresh rollouts according to
        the current set of RolloutRequests.
        """
        requests = self._requests_fn()
        return await self._engine.generate_batch(
            requests=requests,
            max_steps=self._max_steps,
            credit_assigner=self._credit_assigner,
            timeout_s=self._timeout_s,
            concurrency=self._concurrency,
            retokenize=self._retokenize,
            tokenize=self._tokenize,
        )


# ---------------------------------------------------------------------------
# GRPO-specific BatchSource
# ---------------------------------------------------------------------------


class GRPOBatchSource:
    """
    A specialized BatchSource for GRPO (Group Relative Policy Optimization).

    This class wraps a user's simple `requests_fn` and handles the
    "G-sampling" logic automatically, making it much easier to use.

    - The user's `requests_fn` should return `N` requests, one for each
      prompt/group.
    - This class expands that list into `N * G` requests by:
        1. Forcing the same `env.reset(seed=...)` for all `G` requests
           in a group (using `RolloutRequest.seed`).
        2. Forcing a different `sampling_args["seed"]` for each of the `G`
           requests to ensure diverse outputs.
    """

    def __init__(
        self,
        *,
        orchestrator: RolloutEngine,
        credit_assigner: CreditAssigner,
        requests_fn: Callable[[], List[RolloutRequest]],
        group_size: int,  # The 'G' in GRPO
        max_steps: int,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
        retokenize: bool = False,
        tokenize: Optional[TokenizeFn] = None,
    ) -> None:
        if retokenize and tokenize is None:
            raise ValueError(
                "GRPOBatchSource: retokenize=True requires a tokenize() function."
            )
        if group_size <= 0:
            raise ValueError("GRPOBatchSource: group_size must be > 0.")

        self._engine = orchestrator
        self._credit_assigner = credit_assigner
        self._requests_fn = requests_fn
        self._group_size = group_size
        self._max_steps = max_steps
        self._timeout_s = timeout_s
        self._concurrency = concurrency
        self._retokenize = retokenize
        self._tokenize = tokenize

        # Used for generating unique seeds
        self._rng = random.Random()

    def _get_group_env_seed(self, base_req: RolloutRequest) -> int:
        """Determines the single env seed for a group."""
        if base_req.seed is not None:
            return base_req.seed
        # If no seed, create a new random one for this group
        return self._rng.randint(0, 2**32 - 1)

    def _expand_requests(
        self, base_requests: List[RolloutRequest]
    ) -> List[RolloutRequest]:
        """
        Expands a list of N base requests into N * G requests.
        """
        expanded_requests: List[RolloutRequest] = []

        for base_req in base_requests:
            # 1. Determine the single environment seed for this group
            group_env_seed = self._get_group_env_seed(base_req)

            # 2. Get the base sampling seed (if any)
            base_sampling_args = base_req.sampling_args or {}
            base_sampling_seed = base_sampling_args.get(
                "seed", self._rng.randint(0, 2**32 - 1)
            )

            # 3. Create G requests for this group
            for i in range(self._group_size):
                # Create new sampling args with a *different* seed
                new_sampling_args = {
                    **base_sampling_args,
                    "seed": base_sampling_seed + i,
                }

                # Create a copy of the request, forcing the
                # group env seed and new sampling args.
                new_req = replace(
                    base_req,
                    seed=group_env_seed,
                    sampling_args=new_sampling_args,
                    num_episodes=1,  # Each request is now 1 episode
                )
                expanded_requests.append(new_req)

        return expanded_requests

    async def next_batch(self) -> SAWBatch:
        """
        Produce one SAWBatch by generating N*G rollouts.
        """
        # 1. Get the N base requests from the user
        base_requests = self._requests_fn()

        # 2. Expand them to N * G requests
        expanded_requests = self._expand_requests(base_requests)

        # 3. Run the expanded batch
        return await self._engine.generate_batch(
            requests=expanded_requests,
            max_steps=self._max_steps,
            credit_assigner=self._credit_assigner,
            timeout_s=self._timeout_s,
            concurrency=self._concurrency,
            retokenize=self._retokenize,
            tokenize=self._tokenize,
        )