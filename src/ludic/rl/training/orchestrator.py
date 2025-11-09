from __future__ import annotations
import asyncio, json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, List

from ludic.agent.base import Agent
from ludic.env import Env
from ludic.interaction import run_episode
from ludic.types import Rollout, SamplingArgs
from ludic.context.base import ContextStrategy
from ludic.context.full_dialog import FullDialog

# Build a fresh Env each episode
EnvFactory = Callable[..., Env]
CtxFactory = Callable[[], ContextStrategy]

@dataclass
class OrchestratorConfig:
    episodes: int
    max_steps: int = 64
    sampling_args: Optional[SamplingArgs] = None
    concurrency: int = 8
    timeout_s: Optional[float] = None
    system_prompt: Optional[str] = None
    jsonl_path: Optional[str] = None  # if set, append each rollout as JSONL

class Orchestrator:
    """
    Dumb, reliable orchestrator:
      - spawns N episodes
      - runs them with an asyncio.Semaphore
      - returns List[Rollout]
      - optionally writes each rollout to JSONL
    """

    def __init__(
        self,
        env_factory: EnvFactory,
        agent: Agent,
        *,
        cfg: OrchestratorConfig,
        ctx_factory: Optional[CtxFactory] = None,
    ) -> None:
        self.env_factory = env_factory
        self.agent = agent
        self.cfg = cfg
        self.ctx_factory = ctx_factory or (lambda: FullDialog())

        if self.cfg.jsonl_path:
            Path(os.path.dirname(self.cfg.jsonl_path) or ".").mkdir(parents=True, exist_ok=True)

    async def _run_one(self, idx: int, sem: asyncio.Semaphore) -> Rollout:
        async with sem:
            env = self.env_factory()
            ctx = self.ctx_factory()
            rollout = await run_episode(
                env=env,
                agent=self.agent,
                max_steps=self.cfg.max_steps,
                sampling_args=self.cfg.sampling_args or {},
                ctx=ctx,
                system_prompt=self.cfg.system_prompt,
                timeout_s=self.cfg.timeout_s,
            )
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

    async def generate(self) -> List[Rollout]:
        sem = asyncio.Semaphore(max(1, self.cfg.concurrency))
        tasks = [self._run_one(i, sem) for i in range(self.cfg.episodes)]
        return await asyncio.gather(*tasks)

    def generate_sync(self) -> List[Rollout]:
        try:
            loop = asyncio.get_running_loop()
            # If already in an event loop (e.g., notebook), enable nested run
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            return loop.run_until_complete(self.generate())  # type: ignore[return-value]
        except RuntimeError:
            return asyncio.run(self.generate())
