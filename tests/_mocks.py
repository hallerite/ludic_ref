from __future__ import annotations

import torch

from typing import Any, Optional, List, Tuple, Mapping, Dict

from ludic.types import Message, StepOutcome, Observation, Info
from ludic.inference.client import ChatResponse  # protocol impl not required
from ludic.inference.sampling import SamplingConfig
from ludic.agent import Agent
from ludic.env import Env

# ---- Mock client ---------------------------------------------------------

class MockClient:
    def __init__(self, text: str = "1") -> None:
        self._text = text

    async def complete(
        self,
        *,
        model: str,
        messages: List[Message],
        sampling: SamplingConfig,
    ) -> tuple[ChatResponse, Dict[str, Any]]:
        return ChatResponse(text=self._text), {"used_args": sampling}

    def push_update_atomic(self, params: Mapping[str, torch.Tensor], **kwargs) -> str:  # type: ignore[name-defined]
        return "mock-version"

class MockAgent(Agent):
    """
    Real Agent wired to the MockClient.
    No need to override act(); base Agent uses client.complete().
    """
    def __init__(self) -> None:
        super().__init__(client=MockClient(), model="mock")


# ---- Mock env ------------------------------------------------------------

class MockEnv(Env):
    """
    Minimal env for exercising the interaction loop only.

    - reset() -> returns a fixed instruction.
    - step("1") -> reward=1.0, terminated=True, obs="✅ done".
    - step(other) -> reward=-0.1, keep going with obs="❌ try again {t}/{max}".
    - Truncates after `max_steps`.
    - No snapshot support on purpose.
    """

    def __init__(self, *, max_steps: int = 8, target: str = "1") -> None:
        self.max_steps = max_steps
        self.target = target
        self._t = 0
        self._obs: Observation = "Reply with '1' to finish."

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return "You are terse. Output only the final answer."

    def reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        self._t = 0
        self._obs = "Reply with '1' to finish."
        return self._obs, {}

    def step(self, action: str) -> StepOutcome:
        if self._t >= self.max_steps:
            return StepOutcome(obs=self._obs, reward=0.0, truncated=True, terminated=False, info={})

        self._t += 1
        if action.strip() == self.target:
            self._obs = "✅ done"
            return StepOutcome(obs=self._obs, reward=1.0, truncated=False, terminated=True, info={})

        self._obs = f"❌ try again {self._t}/{self.max_steps}"
        truncated = self._t >= self.max_steps
        return StepOutcome(
            obs=self._obs, reward=-0.1, truncated=truncated, terminated=False, info={"attempt": self._t}
        )

    def current_obs(self) -> Observation:
        return self._obs
