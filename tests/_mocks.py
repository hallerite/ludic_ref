from __future__ import annotations

import torch

from typing import Any, Optional, List, Tuple, Mapping, Dict
from dataclasses import asdict

from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.types import Message, StepOutcome, Observation, Info
from ludic.inference.client import ChatResponse, ChatClient
from ludic.inference.sampling import SamplingConfig
from ludic.agents.base_agent import Agent
from ludic.context.base import ContextStrategy
from ludic.context.full_dialog import FullDialog
from ludic.parsers import Parser, ParseResult

# ---- A default pass-through parser for mocks ----
def _mock_parser(raw: str) -> ParseResult:
    """A simple parser that just passes the text through."""
    return ParseResult(action=raw, reward=0.0, obs=None)

# ---- Mock client ---------------------------------------------------------

class MockClient(ChatClient):
    def __init__(self, text: str = "1") -> None:
        self._text = text

    async def complete(
        self,
        *,
        model: str,
        messages: List[Message],
        sampling: SamplingConfig,
    ) -> tuple[ChatResponse, Dict[str, Any]]:
        return ChatResponse(text=self._text), {"used_args": asdict(sampling)}

    def sync_weights(self, params: Mapping[str, torch.Tensor], **kwargs) -> str:  # type: ignore[name-defined]
        return "mock-version"

class MockAgent(Agent):
    """
    Real Agent wired to the MockClient.
    Uses FullDialog and a pass-through parser by default.
    """
    def __init__(
        self, 
        client: ChatClient = MockClient(), 
        model: str = "mock",
        ctx: ContextStrategy = FullDialog(),
        parser: Parser = _mock_parser
    ) -> None:
        super().__init__(client=client, model=model, ctx=ctx, parser=parser)


# ---- Seedable Mock Client for GRPO Test ----

class SeedableMockClient(ChatClient):
    """
    A mock client that returns a deterministic response based on the
    sampling_seed provided. Also returns mock token IDs.
    """
    def __init__(self, seed_map: Dict[int, str]) -> None:
        self._seed_map = seed_map

    async def complete(
        self,
        *,
        model: str,
        messages: List[Message],
        sampling: SamplingConfig,
    ) -> tuple[ChatResponse, Dict[str, Any]]:
        
        # Get the deterministic text output based on the sampling seed
        text_out = self._seed_map.get(sampling.seed, "DEFAULT_FALLBACK")
        
        resp = ChatResponse(
            text=text_out,
            # Add mock token IDs so retokenize=False path passes
            prompt_token_ids=[1, 2, 3],
            completion_token_ids=[10, 11] # Action
        )
        # Return the serializable dict, not the object
        return resp, {"used_args": asdict(sampling)}

    def sync_weights(self, params: Mapping[str, torch.Tensor], **kwargs) -> str:
        # Not needed for this test
        return "mock-version"

class SeedableMockAgent(Agent):
    """
    An agent wired to the SeedableMockClient.
    """
    def __init__(
        self, 
        seed_map: Dict[int, str],
        ctx: ContextStrategy = FullDialog(),
        parser: Parser = _mock_parser
    ) -> None:
        super().__init__(
            client=SeedableMockClient(seed_map), 
            model="seedable_mock",
            ctx=ctx,
            parser=parser
        )


# ---- Mock env ------------------------------------------------------------

class MockEnv(SingleAgentEnv):
    """
    Minimal env for exercising the interaction loop only.

    - reset() -> returns a fixed instruction.
    - step("1") -> reward=1.0, terminated=True, obs="✅ done".
    - step(other) -> reward=-0.1, keep going with obs="❌ try again {t}/{max}".
    - Truncates after `max_steps`.
    - No snapshot support on purpose.
    """

    def __init__(self, *, max_steps: int = 8, target: str = "1") -> None:
        super().__init__()  # <-- Added call to super()
        self.max_steps = max_steps
        self.target = target
        self._t = 0
        self._obs: Observation = "Reply with '1' to finish."

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return "You are terse. Output only the final answer."

    def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:  # <-- Renamed
        self._t = 0
        self._obs = "Reply with '1' to finish."
        return self._obs, {}

    def env_step(self, action: str) -> StepOutcome:  # <-- Renamed
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

    def env_current_obs(self) -> Observation:  # <-- Renamed
        return self._obs