from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from ludic.env import Env
from ludic.context.base import ContextStrategy
from ludic.types import JSON, Rollout, SamplingArgs, Step


# ---------------------------------------------------------------------------
# Rollout-level configuration / identification
# ---------------------------------------------------------------------------

# (rollout_id, step_index)
RolloutStepKey = Tuple[str, int]


@dataclass
class RolloutRequest:
    """
    Configuration for a single rollout.

    Produced by a RolloutPolicy and consumed by the Orchestrator.
    """
    env: Env
    ctx: ContextStrategy
    sampling_args: SamplingArgs
    system_prompt: Optional[str]
    meta: Dict[str, JSON]


class RolloutPolicy(Protocol):
    """
    Controls how each rollout is configured (env, context, sampling, meta).
    """

    def make_rollout(self, episode_idx: int) -> RolloutRequest:
        ...


# ---------------------------------------------------------------------------
# Credit assignment / weighting
# ---------------------------------------------------------------------------


class WeightingStrategy(Protocol):
    """
    Computes a scalar weight for each (rollout, step) in a batch.
    """

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        ...


# ---------------------------------------------------------------------------
# State–Action–Weight representation
# ---------------------------------------------------------------------------


@dataclass
class SAWItem:
    """
    State–Action–Weight sample with masks.

    - input_ids: tokenized [state || action]
    - attention_mask: 1/0 attention mask
    - action_mask: 1 on action tokens, 0 on state tokens
    - weight: scalar credit for this sample
    - meta: arbitrary rollout/step metadata
    """
    input_ids: List[int]
    attention_mask: List[int]
    action_mask: List[int]
    weight: float
    meta: Dict[str, JSON]


# ---------------------------------------------------------------------------
# Helper aliases
# ---------------------------------------------------------------------------

TokenizeFn = Callable[[str], List[int]]
StateFromStepFn = Callable[[Rollout, int, Step], str]
