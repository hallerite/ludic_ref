from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Union, Optional, TypedDict
import time
import uuid
import json

JSON = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


# ----- Agent level types -----

class SamplingArgs(TypedDict, total=False):
    seed: int
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: Union[str, List[str]]
    extras: dict[str, Any] # for vendor-specific stuff

# Chat-style message schema
Message = Dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


# ----- Environment level types -----

Observation = str
Info   = Dict[str, JSON]

@dataclass
class StepOutcome:
    obs: str
    reward: float
    truncated: bool
    terminated: bool
    info: Info = field(default_factory=dict)

@dataclass(frozen=True)
class Snapshot:
    """
    A portable checkpoint of an Environment, not just the outward State.
    `env_kind` and `version` let you refuse incompatible restores.
    """
    env_kind: str
    version: str           # bump on breaking schema changes
    episode_id: str        # source episode lineage
    obs: str           # outward state snapshot
    world: Dict[str, JSON] # extra hidden/internal stuff needed to resume
    created_ns: int = field(default_factory=lambda: time.time_ns())

    def to_json(self) -> str:
        return json.dumps(asdict(self))

# ----- Environment-Agent-Interaction level types -----

@dataclass
class Step:
    index: int
    prev_obs: Observation
    action: str
    next_obs: Optional[Observation]  # may be None on terminal steps
    reward: float
    truncated: bool
    terminated: bool
    info: Info = field(default_factory=dict)
    ts_ns: int = field(default_factory=lambda: time.time_ns())

@dataclass
class Rollout:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Step] = field(default_factory=list)
    meta: Dict[str, JSON] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def start_ns(self) -> Optional[int]:
        return self.steps[0].ts_ns if self.steps else None

    @property
    def end_ns(self) -> Optional[int]:
        return self.steps[-1].ts_ns if self.steps else None

    @property
    def duration_ns(self) -> Optional[int]:
        start = self.start_ns
        end = self.end_ns
        if start is None or end is None:
            return None
        return end - start

    @property
    def duration_s(self) -> Optional[float]:
        ns = self.duration_ns
        return None if ns is None else ns / 1e9
