from __future__ import annotations
from collections import defaultdict
import uuid
from typing import Dict, List, Any

from ludic.types import Rollout, Step

class TraceCollector:
    """
    Accumulates local histories for multiple agents during a single global episode.
    
    This acts as a buffer between the raw execution of a multi-agent protocol
    and the strict, flattened `Rollout` format required by the Trainer.
    """
    
    def __init__(self, **global_meta: Any) -> None:
        """
        Args:
            **global_meta: Metadata shared across all rollouts generated 
                           from this episode (e.g. env_name, protocol_name).
        """
        self._traces: Dict[str, List[Step]] = defaultdict(list)
        self._global_meta = global_meta

    def add(self, agent_id: str, step: Step) -> None:
        """
        Record one completely processed step for a specific agent.
        
        Args:
            agent_id: The identifier for the agent (must match the ID used in
                      Rollout.meta["agent_id"]).
            step: The Step object containing ONLY that agent's view:
                  - prev_obs (what THEY saw)
                  - action (what THEY output, as a raw string)
                  - reward (what THEY received)
        """
        self._traces[agent_id].append(step)

    def extract_rollouts(self) -> List[Rollout]:
        """
        Convert all collected traces into separate, flat Rollout objects.
        
        Each Rollout represents the single-agent trajectory of one participant
        in the multi-agent episode.
        
        Returns:
            A list of Rollout objects, one per agent that generated at least one step.
        """
        rollouts = []
        for agent_id, steps in self._traces.items():
            if not steps:
                continue
            
            # Create a clean, single-agent rollout
            # We treat each agent's trace as a distinct episode for training purposes.
            r = Rollout(
                id=str(uuid.uuid4()),
                steps=steps,
                meta={
                    **self._global_meta,
                    "agent_id": agent_id,
                }
            )
            rollouts.append(r)
            
        return rollouts