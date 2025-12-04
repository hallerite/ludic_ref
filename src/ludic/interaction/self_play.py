from __future__ import annotations
from typing import Optional

from ludic.envs.env import LudicEnv
from ludic.agent import Agent
from ludic.types import Rollout, SamplingArgs
from .base import InteractionProtocol
from .multi_agent import MultiAgentProtocol

class SelfPlayProtocol(InteractionProtocol):
    """
    A "policy" protocol that makes a single agent play against
    itself in any multi-agent environment.
    
    It maps the *same* Agent object to *all* agent roles
    defined by the environment.
    """
    
    def __init__(self, agent: Agent):
        """
        Initializes the protocol with the single agent
        that will be used for all roles.
        """
        self.agent = agent

    async def run(
        self,
        *,
        env: LudicEnv,
        max_steps: int,
        seed: Optional[int] = None,
        sampling_args: Optional[SamplingArgs] = None,
        timeout_s: Optional[float] = None,
    ) -> Rollout:
        
        # 1. Get all roles from the environment
        agent_ids = env.agent_ids
        if len(agent_ids) < 2:
            raise ValueError(
                f"SelfPlayProtocol requires an environment with "
                f"at least two agents, but {env.__class__.__name__} "
                f"only has {len(agent_ids)}."
            )
            
        # 2. Map the *same* agent to *all* roles
        agent_map = {agent_id: self.agent for agent_id in agent_ids}
        
        # 3. Create and delegate to the standard MultiAgentProtocol
        protocol = MultiAgentProtocol(agents=agent_map)
        
        rollout = await protocol.run(
            env=env,
            max_steps=max_steps,
            seed=seed,
            sampling_args=sampling_args,
            timeout_s=timeout_s
        )
        
        # Add self-play metadata
        rollout.meta["protocol"] = self.__class__.__name__
        return rollout