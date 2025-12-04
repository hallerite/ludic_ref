from __future__ import annotations
import asyncio
from typing import Optional, Dict

from ludic.envs.env import LudicEnv
from ludic.agent import Agent
from ludic.types import Rollout, Step, StepOutcome, SamplingArgs
from .base import InteractionProtocol

class MultiAgentProtocol(InteractionProtocol):
    """
    Handles an interaction with multiple agents.
    
    This protocol queries the environment for which agents are
    'active' at each step and only gathers actions from them.
    """
    
    def __init__(self, agents: Dict[str, Agent]):
        """
        Initializes the protocol with a mapping of Agent IDs
        from the environment (e.g., "player_1") to the
        actual Agent objects that will fill those roles.
        """
        self.agent_map = agents

    async def run(
        self,
        *,
        env: LudicEnv,
        max_steps: int,
        seed: Optional[int] = None,
        sampling_args: Optional[SamplingArgs] = None,
        timeout_s: Optional[float] = None,
    ) -> Rollout:
        
        sargs: SamplingArgs = sampling_args or {}
        rollout = Rollout(meta={
            "protocol": self.__class__.__name__,
            "agents": list(self.agent_map.keys()),
            "env_name": env.__class__.__name__,
        })

        # 1. --- Reset Env and all managed Agents ---
        obs_info_dict = env.reset(seed=seed)
        sys_prompt = getattr(env, "suggested_sysprompt", None)
        
        for agent_id, agent in self.agent_map.items():
            obs, info = obs_info_dict.get(agent_id, (None, {}))
            if obs is not None:
                agent.reset(system_prompt=sys_prompt)
                agent.on_env_reset(obs, info)
        
        current_obs = {k: v[0] for k, v in obs_info_dict.items()}

        # 2. --- Run Interaction Loop ---
        for t in range(max_steps):
            
            # --- A. Identify active agents and gather actions ---
            active_ids = env.active_agents
            
            # Find which of the active agents are managed by this protocol
            agents_to_poll = {
                agent_id: self.agent_map[agent_id] 
                for agent_id in active_ids 
                if agent_id in self.agent_map
            }
            
            if not agents_to_poll:
                # No managed agents are active, this might be a
                # bot-vs-bot turn, or the game ended weirdly.
                break 

            # Gather actions in parallel
            tasks = [
                agent.act(sampling_args=sargs, timeout_s=timeout_s)
                for agent in agents_to_poll.values()
            ]
            results = await asyncio.gather(*tasks)
            
            actions_to_take: Dict[str, str] = {}
            raw_actions: Dict[str, str] = {}
            client_infos: Dict[str, dict] = {}
            parser_failures: Dict[str, StepOutcome] = {}
            
            for agent_id, (parse_result, raw, info) in zip(agents_to_poll.keys(), results):
                raw_actions[agent_id] = raw
                client_infos[agent_id] = info
                
                if parse_result.action is None:
                    # Parser failed. Create a synthetic outcome.
                    parser_failures[agent_id] = StepOutcome(
                        obs=parse_result.obs or "Invalid action.",
                        reward=parse_result.reward,
                        truncated=False,
                        terminated=False, # Parser failure shouldn't end the game
                        info={"parse_error": True, **info}
                    )
                else:
                    actions_to_take[agent_id] = parse_result.action

            # --- B. Step the environment ---
            # `actions_to_take` only contains actions from *this* protocol's
            # agents. The env kernel is responsible for handling any
            # unmanaged agents (e.g., internal bots).
            outcomes_dict = env.step(actions_to_take)

            # --- C. Merge outcomes and handle parser failures ---
            # If an agent failed parsing, its synthetic outcome
            # overrides the one from the environment.
            final_outcomes = {**outcomes_dict, **parser_failures}

            # --- D. Log the step (complex in MA, so we simplify) ---
            # We log a "flattened" view of the step.
            # This is a limitation of the current `Step` object.
            
            # Find the first active agent we manage to use for logging
            log_agent_id = next(iter(agents_to_poll.keys()), env.agent_ids[0])

            rollout.steps.append(Step(
                index=t,
                prev_obs=str(current_obs.get(log_agent_id, "")),
                action=str(raw_actions), # Log all raw actions
                next_obs=str(final_outcomes[log_agent_id].obs),
                reward=sum(o.reward for o in final_outcomes.values()),
                truncated=any(o.truncated for o in final_outcomes.values()),
                terminated=any(o.terminated for o in final_outcomes.values()),
                info={"client_infos": client_infos, "agent_outcomes": {
                    k: o.__dict__ for k, o in final_outcomes.items()
                }}
            ))
            
            # --- E. Check for termination ---
            if any(o.terminated or o.truncated for o in final_outcomes.values()):
                break
                
            # --- F. Feed new observations to agents ---
            current_obs = {}
            for agent_id, agent in self.agent_map.items():
                if agent_id in final_outcomes:
                    outcome = final_outcomes[agent_id]
                    current_obs[agent_id] = outcome.obs
                    if not (outcome.terminated or outcome.truncated):
                        agent.on_after_step(outcome.obs, outcome.info)
                        
        return rollout