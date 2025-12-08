from __future__ import annotations
from typing import Optional, List

from ludic.envs.env import LudicEnv
from ludic.agent import Agent
from ludic.types import Rollout, Step, StepOutcome, SamplingArgs
from .base import InteractionProtocol

class SingleAgentSyncProtocol(InteractionProtocol):
    """
    Implements the standard single-agent, synchronous interaction loop.
    
    This protocol consumes a LudicEnv but ASSUMES it has exactly
    one agent and that this agent is active every step.
    
    It works perfectly with any env inheriting from SingleAgentEnv.
    """
    
    def __init__(self, agent: Agent, prompt: Optional[str] = None):
        """
        Initializes the protocol.
        
        Args:
            agent: A fully-configured Agent instance.
            prompt: An optional system prompt override. 
                    If set, this takes priority over the environment's 
                    suggested_sysprompt. This allows you to decouple 
                    prompt engineering from environment logic.
        """
        self.agent = agent
        self.prompt = prompt

    async def run(
        self,
        *,
        env: LudicEnv,
        max_steps: int,
        seed: Optional[int] = None,
        sampling_args: Optional[SamplingArgs] = None,
        timeout_s: Optional[float] = None,
    ) -> List[Rollout]:
        
        agent = self.agent
        sargs: SamplingArgs = sampling_args or {}

        # 1. --- Validate Env and get Agent ID ---
        agent_ids = env.agent_ids
        if len(agent_ids) != 1:
            raise ValueError(
                f"SingleAgentSyncProtocol requires a LudicEnv with "
                f"exactly one agent, but found {len(agent_ids)}."
            )
        agent_id = agent_ids[0]

        # 2. --- Reset Env ---
        # env.reset() returns a dict
        obs_info_dict = env.reset(seed=seed)
        obs, info = obs_info_dict[agent_id]
        
        # 3. --- Reset Agent & Feed First Obs ---
        # Protocol prompt takes priority over Env prompt
        if self.prompt is not None:
            sys_prompt = self.prompt
        else:
            sys_prompt = getattr(env, "suggested_sysprompt", None)

        agent.reset(system_prompt=sys_prompt)
        agent.on_env_reset(obs, info)
        
        # Accumulate steps locally first
        steps: List[Step] = []

        # 4. --- Run Interaction Loop ---
        for t in range(max_steps):
            
            # Check that our agent is the one expected to act
            active_agents = env.active_agents
            if agent_id not in active_agents:
                # This env is not a simple single-agent env, stop.
                break 

            current_obs_for_step = obs
            
            # --- A. Call the Agent ---
            parse_result, raw_action, client_info = await agent.act(
                sampling_args=sargs,
                timeout_s=timeout_s
            )

            # --- B. Handle Parser Failure ---
            if parse_result.action is None:
                synthetic_obs = parse_result.obs or "Invalid action."
                parser_reward = parse_result.reward

                # Create a synthetic outcome
                outcome = StepOutcome(
                    obs=synthetic_obs,
                    reward=parser_reward,
                    truncated=False,
                    terminated=False,
                    info={"parse_error": True, **client_info}
                )
                
                # Log this failure step
                steps.append(Step(
                    index=t,
                    prev_obs=current_obs_for_step,
                    action=raw_action,
                    next_obs=synthetic_obs,
                    reward=parser_reward,
                    truncated=False,
                    terminated=False,
                    info=outcome.info,
                ))

            # --- C. Handle Parser Success (Step Env) ---
            else:
                parsed_action = parse_result.action
                parser_reward = parse_result.reward

                # Send action to env in the required dict format
                actions_dict = {agent_id: parsed_action}
                outcomes_dict = env.step(actions_dict)

                # Unwrap the outcome for our agent
                env_outcome = outcomes_dict[agent_id]
                
                # Combine parser and env rewards
                total_reward = env_outcome.reward + parser_reward
                
                # Merge info dicts
                step_info = {
                    **client_info,
                    **env_outcome.info,
                    "parsed_action": parsed_action,
                }

                # Create the final, combined outcome
                outcome = StepOutcome(
                    obs=env_outcome.obs,
                    reward=total_reward,
                    truncated=env_outcome.truncated,
                    terminated=env_outcome.terminated,
                    info=step_info
                )

                # Log this success step
                logged_next_obs = None
                if not (outcome.terminated or outcome.truncated):
                    logged_next_obs = outcome.obs
                
                steps.append(Step(
                    index=t,
                    prev_obs=current_obs_for_step,
                    action=raw_action,
                    next_obs=logged_next_obs,
                    reward=total_reward,
                    truncated=outcome.truncated,
                    terminated=outcome.terminated,
                    info=step_info,
                ))

            # --- D. Update state for next loop ---
            obs = outcome.obs
            info = outcome.info

            if outcome.terminated or outcome.truncated:
                break # Exit loop
            else:
                # Feed the new observation to the agent
                agent.on_after_step(obs, info)

        rollout = Rollout(
            steps=steps,
            meta={
                "agent_name": getattr(agent, "name", "unknown"),
                "env_name": env.__class__.__name__,
            }
        )

        return [rollout]