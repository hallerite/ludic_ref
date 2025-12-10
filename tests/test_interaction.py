from typing import Dict, List, Tuple, Optional
import pytest

from ludic.context.full_dialog import FullDialog
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.interaction.multi_agent import MultiAgentProtocol
from ludic.agent import Agent
from ludic.inference.client import ChatResponse
from ludic.parsers import (
    cot_prefix_parser,
    xml_move_parser,
    compose_parsers,
    Parser,
)
from ludic.types import StepOutcome, Info
from ludic.envs.env import LudicEnv
from tests._mocks import MockEnv, MockClient, MockAgent


# ---------------------------------------------------------------------
# Basic env/agent termination cases (Single Agent)
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_terminates_immediately():
    env = MockEnv(max_steps=3, target="1")
    # MockAgent provides a default ctx and a pass-through parser
    agent = MockAgent(client=MockClient(text="1"))
    protocol = SingleAgentSyncProtocol(agent=agent)

    # run() now returns List[Rollout]
    rollouts = await protocol.run(
        env=env,
        max_steps=5,
        sampling_args={},
    )

    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.steps[-1].terminated is True
    assert rollout.total_reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_truncation_when_agent_is_wrong():
    class WrongClient(MockClient):
        async def complete(self, *, model, messages, sampling, **kwargs):
            return ChatResponse(text="nope"), {"used_args": sampling}

    env = MockEnv(max_steps=2, target="1")
    agent = MockAgent(client=WrongClient())
    protocol = SingleAgentSyncProtocol(agent=agent)

    rollouts = await protocol.run(
        env=env,
        max_steps=10,
        sampling_args={},
    )

    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.steps[-1].truncated is True
    assert rollout.total_reward < 0.0


# ---------------------------------------------------------------------
# Parser integration test
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_episode_uses_action_parser_and_logs_parsed_action():
    """
    Ensure that:
      - protocol.run() uses the agent's configured parser
      - Step.action keeps the raw LLM text
      - Step.info['parsed_action'] is the parsed action
      - parser reward is added to env reward
      - env.step receives the parsed action
    """

    env = MockEnv(max_steps=3, target="A1")

    # LLM emits a valid CoT-prefixed XML move
    raw_llm_output = "<think>some reasoning</think>\n<move>  A1  </move>"

    action_parser: Parser = compose_parsers(
        cot_prefix_parser,
        xml_move_parser,
    )

    agent = Agent(
        client=MockClient(text=raw_llm_output),
        model="mock",
        ctx=FullDialog(),
        parser=action_parser
    )
    
    protocol = SingleAgentSyncProtocol(agent=agent)

    rollouts = await protocol.run(
        env=env,
        max_steps=5,
        sampling_args={},
    )

    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.length >= 1
    step = rollout.steps[-1]

    # Raw LLM text must be preserved in the main action field
    assert "<think>" in step.action
    assert "<move>" in step.action

    # Parsed action must be logged in info
    assert step.info["parsed_action"] == "A1"

    # Env should terminate because parsed action == target "A1"
    assert step.terminated is True

    # Parser reward: 0.0 ; Env reward: 1.0
    assert rollout.total_reward == pytest.approx(1.0)


# ---------------------------------------------------------------------
# Multi-Agent Tests
# ---------------------------------------------------------------------

class MockTurnBasedEnv(LudicEnv[str, str, str]):
    """
    A simple 2-agent turn-based env for testing MultiAgentProtocol.
    Agent A goes, then Agent B goes.
    """
    def __init__(self):
        self._turn = 0
        self._agents = ["agent_A", "agent_B"]
    
    @property
    def agent_ids(self) -> List[str]:
        return self._agents

    @property
    def active_agents(self) -> List[str]:
        # Return the ID of the agent whose turn it is
        return [self._agents[self._turn % 2]]

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, Tuple[str, Info]]:
        self._turn = 0
        return {
            "agent_A": ("Start A", {}),
            "agent_B": ("Start B", {})
        }

    def step(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
        active = self.active_agents[0]
        
        # In this mock, we just terminate after Agent B (turn 1) acts.
        # Turn 0: A acts
        # Turn 1: B acts -> terminate
        
        terminated = (self._turn >= 1)
        reward = 1.0 if active == "agent_A" else 2.0
        
        outcome = StepOutcome(
            obs=f"Obs for {active} after turn {self._turn}",
            reward=reward,
            truncated=False,
            terminated=terminated,
            info={"turn": self._turn}
        )
        
        self._turn += 1
        return {active: outcome}


@pytest.mark.asyncio
async def test_multi_agent_returns_multiple_rollouts():
    """
    Test that MultiAgentProtocol correctly generates distinct rollouts
    for different agents in the same episode.
    """
    env = MockTurnBasedEnv()
    
    agent_A = MockAgent(client=MockClient(text="Move A"))
    agent_B = MockAgent(client=MockClient(text="Move B"))
    
    agents = {"agent_A": agent_A, "agent_B": agent_B}
    protocol = MultiAgentProtocol(agents)

    # Run for enough steps to let both act
    rollouts = await protocol.run(env=env, max_steps=5)

    # Expecting 2 rollouts: one for A, one for B
    assert len(rollouts) == 2
    
    # Identify which is which
    r_a = next(r for r in rollouts if r.meta["agent_id"] == "agent_A")
    r_b = next(r for r in rollouts if r.meta["agent_id"] == "agent_B")

    # Verify Agent A's trace
    # Agent A acted at global turn 0.
    assert len(r_a.steps) == 1
    assert r_a.steps[0].action == "Move A"
    assert r_a.steps[0].reward == 1.0
    assert r_a.steps[0].prev_obs == "Start A"

    # Verify Agent B's trace
    # Agent B acted at global turn 1.
    assert len(r_b.steps) == 1
    assert r_b.steps[0].action == "Move B"
    assert r_b.steps[0].reward == 2.0
    # Note: B's prev_obs comes from reset(), as B hasn't seen an outcome yet when it acts
    assert r_b.steps[0].prev_obs == "Start B"

class MockMixedBotEnv(LudicEnv[str, str, str]):
    """
    Turn 0: 'bot' (unmanaged) acts.
    Turn 1: 'hero' (managed) acts.
    """
    def __init__(self):
        self._turn = 0
    
    @property
    def agent_ids(self) -> List[str]:
        return ["bot", "hero"]

    @property
    def active_agents(self) -> List[str]:
        return ["bot"] if self._turn == 0 else ["hero"]

    def reset(self, *, seed=None) -> Dict[str, Tuple[str, Info]]:
        self._turn = 0
        return {"hero": ("Start", {}), "bot": ("Start", {})}

    def step(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
        # If it's the bot's turn, actions should be empty coming from the protocol
        if self._turn == 0:
            assert actions == {}, "Protocol should not send actions for unmanaged bot"
            outcome = StepOutcome(obs="Bot attacked!", reward=0.0, truncated=False, terminated=False, info={})
            self._turn = 1
            # Both see the bot's move
            return {"bot": outcome, "hero": outcome}
        
        # Hero's turn
        if self._turn == 1:
            assert "hero" in actions
            outcome = StepOutcome(obs="Hero won", reward=10.0, truncated=False, terminated=True, info={})
            self._turn = 2
            return {"hero": outcome}
            
        return {}

@pytest.mark.asyncio
async def test_multi_agent_handles_unmanaged_bot_turns():
    """
    Ensure the protocol steps the env with empty actions when only
    unmanaged agents (bots) are active, but still updates managed agents.
    """
    env = MockMixedBotEnv()
    hero_agent = MockAgent(client=MockClient(text="Hero Attack"))
    
    # We only manage the 'hero', not the 'bot'
    protocol = MultiAgentProtocol(agents={"hero": hero_agent})

    rollouts = await protocol.run(env=env, max_steps=5)

    assert len(rollouts) == 1
    r = rollouts[0]
    
    assert r.meta["agent_id"] == "hero"
    # Hero should have 1 step (Turn 1), but the prev_obs should reflect Turn 0
    assert len(r.steps) == 1
    
    step = r.steps[0]
    # The hero's observation *before* acting should be the result of the bot's turn
    assert step.prev_obs == "Bot attacked!"
    assert step.action == "Hero Attack"
    assert step.reward == 10.0
    assert step.terminated is True