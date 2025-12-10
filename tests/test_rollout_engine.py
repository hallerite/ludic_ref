from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional

import pytest

from ludic.agent import Agent
from ludic.inference.client import ChatResponse
from ludic.interaction.base import InteractionProtocol
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.context.full_dialog import FullDialog
from ludic.envs.env import LudicEnv
from ludic.inference.sampling import SamplingConfig

from ludic.training.batching import (
    RolloutEngine,
    RolloutBatchSource,
)
from ludic.training.types import (
    EnvSpec,
    ProtocolSpec,
    RolloutRequest,
)
from ludic.training.credit_assignment import MonteCarloReturn
from ludic.types import Rollout, SamplingArgs, Step

from tests._mocks import MockClient, _mock_parser, MockAgent

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# ---------------------------------------------------------------------
# Helper / local mocks
# ---------------------------------------------------------------------

class TokenClient(MockClient):
    """
    Mock client that returns both prompt and completion token IDs so that
    RolloutEngine.generate_batch can consume them without retokenization.
    """

    async def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        sampling: SamplingConfig,
        **kwargs,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        # Prompt is "some prompt", completion is "1".
        # Tokens are arbitrary but deterministic.
        resp = ChatResponse(
            text="1",
            prompt_token_ids=[10, 11],
            completion_token_ids=[12, 13, 14],
        )
        return resp, {"used_args": sampling.to_openai_kwargs()}


class ConstantCreditAssigner:
    """
    Simple CreditAssigner that gives a constant weight to every step,
    just to exercise the plumbing in generate_batch.
    """

    def __init__(self, value: float = 1.23) -> None:
        self.value = value

    def compute(self, rollouts: List[Rollout]) -> Dict[Tuple[str, int], float]:
        out: Dict[Tuple[str, int], float] = {}
        for r in rollouts:
            for s in r.steps:
                out[(r.id, s.index)] = self.value
        return out


def fake_tokenize(text: str) -> List[int]:
    """
    Extremely dumb tokenizer for retokenize=True path:
    converts each character to its ord().
    """
    return [ord(c) for c in text]

# ---------------------------------------------------------------------
# Mock Protocol that produces MULTIPLE rollouts per run()
# ---------------------------------------------------------------------

class MultiTraceMockProtocol(InteractionProtocol):
    """A mock protocol that simulates 2 agents generating traces in 1 episode."""
    async def run(
        self,
        *,
        env: LudicEnv,
        max_steps: int,
        seed: Optional[int] = None,
        sampling_args: Optional[SamplingArgs] = None,
        timeout_s: Optional[float] = None,
    ) -> List[Rollout]:
        # Simulate Agent A
        r1 = Rollout(steps=[Step(0, "obsA", "actA", "nextA", 1.0, False, True)])
        # Simulate Agent B
        r2 = Rollout(steps=[Step(0, "obsB", "actB", "nextB", -0.5, False, True)])
        return [r1, r2]

# ---------------------------------------------------------------------
# generate_rollouts: env/ctx registry + metadata + concurrency
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_rollouts_basic_metadata_and_termination(
    env_registry,
    mock_agent,
) -> None:
    protocol_registry: ProtocolRegistry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=mock_agent)
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={"max_steps": 3, "target": "1"}),
        protocol=ProtocolSpec(kind="mock_protocol"),
        num_episodes=2,
        meta={"tag": "test"},
    )

    # Calling with single-agent protocol, should get 2 rollouts (1 per episode)
    rollouts = await engine.generate_rollouts(
        requests=[request],
        max_steps=5,
        timeout_s=None,
        concurrency=4,
    )

    assert len(rollouts) == 2

    seen_episode_idx = set()

    for r in rollouts:
        # should terminate successfully on correct action "1"
        assert r.steps[-1].terminated is True
        assert r.total_reward == pytest.approx(1.0)

        # engine metadata
        assert r.meta["engine"]["max_steps"] == 5
        assert r.meta["engine"]["timeout_s"] is None
        assert r.meta["engine"]["env_kind"] == "mock"
        assert r.meta["engine"]["protocol_kind"] == "mock_protocol"

        # request-level metadata propagated
        assert r.meta["request_meta"]["tag"] == "test"

        # unique episode indices
        episode_idx = r.meta["episode_idx"]
        assert isinstance(episode_idx, int)
        seen_episode_idx.add(episode_idx)

    assert seen_episode_idx == {0, 1}


@pytest.mark.asyncio
async def test_generate_rollouts_flattens_multi_trace_protocols(
    env_registry,
) -> None:
    """
    Verifies that if a protocol returns [RolloutA, RolloutB] for one episode,
    the engine correctly flattens them into the result list.
    """
    protocol_registry = {
        "multi_trace": lambda: MultiTraceMockProtocol()
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={}),
        protocol=ProtocolSpec(kind="multi_trace"),
        num_episodes=2, # 2 Global Episodes
    )

    # 2 Episodes * 2 Traces per Episode = 4 Total Rollouts
    rollouts = await engine.generate_rollouts(
        requests=[request],
        max_steps=5,
    )

    assert len(rollouts) == 4
    
    # Check that we have 2 from episode 0 and 2 from episode 1
    ep_indices = [r.meta["episode_idx"] for r in rollouts]
    assert ep_indices.count(0) == 2
    assert ep_indices.count(1) == 2


@pytest.mark.asyncio
async def test_generate_rollouts_unknown_env_raises(
    env_registry,
    mock_agent,
) -> None:
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=mock_agent)
    }
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    request = RolloutRequest(
        env=EnvSpec(kind="does_not_exist", kwargs={}),
        protocol=ProtocolSpec(kind="mock_protocol"),
        num_episodes=1,
    )

    with pytest.raises(KeyError, match="Unknown env kind"):
        await engine.generate_rollouts(
            requests=[request],
            max_steps=3,
        )


@pytest.mark.asyncio
async def test_generate_rollouts_unknown_protocol_raises(
    env_registry,
) -> None:
    """Verifies that an unknown protocol kind raises a KeyError."""
    protocol_registry = {
        # No protocols registered
    }
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={}),
        protocol=ProtocolSpec(kind="does_not_exist"),
        num_episodes=1,
    )

    with pytest.raises(KeyError, match="Unknown protocol kind"):
        await engine.generate_rollouts(
            requests=[request],
            max_steps=3,
        )


@pytest.mark.asyncio
async def test_generate_rollouts_heterogeneous_protocols(
    env_registry,
) -> None:
    """
    Tests that the RolloutEngine can run a batch containing
    requests for two different protocols.
    """
    # Define two different agent/protocol setups
    agent_A = MockAgent(client=MockClient(text="Agent A says hi"))
    protocol_A = SingleAgentSyncProtocol(agent=agent_A)

    agent_B = MockAgent(client=MockClient(text="Agent B says hi"))
    protocol_B = SingleAgentSyncProtocol(agent=agent_B)

    protocol_registry = {
        "protocol_A": lambda: protocol_A,
        "protocol_B": lambda: protocol_B,
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    # Request 1 for Protocol A
    req_A = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={}),
        protocol=ProtocolSpec(kind="protocol_A"),
        num_episodes=1,
    )
    
    # Request 2 for Protocol B
    req_B = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={}),
        protocol=ProtocolSpec(kind="protocol_B"),
        num_episodes=1,
    )

    rollouts = await engine.generate_rollouts(
        requests=[req_A, req_B],
        max_steps=2,
        concurrency=2,
    )

    assert len(rollouts) == 2

    # Check that the rollouts were generated by the correct protocols
    protocol_kinds = sorted([r.meta["engine"]["protocol_kind"] for r in rollouts])
    assert protocol_kinds == ["protocol_A", "protocol_B"]

    # Check that the correct agent was used for each
    for r in rollouts:
        if r.meta["engine"]["protocol_kind"] == "protocol_A":
            assert r.steps[0].action == "Agent A says hi"
        elif r.meta["engine"]["protocol_kind"] == "protocol_B":
            assert r.steps[0].action == "Agent B says hi"


@pytest.mark.asyncio
async def test_generate_rollouts_writes_jsonl(
    tmp_path,
    env_registry,
    mock_agent,
) -> None:
    jsonl_path = tmp_path / "rollouts.jsonl"
    
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=mock_agent)
    }

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=str(jsonl_path),
    )

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={"max_steps": 2, "target": "1"}),
        protocol=ProtocolSpec(kind="mock_protocol"),
        num_episodes=2,
        meta={"foo": "bar"},
    )

    rollouts = await engine.generate_rollouts(
        requests=[request],
        max_steps=4,
    )

    assert len(rollouts) == 2
    assert jsonl_path.exists()

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    for line in lines:
        payload = json.loads(line)
        assert "id" in payload
        assert "meta" in payload
        assert "steps" in payload
        assert payload["meta"]["request_meta"]["foo"] == "bar"
        assert payload["meta"]["engine"]["env_kind"] == "mock"
        assert payload["meta"]["engine"]["protocol_kind"] == "mock_protocol"
        assert payload["length"] == len(payload["steps"])


# ---------------------------------------------------------------------
# generate_batch: with model token IDs (no retokenize)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_batch_uses_model_token_ids_when_available(
    env_registry,
) -> None:
    # Agent whose client returns prompt_token_ids + token_ids
    agent = Agent(
        client=TokenClient(),
        model="mock",
        ctx=FullDialog(),
        parser=_mock_parser
    )
    
    protocol_registry = {
        "token_protocol": lambda: SingleAgentSyncProtocol(agent=agent)
    }

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    credit_assigner = ConstantCreditAssigner(value=1.23)

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={"max_steps": 1, "target": "1"}),
        protocol=ProtocolSpec(kind="token_protocol"),
        num_episodes=1,
    )

    batch = await engine.generate_batch(
        requests=[request],
        max_steps=3,
        credit_assigner=credit_assigner,
        timeout_s=None,
        concurrency=1,
        retokenize=False,  # MUST not retokenize; should use model IDs
    )

    # Single rollout, single step
    assert batch.meta["batch_size"] == 1
    assert batch.meta["total_items"] == 1
    # Env gives +1 reward when guess is correct
    assert batch.meta["avg_total_reward"] == pytest.approx(1.0)

    assert len(batch.items) == 1
    item = batch.items[0]

    # IDs must be prompt_ids + completion_ids
    assert item.input_ids == [10, 11, 12, 13, 14]
    assert item.attention_mask == [1, 1, 1, 1, 1]
    assert item.action_mask == [0, 0, 1, 1, 1]
    assert item.weight == pytest.approx(1.23)

    # meta carries rollout/step info
    assert "rollout_id" in item.meta
    assert "step_index" in item.meta
    assert item.meta["total_reward"] == pytest.approx(1.0)


# ---------------------------------------------------------------------
# generate_batch: missing token IDs and retokenize behaviour
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_batch_raises_if_no_token_ids_and_no_retokenize(
    env_registry,
    mock_agent,
) -> None:
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=mock_agent)
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    credit_assigner = ConstantCreditAssigner(value=1.0)

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={"max_steps": 1, "target": "1"}),
        protocol=ProtocolSpec(kind="mock_protocol"),
        num_episodes=1,
    )

    with pytest.raises(ValueError, match="Missing model token IDs"):
        await engine.generate_batch(
            requests=[request],
            max_steps=2,
            credit_assigner=credit_assigner,
            timeout_s=None,
            concurrency=1,
            retokenize=False,
        )


@pytest.mark.asyncio
async def test_generate_batch_retokenize_path_uses_custom_tokenizer(
    env_registry,
    mock_agent,
) -> None:
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=mock_agent)
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    # Use a real credit assigner here just to check it integrates fine
    credit_assigner = MonteCarloReturn(gamma=1.0)

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={"max_steps": 1, "target": "1"}),
        protocol=ProtocolSpec(kind="mock_protocol"),
        num_episodes=1,
    )

    batch = await engine.generate_batch(
        requests=[request],
        max_steps=2,
        credit_assigner=credit_assigner,
        timeout_s=None,
        concurrency=1,
        retokenize=True,
        tokenize=fake_tokenize,
    )

    assert batch.meta["batch_size"] == 1
    assert batch.meta["total_items"] == 1
    assert batch.meta["avg_total_reward"] == pytest.approx(1.0)

    assert len(batch.items) == 1
    item = batch.items[0]

    # State text is the initial observation from MockEnv.reset()
    state_text = "Reply with '1' to finish."
    state_len = len(state_text)

    # We used per-character ord() tokenization
    assert item.input_ids[:state_len] == [ord(c) for c in state_text]

    # Action is "1" from MockClient; so last token should be ord("1")
    assert item.input_ids[-1] == ord("1")

    # Action mask should be 0 over state region, 1 over action region
    assert all(v == 0 for v in item.action_mask[:state_len])
    assert all(v == 1 for v in item.action_mask[state_len:])

    assert item.weight == pytest.approx(batch.items[0].weight)


# ---------------------------------------------------------------------
# RolloutBatchSource integration: delegates to RolloutEngine.generate_batch
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rollout_batch_source_next_batch_integration(
    env_registry,
    mock_agent,
) -> None:
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=mock_agent)
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    credit_assigner = MonteCarloReturn(gamma=0.9)

    def requests_fn() -> List[RolloutRequest]:
        # Two episodes per call
        return [
            RolloutRequest(
                env=EnvSpec(kind="mock", kwargs={"max_steps": 2, "target": "1"}),
                protocol=ProtocolSpec(kind="mock_protocol"),
                num_episodes=2,
                meta={"batch_source": True},
            )
        ]

    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=credit_assigner,
        requests_fn=requests_fn,
        max_steps=3,
        timeout_s=None,
        concurrency=2,
        retokenize=True,
        tokenize=fake_tokenize,
    )

    saw_batch = await batch_source.next_batch()

    # We asked for num_episodes=2 => 2 rollouts; each should have at least 1 step
    assert saw_batch.meta["batch_size"] == 2
    assert saw_batch.meta["total_items"] >= 2

    # All items should carry rollout metadata including request_meta
    assert len(saw_batch.items) == saw_batch.meta["total_items"]
    for item in saw_batch.items:
        assert item.meta.get("request_meta", {}).get("batch_source") is True
        assert "rollout_id" in item.meta
        assert "step_index" in item.meta