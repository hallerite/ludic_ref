from __future__ import annotations

import pytest

from ludic.training.orchestrator import Orchestrator
from ludic.training.types import RolloutStepKey, WeightingStrategy, SAWBatch
from ludic.types import Rollout
from tests._mocks import MockEnv, MockAgent


@pytest.mark.asyncio
async def test_generate_async_n_rollouts() -> None:
    o = Orchestrator(env_factory=lambda: MockEnv(max_steps=2), agent=MockAgent())

    rs = await o.generate(
        batch_size=4,
        max_steps=2,
        timeout_s=1.0,
        concurrency=2,
    )

    assert len(rs) == 4
    assert all(r.length >= 1 for r in rs)


def test_generate_sync_n_rollouts() -> None:
    o = Orchestrator(env_factory=lambda: MockEnv(max_steps=2), agent=MockAgent())

    rs = o.generate_sync(
        batch_size=3,
        max_steps=3,
        timeout_s=1.0,
        concurrency=1,
    )

    assert len(rs) == 3
    assert all(r.length >= 1 for r in rs)


class RewardWeighting(WeightingStrategy):
    """
    Simple weighting strategy: weight == step reward.
    """

    def compute(self, rollouts: list[Rollout]) -> dict[RolloutStepKey, float]:
        out: dict[RolloutStepKey, float] = {}
        for r in rollouts:
            for s in r.steps:
                out[(r.id, s.index)] = float(s.reward)
        return out


def simple_tokenize(text: str) -> list[int]:
    """
    Trivial tokenizer for tests: 1 token per character.

    This makes it easy to reason about action_mask lengths: the number of
    ones in action_mask == len(action_text).
    """
    return [1] * len(text)


@pytest.mark.asyncio
async def test_generate_batch_builds_sawbatch_with_weights_and_masks() -> None:
    # Env/agent setup: MockAgent always outputs "1", so each episode
    # terminates in a single step with reward=1.0.
    env_factory = lambda: MockEnv(max_steps=4, target="1") # noqa: E731
    agent = MockAgent()

    batch_size = 3
    max_steps = 4
    timeout_s = 1.0

    orch = Orchestrator(env_factory=env_factory, agent=agent)
    weighting = RewardWeighting()

    saw_batch: SAWBatch = await orch.generate_batch(
        batch_size=batch_size,
        max_steps=max_steps,
        weighting=weighting,
        tokenize=simple_tokenize,
        timeout_s=timeout_s,
        use_model_token_ids=False,  # force retokenize path
        retokenize=True,
    )

    # ---- batch-level checks ------------------------------------------

    # We expect one step per episode (MockAgent hits the target immediately).
    assert saw_batch.meta["batch_size"] == batch_size
    assert saw_batch.meta["total_items"] == batch_size

    # With reward=1.0 per episode, avg_total_reward should be ~1.0.
    assert pytest.approx(saw_batch.meta["avg_total_reward"], rel=1e-6) == 1.0

    assert len(saw_batch.items) == batch_size

    # ---- item-level checks -------------------------------------------

    for item in saw_batch.items:
        # Weight should match the reward coming from MockEnv (1.0).
        assert pytest.approx(item.weight, rel=1e-6) == 1.0

        # Basic shape consistency.
        n = len(item.input_ids)
        assert n > 0
        assert len(item.attention_mask) == n
        assert len(item.action_mask) == n

        # With simple_tokenize (1 token per char), the action region should
        # contain exactly one token ("1"), so the sum of the action_mask
        # must be 1.0.
        assert sum(item.action_mask) == 1

        # Meta should at least carry rollout/step identity and reward info.
        assert "rollout_id" in item.meta
        assert "step_index" in item.meta
        assert item.meta["reward"] == pytest.approx(1.0, rel=1e-6)
