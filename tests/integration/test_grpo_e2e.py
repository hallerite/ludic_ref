from __future__ import annotations

from collections import defaultdict
from typing import Optional, Tuple, List, Dict, Any

import pytest

from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.types import Observation, Info, StepOutcome, SamplingArgs
from ludic.training.types import (
    RolloutRequest,
    ProtocolSpec,
    EnvSpec,
)
from ludic.training.batching import (
    RolloutEngine,
    RolloutBatchSource,
    GRPORequestStrategy,
)
from ludic.training.credit_assignment import (
    GroupNormalizedReturn,
)
from ludic.interaction.base import InteractionProtocol
from ludic.interaction.single_agent import SingleAgentSyncProtocol

from tests._mocks import SeedableMockAgent

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# 1. A mock environment that is deterministic based on seed
# ---------------------------------------------------------------------------
class SeedableMockEnv(SingleAgentEnv):
    """
    A mock env that is deterministic based on the reset seed.
    - reset() obs includes the seed.
    - step() gives +1 for a "correct" action, -0.1 otherwise.
    """

    def __init__(self, correct_action: str = "A") -> None:
        super().__init__()
        self.correct_action = correct_action
        self._obs: Observation = "Not reset"
        self._t = 0

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return f"You must reply with only one letter. The correct letter is {self.correct_action}."

    def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        self._t = 0
        self._obs = f"Start state for seed {seed}. Correct action is {self.correct_action}."
        return self._obs, {"seed": seed}

    def env_step(self, action: str) -> StepOutcome:
        self._t += 1
        action_clean = action.strip().upper()

        if action_clean == self.correct_action:
            reward = 1.0
            terminated = True
            self._obs = "✅ Correct"
        else:
            reward = -0.1
            terminated = True  # End episode on first action for simplicity
            self._obs = f"❌ Incorrect (got {action_clean})"

        return StepOutcome(
            obs=self._obs,
            reward=reward,
            truncated=False,
            terminated=terminated,
            info={"action_taken": action_clean},
        )

    def env_current_obs(self) -> Observation:
        return self._obs


# ---------------------------------------------------------------------------
# 2. The End-to-End Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grpo_e2e_seed_grouping_and_credit() -> None:
    """
    Tests the full GRPO data pipeline:
    1. GRPORequestStrategy expands N requests to N*G.
    2. RolloutEngine uses env_seed for reset() and sampling_seed for act().
    3. The Env is deterministic on env_seed (same obs).
    4. The *Mock* Agent is deterministic on sampling_seed (different actions).
    5. GroupNormalizedReturn computes advantages based on the seed groups.
    """

    # ---- 1. Arrange ----

    N_GROUPS = 2
    G_PER_GROUP = 2
    seed_group_A = 100
    seed_group_B = 200
    env_registry = {
        "env_A": lambda **kwargs: SeedableMockEnv(correct_action="A"),
        "env_B": lambda **kwargs: SeedableMockEnv(correct_action="B"),
    }
    # ctx_registry is no longer needed here

    base_seed_A = 9000
    base_seed_B = 8000

    # This map defines the *guaranteed* agent behavior
    seed_to_action_map = {
        base_seed_A + 0: "A",  # Group A, Rollout 0 -> Correct (1.0)
        base_seed_A + 1: "Z",  # Group A, Rollout 1 -> Incorrect (-0.1)
        base_seed_B + 0: "B",  # Group B, Rollout 0 -> Correct (1.0)
        base_seed_B + 1: "Y",  # Group B, Rollout 1 -> Incorrect (-0.1)
    }

    # Use the new SeedableMockAgent
    # agent = SeedableMockAgent(seed_map=seed_to_action_map) # <-- Removed

    credit_assigner = GroupNormalizedReturn(normalize_adv=True)

    # --- Create the Protocol Factory ---
    # This creates a new agent worker for each concurrent rollout
    def create_protocol() -> InteractionProtocol:
        agent = SeedableMockAgent(seed_map=seed_to_action_map)
        return SingleAgentSyncProtocol(agent=agent)
    # -----------------------------------
    
    protocol_registry = {
        "grpo_protocol": create_protocol
    }

    engine = RolloutEngine(
        protocol_registry=protocol_registry,
        env_registry=env_registry,
    )

    def make_expanded_requests() -> List[RolloutRequest]:
        # 1. Define Base Requests
        s_args_A: SamplingArgs = {
            "temperature": 0.7,
            "max_tokens": 5,
            "seed": base_seed_A,
            "extras": {"extra_body": {"return_token_ids": True}},
        }
        s_args_B: SamplingArgs = {
            "temperature": 0.7,
            "max_tokens": 5,
            "seed": base_seed_B,
            "extras": {"extra_body": {"return_token_ids": True}},
        }

        req_A = RolloutRequest(
            env=EnvSpec(kind="env_A", kwargs={}),
            protocol=ProtocolSpec(kind="grpo_protocol", kwargs={}),
            seed=seed_group_A,  # Force env seed for Group A
            sampling_args=s_args_A,
        )
        req_B = RolloutRequest(
            env=EnvSpec(kind="env_B", kwargs={}),
            protocol=ProtocolSpec(kind="grpo_protocol", kwargs={}),
            seed=seed_group_B,  # Force env seed for Group B
            sampling_args=s_args_B,
        )
        
        # 2. Expand using GRPO Strategy
        strategy = GRPORequestStrategy(group_size=G_PER_GROUP)
        return strategy.expand([req_A, req_B])

    # NOTE: We use the STANDARD RolloutBatchSource, effectively injecting
    # the GRPO logic via the expansion function above.
    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=credit_assigner,
        requests_fn=make_expanded_requests,
        max_steps=3,
        retokenize=False,
        tokenize=None,
    )

    # ---- 2. Act ----

    saw_batch = await batch_source.next_batch()

    # ---- 3. Assert ----

    assert saw_batch.meta["batch_size"] == N_GROUPS * G_PER_GROUP
    assert len(saw_batch.items) == N_GROUPS * G_PER_GROUP

    rollouts: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for item in saw_batch.items:
        rollout_id = item.meta["rollout_id"]
        rollouts[rollout_id]["id"] = rollout_id
        rollouts[rollout_id]["used_seed"] = item.meta["engine"]["used_seed"]
        rollouts[rollout_id]["prev_obs"] = item.meta["prev_obs"]
        rollouts[rollout_id]["reward"] = item.meta["reward"]
        rollouts[rollout_id]["weight"] = item.weight

        assert "prompt_token_ids" in item.meta
        assert "completion_token_ids" in item.meta
        assert item.meta["engine"]["protocol_kind"] == "grpo_protocol"

    rollout_list = list(rollouts.values())
    assert len(rollout_list) == 4

    # --- Assert 1: Correct Grouping by Env Seed ---
    seeds_seen = {r["used_seed"] for r in rollout_list}
    assert seeds_seen == {seed_group_A, seed_group_B}

    rollouts_in_A = [r for r in rollout_list if r["used_seed"] == seed_group_A]
    rollouts_in_B = [r for r in rollout_list if r["used_seed"] == seed_group_B]

    assert len(rollouts_in_A) == G_PER_GROUP
    assert len(rollouts_in_B) == G_PER_GROUP

    # --- Assert 2: Deterministic Env Start ---
    obs_A = {r["prev_obs"] for r in rollouts_in_A}
    assert len(obs_A) == 1
    assert f"seed {seed_group_A}" in list(obs_A)[0]
    assert "Correct action is A" in list(obs_A)[0]

    obs_B = {r["prev_obs"] for r in rollouts_in_B}
    assert len(obs_B) == 1
    assert f"seed {seed_group_B}" in list(obs_B)[0]
    assert "Correct action is B" in list(obs_B)[0]

    assert obs_A != obs_B

    # --- Assert 3: Deterministic Agent Actions ---
    rewards_A = {r["reward"] for r in rollouts_in_A}
    rewards_B = {r["reward"] for r in rollouts_in_B}

    assert rewards_A == {1.0, -0.1}
    assert rewards_B == {1.0, -0.1}

    # --- Assert 4: Correct Credit Assignment ---
    # Group A: rewards = [1.0, -0.1].
    #   mean = 0.45, adv = [0.55, -0.55], std = 0.55, norm_adv = [1.0, -1.0]
    weights_A = {r["weight"] for r in rollouts_in_A}
    assert len(weights_A) == 2
    assert 1.0 in {round(w, 4) for w in weights_A}
    assert -1.0 in {round(w, 4) for w in weights_A}

    # Group B: rewards = [1.0, -0.1]. Same calculation.
    weights_B = {r["weight"] for r in rollouts_in_B}
    assert len(weights_B) == 2
    assert 1.0 in {round(w, 4) for w in weights_B}
    assert -1.0 in {round(w, 4) for w in weights_B}