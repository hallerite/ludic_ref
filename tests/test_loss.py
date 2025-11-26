from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor

from ludic.training.loss import (
    Loss,
    Batch,
    compute_logp_action,
    ReinforceLoss,
    ReinforceBaselineLoss,
    CompositeLoss,
    LossTerm,
)

# ---- Mocks ----

@dataclass
class MockLoss(Loss):
    """A mock Loss object that returns a fixed loss and stats."""
    loss_val: float
    stats: Dict[str, Any]

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        return torch.tensor(self.loss_val, dtype=torch.float32), self.stats


# ---- test_compute_logp_action ----

def test_compute_logp_action():
    """Unit test the core log-probability calculation."""
    # B=1, T=3, V=4
    logits = torch.tensor([[
        [1.0, 1.0, 3.0, 1.0],  # pos 0
        [4.0, 2.0, 1.0, 1.0],  # pos 1
        [1.0, 5.0, 1.0, 1.0],  # pos 2
    ]], dtype=torch.float32)

    # B=1, T=3
    # Action tokens are at indices 2, 0, 1
    input_ids = torch.tensor([[2, 0, 1]], dtype=torch.long)

    # B=1, T=3
    # State = pos 0, Action = pos 1, 2
    action_mask = torch.tensor([[0, 1, 1]], dtype=torch.float32)

    expected_logp_action = torch.tensor([-0.2645], dtype=torch.float32)

    logp_action = compute_logp_action(logits, input_ids, action_mask)

    assert logp_action.shape == (1,)
    assert torch.allclose(logp_action, expected_logp_action, atol=1e-3)


# ---- test_reinforce_loss ----

def test_reinforce_loss():
    loss_fn = ReinforceLoss()

    # B=1, T=2, V=2
    logits = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]], dtype=torch.float32)
    # logprobs -> [[[-1.313, -0.313], [-0.127, -2.127]]]
    
    batch = {
        "input_ids": torch.tensor([[1, 0]], dtype=torch.long),
        "action_mask": torch.tensor([[0, 1]], dtype=torch.float32),
        "weight": torch.tensor([2.0], dtype=torch.float32), # Advantages
    }

    # logp_action = logp[0, 1, 0] = -0.127
    # advantages = 2.0
    # loss = - (adv * logp_action).mean() = - (2.0 * -0.127) = 0.254
    expected_loss = 0.254

    loss, stats = loss_fn.compute(logits, batch)

    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-3)
    assert stats["adv_mean"] == pytest.approx(2.0)
    assert stats["logp_mean"] == pytest.approx(-0.127, abs=1e-3)


# ---- test_reinforce_baseline_loss ----

@pytest.mark.parametrize("normalize", [False, True])
def test_reinforce_baseline_loss(normalize):
    loss_fn = ReinforceBaselineLoss(normalize=normalize)

    # B=2, T=2, V=2
    logits = torch.tensor([
        [[1.0, 2.0], [3.0, 1.0]], # logp(id=1,0 | mask=0,1) = -0.127
        [[1.0, 2.0], [1.0, 2.0]], # logp(id=0,0 | mask=0,1) = -1.313
    ], dtype=torch.float32)

    batch = {
        "input_ids": torch.tensor([[1, 0], [0, 0]], dtype=torch.long),
        "action_mask": torch.tensor([[0, 1], [0, 1]], dtype=torch.float32),
        "weight": torch.tensor([1.5, 0.5], dtype=torch.float32), # Raw returns
    }

    # logp_actions = [-0.127, -1.313]
    # raw_returns = [1.5, 0.5]
    # baseline = raw_returns.mean() = 1.0
    # advantages = [1.5 - 1.0, 0.5 - 1.0] = [0.5, -0.5]
    
    if normalize:
        # std = advantages.std(unbiased=False) = 0.5
        # advantages = [0.5 / 0.5, -0.5 / 0.5] = [1.0, -1.0]
        # loss = - (advantages * logp_action).mean()
        # loss = - ([1.0, -1.0] * [-0.127, -1.313]).mean()
        # loss = - ([-0.127, 1.313]).mean() = - (1.186 / 2) = -0.593
        expected_loss = -0.593
    else:
        # advantages = [0.5, -0.5]
        # loss = - (advantages * logp_action).mean()
        # loss = - ([0.5, -0.5] * [-0.127, -1.313]).mean()
        # loss = - ([-0.0635, 0.6565]).mean() = - (0.593 / 2) = -0.2965
        expected_loss = -0.2965

    loss, stats = loss_fn.compute(logits, batch)

    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-3)
    assert stats["baseline"] == pytest.approx(1.0)
    if normalize:
        assert stats["adv_mean"] == pytest.approx(0.0)
    else:
        assert stats["adv_mean"] == pytest.approx(0.0) # mean is 0 before norm too


# ---- test_composite_loss ----

def test_composite_loss():
    """Tests that CompositeLoss combines losses and stats correctly."""
    
    # Create two mock loss terms
    term1 = LossTerm(
        name="ppo",
        loss=MockLoss(loss_val=10.0, stats={"clip_frac": 0.1}),
        weight=1.0
    )
    term2 = LossTerm(
        name="kl",
        loss=MockLoss(loss_val=4.0, stats={"kl_mean": 0.5}),
        weight=0.5
    )
    
    composite_loss = CompositeLoss(terms=[term1, term2])

    # These inputs don't matter since the mock losses ignore them
    dummy_logits = torch.empty(0)
    dummy_batch = {}

    loss, stats = composite_loss.compute(dummy_logits, dummy_batch)

    # Total loss = (10.0 * 1.0) + (4.0 * 0.5) = 10.0 + 2.0 = 12.0
    assert loss == pytest.approx(12.0)
    
    # Check that stats are correctly namespaced
    expected_stats = {
        "loss": 12.0,
        "ppo/loss": 10.0,
        "ppo/weight": 1.0,
        "ppo/clip_frac": 0.1,
        "kl/loss": 4.0,
        "kl/weight": 0.5,
        "kl/kl_mean": 0.5,
    }
    assert stats == expected_stats