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

    # With causal shift, tokens at positions 1 and 2 are scored.
    expected_logp_action = torch.tensor([-4.5518], dtype=torch.float32)

    logp_action = compute_logp_action(logits, input_ids, action_mask)

    assert logp_action.shape == (1,)
    assert torch.allclose(logp_action, expected_logp_action, atol=1e-3)


def test_compute_logp_action_multitoken_and_norm():
    """
    Verifies causal shift when the action spans multiple tokens, and length normalization.
    """
    logits = torch.tensor([[
        [0.0, 0.0],  # predicts token at pos 1
        [1.0, 0.0],  # predicts token at pos 2
        [0.0, 2.0],  # predicts token at pos 3
        [3.0, 0.0],  # unused (no target after final token)
    ]], dtype=torch.float32)

    input_ids = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
    action_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)

    # Manual expectation: only positions 2 and 3 are actions, scored with logits at 1 and 2.
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, 3, 2]
    expected = logp[0, 1, 0] + logp[0, 2, 1]  # action tokens at ids 0 and 1

    logp_action = compute_logp_action(logits, input_ids, action_mask)
    assert logp_action.shape == (1,)
    assert torch.allclose(logp_action, expected.unsqueeze(0), atol=1e-4)

    logp_action_norm = compute_logp_action(
        logits, input_ids, action_mask, length_normalize=True
    )
    assert torch.allclose(logp_action_norm, expected.unsqueeze(0) / 2, atol=1e-4)


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

    # With causal shift, only position 0 logits are used to score token at pos 1
    # logp_action = logp[0, 0, 0] = -1.3133
    # advantages = 2.0
    # loss = - (adv * logp_action).mean() = - (2.0 * -1.3133) = 2.6266
    expected_loss = 2.6266

    loss, stats = loss_fn.compute(logits, batch)

    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-3)
    assert stats["adv_mean"] == pytest.approx(2.0)
    assert stats["logp_mean"] == pytest.approx(-1.3133, abs=1e-3)


# ---- test_reinforce_baseline_loss ----

@pytest.mark.parametrize("normalize", [False, True])
def test_reinforce_baseline_loss(normalize):
    loss_fn = ReinforceBaselineLoss(normalize=normalize)

    # B=2, T=2, V=2
    logits = torch.tensor([
        [[1.0, 2.0], [3.0, 1.0]],  # sample 0: logp(id=0 | pos0) = -1.313
        [[2.0, 1.0], [1.0, 2.0]],  # sample 1: logp(id=0 | pos0) = -0.313
    ], dtype=torch.float32)

    batch = {
        "input_ids": torch.tensor([[1, 0], [0, 0]], dtype=torch.long),
        "action_mask": torch.tensor([[0, 1], [0, 1]], dtype=torch.float32),
        "weight": torch.tensor([1.5, 0.5], dtype=torch.float32), # Raw returns
    }

    # With causal shift, both samples score the token at position 1 using logits from position 0
    # logp_actions = [-1.313, -0.313]
    # raw_returns = [1.5, 0.5]
    # baseline = raw_returns.mean() = 1.0
    # advantages = [1.5 - 1.0, 0.5 - 1.0] = [0.5, -0.5]
    
    if normalize:
        # std = advantages.std(unbiased=False) = 0.5
        # advantages = [0.5 / 0.5, -0.5 / 0.5] = [1.0, -1.0]
        # loss = - (advantages * logp_action).mean()
        # loss = - ([1.0, -1.0] * [-1.313, -0.313]).mean()
        # loss = - ([-1.313, 0.313]).mean() = 0.5
        expected_loss = 0.5
    else:
        # advantages = [0.5, -0.5]
        # loss = - (advantages * logp_action).mean()
        # loss = - ([0.5, -0.5] * [-1.313, -0.313]).mean()
        # loss = - ([-0.6565, 0.1565]).mean() = 0.25
        expected_loss = 0.25

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
