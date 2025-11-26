from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Tuple, List

import torch
from torch import Tensor


Batch = Mapping[str, Tensor]


class Loss(Protocol):
    """
    Generic loss: given model outputs (logits) and a collated batch, return
    (scalar_loss, stats).
    """

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        ...


def _check_shape_2d(name: str, t: Tensor) -> None:
    if t.ndim != 2:
        raise ValueError(f"{name} must be 2D [B, T], got shape {tuple(t.shape)}")


def compute_logp_action(
    logits: Tensor,
    input_ids: Tensor,
    action_mask: Tensor,
) -> Tensor:
    """
    Compute log π(a|s) given token-level logits and an action mask.

    Args:
        logits: [B, T, V] float tensor of unnormalized logits.
        input_ids: [B, T] long tensor of token ids actually sampled.
        action_mask: [B, T] {0,1} mask; 1 where tokens belong to the "action".

    Returns:
        logp_action: [B] log-prob of the entire action sequence per sample.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B, T, V], got {tuple(logits.shape)}")
    _check_shape_2d("input_ids", input_ids)
    _check_shape_2d("action_mask", action_mask)

    if input_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"input_ids shape {tuple(input_ids.shape)} incompatible with logits "
            f"shape {tuple(logits.shape)}"
        )
    if action_mask.shape != logits.shape[:2]:
        raise ValueError(
            f"action_mask shape {tuple(action_mask.shape)} incompatible with logits "
            f"shape {tuple(logits.shape)}"
        )

    # [B, T, V]
    logprobs = torch.log_softmax(logits, dim=-1)

    # Gather log-prob of the actual token at each position: [B, T]
    token_logp = logprobs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    # Sum log-probs over the action region only: [B]
    amask = action_mask.to(token_logp.dtype)
    logp_action = (token_logp * amask).sum(dim=1)

    return logp_action


# ---------------------------------------------------------------------------
# REINFORCE family
# ---------------------------------------------------------------------------


@dataclass
class ReinforceLoss:
    """
    Vanilla REINFORCE:

        loss = - E[ A * log π(a|s) ]

    where A is taken from `batch["weight"]`.
    """

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]            # [B, T]
        action_mask = batch["action_mask"]        # [B, T]
        advantages = batch["weight"]              # [B]

        logp_action = compute_logp_action(logits, input_ids, action_mask)  # [B]

        loss = - (advantages * logp_action).mean()

        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "adv_mean": float(advantages.mean().detach().cpu()),
            "adv_std": float(advantages.std(unbiased=False).detach().cpu()),
            "logp_mean": float(logp_action.mean().detach().cpu()),
        }
        return loss, stats


@dataclass
class ReinforceBaselineLoss:
    """
    REINFORCE with batch-mean baseline:

        A_i = adv_i - mean(adv)
        loss = - E[ A_i * log π(a_i|s_i) ]

    where adv_i is batch["weight"].
    """

    normalize: bool = False

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]
        action_mask = batch["action_mask"]
        adv_raw = batch["weight"]                # [B]

        logp_action = compute_logp_action(logits, input_ids, action_mask)  # [B]

        baseline = adv_raw.mean()
        advantages = adv_raw - baseline

        if self.normalize:
            std = advantages.std(unbiased=False)
            advantages = advantages / (std + 1e-8)

        loss = - (advantages * logp_action).mean()

        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "baseline": float(baseline.detach().cpu()),
            "adv_mean": float(advantages.mean().detach().cpu()),
            "adv_std": float(advantages.std(unbiased=False).detach().cpu()),
            "logp_mean": float(logp_action.mean().detach().cpu()),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# PPO clipped policy loss (no value term here)
# ---------------------------------------------------------------------------


@dataclass
class PPOLoss:
    """
    PPO clipped policy loss (actor part only):

        r = π_new(a|s) / π_old(a|s)
        L_clip = - E[ min(r * A, clip(r, 1 - eps, 1 + eps) * A) ]

    Expects:
        - batch["weight"]:       A  (advantages)      [B]
        - batch[old_logp_key]:   log π_old(a|s)      [B]
        - input_ids / attention_mask / action_mask for π_new.
    """

    clip_eps: float = 0.2
    old_logp_key: str = "old_logp_action"

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]
        action_mask = batch["action_mask"]
        advantages = batch["weight"]              # [B]
        old_logp = batch[self.old_logp_key]       # [B]

        logp_action = compute_logp_action(logits, input_ids, action_mask)  # [B]

        # ratio = π_new / π_old
        ratio = torch.exp(logp_action - old_logp)                          # [B]

        # unclipped and clipped objectives
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

        obj = torch.min(unclipped, clipped)
        loss = -obj.mean()

        clip_frac = ((ratio > 1.0 + self.clip_eps) | (ratio < 1.0 - self.clip_eps)).float().mean()

        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "ratio_mean": float(ratio.mean().detach().cpu()),
            "ratio_std": float(ratio.std(unbiased=False).detach().cpu()),
            "clip_frac": float(clip_frac.detach().cpu()),
            "adv_mean": float(advantages.mean().detach().cpu()),
            "adv_std": float(advantages.std(unbiased=False).detach().cpu()),
            "logp_mean": float(logp_action.mean().detach().cpu()),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# KL penalty and entropy bonus
# ---------------------------------------------------------------------------


@dataclass
class KLLoss:
    """
    KL penalty between π_new and a reference policy whose log-prob is stored as
    batch[old_logp_key].

    We use the standard policy-gradient surrogate estimate:

        KL(π_new || π_old) ≈ E_{a ~ π_new} [ log π_new(a|s) - log π_old(a|s) ]

    Loss is:

        loss = coeff * mean(kl)

    (You usually *add* this to the overall loss; coeff > 0 makes it a penalty.)
    """

    coeff: float = 1.0
    old_logp_key: str = "old_logp_action"

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]
        action_mask = batch["action_mask"]
        old_logp = batch[self.old_logp_key]       # [B]

        logp_new = compute_logp_action(logits, input_ids, action_mask)     # [B]

        kl = logp_new - old_logp                                           # [B]
        loss = self.coeff * kl.mean()

        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "kl_mean": float(kl.mean().detach().cpu()),
            "kl_std": float(kl.std(unbiased=False).detach().cpu()),
        }
        return loss, stats


@dataclass
class EntropyBonus:
    """
    Entropy bonus over the action region.

    Computes token-level entropy H(π(·|token)) and averages over tokens where
    action_mask == 1. Loss is:

        loss = - coeff * mean_entropy

    So with coeff > 0, this *reduces* the total loss (encourages exploration).
    """

    coeff: float = 0.01

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]

        logprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)

        # token entropy: [B, T]
        token_entropy = -(probs * logprobs).sum(dim=-1)

        mask = action_mask.to(token_entropy.dtype)

        masked_entropy = token_entropy * mask   # [B, T]
        # avoid divide-by-zero if mask is all zeros
        denom = mask.sum()
        if denom.item() == 0:
            mean_entropy = torch.zeros((), device=logits.device, dtype=logits.dtype)
        else:
            mean_entropy = masked_entropy.sum() / denom

        loss = -self.coeff * mean_entropy

        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "entropy_mean": float(mean_entropy.detach().cpu()),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------


@dataclass
class LossTerm:
    """
    Single term inside a CompositeLoss.

    - name:   short identifier for logging
    - loss:   loss object implementing Loss protocol
    - weight: scalar multiplier applied to that loss
    """
    name: str
    loss: Loss
    weight: float = 1.0


@dataclass
class CompositeLoss:
    """
    Combine multiple Loss terms into a single scalar loss:

        total_loss = sum_i weight_i * loss_i

    Stats are merged with hierarchical keys:

        "{name}/loss", "{name}/<stat_key>", ...

    and a top-level "loss" key for the final combined loss.
    
    This class expects logits to be passed in, and it passes them
    down to all child terms.
    """

    terms: List[LossTerm]

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        if not self.terms:
            raise ValueError("CompositeLoss.terms must be non-empty")

        total_loss: Tensor | None = None
        stats: Dict[str, Any] = {}

        for term in self.terms:
            # Pass the pre-computed logits down to the child term
            raw_loss, term_stats = term.loss.compute(logits, batch)
            scaled_loss = term.weight * raw_loss

            if total_loss is None:
                total_loss = scaled_loss
            else:
                total_loss = total_loss + scaled_loss

            # per-term stats
            stats[f"{term.name}/loss"] = float(raw_loss.detach().cpu())
            stats[f"{term.name}/weight"] = term.weight
            for k, v in term_stats.items():
                stats[f"{term.name}/{k}"] = v

        assert total_loss is not None
        stats["loss"] = float(total_loss.detach().cpu())

        return total_loss, stats