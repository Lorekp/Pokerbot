"""Trinal-Clip PPO for imperfect-information poker (AlphaHoldem-inspired)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pokerbot.models.policy import PseudoSiamesePolicy, PolicyOutput


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    value_clip: float = 0.2  # Trinal-Clip: extra clip on value loss
    policy_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[list[float], list[float]]:
    advantages: list[float] = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def trinal_clip_ppo_loss(
    model: PseudoSiamesePolicy,
    batch: dict[str, torch.Tensor],
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(
        batch["weva_ids"],
        batch["actions"],
        batch["position"],
        batch.get("legal_mask"),
        cards=batch["cards"],
    )
    log_probs = F.log_softmax(out.logits, dim=-1)
    selected = batch["action"].long()
    new_lp = log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)

    ratio = torch.exp(new_lp - old_log_probs)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - config.policy_clip, 1 + config.policy_clip) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    value = out.value
    value_clipped = batch["old_value"] + torch.clamp(
        value - batch["old_value"],
        -config.value_clip,
        config.value_clip,
    )
    value_loss = torch.max(
        F.mse_loss(value, returns),
        F.mse_loss(value_clipped, returns),
    )

    entropy = -(out.probs * log_probs).sum(dim=-1).mean()
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

    stats = {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
    }
    return loss, stats
