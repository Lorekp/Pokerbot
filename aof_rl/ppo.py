"""PPO with optional Nash teacher auxiliary loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aof_rl.policy import AoFPolicyNet


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    teacher_coef: float = 0.3
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 1.0,
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


def ppo_loss(
    model: AoFPolicyNet,
    batch: dict[str, torch.Tensor],
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    obs = batch["obs"]
    out = model(obs)
    log_probs = F.log_softmax(out.logits, dim=-1)
    selected = batch["action"].long()
    new_lp = log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)

    ratio = torch.exp(new_lp - old_log_probs)
    adv = advantages
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    old_v = batch["old_value"]
    v_clipped = old_v + torch.clamp(
        out.value - old_v, -config.value_clip, config.value_clip
    )
    value_loss = 0.5 * torch.max(
        (returns - out.value).pow(2),
        (returns - v_clipped).pow(2),
    ).mean()

    entropy = -(out.probs * log_probs).sum(dim=-1).mean()

    teacher_loss = torch.tensor(0.0, device=obs.device)
    if "teacher_action" in batch:
        teacher_loss = F.cross_entropy(out.logits, batch["teacher_action"].long())

    loss = (
        policy_loss
        + config.value_coef * value_loss
        - config.entropy_coef * entropy
        + config.teacher_coef * teacher_loss
    )

    stats = {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "teacher_loss": float(teacher_loss.item()),
    }
    return loss, stats
