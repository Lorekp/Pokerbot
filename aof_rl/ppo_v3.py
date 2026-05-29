"""PPO v3: policy + value + Q + teacher distillation + uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aof_rl.obs_v3_batch import ObsV3Batch
from aof_rl.policy_v3 import AoFPolicyNetV3


@dataclass
class PPOv3Config:
    clip_eps: float = 0.1
    value_coef: float = 0.5
    q_coef: float = 0.5
    teacher_coef: float = 0.25
    teacher_q_coef: float = 0.0
    entropy_coef: float = 0.01
    uncertainty_coef: float = 0.1
    ppo_epochs: int = 4
    gamma: float = 1.0
    gae_lambda: float = 1.0
    normalize_advantages: bool = True
    max_grad_norm: float = 0.5


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    *,
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


def ppo_v3_loss(
    model: AoFPolicyNetV3,
    batch_obs: ObsV3Batch,
    batch: dict[str, torch.Tensor],
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOv3Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model.forward(batch_obs)
    logits = out.logits
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    selected = batch["action"].long()
    new_lp = log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)

    adv = advantages
    if config.normalize_advantages and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    ratio = torch.exp(new_lp - old_log_probs)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    old_v = batch["old_value"]
    value = out.value
    if value.dim() == 0:
        value = value.unsqueeze(0)
    v_clipped = old_v + torch.clamp(
        value - old_v, -config.clip_eps, config.clip_eps
    )
    value_loss = 0.5 * torch.max(
        (returns - value).pow(2),
        (returns - v_clipped).pow(2),
    ).mean()

    q_pred = out.q_bb
    if q_pred.dim() == 1:
        q_pred = q_pred.unsqueeze(0)
    teacher_q = batch["teacher_q"]
    q_loss = F.mse_loss(q_pred, teacher_q)

    entropy = -(out.probs * log_probs).sum(dim=-1).mean()

    teacher_loss = torch.tensor(0.0, device=logits.device)
    if "teacher_action" in batch:
        teacher_loss = F.cross_entropy(logits, batch["teacher_action"].long())

    unc = out.uncertainty
    if unc.dim() == 0:
        unc = unc.unsqueeze(0)
    uncertainty_loss = F.mse_loss(unc, batch["teacher_edge"])

    loss = (
        policy_loss
        + config.value_coef * value_loss
        + (config.q_coef + config.teacher_q_coef) * q_loss
        + config.teacher_coef * teacher_loss
        + config.uncertainty_coef * uncertainty_loss
        - config.entropy_coef * entropy
    )

    stats = {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "q_loss": float(q_loss.item()),
        "entropy": float(entropy.item()),
        "teacher_loss": float(teacher_loss.item()),
        "uncertainty_loss": float(uncertainty_loss.item()),
        "total_loss": float(loss.item()),
    }
    return loss, stats
