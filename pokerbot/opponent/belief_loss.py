"""Joint loss for MADT opponent transformer (belief + policy + MI diversity)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    total: torch.Tensor
    belief: torch.Tensor
    policy: torch.Tensor
    mi: torch.Tensor
    belief_count: float
    policy_count: float


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    ignore_index: int = -1,
) -> tuple[torch.Tensor, float]:
    """
    Cross-entropy normalized by valid mask count.

    logits: (B, C) or (B, N, C)
    targets: same leading dims
    mask: (B,) or (B, N) float/bool — 1 where loss applies
    """
    if logits.dim() == 3:
        b, n, c = logits.shape
        flat_logits = logits.reshape(b * n, c)
        flat_targets = targets.reshape(b * n)
        flat_mask = mask.reshape(b * n)
    else:
        flat_logits = logits
        flat_targets = targets
        flat_mask = mask

    valid = flat_mask.bool() & (flat_targets >= 0)
    count = valid.float().sum()
    if count.item() == 0:
        zero = flat_logits.sum() * 0.0
        return zero, 0.0

    loss = F.cross_entropy(flat_logits[valid], flat_targets[valid], reduction="sum")
    return loss / count.clamp(min=1.0), float(count.item())


def mutual_information_diversity(
    policy_logits: torch.Tensor,
    population_policy_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Encourage diversity: negative mean cosine similarity between population members.

    policy_logits: (B, 5, A) — primary forward
    population_policy_logits: (P, B, 5, A) — P population forwards on same states
    """
    if population_policy_logits is None or population_policy_logits.shape[0] < 2:
        return policy_logits.sum() * 0.0

    probs = F.softmax(policy_logits, dim=-1)
    pop_probs = F.softmax(population_policy_logits, dim=-1)

    p = pop_probs.shape[0]
    sims = []
    for i in range(p):
        for j in range(i + 1, p):
            pi = pop_probs[i].reshape(-1, pop_probs.shape[-1])
            pj = pop_probs[j].reshape(-1, pop_probs.shape[-1])
            cos = F.cosine_similarity(pi, pj, dim=-1).mean()
            sims.append(cos)

    if not sims:
        return policy_logits.sum() * 0.0
    return -torch.stack(sims).mean()


def opponent_transformer_loss(
    out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    lambda_policy: float = 1.0,
    lambda_belief: float = 1.0,
    lambda_mi: float = 0.1,
    population_policy_logits: torch.Tensor | None = None,
) -> LossBreakdown:
    """
    Combined loss with per-head masking.

    targets keys:
        belief_targets: (B, 5) int, -1 ignored
        policy_targets: (B, 5) int, -1 ignored
        cards_known_mask: (B, 5) float
        action_known_mask: (B, 5) float
    """
    belief_logits = out["belief_logits"]
    policy_logits = out["policy_logits"]

    belief_loss, belief_count = masked_cross_entropy(
        belief_logits,
        targets["belief_targets"],
        targets["cards_known_mask"],
    )
    policy_loss, policy_count = masked_cross_entropy(
        policy_logits,
        targets["policy_targets"],
        targets["action_known_mask"],
    )
    mi_loss = mutual_information_diversity(policy_logits, population_policy_logits)

    total = (
        lambda_belief * belief_loss
        + lambda_policy * policy_loss
        + lambda_mi * mi_loss
    )

    return LossBreakdown(
        total=total,
        belief=belief_loss,
        policy=policy_loss,
        mi=mi_loss,
        belief_count=belief_count,
        policy_count=policy_count,
    )
