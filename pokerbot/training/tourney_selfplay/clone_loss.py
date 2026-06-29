"""Clone loss — masked next-action CE (primary) + belief CE + an aux multi-hot BCE.

The HumanCloneNet is trained by Maia-style supervised imitation: predict the ACTUAL human next action
(``policy_targets``), with a hole-card belief auxiliary at showdowns (``belief_targets``) and a small
auxiliary multi-hot signal. This REUSES :func:`pokerbot.opponent.belief_loss.masked_cross_entropy`
verbatim (the per-head masking that normalizes by the valid-count), and mirrors
``opponent_transformer_loss`` — but **drops the SCAPT mutual-information diversity term** (component #4
of the design): for a single imitation anchor we want faithful prediction, not population diversity
(diversity is the LEAGUE's job, via piKL λ + the exploiter KL gate, not the clone's).

Loss = ``L_policy + λ_belief·L_belief + λ_aux·L_aux`` with the design defaults λ_belief=0.3,
λ_aux=0.1. ``L_aux`` is a masked multi-hot BCE over a coarse action-class target derived from the
policy target (the K=4 outcome-bin one-hot from :mod:`pokerbot.opponent.context`) — a cheap auxiliary
that regularizes the policy head toward the right *macro* action (fold / passive / agg-small /
agg-large) even when the exact size bucket is wrong (Maia-2 uses an analogous auxiliary head). It is
masked exactly like the policy loss (only where the human action is known).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.opponent.belief_loss import masked_cross_entropy
from pokerbot.opponent.context import NUM_BINS, action_bin


@dataclass
class CloneLossBreakdown:
    total: torch.Tensor
    policy: torch.Tensor
    belief: torch.Tensor
    aux: torch.Tensor
    policy_count: float
    belief_count: float


# Precompute action-index -> K=4 outcome bin (fold / passive / agg-small / agg-large), the aux target.
_ACTION_TO_BIN = torch.tensor([action_bin(a) for a in range(NUM_ACTIONS)], dtype=torch.long)


def _aux_multihot_targets(policy_targets: torch.Tensor) -> torch.Tensor:
    """Map (B,5) action-index targets to (B,5,NUM_BINS) multi-hot outcome-bin targets.

    -1 (no action) stays all-zero (and is masked off by the action mask anyway). The bin one-hot is a
    valid multi-hot with exactly one set bit; BCE-with-logits then teaches the macro class.
    """
    b, n = policy_targets.shape
    out = torch.zeros(b, n, NUM_BINS, dtype=torch.float32, device=policy_targets.device)
    valid = policy_targets >= 0
    if valid.any():
        idx = policy_targets.clamp(min=0)
        bins = _ACTION_TO_BIN.to(policy_targets.device)[idx]  # (B,5)
        out.scatter_(2, bins.unsqueeze(-1), 1.0)
        out = out * valid.unsqueeze(-1).float()
    return out


def _masked_bce(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Multi-hot BCE normalized by the number of valid (masked) rows. logits/targets: (B,5,NUM_BINS)."""
    valid = mask.bool()
    count = valid.float().sum()
    if count.item() == 0:
        return logits.sum() * 0.0, 0.0
    flat_logits = logits[valid]          # (M, NUM_BINS)
    flat_targets = targets[valid]        # (M, NUM_BINS)
    loss = F.binary_cross_entropy_with_logits(flat_logits, flat_targets, reduction="sum")
    return loss / count.clamp(min=1.0) / max(1, logits.shape[-1]), float(count.item())


def clone_loss(
    out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    lambda_belief: float = 0.3,
    lambda_aux: float = 0.1,
    aux_logits: torch.Tensor | None = None,
) -> CloneLossBreakdown:
    """Clone training loss (design A4): ``L_policy + 0.3·L_belief + 0.1·L_aux``.

    Args:
        out: forward dict with ``policy_logits`` (B,5,A) and ``belief_logits`` (B,5,1326).
        targets: ``policy_targets`` / ``belief_targets`` (B,5 int, -1 ignored) +
            ``action_known_mask`` / ``cards_known_mask`` (B,5 float).
        aux_logits: optional (B,5,NUM_BINS) aux head logits. When None, the aux head is derived by
            mean-pooling the policy logits into the K=4 bins (so a single-head clone still gets the aux
            regularizer without an extra parameter tensor) — this keeps the skeleton a drop-in over the
            base policy head, and a real aux head can be added later by passing ``aux_logits``.
    """
    policy_logits = out["policy_logits"]
    belief_logits = out["belief_logits"]

    policy_loss, policy_count = masked_cross_entropy(
        policy_logits, targets["policy_targets"], targets["action_known_mask"]
    )
    belief_loss, belief_count = masked_cross_entropy(
        belief_logits, targets["belief_targets"], targets["cards_known_mask"]
    )

    # Aux multi-hot BCE over the K=4 outcome bins.
    if aux_logits is None:
        aux_logits = _pool_policy_to_bins(policy_logits)
    aux_targets = _aux_multihot_targets(targets["policy_targets"])
    aux_loss, _ = _masked_bce(aux_logits, aux_targets, targets["action_known_mask"])

    total = policy_loss + lambda_belief * belief_loss + lambda_aux * aux_loss
    return CloneLossBreakdown(
        total=total,
        policy=policy_loss,
        belief=belief_loss,
        aux=aux_loss,
        policy_count=policy_count,
        belief_count=belief_count,
    )


def _pool_policy_to_bins(policy_logits: torch.Tensor) -> torch.Tensor:
    """Mean-pool the (B,5,A) policy logits into (B,5,NUM_BINS) by the action->bin partition.

    A parameter-free aux head: the bin logit is the mean of the action logits assigned to that bin.
    Used when no dedicated aux head is supplied (keeps the clone a single-head drop-in).
    """
    b, n, a = policy_logits.shape
    bins = _ACTION_TO_BIN.to(policy_logits.device)
    out = torch.full((b, n, NUM_BINS), float("-inf"), device=policy_logits.device)
    for k in range(NUM_BINS):
        sel = bins == k
        if sel.any():
            out[:, :, k] = policy_logits[:, :, sel].mean(dim=-1)
        else:
            out[:, :, k] = policy_logits.mean(dim=-1)
    return out
