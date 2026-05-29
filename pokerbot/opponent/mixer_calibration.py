"""Calibrate HybridPolicy belief-confidence threshold from held-out showdown trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CalibrationCurve:
    thresholds: list[float]
    mean_nll_below: list[float]
    mean_nll_above: list[float]
    mean_top1_below: list[float]
    mean_top1_above: list[float]
    chosen_threshold: float

    def as_dict(self) -> dict:
        return {
            "thresholds": self.thresholds,
            "mean_nll_below": self.mean_nll_below,
            "mean_nll_above": self.mean_nll_above,
            "mean_top1_below": self.mean_top1_below,
            "mean_top1_above": self.mean_top1_above,
            "chosen_threshold": self.chosen_threshold,
        }


def _belief_confidence_from_logits(belief_logits: torch.Tensor) -> torch.Tensor:
    belief = F.softmax(belief_logits, dim=-1)
    entropy = -(belief * torch.log(belief + 1e-8)).sum(dim=-1)
    max_entropy = float(np.log(belief.shape[-1]))
    return 1.0 - entropy / max(max_entropy, 1e-8)


def calibrate_belief_threshold(
    belief_logits: torch.Tensor,
    belief_targets: torch.Tensor,
    cards_known_mask: torch.Tensor,
    *,
    candidate_thresholds: list[float] | None = None,
    nll_improvement: float = 0.05,
) -> CalibrationCurve:
    """
    Pick the lowest confidence threshold where held-out showdown belief NLL
    beats the below-threshold NLL by ``nll_improvement`` nats on average.
    """
    candidate_thresholds = candidate_thresholds or [
        round(x, 2) for x in np.linspace(0.05, 0.95, 19)
    ]
    valid = cards_known_mask.bool() & (belief_targets >= 0)
    if not valid.any():
        return CalibrationCurve([], [], [], [], [], [], 0.35)

    flat_logits = belief_logits.reshape(-1, belief_logits.shape[-1])[valid.reshape(-1)]
    flat_targets = belief_targets.reshape(-1)[valid.reshape(-1)]
    conf = _belief_confidence_from_logits(flat_logits)

    nll_below: list[float] = []
    nll_above: list[float] = []
    top1_below: list[float] = []
    top1_above: list[float] = []
    chosen = candidate_thresholds[-1]

    baseline_nll = F.cross_entropy(flat_logits, flat_targets, reduction="mean").item()

    for thr in candidate_thresholds:
        above = conf >= thr
        below = ~above
        if above.sum() == 0 or below.sum() == 0:
            nll_below.append(baseline_nll)
            nll_above.append(baseline_nll)
            top1_below.append(0.0)
            top1_above.append(0.0)
            continue
        nll_b = F.cross_entropy(flat_logits[below], flat_targets[below], reduction="mean").item()
        nll_a = F.cross_entropy(flat_logits[above], flat_targets[above], reduction="mean").item()
        acc_b = (flat_logits[below].argmax(-1) == flat_targets[below]).float().mean().item()
        acc_a = (flat_logits[above].argmax(-1) == flat_targets[above]).float().mean().item()
        nll_below.append(nll_b)
        nll_above.append(nll_a)
        top1_below.append(acc_b)
        top1_above.append(acc_a)
        if nll_a <= nll_b - nll_improvement and acc_a >= acc_b:
            chosen = thr
            break

    return CalibrationCurve(
        thresholds=candidate_thresholds,
        mean_nll_below=nll_below,
        mean_nll_above=nll_above,
        mean_top1_below=top1_below,
        mean_top1_above=top1_above,
        chosen_threshold=float(chosen),
    )
