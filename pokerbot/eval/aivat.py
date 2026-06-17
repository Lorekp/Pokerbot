"""AIVAT-style control variate for the exploitation battery (variance reduction).

Two zero-mean control variates remove variance from the per-hand chip outcome (Burch et al. AIVAT,
AAAI 2018):

  * DECISION term — the hero randomizes its action, which adds variance. At a hero decision with action
    probabilities π and a baseline value bv(a):

        c_H = bv(taken) − Σ_a π(a)·bv(a)        ⟶  E[c_H] = 0  over the hero's own randomization.

  * CHANCE term — card reveals (flop/turn/river) add variance. At a reveal with realized post-reveal
    value u(after) and its conditional expectation E[u(after) | before] over the possible reveals:

        c_C = u(after) − E[u(after) | before]   ⟶  E[c_C] = 0  over the deal.

The adjusted per-hand score = raw_chips − Σ_H c_H − Σ_C c_C is UNBIASED for ANY baseline, and
lower-variance when the baseline correlates with the hand outcome. ``baseline_from_leaf_net`` supplies
the rigorous per-action baseline bv(a) = Q(h,a) from the trained HistoryValueNet (the chance term is
the larger half of the reduction); a cheap hand-strength × commitment proxy is the checkpoint-free
fallback. Combined with common-random-number (duplicate) dealing in the battery, this gives the
variance reduction needed for tight uplift estimates in a few thousand hands.
"""

from __future__ import annotations

import numpy as np


def decision_correction(probs, baseline_values, taken_idx: int) -> float:
    """Zero-mean control-variate term for one hero decision."""
    p = np.asarray(probs, dtype=np.float64)
    bv = np.asarray(baseline_values, dtype=np.float64)
    s = p.sum()
    if s <= 0 or not (0 <= taken_idx < len(bv)):
        return 0.0
    p = p / s
    return float(bv[taken_idx] - float((p * bv).sum()))


def baseline_from_leaf_net(q_row, legal_mask=None) -> np.ndarray:
    """Per-action AIVAT baseline bv(a) from the trained leaf value net's Q-row.

    The rigorous upgrade over the hand-strength × commitment proxy: bv(a) = Q(h, a), the leaf net's
    value of taking action a (from ``HistoryValueEvaluator.evaluate_batch``). Illegal actions are
    zeroed — the hero's policy puts no mass there, so they never enter the zero-mean ``decision_
    correction``. The estimator is unbiased for ANY bv; a Q-row that correlates with the realized
    outcome is what buys the variance reduction.
    """
    bv = np.asarray(q_row, dtype=np.float64).reshape(-1)
    if legal_mask is not None:
        lm = np.asarray(legal_mask).reshape(-1).astype(bool)
        if lm.shape == bv.shape:
            bv = np.where(lm, bv, 0.0)
    return bv


def chance_correction(expected_value, realized_value) -> float:
    """Zero-mean control-variate term for one chance event (board reveal).

    ``c_C = realized_value − expected_value`` where ``expected_value`` MUST be the conditional
    expectation E[value | pre-reveal] of ``realized_value`` over the possible reveals (e.g. the leaf-net
    value averaged over the remaining deck), and ``realized_value`` is the value on the board actually
    dealt. E[c_C] = 0 over the deal, so subtracting Σc_C removes card-luck variance while staying
    unbiased. The caller is responsible for the expectation matching the realized event.
    """
    return float(realized_value) - float(expected_value)


class AIVATAccumulator:
    """Collects per-hand raw scores + control-variate corrections; reports raw vs adjusted stats."""

    def __init__(self) -> None:
        self._raw: list[float] = []
        self._adj: list[float] = []

    def add_hand(self, raw_chips: float, corrections=None, chance_corrections=None) -> None:
        """Record a hand. ``corrections`` = the hero's per-decision control variates; the optional
        ``chance_corrections`` = the per-reveal (board) control variates. Both are summed and removed."""
        c = float(sum(corrections)) if corrections else 0.0
        c += float(sum(chance_corrections)) if chance_corrections else 0.0
        self._raw.append(float(raw_chips))
        self._adj.append(float(raw_chips) - c)

    def summary(self) -> dict:
        raw = np.asarray(self._raw, dtype=np.float64)
        adj = np.asarray(self._adj, dtype=np.float64)
        rstd = float(raw.std(ddof=1)) if len(raw) > 1 else 0.0
        astd = float(adj.std(ddof=1)) if len(adj) > 1 else 0.0
        return {
            "hands": len(raw),
            "raw_mean_bb": float(raw.mean()) if len(raw) else 0.0,
            "adj_mean_bb": float(adj.mean()) if len(adj) else 0.0,  # ~ same as raw (unbiased)
            "raw_std": rstd,
            "adj_std": astd,
            "variance_reduction": (1.0 - (astd / rstd) ** 2) if rstd > 0 else 0.0,
        }
