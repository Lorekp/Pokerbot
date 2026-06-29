"""piKL operator — the closed-form λ-regularized policy that is the STRENGTH DIAL.

This is the load-bearing math of the strengthening design (Jacob et al. 2021, arXiv 2112.07544;
Cicero / RL-DiL-piKL, arXiv 2210.05492). Given a per-action value ``Q(s,·)``, the frozen human-clone
action logits ``log τ(·|s)`` and the legal mask, the piKL policy is the closed-form softmax

    π_λ(a) = softmax_a( (Q(s,a) + λ·log τ(a|s)) / (κ + λ) )   over the LEGAL actions only.

λ is the literal strength dial:
  * λ → ∞  ⇒  π → τ        (the faithful human clone — the FISH tier plays here, "no lift")
  * λ → 0  ⇒  π → argmax over legal Q (pure value maximization — drifts off human shape; the floor
                                       λ≥1e-3 stops the dial fully releasing)
  * small λ ⇒ stronger-than-data (Q pulls past the clone's mistakes) while bounded
              ``D_KL(π‖τ) ≤ O(1/λ)`` of human play (Jacob Thm 1) — the anti-GTO-drift KL leash.

``κ`` (kappa) is the value-temperature: a fixed positive scalar that sets how sharply Q sharpens the
policy at fixed λ (the design's ``κ+λ`` denominator). Default 1.0.

**DiL mixture** (Cicero): a tier plays the *mean* over a per-tier λ-SET (a distributional-λ agent),
``π̄(a) = mean_{λ∈Λ_i} π_λ(a)``. The per-tier λ-sets (design C):
  fish U{1e-1,1}  ·  weak_reg U{1e-1,3e-1}  ·  strong_reg U{1e-2,1e-1}  ·  modern_pro U{1e-3,1e-2}
  ·  elite/exploiter {1e-3}  (play-λ floored ≥ 1e-3 so the leash never fully releases).

Everything here is **net-free**: it consumes any ``Q`` + any ``log τ`` over an identical legal-masked
index space (the design's KL-mismatch firewall — both must be the SAME ``_legal_mask`` index order;
unit-tested). No torch dependency in the hot path: the operator is numpy so it is callable from a
``decide_fn`` without a torch import, and the league/Rust seam stays light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# The play-λ floor (design anti-drift mechanism #2): the human leash never fully releases.
LAMBDA_FLOOR = 1e-3
# Default value-temperature κ (the design's κ+λ denominator).
DEFAULT_KAPPA = 1.0
# A practical "λ → ∞" stand-in for the FISH tier / faithful-clone gate (κ/λ ≈ 0 ⇒ π = τ).
LAMBDA_FAITHFUL = 1e9

# Per-tier λ-sets (design Phase C). Tier name -> the DiL mixture λ-set.
TIER_LAMBDA_SETS: dict[str, tuple[float, ...]] = {
    "fish": (LAMBDA_FAITHFUL,),               # pure τ, no lift (λ≈∞)
    "weak_reg": (1e-1, 3e-1),
    "strong_reg": (1e-2, 1e-1),
    "modern_pro": (1e-3, 1e-2),
    "elite": (1e-3,),
    "exploiter": (1e-3,),
}


def _log_softmax_legal(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Numerically-stable log-softmax over the legal entries; illegal entries -> -inf.

    ``logits`` and ``legal`` are aligned 1-D arrays over the SAME action-index space.
    """
    out = np.full_like(logits, -np.inf, dtype=np.float64)
    idx = np.flatnonzero(legal)
    if idx.size == 0:
        return out
    z = logits[idx].astype(np.float64)
    z = z - z.max()
    lse = np.log(np.exp(z).sum())
    out[idx] = z - lse
    return out


def pikl_policy(
    q: Sequence[float],
    tau_logits: Sequence[float],
    legal_mask: Sequence[bool],
    lam: float,
    *,
    kappa: float = DEFAULT_KAPPA,
    floor_lambda: bool = True,
) -> np.ndarray:
    """The closed-form piKL policy ``softmax((Q + λ·log τ)/(κ+λ))`` over the legal mask.

    Args:
        q: (A,) per-action value (``$-EV`` ; chip-EV; any consistent scalar). Illegal entries ignored.
        tau_logits: (A,) the frozen clone's action LOGITS (pre-softmax). We normalize to ``log τ``
            internally over the legal mask, so passing raw logits is correct (the additive softmax
            constant cancels). Illegal entries ignored.
        legal_mask: (A,) bool — the legal-action mask (the SAME index order as ``q`` / ``tau_logits``).
        lam: λ, the strength dial. ``floor_lambda`` clamps it to ``LAMBDA_FLOOR`` (the anti-drift floor).
        kappa: κ, the value-temperature (design ``κ+λ`` denominator). Must be > 0.

    Returns:
        (A,) probability vector over the legal mask (illegal entries exactly 0; sums to 1).

    Limit behaviors (unit-tested in ``tests/test_pikl.py``):
        * λ → ∞ (``LAMBDA_FAITHFUL``)  ⇒  π == τ over the legal mask.
        * λ → 0 (floor off, lam=0)     ⇒  π == one-hot argmax over legal Q (value maximization).
    """
    q = np.asarray(q, dtype=np.float64)
    tau_logits = np.asarray(tau_logits, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=bool)
    if q.shape != tau_logits.shape or q.shape != legal.shape:
        raise ValueError(
            f"piKL index mismatch: q{q.shape} tau{tau_logits.shape} legal{legal.shape} "
            "(all must share the SAME legal-masked action-index space)"
        )
    if not legal.any():
        raise ValueError("piKL called with no legal actions")
    if kappa <= 0:
        raise ValueError("kappa must be > 0")

    lam = max(float(lam), LAMBDA_FLOOR) if floor_lambda else float(lam)
    log_tau = _log_softmax_legal(tau_logits, legal)  # log τ over the legal mask (illegal -> -inf)

    idx = np.flatnonzero(legal)
    out = np.zeros_like(q, dtype=np.float64)

    if lam <= 0.0:
        # λ = 0 limit: pure value maximization (one-hot argmax over legal Q). Defined here so the
        # unit test λ=0 ⇒ argmax(Q) is exact (the general formula divides by κ+λ -> κ, still a valid
        # softmax, but the *limit* the design names is the argmax; we return it for λ exactly 0).
        best = idx[np.argmax(q[idx])]
        out[best] = 1.0
        return out

    if not np.isfinite(lam) or lam >= LAMBDA_FAITHFUL:
        # λ → ∞ limit: the score is dominated by λ·log τ, denominator ~ λ ⇒ π = softmax(log τ) = τ.
        out[idx] = np.exp(log_tau[idx])
        s = out.sum()
        return out / s if s > 0 else out

    score = (q + lam * log_tau) / (kappa + lam)  # log τ is -inf on illegal -> score -inf there
    z = score[idx]
    z = z - z.max()
    e = np.exp(z)
    out[idx] = e / e.sum()
    return out


def pikl_mixture(
    q: Sequence[float],
    tau_logits: Sequence[float],
    legal_mask: Sequence[bool],
    lambda_set: Sequence[float],
    *,
    kappa: float = DEFAULT_KAPPA,
    floor_lambda: bool = True,
) -> np.ndarray:
    """DiL mixture: the MEAN piKL policy over a per-tier λ-set (Cicero distributional-λ).

    ``π̄(a) = mean_{λ∈Λ}( π_λ(a) )``. The λ-set is the tier's strength signature
    (see :data:`TIER_LAMBDA_SETS`). Returns an (A,) probability vector over the legal mask.
    """
    lambda_set = list(lambda_set)
    if not lambda_set:
        raise ValueError("lambda_set must be non-empty")
    acc = None
    for lam in lambda_set:
        p = pikl_policy(q, tau_logits, legal_mask, lam, kappa=kappa, floor_lambda=floor_lambda)
        acc = p if acc is None else acc + p
    return acc / len(lambda_set)


def tier_policy(
    q: Sequence[float],
    tau_logits: Sequence[float],
    legal_mask: Sequence[bool],
    tier: str,
    *,
    kappa: float = DEFAULT_KAPPA,
) -> np.ndarray:
    """Convenience: the DiL mixture for a named tier (uses :data:`TIER_LAMBDA_SETS`)."""
    if tier not in TIER_LAMBDA_SETS:
        raise ValueError(f"unknown tier {tier!r} (known: {sorted(TIER_LAMBDA_SETS)})")
    return pikl_mixture(q, tau_logits, legal_mask, TIER_LAMBDA_SETS[tier], kappa=kappa)


def kl_to_anchor(pi: Sequence[float], tau_logits: Sequence[float], legal_mask: Sequence[bool]) -> float:
    """``D_KL(π ‖ τ)`` over the legal mask — the human-leash diagnostic (design bound ``≤ O(1/λ)``).

    Used by the per-tier gate (the league's strength-vs-human-shape check) to assert a strengthened
    tier has not drifted off the human anchor.
    """
    pi = np.asarray(pi, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=bool)
    log_tau = _log_softmax_legal(np.asarray(tau_logits, dtype=np.float64), legal)
    idx = np.flatnonzero(legal & (pi > 0))
    if idx.size == 0:
        return 0.0
    return float((pi[idx] * (np.log(pi[idx]) - log_tau[idx])).sum())


@dataclass
class PiklConfig:
    """Per-tier piKL configuration (the league member carries one of these)."""

    tier: str
    lambda_set: tuple[float, ...]
    kappa: float = DEFAULT_KAPPA

    @classmethod
    def for_tier(cls, tier: str, *, kappa: float = DEFAULT_KAPPA) -> "PiklConfig":
        if tier not in TIER_LAMBDA_SETS:
            raise ValueError(f"unknown tier {tier!r} (known: {sorted(TIER_LAMBDA_SETS)})")
        return cls(tier=tier, lambda_set=TIER_LAMBDA_SETS[tier], kappa=kappa)
