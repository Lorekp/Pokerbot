"""LiftedAgent — a frozen clone tier wrapped with the piKL operator + a Q source.

One ``LiftedAgent`` per league member (design component #6). It holds:
  * a frozen clone ``τ`` (a ``HumanCloneNet`` at the tier's skill bin) — supplied as a *callable*
    ``tau_logits_fn(obs) -> (A,) logits`` so the agent is net-agnostic (the league can pass a torch
    net, a tabular policy, or a stub for tests / the heuristic fallback);
  * a Q source ``q_fn(obs) -> (A,) values`` — the 1-ply ``settle_leaf`` $-EV CV rollout in the real
    system (a ``$-EV`` value head later); a NEUTRAL/zero Q in the skeleton, documented below;
  * a :class:`pikl.PiklConfig` (the tier's λ-set + κ) — the strength dial.

``policy(obs, legal_mask) -> (A,) probs`` runs ``pikl.pikl_mixture`` and is the serve policy. For the
FISH tier (λ≈∞) this returns ``τ`` verbatim (no lift). For a non-fish tier the Q pulls ``τ`` toward
high-$-EV actions with the tier's λ strength.

This mirrors the ``neural_serve`` fail-safe pattern: a Q source or τ source that raises (no net, bad
ckpt) is caught and the agent falls back to the pure clone (``τ``) rather than fabricating a value —
exactly the design's "abstain, never fabricate" rule. The Rust sim's native push/fold-ICM agent is the
λ-small, native-Q analogue of this class (same operator, Q computed in Rust).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from pokerbot.training.tourney_selfplay import pikl

# An observation is opaque to this layer — it is whatever the τ/Q callables agree on (a SpotContext, a
# torch batch, an engine state). The LiftedAgent never inspects it; it only routes it to the callables.
Obs = object
TauLogitsFn = Callable[[Obs], Sequence[float]]
QFn = Callable[[Obs], Sequence[float]]
LegalMaskFn = Callable[[Obs], Sequence[bool]]


def neutral_q(num_actions: int) -> QFn:
    """A zero Q source (the lift collapses to τ at any λ — the safe skeleton default).

    DOCUMENTED TODO (design risk #5 / Phase C): replace with the 1-ply ``eval_score.settle_leaf``
    $-EV CV rollout so the lift actually strengthens the tier. Until that Q exists, the LiftedAgent is
    a faithful clone at every λ (it cannot mis-rank a bubble spot because it ranks nothing) — which is
    the conservative, never-ship-an-unaudited-Q behavior the design demands.
    """

    def fn(_obs: Obs) -> np.ndarray:
        return np.zeros(num_actions, dtype=np.float64)

    return fn


@dataclass
class LiftedAgent:
    """A single league tier's serve policy = frozen clone ``τ`` + piKL(``Q``, λ-set)."""

    tier: str
    tau_logits_fn: TauLogitsFn
    q_fn: QFn
    config: pikl.PiklConfig
    num_actions: int

    @classmethod
    def from_tier(
        cls,
        tier: str,
        tau_logits_fn: TauLogitsFn,
        num_actions: int,
        *,
        q_fn: QFn | None = None,
        kappa: float = pikl.DEFAULT_KAPPA,
    ) -> "LiftedAgent":
        """Build a LiftedAgent for a named tier. ``q_fn`` defaults to :func:`neutral_q` (skeleton)."""
        return cls(
            tier=tier,
            tau_logits_fn=tau_logits_fn,
            q_fn=q_fn or neutral_q(num_actions),
            config=pikl.PiklConfig.for_tier(tier, kappa=kappa),
            num_actions=num_actions,
        )

    def tau_probs(self, obs: Obs, legal_mask: Sequence[bool]) -> np.ndarray:
        """The raw clone policy ``τ`` over the legal mask (λ→∞ reference; the FISH tier's serve)."""
        logits = np.asarray(self.tau_logits_fn(obs), dtype=np.float64)
        return pikl.pikl_policy(
            np.zeros_like(logits), logits, legal_mask, pikl.LAMBDA_FAITHFUL, kappa=self.config.kappa
        )

    def policy(self, obs: Obs, legal_mask: Sequence[bool]) -> np.ndarray:
        """The strengthened serve policy ``π̄ = DiL-mixture_λ piKL(Q, log τ)`` over the legal mask.

        Fail-safe: if the Q source raises, fall back to the pure clone ``τ`` (never fabricate a value).
        """
        logits = np.asarray(self.tau_logits_fn(obs), dtype=np.float64)
        try:
            q = np.asarray(self.q_fn(obs), dtype=np.float64)
            if q.shape != logits.shape:
                raise ValueError("Q / τ action-index shape mismatch")
        except Exception:  # noqa: BLE001 — abstain to the clone, never raise into the sim
            q = np.zeros_like(logits)
        return pikl.pikl_mixture(
            q, logits, legal_mask, self.config.lambda_set, kappa=self.config.kappa
        )

    def sample(self, obs: Obs, legal_mask: Sequence[bool], rng: np.random.Generator) -> int:
        """Sample a legal action index from the strengthened policy."""
        p = self.policy(obs, legal_mask)
        return int(rng.choice(len(p), p=p))

    def kl_to_anchor(self, obs: Obs, legal_mask: Sequence[bool]) -> float:
        """``D_KL(π̄ ‖ τ)`` at this obs — the per-tier human-shape diagnostic (design gate)."""
        logits = self.tau_logits_fn(obs)
        pi = self.policy(obs, legal_mask)
        return pikl.kl_to_anchor(pi, logits, legal_mask)
