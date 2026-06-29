"""tournament_reward — wire ICM/PKO $-EV + the ICM control-variate baseline into the GAE/return path.

Replaces the cash-bb reward with the tournament $-EV signal (design component #10). The hero PPO
trainer computes advantages from per-terminal-hand $-deltas; this module supplies:

  * :func:`hand_dollar_reward` — one hand's per-seat chip deltas → the hero's $-EV delta via
    ``eval_score.score_chip_to_dollar`` (ICM/PKO routing).
  * :func:`cv_baseline` — the hero's pre-hand ICM equity (``eval_score.score_with_cv`` baseline) used
    as the GAE value baseline so the all-or-nothing finish variance is subtracted (design risk #6).
  * :func:`returns_and_advantages` — a thin adapter that feeds the $-reward series + the ICM CV
    baseline into ``ppo.compute_gae`` (reused verbatim), so the hero's advantage is the
    variance-reduced ``$-reward − ICM(pre)`` signal the strength dial needs to be measurable.

This is dependency-light (eval_score + ppo only); it does NOT import the hero net or the engine search
path, so the $-EV reward definition is certifiable in isolation (mirrors ``eval_score``'s firewall).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pokerbot.tournament import eval_score as _eval_score


@dataclass
class HandOutcome:
    """One terminal hand's accounting for the hero (the reward unit)."""

    pre_stacks: Sequence[float]
    chip_deltas: Sequence[float]
    payouts: Sequence[float]
    hero: int
    fmt=None  # payouts.TournamentFormat; None -> NORMAL
    # PKO extras (only consulted for fmt == PKO)
    villain: int | None = None
    hero_equity: float | None = None
    villain_bounty: float = 0.0


def _fmt_or_default(fmt):
    if fmt is None:
        from pokerbot.tournament.payouts import TournamentFormat

        return TournamentFormat.NORMAL
    return fmt


def hand_dollar_reward(outcome: HandOutcome) -> float:
    """The hero's $-EV delta for one terminal hand (ICM/PKO via ``score_chip_to_dollar``)."""
    fmt = _fmt_or_default(outcome.fmt)
    deltas = _eval_score.score_chip_to_dollar(
        outcome.pre_stacks,
        outcome.chip_deltas,
        outcome.payouts,
        fmt,
        hero=outcome.hero,
        villain=outcome.villain,
        hero_equity=outcome.hero_equity,
        villain_bounty=outcome.villain_bounty,
    )
    return float(deltas[outcome.hero])


def cv_baseline(pre_stacks: Sequence[float], payouts: Sequence[float], hero: int) -> float:
    """The ICM control-variate baseline = hero's pre-hand ICM equity (the GAE value baseline)."""
    from pokerbot.tournament.icm import icm_equity

    return float(icm_equity(pre_stacks, payouts)[hero])


def cv_reward(outcome: HandOutcome) -> float:
    """The variance-reduced per-hand signal: ``$-reward − ICM(pre)[hero]`` (``eval_score.score_with_cv``).

    This is the reward the strength dial is measured on; the mean equals the realized-$ mean (ICM is
    mean-correct) but the variance is far lower (design risk #6).
    """
    realized = hand_dollar_reward(outcome)
    return _eval_score.score_with_cv(
        realized, outcome.pre_stacks, outcome.payouts, outcome.hero
    )


def returns_and_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float = 0.999,
    lam: float = 0.95,
):
    """Compute GAE returns + advantages over a $-reward series (reuses ``ppo.compute_gae``).

    ``rewards`` are per-step $-EV deltas (e.g. :func:`cv_reward` per terminal hand), ``values`` the
    hero net's value estimates, ``dones`` the per-step terminal flags. ``gamma`` defaults near 1.0 (a
    tournament is a long-horizon $-game). Returns ``(returns, advantages)`` from the repo's PPO GAE.

    Reuses ``ppo.compute_gae`` VERBATIM (plain-list signature) — the tournament $-EV reward + ICM CV
    baseline simply replace the cash-bb reward + chip-value baseline; the GAE machinery is unchanged.
    """
    from pokerbot.training.ppo import compute_gae

    return compute_gae(
        [float(x) for x in rewards],
        [float(x) for x in values],
        [bool(x) for x in dones],
        gamma=gamma,
        lam=lam,
    )
