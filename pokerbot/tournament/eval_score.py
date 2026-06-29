"""The shared chip->$ scoring + variance-reduction CORE every tournament-eval consumer imports.

A tournament is a **$-EV** game, not the chip-EV game the cash engine solves: a hand's chip result only
matters through the payout structure (ICM, satellite step-payouts, PKO prize+bounty). This module is the
single, dependency-light place that turns a chip outcome into a per-seat **$-equity** delta AND supplies the
two variance-reduction levers the high-variance tournament estimators need to converge:

  * the **ICM control variate** (``score_with_cv``) -- replace the brutal "did I bust this hand" indicator
    with its *expectation* (hero's ICM equity at the pre-hand stacks), so a realized-$ series has the
    enormous all-or-nothing finish variance subtracted off (an unbiased baseline: ICM is mean-correct).
  * the **mean-preserving all-in leaf** (``settle_leaf``) -- wrap ``rollout_leaf.equity_allin_settle`` (an
    equity average over many board runouts, *not* one runout) so a paired arm can cancel under common
    random numbers via a shared ``settle_seed``.

Five public functions, all pure-Python over the *certified* tournament primitives (``icm``, ``bounty``,
``payouts``, ``rollout_leaf``) + numpy. NO net, NO jax, NO GPU -- so it is certifiable in isolation and
imports nothing from the trained model or the engine search path.

  1. :func:`score_chip_to_dollar` -- chip deltas -> per-seat $-equity deltas, routing NORMAL/SNG/SATELLITE
     through :func:`icm.icm_delta` and PKO through :func:`bounty.pko_call_dollar_ev` (prize-$ + bounty-$).
  2. :func:`score_with_cv` -- realized-$ minus the ICM baseline ``icm_equity(pre)[hero]`` (the CV / chip-EV
     baseline subtraction). The output's *mean* equals the realized-$ mean (the baseline has the same
     expectation as the realized bust term), but its *variance* is far lower.
  3. :func:`settle_leaf` -- mean-preserving all-in equity settle of a state -> per-seat $-deltas, with a
     shared ``settle_seed`` so paired arms (e.g. exploit-on vs exploit-off) share runouts under CRN.
  4. :func:`stat_block` -- ``{n, mean, std, se, bb_per_100}`` mirroring ``exploitation_battery._stat``.
  5. :func:`verdict` -- ``'POSITIVE' | 'NEGATIVE' | 'INCONCLUSIVE (CI spans 0)'`` (the ``eval_cv_realistic``
     idiom), the standard CI->headline reducer.

See ``tests/test_eval_score.py`` for the conservation / mean-preservation / CV-identity proofs.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from pokerbot.tournament import bounty as _bounty
from pokerbot.tournament import payouts as _payouts
from pokerbot.tournament.icm import icm_delta, icm_equity
from pokerbot.tournament.rollout_leaf import equity_allin_settle

TournamentFormat = _payouts.TournamentFormat

# Default number of mean-preserving board runouts for the all-in leaf (matches the spec). Override per call.
_DEFAULT_RUNOUTS = 120


def score_chip_to_dollar(
    pre_stacks: Sequence[float],
    chip_deltas: Sequence[float],
    payouts: Sequence[float],
    fmt: TournamentFormat,
    *,
    hero: int | None = None,
    villain: int | None = None,
    hero_equity: float | None = None,
    villain_bounty: float = 0.0,
    progressive_frac: float = 0.5,
    growth_value: float = 0.5,
) -> np.ndarray:
    """Map a hand's per-seat chip deltas to per-seat **$-equity** deltas under the payout structure.

    Routing (the format is the payout *regime*, never the engine):

    * ``NORMAL`` / ``SATELLITE`` -> :func:`icm.icm_delta(pre_stacks, chip_deltas, payouts)` -- the post-hand
      ICM equity minus the pre-hand ICM equity (busts clamped to 0 inside ``icm_delta``). For a SATELLITE
      the ``payouts`` vector is the flat ``[seat_value]*seats`` ladder, so the same ICM wrap correctly prices
      the step-payout (every paid finish is worth one identical seat). SNG is a NORMAL ladder -> same path.

    * ``PKO`` -> the prize side via the SAME ICM win/lose post-stacks PLUS the bounty cash collected on a win,
      using :func:`bounty.pko_call_dollar_ev`. This requires the all-in pair (``hero``, ``villain``), hero's
      showdown ``hero_equity`` (P(win)), and the ``villain_bounty``. ``pko_call_dollar_ev`` returns hero's
      total CALL $-EV (prize + bounty); we subtract hero's pre-hand fold equity ``icm_equity(pre)[hero]`` so
      the result is a $-equity DELTA on hero's index (the other seats keep the pure ICM-prize delta, since
      only the eliminator collects the bounty). Reduces EXACTLY to the ICM-only delta when
      ``villain_bounty == 0`` (``pko_call_dollar_ev`` collapses to the chip-EV/ICM call value).

    Parameters
    ----------
    pre_stacks, chip_deltas, payouts
        Per-seat pre-hand chips, per-seat net chip change (sum ~= 0), best-first currency payout vector.
    fmt
        A :class:`payouts.TournamentFormat`. Only ``PKO`` consults the bounty arguments.
    hero, villain, hero_equity, villain_bounty, progressive_frac, growth_value
        PKO-only: the all-in pair, hero's win probability, and the bounty accounting (CoinPoker
        ``progressive_frac=0.5``; ``growth_value`` discounts the half that grows hero's own bounty).

    Returns
    -------
    np.ndarray
        Per-seat $-equity delta. For NORMAL/SATELLITE this conserves the prize pool change to ~0
        (``sum == 0`` up to float tolerance, since ``icm_equity`` is pool-conserving at fixed ``payouts``).
    """
    pre = [float(s) for s in pre_stacks]
    deltas = icm_delta(pre, chip_deltas, payouts)

    if fmt is not TournamentFormat.PKO:
        return deltas

    # PKO: override hero's index with the prize+bounty call $-EV delta. The prize side of
    # pko_call_dollar_ev is ICM over the win/lose post-stacks of the all-in for the SMALLER stack, which
    # equals the chip-delta ICM on hero's seat (the all-in moves exactly that many chips); the bounty side
    # adds the cash. When villain_bounty==0 this is identical to deltas[hero] (asserted in the tests).
    if hero is None or villain is None or hero_equity is None:
        raise ValueError("PKO scoring needs hero, villain, and hero_equity (the all-in pair + win prob)")
    call_ev = _bounty.pko_call_dollar_ev(
        pre, payouts, hero, villain, hero_equity,
        villain_bounty=villain_bounty, progressive_frac=progressive_frac, growth_value=growth_value,
    )
    fold_eq = float(icm_equity(pre, payouts)[hero])
    out = np.asarray(deltas, dtype=float).copy()
    out[hero] = call_ev - fold_eq
    return out


def score_with_cv(
    realized_dollar: float,
    pre_stacks: Sequence[float],
    payouts: Sequence[float],
    hero: int,
) -> float:
    """ICM control-variate / chip-EV baseline subtraction: ``realized_$ - icm_equity(pre_stacks)[hero]``.

    The realized-$ result of a hand for ``hero`` carries the entire all-or-nothing "did I bust" variance.
    Its expectation, conditional on the pre-hand stacks and the eventual play, is hero's pre-hand ICM equity
    (ICM is mean-correct by the prize-conservation identity). Subtracting that *expectation* leaves a
    zero-mean, low-variance residual whose sample mean is an UNBIASED estimate of ``E[realized_$] - E[ICM]``;
    add ``E[ICM]`` back (a cheap, near-exact quantity at fixed stacks) to recover the headline. On a no-op
    hand where ``realized_$`` already equals ``icm_equity(pre)[hero]`` (e.g. pre == post, nobody moves), this
    returns exactly 0.0 -- the residual carries no spurious signal.

    This mirrors ``cv_absolute``'s ``E[S] = E[S - B] + E[B]`` with ``B = ICM(pre)[hero]``: the high-variance
    term is replaced by its expectation, sharpening (never biasing) the estimate.
    """
    baseline = float(icm_equity(pre_stacks, payouts)[hero])
    return float(realized_dollar) - baseline


def settle_leaf(
    state,
    payouts: Sequence[float],
    pre_stacks: Sequence[float],
    fmt: TournamentFormat = TournamentFormat.NORMAL,
    *,
    settle_seed: int = 0,
    n_runouts: int = _DEFAULT_RUNOUTS,
    hero: int | None = None,
    villain: int | None = None,
    hero_equity: float | None = None,
    villain_bounty: float = 0.0,
    progressive_frac: float = 0.5,
    growth_value: float = 0.5,
) -> dict[int, float]:
    """Mean-preserving all-in settle of ``state`` -> per-seat **$-equity** deltas.

    Wraps :func:`rollout_leaf.equity_allin_settle(state, n_runouts, rng=random.Random(settle_seed))` to get
    the LOW-VARIANCE per-seat chip deltas (each contender's equity share averaged over ``n_runouts`` board
    completions; exact when the board is already complete), then maps those chip deltas to $-equity deltas
    via :func:`score_chip_to_dollar`. The shared ``settle_seed`` is the CRN hook: two paired arms (e.g.
    exploit-on vs exploit-off) that reach the same all-in pass the SAME ``settle_seed`` -> identical runout
    samples -> the runout noise cancels in the paired difference.

    Returns ``{seat: dollar_delta}`` (dense over ``range(len(state.players))``). The chip leaf is zero-sum;
    the $-mapping is pool-conserving for NORMAL/SATELLITE (sum ~= 0). PKO routes hero's seat through the
    bounty arm (needs ``hero``/``villain``/``hero_equity``).
    """
    rng = random.Random(settle_seed)
    chip = equity_allin_settle(state, n_runouts=n_runouts, rng=rng)
    n = len(state.players)
    chip_vec = [float(chip.get(i, 0.0)) for i in range(n)]
    dollar = score_chip_to_dollar(
        pre_stacks, chip_vec, payouts, fmt,
        hero=hero, villain=villain, hero_equity=hero_equity, villain_bounty=villain_bounty,
        progressive_frac=progressive_frac, growth_value=growth_value,
    )
    return {i: float(dollar[i]) for i in range(n)}


def stat_block(vals: Sequence[float]) -> dict:
    """``{n, mean, std, se, bb_per_100}`` for a per-spot series, mirroring ``exploitation_battery._stat``.

    ``std`` is the sample std (``ddof=1``); ``se = std/sqrt(n)``; ``bb_per_100 = mean*100`` (the standard
    scale-up of a per-hand/per-spot mean -- here a per-spot $-equity if the series is $-valued, or chip
    bb/100 if it is chip-valued; the caller chooses the unit of ``vals``). Empty/singleton inputs report
    ``std == se == 0``."""
    a = np.asarray(vals, dtype=np.float64)
    n = int(a.size)
    mean = float(a.mean()) if n else 0.0
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    se = float(std / np.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "se": se, "bb_per_100": mean * 100.0}


def verdict(ci_lo: float, ci_hi: float) -> str:
    """Reduce a 95% CI ``[ci_lo, ci_hi]`` to a headline (the ``eval_cv_realistic.py:234`` idiom).

    ``'POSITIVE'`` if the whole interval is above 0, ``'NEGATIVE'`` if entirely below 0, else
    ``'INCONCLUSIVE (CI spans 0)'`` (the interval straddles 0 -> no significant sign).
    """
    if ci_lo > 0.0:
        return "POSITIVE"
    if ci_hi < 0.0:
        return "NEGATIVE"
    return "INCONCLUSIVE (CI spans 0)"
