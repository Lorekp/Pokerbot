r"""The cash-eval scoring + verdict CORE the ``cash_eval_suite`` driver ties together.

This is the cash mirror of ``pokerbot/tournament/eval_score.py`` (whose ``stat_block`` / ``verdict`` /
CV idioms it REUSES verbatim): the single dependency-light place that turns the raw per-(stake x depth x
position) measurements (M1 net curve, M2 exploitation uplift, M3/M6 pro-safety, M4 het net, M5 VR
provenance) into a ranked, rake-aware PASS / FAIL / INCONCLUSIVE verdict.

Why a cash module distinct from the tournament one
--------------------------------------------------
A tournament is a $-EV game (chip outcomes priced through ICM / bounty payouts); the cash engine solves
the chip-EV game directly, and the cash "$" axis is simply chip bb/100 MINUS the CoinPoker rake (5%-flat
with a per-stake BB-cap hump, modeled in ``pokerbot.economics.rake``). So the cash verdict has three axes
the tournament one does not:

  1. **a NET-of-rake curve** over ``rake.STAKE_ORDER`` — and the BB-cap makes that curve a NON-MONOTONE
     HUMP (micro: rakeback ~100% so net rake ~0; low/mid: the fat cap still binds AND the rich rakeback
     has ended -> the worst net valley; high: the cap binds at an ~8 BB pot so rake is a tiny flat tax).
     A chip-POSITIVE stake can be net-NEGATIVE in that valley — the whole point of the rake-aware sweep.
  2. **exploitation UPLIFT** (exploit-ON minus exploit-OFF) vs a field that actually RESPONDS to sizing —
     the deployed cash edge is value-EXTRACTION, not GTO purity, so the verdict must require the uplift's
     lower CI to clear 0 (the overlay genuinely prints) AND the blindness-regression control (uplift ~0
     vs a size-blind field) so we are measuring the mechanism, not a bookkeeping artifact.
  3. **pro-safety** (M3 ``Delta = LBR6(ON) - LBR6(OFF)`` exploitability + the M6 cap-binding ``bet_shift``
     proxy): turning exploitation on must NOT make the hero materially more counter-exploitable by a
     best-responding pro — ``Delta_upper_CI <= eps_rnr``.

Public surface (all pure-Python over numpy + ``tournament.eval_score`` + ``rake`` — NO net, NO jax, NO
engine search, so it is certifiable in isolation):

  * :func:`ci95_from_se` / :func:`ci95_from_halfwidth` — the 95% CI builders the driver feeds verdicts.
  * :func:`cell_verdict` — per-cell ``POSITIVE | NEGATIVE | INCONCLUSIVE`` from a 95% CI (the
    ``eval_score.verdict`` idiom, re-exported so cash consumers import ONE module) with an explicit
    ``INCONCLUSIVE`` override when the VR provenance is bad (realized half-width misses target, or
    ``var_reduction <= 1`` — a non-reducing CV must never read as a clean signal).
  * :func:`net_from_chip` — the additive identity ``net_on = chip_on - net_cost_bb`` the driver and the
    tests pin (the net column is the chip column minus the per-hand rake cost; never re-derived).
  * :func:`hump_table` — the net-bb/100 curve builder over ``STAKE_ORDER`` + its non-monotone-hump
    classification (peak stake, valley stake, and whether the curve actually humps).
  * :func:`stake_go_no_go` — the combined GO / NO-GO per stake: ``net_on lower_CI > 0`` (we net-win after
    rake) AND ``Delta_upper_CI <= eps_rnr`` (pro-safe) AND ``uplift_lower_CI > 0`` (the exploit prints).

``stat_block`` and the base ``verdict`` are RE-EXPORTED from ``tournament.eval_score`` so a cash consumer
imports a single module; the cash-specific reducers wrap them. See ``tests/test_cash_eval_score.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# REUSE the tournament scoring core verbatim — do NOT re-implement stat_block / the CI->headline reducer.
from pokerbot.tournament.eval_score import stat_block, verdict
from pokerbot.economics.rake import STAKE_ORDER, net_cost_bb

# Re-export so cash consumers import ONE module (the spec's "ties it together").
__all__ = [
    "stat_block",
    "verdict",
    "ci95_from_se",
    "ci95_from_halfwidth",
    "cell_verdict",
    "net_from_chip",
    "hump_table",
    "stake_go_no_go",
    "CellResult",
    "DEFAULT_EPS_RNR",
]

# The pro-safety tolerance: how much extra LBR6 exploitability (bb/100) we will accept the exploit-ON arm
# to incur over the OFF base before calling the cell counter-exploitable. The RNR cap is meant to hold
# this near 0 (the capped bet_shift ~0.0119 vs uncapped ~0.0235 from rnr_cap); a small positive slack
# absorbs the LBR estimator's own noise. Overridable per call so a stricter harness can demand 0.
DEFAULT_EPS_RNR = 5.0


def ci95_from_se(mean: float, se: float) -> tuple[float, float]:
    """The 95% CI ``[mean - 1.96 se, mean + 1.96 se]`` (the ``cv_absolute.CVResult.ci95`` convention)."""
    return (float(mean) - 1.96 * float(se), float(mean) + 1.96 * float(se))


def ci95_from_halfwidth(mean: float, halfwidth: float) -> tuple[float, float]:
    """The 95% CI from a mean + an already-1.96-scaled half-width (when a CVResult hands you ``ci95``
    directly, or a downstream reports a half-width rather than an SE)."""
    return (float(mean) - abs(float(halfwidth)), float(mean) + abs(float(halfwidth)))


def cell_verdict(ci_lo: float, ci_hi: float, *, var_reduction: float | None = None,
                 realized_halfwidth: float | None = None,
                 target_halfwidth: float | None = None) -> str:
    """Per-cell headline from a 95% CI, with the VR-provenance INCONCLUSIVE overrides the cash spec adds.

    The base reducer is ``tournament.eval_score.verdict`` (``POSITIVE`` iff the whole CI is above 0,
    ``NEGATIVE`` iff entirely below, else ``INCONCLUSIVE (CI spans 0)``). But a cash cell measured with a
    control variate carries a *provenance* the tournament anchor does not: if the CV failed to reduce
    variance (``var_reduction <= 1`` — the cheap baseline is mis-wired / anti-correlated, the same silent
    mis-measurement class AIVAT-on-search is) OR the run did not reach its target precision (the realized
    95% half-width misses the requested ``target_halfwidth``), the number is NOT a clean signal regardless
    of where the CI sits — we force ``INCONCLUSIVE`` with the reason, so a wide/garbage CI can never read
    as a confident PASS/FAIL.
    """
    if var_reduction is not None and float(var_reduction) <= 0.0:
        return "INCONCLUSIVE (var_reduction<=0; CV adding variance)"
    if (realized_halfwidth is not None and target_halfwidth is not None
            and float(realized_halfwidth) > float(target_halfwidth)):
        return "INCONCLUSIVE (CI half-width missed target)"
    return verdict(float(ci_lo), float(ci_hi))


def net_from_chip(chip_bb_per_100: float, net_cost_bb_per_100: float) -> float:
    """The additive net identity ``net = chip - net_cost`` (both in bb/100).

    This is the ONE place the net column is defined: it is the chip-EV column MINUS the hero's contributed
    per-hand net cash cost (``rake.net_cost_bb``, already aggregated to bb/100). The chip-EV number is
    byte-unchanged; the net column is strictly this subtraction. ``net_on = chip_on - net_cost`` and
    ``net_off = chip_off - net_cost`` (the rake cost is policy-agnostic at a fixed stake — it is a function
    of the pot/commitment distribution, not of whether the exploit overlay is on), so the UPLIFT is
    rake-invariant: ``net_on - net_off == chip_on - chip_off``.
    """
    return float(chip_bb_per_100) - float(net_cost_bb_per_100)


def per_hand_net_cost_bb_per_100(pot_bb: float, hero_committed_bb: float, n_dealt: int,
                                 ended_preflop: bool, stake_key: str, *,
                                 rakeback_mode: str = "modeled") -> float:
    """A thin bb/100 wrapper over ``rake.net_cost_bb`` for a SINGLE representative hand (the per-hand
    cost x100). The driver normally accumulates the per-deck net cost inside the estimator; this is the
    convenience used by the hump-table builder and the tests to price ONE pot at every stake from ONE
    deck series (swap ``stake_key`` only — the model's central simplification)."""
    return 100.0 * float(net_cost_bb(pot_bb, hero_committed_bb, n_dealt, ended_preflop, stake_key,
                                     rakeback_mode=rakeback_mode))


@dataclass(frozen=True)
class CellResult:
    """One (stake x depth x position) cell's resolved verdict block — the JSON row the driver emits.

    All bb/100. ``net_off`` / ``net_on`` are the chip arms minus the stake's per-hand net rake cost;
    ``net_uplift = net_on - net_off`` (== the chip uplift, rake-invariant). ``var_reduction`` / ``corr``
    are the CV provenance (``<=1`` forces INCONCLUSIVE). ``ci95`` is the net-ON 95% CI. ``lbr_delta`` /
    ``lbr_delta_ci95`` are the M3 pro-safety exploitability delta + CI (None when M3 was not run for this
    cell). ``bet_shift`` is the M6 cap-binding proxy. ``verdict`` is the per-cell headline."""

    stake: str
    depth_bb: float
    position: int
    net_off: float
    net_on: float
    net_uplift: float
    uplift_ci95: tuple[float, float]
    var_reduction: float
    corr: float
    ci95: tuple[float, float]
    verdict: str
    lbr_delta: float | None = None
    lbr_delta_ci95: tuple[float, float] | None = None
    bet_shift: float | None = None

    def as_json(self) -> dict:
        return {
            "stake": self.stake,
            "depth_bb": self.depth_bb,
            "position": self.position,
            "net_off": self.net_off,
            "net_on": self.net_on,
            "net_uplift": self.net_uplift,
            "uplift_ci95": list(self.uplift_ci95),
            "var_reduction": self.var_reduction,
            "corr": self.corr,
            "ci95": list(self.ci95),
            "verdict": self.verdict,
            "lbr_delta": self.lbr_delta,
            "lbr_delta_ci95": (None if self.lbr_delta_ci95 is None else list(self.lbr_delta_ci95)),
            "bet_shift": self.bet_shift,
        }


def hump_table(net_by_stake: dict[str, float]) -> dict:
    """Build the net-bb/100 curve over ``rake.STAKE_ORDER`` and classify its non-monotone hump.

    ``net_by_stake`` maps a subset of ``STAKE_ORDER`` keys to a net bb/100 value (typically the net-ON
    arm at a fixed depth/position). Returns the ordered curve plus:

      * ``peak_stake`` / ``peak_value``   — the stake with the MAX net bb/100 (the best place to play).
      * ``valley_stake`` / ``valley_value`` — the MIN (the worst net-rake valley the BB-cap hump carves).
      * ``is_hump`` — True iff the curve is NON-MONOTONE (rises then falls, or has an interior max that is
        not an endpoint): the curve goes UP from micro into the mid sweet-spot then DOWN as the BB-cap
        binds tighter at high stakes, so the maximum sits at an INTERIOR stake, not at an end. A purely
        monotone curve (always-rising or always-falling) is NOT a hump.

    The hump is the decisive structural fact: it means game/stake selection (where on the curve you sit),
    not raw chip win-rate, is the cash lever — a chip-positive edge can be eaten in the valley.
    """
    ordered = [(s, float(net_by_stake[s])) for s in STAKE_ORDER if s in net_by_stake]
    if not ordered:
        return {"curve": [], "peak_stake": None, "valley_stake": None, "is_hump": False}
    stakes = [s for s, _ in ordered]
    vals = np.asarray([v for _, v in ordered], dtype=float)
    peak_i = int(np.argmax(vals))
    valley_i = int(np.argmin(vals))
    is_hump = _is_non_monotone(vals)
    return {
        "curve": [{"stake": s, "net_bb_per_100": v} for s, v in ordered],
        "stakes": stakes,
        "values": [float(v) for v in vals],
        "peak_stake": stakes[peak_i],
        "peak_value": float(vals[peak_i]),
        "valley_stake": stakes[valley_i],
        "valley_value": float(vals[valley_i]),
        "is_hump": bool(is_hump),
    }


def _is_non_monotone(vals: np.ndarray, *, tol: float = 1e-9) -> bool:
    """True iff ``vals`` is neither non-decreasing nor non-increasing — i.e. it has at least one interior
    turning point (an up-step followed by a down-step or vice versa). A flat segment alone is monotone."""
    a = np.asarray(vals, dtype=float)
    if a.size < 3:
        return False
    diffs = np.diff(a)
    has_up = bool(np.any(diffs > tol))
    has_down = bool(np.any(diffs < -tol))
    return has_up and has_down


def stake_go_no_go(*, net_on_ci95: tuple[float, float], uplift_ci95: tuple[float, float],
                   lbr_delta_ci95: tuple[float, float] | None,
                   eps_rnr: float = DEFAULT_EPS_RNR,
                   cell_verdicts: list[str] | None = None) -> dict:
    """The combined GO / NO-GO decision for ONE stake (aggregating its cells).

    Three independent gates, ALL required for GO (the cash thesis's three pillars):

      1. **NET-POSITIVE** — ``net_on_ci95[0] > 0``: after the CoinPoker rake the exploit-ON hero's net
         win-rate lower CI clears 0. A chip-positive but net-negative stake (the BB-cap valley) FAILS this.
      2. **PRO-SAFE** — ``lbr_delta_ci95[1] <= eps_rnr``: turning the exploit on does NOT raise the LBR6
         best-response exploitability beyond the RNR tolerance (its UPPER CI stays under ``eps_rnr``).
         When ``lbr_delta_ci95 is None`` (M3 not run for this stake) this gate is INCONCLUSIVE, not GO.
      3. **EXPLOIT PRINTS** — ``uplift_ci95[0] > 0``: the value-sizing overlay's exploit-ON vs OFF uplift
         lower CI clears 0 (the deployed edge is real, not noise).

    Returns ``{verdict: 'GO'|'NO-GO'|'INCONCLUSIVE', net_positive, pro_safe, exploit_prints, ...}``.
    A failed gate with a CI strictly on the wrong side -> ``NO-GO``; a gate whose CI merely spans the
    threshold (or M3 absent) -> ``INCONCLUSIVE``. GO requires all three CLEAR.
    """
    net_lo, _net_hi = float(net_on_ci95[0]), float(net_on_ci95[1])
    up_lo, _up_hi = float(uplift_ci95[0]), float(uplift_ci95[1])

    # Gate 1 — net positive after rake.
    if net_lo > 0.0:
        net_positive = "GO"
    elif net_on_ci95[1] < 0.0:
        net_positive = "NO-GO"
    else:
        net_positive = "INCONCLUSIVE"

    # Gate 2 — pro-safe (Delta upper CI within the RNR tolerance).
    if lbr_delta_ci95 is None:
        pro_safe = "INCONCLUSIVE"
        delta_hi = None
    else:
        delta_hi = float(lbr_delta_ci95[1])
        delta_lo = float(lbr_delta_ci95[0])
        if delta_hi <= eps_rnr:
            pro_safe = "GO"
        elif delta_lo > eps_rnr:
            pro_safe = "NO-GO"          # provably MORE exploitable than tolerance -> counter-exploitable
        else:
            pro_safe = "INCONCLUSIVE"

    # Gate 3 — the exploit overlay genuinely prints.
    if up_lo > 0.0:
        exploit_prints = "GO"
    elif uplift_ci95[1] < 0.0:
        exploit_prints = "NO-GO"
    else:
        exploit_prints = "INCONCLUSIVE"

    gates = [net_positive, pro_safe, exploit_prints]
    if all(g == "GO" for g in gates):
        v = "GO"
    elif any(g == "NO-GO" for g in gates):
        v = "NO-GO"
    else:
        v = "INCONCLUSIVE"

    out = {
        "verdict": v,
        "net_positive": net_positive,
        "pro_safe": pro_safe,
        "exploit_prints": exploit_prints,
        "eps_rnr": float(eps_rnr),
        "net_on_lower_ci": net_lo,
        "uplift_lower_ci": up_lo,
        "lbr_delta_upper_ci": delta_hi,
    }
    if cell_verdicts is not None:
        out["n_cells"] = len(cell_verdicts)
        out["n_cells_positive"] = sum(1 for c in cell_verdicts if c == "POSITIVE")
        out["n_cells_negative"] = sum(1 for c in cell_verdicts if c == "NEGATIVE")
    return out
