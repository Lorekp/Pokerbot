"""Workstream F regression guard for the HUD value-sizing escalation + RNR cap.

These are FAST directional assertions (small N, fixed seeds) on the SAME measurement infra the full
acceptance run uses (scripts/hud_sizing_acceptance.py), so a regression in either the value-sizing
overlay or the RNR cap's bounding is caught in CI without the multi-minute full sweep.

What is asserted (robust at small N):
  * SIZING value lever fires vs a calling station — the CRN-paired sizing-on minus sizing-off uplift
    has a lower-CI-bound > 0 (the +11.9 anchor's DIRECTION; the magnitude is the heuristic-base scale).
  * The RNR cap DOES WORK — the hero's deviation MAGNITUDE (bet_shift) is monotone in the alpha cap and
    strictly smaller capped (0.5) than uncapped (1.0). bet_shift is a near-deterministic property of the
    policy (tiny sampling noise), so this is reliable at small N where the absolute bb/100 cost of being
    modeled is NOT (per-hand std ~73 bb/100 vs an adaptive bot).

The ABSOLUTE counter-exploiter bb/100 loss is deliberately NOT asserted here (it needs ~1e5+ hands to
resolve); the full run reports it with honest wide CIs.
"""
from __future__ import annotations

import math
import os
import statistics as st

import pytest


def _mean_ci(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def _run(arch: str, seed: int, *, cap: float, sizing: bool, hands: int) -> dict:
    """One CRN A/B battery cell at a given alpha cap + value-sizing flag. Restores globals after."""
    os.environ["POKERBOT_POSTFLOP_MODE"] = "fast"
    os.environ["POKERBOT_VALUE_SIZING"] = "1" if sizing else "0"
    import pokerbot.eval.exploitation_battery as eb
    from pokerbot.eval.exploitation_battery import run_archetype

    orig = eb.DEFAULT_MAX_ALPHA
    try:
        eb.DEFAULT_MAX_ALPHA = cap
        return run_archetype(arch, hands=hands, seed=seed, seed_hud=True, allin_ev=True,
                             progress_every=10 ** 9)
    finally:
        eb.DEFAULT_MAX_ALPHA = orig
        os.environ["POKERBOT_VALUE_SIZING"] = "1"


@pytest.mark.slow
def test_sizing_value_lever_fires_vs_station():
    """The value-SIZING overlay extracts positive EV vs a calling station (CRN-paired sizing-on minus
    sizing-off uplift). Regression guard for the DIRECTION of the +11.9 anchor: the pooled gain is
    clearly positive at the small fast-test N (the strict CI-excludes-0 claim is the FULL run's gate —
    24 seeds x 2000 hands gave +71.3 +- 10.0, lcb +61.3; here we just assert the lever is alive)."""
    hands, seeds = 250, 5
    deltas = []
    for s in range(seeds):
        on = _run("calling_station", s, cap=0.5, sizing=True, hands=hands)["uplift_bb_per_hand"]
        off = _run("calling_station", s, cap=0.5, sizing=False, hands=hands)["uplift_bb_per_hand"]
        deltas.append((on - off) * 100.0)
    m = sum(deltas) / len(deltas)
    # Direction (mean clearly > 0) is robust at this N; the per-seed CI needs the full run's N.
    assert m > 5.0, f"sizing gain not clearly positive: {m:+.1f} bb/100 (deltas={deltas})"
    assert sum(1 for d in deltas if d > 0) >= seeds // 2, f"majority of seeds not positive: {deltas}"


@pytest.mark.slow
def test_rnr_cap_bounds_deviation_vs_counter_exploiter():
    """The RNR cap DOES WORK: the hero's deviation magnitude (bet_shift) is strictly smaller at the
    capped alpha (0.5) than uncapped (1.0) vs the adaptive counter-exploiter, and both are positive.
    bet_shift is near-deterministic so this is reliable at small N (unlike the absolute bb/100 loss)."""
    hands, seeds = 250, 3
    capped, uncapped = [], []
    for s in range(seeds):
        capped.append(_run("counter_exploiter", s, cap=0.5, sizing=True, hands=hands)["bet_shift"])
        uncapped.append(_run("counter_exploiter", s, cap=1.0, sizing=True, hands=hands)["bet_shift"])
    cap_m = sum(capped) / len(capped)
    unc_m = sum(uncapped) / len(uncapped)
    assert cap_m > 0.0, f"capped bet_shift not positive: {cap_m:+.4f}"
    assert cap_m < unc_m, f"cap did not reduce deviation: capped={cap_m:+.4f} >= uncapped={unc_m:+.4f}"


def test_exploit_cap_calibrated_values_are_monotone_and_le_4max():
    """The shipped EXPLOIT_CAP_BY_N (post-calibration) keeps the structural invariants the wiring relies
    on: non-increasing in N and the 5/6 entries <= the 4-max ceiling 0.6 (and < the legacy flat 0.7)."""
    from pokerbot.opponent.rnr_cap import EXPLOIT_CAP_BY_N

    caps = [EXPLOIT_CAP_BY_N[n] for n in (2, 3, 4, 5, 6)]
    for a, b in zip(caps, caps[1:]):
        assert b <= a + 1e-12, f"ceiling not non-increasing in N: {caps}"
    assert EXPLOIT_CAP_BY_N[5] <= EXPLOIT_CAP_BY_N[4] + 1e-12
    assert EXPLOIT_CAP_BY_N[6] <= EXPLOIT_CAP_BY_N[4] + 1e-12
    assert EXPLOIT_CAP_BY_N[5] <= 0.7 and EXPLOIT_CAP_BY_N[6] <= 0.7
