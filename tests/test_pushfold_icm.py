"""Tests for the ICM heads-up push/fold solver (pokerbot/tournament/pushfold_icm.py).

Validation strategy:
  * winner-take-all over 2 players == chip-EV Nash push/fold -> reproduce the PUBLISHED Nash charts
    (10bb HU: ~58% SB shove / ~37% BB call), which is the strongest possible known-answer check;
  * convergence: the ICM Nash gap (exploitability) drives to ~0;
  * ICM correctness: heads-up is ICM-neutral (== WTA), but a short-stack bystander on the bubble
    makes BB calling collapse AND SB shoving widen (the bubble bully) -- both sides of ICM.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.tournament.pushfold_icm import solve_hu_icm_pushfold as solve


def _bucket(sol, label):
    return list(sol.labels).index(label)


def test_converges_to_nash():
    for stacks, pay in [([10, 10], [1, 0]), ([10, 10, 2], [50, 30, 20]), ([8, 12], [1, 0])]:
        sol = solve(stacks, pay, sb=0, bb=1, iters=4000)
        assert sol.nash_gap < 1e-4               # exploitability ~ 0 (well-converged)


def test_wta_matches_published_nash_10bb():
    # Winner-take-all 2-player == chip-EV: published Nash push/fold at 10bb is ~58% shove / ~37% call.
    sol = solve([10.0, 10.0], [1.0, 0.0], sb=0, bb=1, iters=5000)
    assert 0.52 <= sol.sb_jam_pct <= 0.64
    assert 0.32 <= sol.bb_call_pct <= 0.43


def test_premium_in_trash_out():
    sol = solve([10.0, 10.0], [1.0, 0.0], sb=0, bb=1, iters=4000)
    for h in ("AA", "KK", "AKs"):
        assert sol.sb_jam[_bucket(sol, h)] > 0.99
        assert sol.bb_call[_bucket(sol, h)] > 0.99
    assert sol.sb_jam[_bucket(sol, "72o")] < 0.01
    assert sol.bb_call[_bucket(sol, "72o")] < 0.01


def test_shorter_stack_is_wider():
    s5 = solve([5, 5], [1, 0], sb=0, bb=1, iters=4000)
    s10 = solve([10, 10], [1, 0], sb=0, bb=1, iters=4000)
    s15 = solve([15, 15], [1, 0], sb=0, bb=1, iters=4000)
    assert s5.sb_jam_pct > s10.sb_jam_pct > s15.sb_jam_pct
    assert s5.bb_call_pct > s10.bb_call_pct > s15.bb_call_pct


def test_heads_up_is_icm_neutral():
    # 2 players left: chips are $-linear, so an ICM payout structure == winner-take-all here.
    wta = solve([10, 10], [1, 0], sb=0, bb=1, iters=5000)
    icm = solve([10, 10], [60, 40], sb=0, bb=1, iters=5000)
    assert icm.sb_jam_pct == pytest.approx(wta.sb_jam_pct, abs=0.02)
    assert icm.bb_call_pct == pytest.approx(wta.bb_call_pct, abs=0.02)


def test_bubble_tightens_calls_and_widens_shoves():
    # Two 10bb stacks battle while a 2bb short stack is about to bust (3 left, laddered payouts).
    hu = solve([10, 10], [60, 40], sb=0, bb=1, iters=5000)
    bub = solve([10, 10, 2], [50, 30, 20], sb=0, bb=1, iters=5000)
    # Calling collapses (don't bust on the bubble) ...
    assert bub.bb_call_pct < hu.bb_call_pct - 0.10
    # ... while shoving widens (fold equity explodes -> the big stack bullies the bubble).
    assert bub.sb_jam_pct > hu.sb_jam_pct + 0.10
    assert bub.nash_gap < 1e-4


def test_antes_widen_hu_ranges():
    # Antes (CoinPoker structure) add dead money -> both players play wider; still converged.
    no_ante = solve([10, 10], [1, 0], sb=0, bb=1, ante=0.0, iters=5000)
    ante = solve([10, 10], [1, 0], sb=0, bb=1, ante=0.125, iters=5000)
    assert ante.sb_jam_pct > no_ante.sb_jam_pct + 0.02
    assert ante.bb_call_pct > no_ante.bb_call_pct + 0.02
    assert ante.nash_gap < 1e-4


def test_hand_to_label():
    from pokerbot.tournament.pushfold_icm import hand_to_label
    assert hand_to_label("AhKd") == "AKo"
    assert hand_to_label("AhKh") == "AKs"
    assert hand_to_label("AsAh") == "AA"
    assert hand_to_label("Kd2h") == "K2o"


def test_decision_api():
    from pokerbot.tournament.pushfold_icm import decide_hu_pushfold
    # 10bb HU (chip-EV via WTA): SB jams AA, folds 72o; BB calls a strong hand.
    jam = decide_hu_pushfold([10, 10], [1, 0], hero=0, villain=1, hero_hand="AsAh", role="sb")
    assert jam["action"] == "jam" and jam["freq"] > 0.99
    fold = decide_hu_pushfold([10, 10], [1, 0], hero=0, villain=1, hero_hand="7c2d", role="sb")
    assert fold["action"] == "fold" and fold["freq"] < 0.01
    call = decide_hu_pushfold([10, 10], [1, 0], hero=0, villain=1, hero_hand="AsKh", role="bb")
    assert call["action"] == "call"
    assert jam["nash_gap"] < 1e-4


def test_range_pcts_are_valid_probabilities():
    sol = solve([10, 10], [60, 40], sb=0, bb=1, iters=2000)
    assert 0.0 <= sol.sb_jam_pct <= 1.0
    assert 0.0 <= sol.bb_call_pct <= 1.0
    assert (sol.sb_jam >= -1e-9).all() and (sol.sb_jam <= 1 + 1e-9).all()
    assert sol.weights.sum() == pytest.approx(1326.0)   # 13*6 + 78*4 + 78*12
