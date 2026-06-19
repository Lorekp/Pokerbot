"""Tests for the ICM open-shove (first-in jam) solver (pokerbot/tournament/open_shove_icm.py).

Validation: it must REDUCE EXACTLY to the certified HU push/fold solver with one player behind
(internal consistency with a known-good anchor), and produce correct positional monotonicity
(SB widest -> UTG tightest), premium/trash behavior, and small measured exploitability.
"""

from __future__ import annotations

import pytest

from pokerbot.tournament.open_shove_icm import solve_open_shove_icm as openjam
from pokerbot.tournament.pushfold_icm import solve_hu_icm_pushfold as husolve


def test_reduces_to_hu_solver():
    # SB open with only BB behind == the exact heads-up push/fold game.
    hu = husolve([10, 10], [1, 0], sb=0, bb=1, iters=4000)
    oj = openjam([10, 10], [1, 0], hero=0, behind=[1], blinds={0: 0.5, 1: 1.0}, iters=3000)
    assert oj.hero_jam_pct == pytest.approx(hu.sb_jam_pct, abs=0.01)
    assert oj.nash_gap < 1e-4


def _positions():
    stacks = [10] * 6
    pay = [40, 30, 20, 10]
    bl = {4: 0.5, 5: 1.0}                      # seats: 0..5 = UTG,HJ,CO,BTN,SB,BB
    out = {}
    for name, seat, behind in [("UTG", 0, [1, 2, 3, 4, 5]), ("HJ", 1, [2, 3, 4, 5]),
                               ("CO", 2, [3, 4, 5]), ("BTN", 3, [4, 5]), ("SB", 4, [5])]:
        out[name] = openjam(stacks, pay, hero=seat, behind=behind, blinds=bl, iters=2500)
    return out


def test_positional_monotonicity():
    sols = _positions()
    pcts = [sols[p].hero_jam_pct for p in ("UTG", "HJ", "CO", "BTN", "SB")]
    assert all(pcts[i] <= pcts[i + 1] + 1e-9 for i in range(len(pcts) - 1))   # widen toward the button
    assert sols["UTG"].hero_jam_pct < 0.35       # UTG is tight at 10bb
    assert sols["SB"].hero_jam_pct > 0.80        # SB jams very wide vs only the BB


def test_converges_small_gap():
    for sol in _positions().values():
        assert sol.nash_gap < 5e-3                # measured residual (multiplayer: not a theorem)


def test_premium_in_trash_out():
    sols = _positions()
    for pos in ("UTG", "BTN", "SB"):
        assert sols[pos].jam_freq("AA") > 0.99
    assert sols["UTG"].jam_freq("72o") < 0.05     # UTG never opens trash at 10bb


def test_antes_widen_ranges():
    # CoinPoker tournaments ante from L1; antes add dead money -> wider opening ranges.
    pay = [40, 30, 20, 10]
    bl = {4: 0.5, 5: 1.0}
    stacks = [10] * 6
    no_ante = openjam(stacks, pay, hero=2, behind=[3, 4, 5], blinds=bl, ante=0.0, iters=3000)
    ante = openjam(stacks, pay, hero=2, behind=[3, 4, 5], blinds=bl, ante=0.25, iters=3000)
    assert ante.hero_jam_pct > no_ante.hero_jam_pct + 0.05
    assert ante.nash_gap < 5e-3


def test_shorter_stack_widens():
    pay = [40, 30, 20, 10]
    bl = {4: 0.5, 5: 1.0}
    short = openjam([6] * 6, pay, hero=2, behind=[3, 4, 5], blinds=bl, iters=2500)   # CO 6bb
    deep = openjam([14] * 6, pay, hero=2, behind=[3, 4, 5], blinds=bl, iters=2500)   # CO 14bb
    assert short.hero_jam_pct > deep.hero_jam_pct
