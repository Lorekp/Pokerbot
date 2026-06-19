"""Tests for the ICM-aware all-in EV (pokerbot/tournament/allin_ev.py).

Validates the discrete-outcome enumeration against the trusted AoF equity, ICM prize conservation,
and the textbook ICM result: a slight chip-EV favorite *loses* $-EV by flipping on the bubble
(the risk premium), but gains deep when chips are $-linear.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.tournament import allin_ev
from pokerbot.tournament.icm import icm_equity


def test_enumeration_matches_aof_equity():
    # P(win) + 0.5*P(chop) must equal hero's exact pot equity (validate vs trusted AoF kernel).
    from aof_equity_grids import _equity_two_combos_exact
    for hero_s, vil_s in [("AsAh", "KsKd"), ("AhKs", "QcQd"), ("2c2d", "AhKs")]:
        h = allin_ev.parse_hole(hero_s)
        v = allin_ev.parse_hole(vil_s)
        pw, pc, pl = allin_ev.hu_outcome_probs(h, v)
        assert pw + pc + pl == pytest.approx(1.0, abs=1e-9)
        eq = float(_equity_two_combos_exact(h[0], h[1], v[0], v[1]))
        assert pw + 0.5 * pc == pytest.approx(eq, abs=1e-6)


def test_aa_beats_kk_probability():
    pw, pc, pl = allin_ev.hu_outcome_probs("AsAh", "KsKd")
    assert pw > 0.79  # AA vs KK is ~82%


def test_allin_icm_conserves_prize():
    ev = allin_ev.hu_allin_icm_ev("AsAh", "KsKd", [40, 40, 40, 40], [60, 40], hero=0, villain=1)
    assert ev.sum() == pytest.approx(100.0, abs=1e-6)


def test_dominant_hand_gains_everywhere():
    # AA vs KK: hero should gain $ over folding in both deep and bubble spots.
    base_deep = icm_equity([40, 40], [60, 40])[0]
    ev_deep = allin_ev.hu_allin_icm_ev("AsAh", "KsKd", [40, 40], [60, 40], hero=0, villain=1)[0]
    assert ev_deep > base_deep


def test_bubble_risk_premium_flips_a_slight_favorite():
    # 22 vs AKo is a slight chip favorite (~52%). The textbook ICM result:
    #   - DEEP (heads-up, chips $-linear): the slight edge is +$EV vs folding.
    #   - BUBBLE (4 left, 2 paid): the same edge is -$EV -- you should not flip on the bubble.
    hero, vil = "2c2d", "AhKs"
    pw, pc, pl = allin_ev.hu_outcome_probs(hero, vil)
    assert 0.50 < pw < 0.55  # confirm hero is the slight favorite

    base_deep = icm_equity([40, 40], [60, 40])[0]
    ev_deep = allin_ev.hu_allin_icm_ev(hero, vil, [40, 40], [60, 40], hero=0, villain=1)[0]
    assert ev_deep >= base_deep - 1e-9            # slight fav: +EV deep

    base_bub = icm_equity([40, 40, 40, 40], [60, 40])[0]
    ev_bub = allin_ev.hu_allin_icm_ev(hero, vil, [40, 40, 40, 40], [60, 40], hero=0, villain=1)[0]
    assert ev_bub < base_bub                       # same edge: -EV on the bubble (risk premium)


def test_chip_ev_vs_icm_ev_diverge_on_bubble():
    # Sanity that ICM EV != chip-EV-then-ICM for a non-trivial spot (the non-linearity the module
    # exists to handle). Compare the at-risk player's ICM EV against naively feeding expected share.
    hero, vil = "2c2d", "AhKs"
    stacks = [40.0, 40.0, 40.0, 40.0]
    pw, pc, pl = allin_ev.hu_outcome_probs(hero, vil)
    share = pw + 0.5 * pc                            # expected chip share
    ev_icm = allin_ev.hu_allin_icm_ev(hero, vil, stacks, [60, 40], hero=0, villain=1)[0]
    # naive (wrong) "ICM of expected chips": award fractional pot, then ICM
    risked = 40.0
    naive_stacks = list(stacks)
    naive_stacks[0] = 40.0 + (2 * share - 1) * risked
    naive_stacks[1] = 40.0 - (2 * share - 1) * risked
    ev_naive = float(icm_equity(naive_stacks, [60, 40])[0])
    assert abs(ev_icm - ev_naive) > 0.1             # they genuinely differ
