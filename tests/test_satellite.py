"""Satellite step-payoff + seat-lock tests (extreme ICM, fold-when-locked)."""

from __future__ import annotations

import math

from pokerbot.tournament.satellite import (
    satellite_payouts,
    seat_bubble_factor,
    seat_locked,
    should_fold_for_seat,
)


def test_payout_vector_is_flat_step():
    assert satellite_payouts(10, 530.0) == [530.0] * 10
    assert satellite_payouts(3) == [1.0, 1.0, 1.0]  # unit-agnostic default
    assert satellite_payouts(0) == []


def test_seat_locked_chip_fraction_theorem():
    # seats=2, total=10000 -> lock threshold = 10000/3 ~= 3333.3
    # hero 6000 >= 3333 -> locked (cannot carve 2 rival piles each > 6000 from the other 4000)
    assert seat_locked([6000, 2000, 2000], 2, 0) is True
    # hero 3000 < 3333 -> NOT locked (adversary could make two stacks > 3000 from the other 7000)
    assert seat_locked([3000, 3500, 3500], 2, 0) is False
    # exactly at the boundary is locked (>=)
    assert seat_locked([4000, 4000, 4000], 2, 0) is True  # 4000 >= 12000/3 == 4000


def test_seat_locked_trivial_and_busted():
    # at most `seats` players remain -> everyone left is paid -> locked
    assert seat_locked([100, 100, 0, 0], 2, 0) is True
    # a busted hero is never locked
    assert seat_locked([0, 5000, 5000], 2, 0) is False


def test_should_fold_when_locked():
    locked = [6000, 2000, 2000]
    assert should_fold_for_seat(locked, 2, 0, action_risks_chips=True) is True
    # a zero-risk action is never forced to fold
    assert should_fold_for_seat(locked, 2, 0, action_risks_chips=False) is False
    # not locked -> do not force a fold
    assert should_fold_for_seat([3000, 3500, 3500], 2, 0, action_risks_chips=True) is False


def test_extreme_icm_under_step_payoff():
    # The flat step payoff (4 players, 2 equal seats) carries STRONG ICM pressure essentially
    # everywhere -- unlike a cash ladder there is no "deep / far from the money" relief, because the
    # next elimination is always a pay jump. So a contested all-in for a large fraction of equal
    # stacks has a bubble factor well above 1 (demand much more equity to stack off).
    bf_equal = seat_bubble_factor([100000, 98000, 102000, 99000], 2, 0, 1, seat_value=530.0)
    assert bf_equal > 1.3, bf_equal
    # a short-vs-short skirmish (risking only the smaller stack) still tightens, BF > 1.
    bf_short = seat_bubble_factor([4000, 3000, 2000, 1000], 2, 2, 3, seat_value=530.0)
    assert bf_short > 1.0
    assert not math.isinf(bf_equal)
