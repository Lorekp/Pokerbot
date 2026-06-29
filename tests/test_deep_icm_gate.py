"""Tests for the deep (>20bb) ICM shove/fold gate (pokerbot/tournament/deep_icm_gate.py).

Locks the measured value-add (deep_icm_grid): near the money, ICM folds a +chip-EV but -$EV shove even at
DEEP stacks; it is a no-op for clear hands and far from the money.
"""

from __future__ import annotations

from pokerbot.tournament.deep_icm_gate import deep_icm_overrides_shove, deep_icm_shove_fold

PAY = [50.0, 30.0, 20.0]
DEEP_NEAR_MONEY = [35.0, 45.0, 27.0, 20.0, 12.0, 7.0]   # hero 35bb (DEEP) near the money (two bubble shorts)


def test_icm_folds_plus_chipev_shove_at_marginal_equity_DEEP():
    # 0.55 equity -> shove is +chip-EV, but near the money ICM folds it -> the deep value-add the grid found
    a, flipped = deep_icm_shove_fold(DEEP_NEAR_MONEY, PAY, hero=0, villain=1, hero_equity=0.55, risked=35.0)
    assert a == "fold" and flipped
    assert deep_icm_overrides_shove(DEEP_NEAR_MONEY, PAY, 0, 1, 0.55, 35.0) is True


def test_clear_monster_shoves_under_both_no_override():
    a, flipped = deep_icm_shove_fold(DEEP_NEAR_MONEY, PAY, hero=0, villain=1, hero_equity=0.85, risked=35.0)
    assert a == "shove" and not flipped
    assert deep_icm_overrides_shove(DEEP_NEAR_MONEY, PAY, 0, 1, 0.85, 35.0) is False


def test_far_from_money_is_a_no_op():
    # equal deep stacks + winner-take-all -> bubble_factor ~1 -> ICM == chip-EV -> never overrides
    eq_stacks = [50.0] * 6
    a, flipped = deep_icm_shove_fold(eq_stacks, [100.0, 0, 0, 0, 0, 0], hero=0, villain=1,
                                     hero_equity=0.55, risked=50.0)
    assert not flipped
    assert deep_icm_overrides_shove(eq_stacks, [100.0, 0, 0, 0, 0, 0], 0, 1, 0.55, 50.0) is False
