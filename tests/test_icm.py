"""Unit tests for the ICM primitive (pokerbot/tournament/icm.py).

Closed-form small-N values, prize conservation, the short-stack ICM premium (concavity),
bubble-factor limits (heads-up covered == 1.0; bubble >> 1), and the icm_delta bust case.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.tournament import icm


# --------------------------------------------------------------------------- #
# icm_equity: closed forms + conservation
# --------------------------------------------------------------------------- #
def test_two_player_closed_form():
    # eq_i = pay0 * s_i/S + pay1 * s_j/S
    eq = icm.icm_equity([3.0, 1.0], [70.0, 30.0])
    assert eq == pytest.approx([60.0, 40.0])


def test_equal_stacks_split_prize():
    eq = icm.icm_equity([1.0, 1.0, 1.0], [60.0, 30.0, 10.0])
    assert eq == pytest.approx([100.0 / 3] * 3)


def test_three_player_known_value():
    # Hand-computed Malmuth-Harville for stacks [50,30,20], payouts [70,20,10].
    eq = icm.icm_equity([50.0, 30.0, 20.0], [70.0, 20.0, 10.0])
    assert eq == pytest.approx([43.393, 31.750, 24.857], abs=1e-2)


def test_prize_conserved():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 8))
        stacks = rng.uniform(1, 100, size=n).tolist()
        npaid = int(rng.integers(1, n + 1))
        payouts = sorted(rng.uniform(1, 100, size=npaid).tolist(), reverse=True)
        eq = icm.icm_equity(stacks, payouts)
        assert eq.sum() == pytest.approx(sum(payouts), rel=1e-9)
        assert (eq >= -1e-9).all()


def test_payouts_padded_when_shorter_than_field():
    # 4 players, only top 2 paid -> 3rd/4th get 0; prize still conserved.
    eq = icm.icm_equity([40, 30, 20, 10], [60, 40])
    assert eq.sum() == pytest.approx(100.0)
    # chip leader has the most equity, shortest the least
    assert eq[0] > eq[1] > eq[2] > eq[3]


def test_short_stack_icm_premium():
    # ICM is concave in chips: the short stack is worth MORE per chip than the leader.
    stacks = [50.0, 30.0, 20.0]
    eq = icm.icm_equity(stacks, [70.0, 20.0, 10.0])
    per_chip = eq / np.asarray(stacks)
    assert per_chip[2] > per_chip[0]            # short stack > leader, per chip
    assert eq[0] < stacks[0] / sum(stacks) * 100  # leader's $ share < chip share


def test_monotone_in_stack():
    base = icm.icm_equity([40, 30, 30], [50, 30, 20])
    bigger = icm.icm_equity([60, 30, 30], [50, 30, 20])
    assert bigger[0] > base[0]                  # more chips -> more equity


# --------------------------------------------------------------------------- #
# bubble_factor: limits + monotonicity
# --------------------------------------------------------------------------- #
def test_bubble_factor_headsup_covered_is_one():
    # Heads-up, both covered, winner-take-most: chips are $-linear -> BF == 1.0 exactly.
    bf = icm.bubble_factor([50.0, 50.0], [60.0, 40.0], hero=0, villain=1)
    assert bf == pytest.approx(1.0, abs=1e-9)


def test_bubble_factor_on_the_bubble_is_large():
    # 4 equal stacks, 3 paid (4th = 0): classic bubble. BF ~ 1.875 (hand-computed).
    bf = icm.bubble_factor([40.0, 40.0, 40.0, 40.0], [50.0, 30.0, 20.0], hero=0, villain=1)
    assert bf == pytest.approx(1.875, abs=2e-2)
    assert bf > 1.5


def test_bubble_factor_monotone_toward_money():
    # Same per-seat stacks + payout shape, but the bubble (4 players, 3 paid) has higher
    # pressure than a heads-up for the title.
    bf_bubble = icm.bubble_factor([40, 40, 40, 40], [50, 30, 20], hero=0, villain=1)
    bf_hu = icm.bubble_factor([40, 40], [50, 30], hero=0, villain=1)
    assert bf_bubble > bf_hu >= 1.0


def test_bubble_factor_infinite_when_no_gain():
    # Hero already holds every chip an opponent could yield nothing extra -> reward 0 -> inf.
    bf = icm.bubble_factor([100.0, 0.0], [60.0, 40.0], hero=0, villain=1)
    assert bf == float("inf")


# --------------------------------------------------------------------------- #
# icm_delta: the chip -> $ wrap, including a bust
# --------------------------------------------------------------------------- #
def test_icm_delta_zero_sum_in_dollars():
    pre = [50.0, 30.0, 20.0]
    chip_delta = [+10.0, -10.0, 0.0]            # A wins 10 from B, nobody busts
    d = icm.icm_delta(pre, chip_delta, [70.0, 20.0, 10.0])
    assert d.sum() == pytest.approx(0.0, abs=1e-9)   # $ is conserved (prize pool fixed)
    assert d[0] > 0 and d[1] < 0


def test_icm_delta_bust_realizes_bottom_payout():
    # A busts C (wins C's 20). Post stacks [70,30,0] -> C finishes 3rd for payout 10.
    pre = [50.0, 30.0, 20.0]
    chip_delta = [+20.0, 0.0, -20.0]
    d = icm.icm_delta(pre, chip_delta, [70.0, 20.0, 10.0])
    post_eq = icm.icm_equity([70.0, 30.0, 0.0], [70.0, 20.0, 10.0])
    assert post_eq[2] == pytest.approx(10.0)    # buster locked into 3rd-place money
    assert d.sum() == pytest.approx(0.0, abs=1e-9)
    assert d[0] > 0 and d[2] < 0                # winner gains $, buster loses the most


def test_icm_delta_chip_neutral_hand_is_dollar_neutral_when_far():
    # Heads-up covered: a chip swing that is later reversed nets ~0; single swing is monotone.
    pre = [100.0, 100.0]
    d = icm.icm_delta(pre, [+20.0, -20.0], [60.0, 40.0])
    assert d.sum() == pytest.approx(0.0, abs=1e-9)
    assert d[0] > 0 > d[1]


def test_icm_delta_length_mismatch_raises():
    with pytest.raises(ValueError):
        icm.icm_delta([50.0, 50.0], [10.0], [60.0, 40.0])
