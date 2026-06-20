"""7-max always-fold calibration anchor + its grounding in the seat-parameterized engine.

The closed-form anchor is validated two ways: (1) it reproduces the empirically-measured 6-max
harness anchor (+128 band) at 6-max no-ante, and (2) its dead-money accounting matches what the real
JAX engine posts at a 7-seat ante table -- so the +203.6 bb/100 7-max number rests on the actual
engine, not just arithmetic.
"""

from __future__ import annotations

import pytest

from pokerbot.tournament.anchor import (
    SIXMAX_NOANTE_BAND,
    always_fold_exploitability,
    sevenmax_tournament_anchor,
    sixmax_anchor,
)


def test_sixmax_matches_empirical_band():
    a = sixmax_anchor()
    assert a == pytest.approx(125.0, abs=1e-9)
    lo, hi = SIXMAX_NOANTE_BAND
    assert lo <= a <= hi  # the formula lands inside the measured +128 band


def test_sevenmax_tournament_anchor_value():
    a = sevenmax_tournament_anchor()  # ante = 0.125 bb
    # (6*1.5 + 7*6*0.125)/7 * 100 = 14.25/7*100 = 203.571...
    assert a == pytest.approx(203.5714, abs=1e-3)
    assert a > sixmax_anchor()  # more folders + ante dead money => higher than 6-max


def test_monotonic_in_seats_and_ante():
    seats = [always_fold_exploitability(n, ante_bb=0.125) for n in range(2, 8)]
    assert all(b > a for a, b in zip(seats, seats[1:]))  # strictly increasing in seat count
    assert always_fold_exploitability(7, ante_bb=0.2) > always_fold_exploitability(7, ante_bb=0.0)


def test_rejects_degenerate_seat_count():
    with pytest.raises(ValueError):
        always_fold_exploitability(1)


def test_engine_grounds_dead_money_at_seven_seats():
    """The anchor's per-hand value is the dead money a fold-around winner collects. Confirm the real
    7-seat JAX engine posts exactly that (within ante display-rounding: 13 chips vs 12.5)."""
    import jax
    import jax.numpy as jnp

    from pokerbot.engine import jax_engine as je

    n = 7
    ante_chips = 13  # round_half_up(BB/8) = round(100/8) = 13 (CoinPoker MTT ante)
    stacks = jnp.full((n,), 200 * je.CHIP_UNIT, dtype=jnp.int32)
    occ = jnp.ones((n,), dtype=bool)
    state = je.init_hand(jax.random.PRNGKey(0), button=0, stacks=stacks, occupancy=occ,
                         ante=ante_chips, n_players=n)

    committed_total = [int(x) for x in state.committed_total]
    # every dealt seat posts the ante; SB and BB additionally post their blinds
    assert sum(committed_total) == n * ante_chips + je.SB_CHIPS + je.BB_CHIPS

    # dead money a NON-blind seat wins on a fold-around = sum of the OTHER seats' committed chips.
    # With button=0, the blinds are seats 1 (SB) and 2 (BB); pick a non-blind winner, e.g. seat 4.
    winner = 4
    others_dead_bb = (sum(committed_total) - committed_total[winner]) / je.CHIP_UNIT
    analytic_nonblind = 0.5 + 1.0 + (n - 1) * 0.125  # sb + bb + others' antes (ideal 12.5-chip ante)
    assert others_dead_bb == pytest.approx(analytic_nonblind, abs=0.05)  # within ante rounding
