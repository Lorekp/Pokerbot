"""Multi-table tournament FIELD model (pokerbot/tournament/mtt_field.py).

The load-bearing correctness check: the scalable Monte-Carlo Malmuth-Harville field ICM must AGREE with the
exact recursive primitive (pokerbot.tournament.icm.icm_equity) at small N — that anchors the MC as the same
model, just tractable at field scale. Plus prize conservation, stack-monotonicity, bubble-factor sanity, and
the table-balancing seat-flow rules.
"""

from __future__ import annotations

import random

import numpy as np

from pokerbot.tournament.icm import icm_equity
from pokerbot.tournament.mtt_field import (
    MttField,
    MttTable,
    apply_moves,
    balance_moves,
    field_bubble_factor,
    field_icm_equity,
    needs_balancing,
)


def test_mc_matches_exact_icm_small_field():
    """MC field ICM converges to the exact Malmuth-Harville equity at small N (the anchoring check)."""
    stacks = [4000.0, 3000.0, 2000.0, 1000.0]
    payouts = [50.0, 30.0, 20.0]
    exact = icm_equity(stacks, payouts)
    rng = random.Random(7)
    mc = np.array([field_icm_equity(stacks, payouts, i, n_sims=30000, rng=rng) for i in range(4)])
    # within ~2% of total prize (MC noise at 30k sims)
    assert np.allclose(mc, exact, atol=0.02 * sum(payouts)), f"mc={mc} exact={exact}"


def test_prize_conservation():
    stacks = [5000.0, 2500.0, 1500.0, 800.0, 200.0]
    payouts = [60.0, 25.0, 15.0]
    rng = random.Random(1)
    total = sum(field_icm_equity(stacks, payouts, i, n_sims=20000, rng=rng) for i in range(len(stacks)))
    assert abs(total - sum(payouts)) < 0.03 * sum(payouts)


def test_equity_monotonic_in_stack():
    stacks = [10000.0, 5000.0, 3000.0, 2000.0, 1000.0, 500.0]
    payouts = [100.0, 60.0, 40.0]
    rng = random.Random(3)
    eqs = [field_icm_equity(stacks, payouts, i, n_sims=20000, rng=rng) for i in range(len(stacks))]
    # bigger stack -> at least as much $-equity (allow tiny MC noise)
    for i in range(len(stacks) - 1):
        assert eqs[i] >= eqs[i + 1] - 0.01 * sum(payouts), f"non-monotone at {i}: {eqs}"


def test_bubble_factor_above_one_near_money():
    # 4 left, 3 paid -> exactly on the bubble: busting is far costlier than doubling -> BF >> 1
    stacks = [4000.0, 3000.0, 2000.0, 1000.0]
    payouts = [50.0, 30.0, 20.0]
    bf = field_bubble_factor(stacks, payouts, hero_idx=2, villain_idx=3, n_sims=8000, rng=random.Random(5))
    assert bf > 1.2, f"expected bubble premium > 1.2, got {bf:.3f}"


def test_bubble_factor_near_one_when_deep():
    # far from the money (everyone deep, only top-1 of a huge field paid relative to stacks) -> ~chip-linear
    stacks = [1000.0] * 9 + [1000.0]   # 10 equal stacks
    payouts = [100.0, 60.0, 40.0, 25.0, 15.0]  # 5 of 10 paid, all equal -> low pressure
    bf = field_bubble_factor(stacks, payouts, hero_idx=0, villain_idx=1, n_sims=8000, rng=random.Random(9))
    assert 0.8 <= bf <= 1.4, f"expected ~chip-linear, got {bf:.3f}"


def test_field_equity_uses_whole_field_not_one_table():
    # two tables; the hero's equity must reflect ALL 12 stacks, not just its 6-handed table
    t0 = MttTable(0, [5000, 1000, 1000, 1000, 1000, 1000])
    t1 = MttTable(1, [5000, 1000, 1000, 1000, 1000, 1000])
    f = MttField([t0, t1], payouts=[100.0, 60.0, 40.0], entrants=12)
    assert f.players_left == 12
    assert not f.in_the_money            # 12 left, 3 paid
    eq_big = f.hero_field_equity(0, 0, n_sims=20000, rng=random.Random(2))   # the 5000 chip leader
    eq_small = f.hero_field_equity(0, 1, n_sims=20000, rng=random.Random(2))  # a 1000 short stack
    assert eq_big > eq_small
    # the chip leader of a 12-player field, 3 paid, holds < the top prize but a real share
    assert 0.0 < eq_small < eq_big < sum(f.payouts)


def test_needs_balancing_and_incremental_move():
    # table A has 9, table B has 6 -> differ by 3 -> needs balancing, move A->B
    a = MttTable(0, [1000.0] * 9)
    b = MttTable(1, [1000.0] * 6 + [0.0] * 3)
    f = MttField([a, b], payouts=[100.0], entrants=15)
    assert needs_balancing(f, seats_per_table=9)
    moves = balance_moves(f, seats_per_table=9)
    assert len(moves) >= 1 and moves[0].from_table == 0 and moves[0].to_table == 1
    g = apply_moves(f, moves)
    sizes = sorted(t.num_players for t in g.tables)
    assert sizes[-1] - sizes[0] <= 1            # balanced to within 1
    assert g.players_left == 15                  # no player lost/created


def test_break_shortest_table_when_field_shrinks():
    # 3 tables, but the field (12) fits on 2 tables of 9 -> break the shortest
    a = MttTable(0, [1000.0] * 5 + [0.0] * 4)
    b = MttTable(1, [1000.0] * 5 + [0.0] * 4)
    c = MttTable(2, [1000.0] * 2 + [0.0] * 7)   # shortest -> should be broken
    f = MttField([a, b, c], payouts=[100.0], entrants=12)
    assert needs_balancing(f, seats_per_table=9)
    moves = balance_moves(f, seats_per_table=9)
    assert all(m.from_table == 2 for m in moves)   # everyone moved OFF the shortest table
    g = apply_moves(f, moves)
    assert g.tables[2].num_players == 0            # table 2 broken
    assert g.players_left == 12


def test_field_icm_tilt_reduces_to_single_table():
    """For a SINGLE-table field, the field tilt must ~match the exact single-table icm_tilt (the field BF
    converges to the single-table BF) — the consistency anchor for the multi-table overlay path."""
    from pokerbot.tournament.icm_overlay import field_icm_tilt, icm_tilt

    stacks = [4000.0, 3000.0, 2000.0, 1000.0]
    payouts = [50.0, 30.0, 20.0]
    chip_values = [100.0, 100.0]      # two actions, same chip-EV
    action_risk = [0.0, 1000.0]       # action 1 risks 1000 chips, action 0 risks nothing
    single = icm_tilt(chip_values, action_risk, stacks, payouts, hero=2, villain=3)
    field = field_icm_tilt(chip_values, action_risk, stacks, payouts, 2, 3,
                           n_sims=20000, rng=random.Random(0))
    # action 0 (no risk) unchanged in both; action 1 (risky) tilted down by ~the same premium
    assert abs(field[0] - single[0]) < 1.0
    assert abs(field[1] - single[1]) < 0.06 * 1000.0   # within MC noise of the exact premium
    assert field[1] < field[0]                          # risk is penalized near the money


def test_field_icm_decide_shifts_off_risk_near_global_bubble():
    """The serve DECISION: deep -> picks the chip-EV argmax; on the global bubble -> the ICM premium flips it to
    the lower-risk action. A marginal spot where the risky action is barely +chip-EV is the test."""
    from pokerbot.tournament.icm_overlay import field_icm_decide

    payouts = [50.0, 30.0, 20.0]
    chip_values = [100.0, 120.0]      # action 1 (risky) is clearly better in chip-EV (+20)
    action_risk = [0.0, 200.0]        # action 1 risks 200 chips
    # deep: huge field left -> ICM premium ~0 -> the +20 chip-EV edge wins -> take the risky action (1)
    deep = [4000.0, 3000.0, 2000.0, 1000.0] + [3000.0] * 32
    assert field_icm_decide(chip_values, action_risk, deep, payouts, 2, 3,
                            n_sims=12000, rng=random.Random(1)) == 1
    # bubble: 4 live, 3 paid -> premium >> 0.1 (BF ~1.3-1.5) -> the risk penalty dwarfs +20 -> safe action (0)
    bubble = [4000.0, 3000.0, 2000.0, 1000.0]
    assert field_icm_decide(chip_values, action_risk, bubble, payouts, 2, 3,
                            n_sims=12000, rng=random.Random(1)) == 0


def test_field_icm_tilt_uses_global_field_position():
    """The field tilt penalizes a risky action MORE when the GLOBAL field is on the bubble than when deep —
    even though the hero's own table looks identical. This is the multi-table value: survival premium is a
    field property, not a table property."""
    from pokerbot.tournament.icm_overlay import field_icm_tilt

    payouts = [50.0, 30.0, 20.0]
    chip_values = [100.0, 100.0]
    action_risk = [0.0, 1000.0]
    # bubble: 4 live total, 3 paid -> high global premium
    bubble = [4000.0, 3000.0, 2000.0, 1000.0]
    tb = field_icm_tilt(chip_values, action_risk, bubble, payouts, 2, 3, n_sims=12000, rng=random.Random(1))
    # deep: many live, same hero/villain stacks but lots of field left -> low premium
    deep = [4000.0, 3000.0, 2000.0, 1000.0] + [3000.0] * 20
    td = field_icm_tilt(chip_values, action_risk, deep, payouts, 2, 3, n_sims=12000, rng=random.Random(1))
    # the risky action is penalized more on the bubble than deep
    assert tb[1] < td[1] - 1.0, (tb[1], td[1])


def test_in_the_money_and_bubble_flags():
    # 3 left, 2 paid -> exactly the bubble (one bust from the money)
    f = MttField([MttTable(0, [1000.0, 1000.0, 1000.0])], payouts=[60.0, 40.0], entrants=4)
    assert not f.in_the_money and f.on_the_bubble
    # 4 left, 2 paid -> two off the money (neither ITM nor bubble)
    f0 = MttField([MttTable(0, [1000.0, 1000.0, 1000.0, 1000.0])], payouts=[60.0, 40.0], entrants=4)
    assert not f0.in_the_money and not f0.on_the_bubble
    # 2 left, 2 paid -> ITM
    f2 = MttField([MttTable(0, [1000.0, 1000.0])], payouts=[60.0, 40.0], entrants=4)
    assert f2.in_the_money and not f2.on_the_bubble
