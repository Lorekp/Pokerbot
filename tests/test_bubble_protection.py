"""Bubble Protection tests: B(N) brackets + the cliff-flattening (protected ranges are WIDER)."""

from __future__ import annotations

from pokerbot.tournament.bubble import (
    bubble_seats,
    hard_bubble_factor,
    protected_bubble_factor,
    protected_payouts,
)


def test_bubble_brackets():
    assert bubble_seats(9) == 0
    assert bubble_seats(50) == 1
    assert bubble_seats(99) == 1
    assert bubble_seats(200) == 2
    assert bubble_seats(1000) == 3
    assert bubble_seats(2000) == 4
    assert bubble_seats(4000) == 5
    assert bubble_seats(6000) == 5  # cap
    assert bubble_seats(0) == 0


def test_protected_payouts_appends_refunds():
    base = [650.0, 350.0]
    # field 50 -> B=1, eligible -> append one buy-in refund rank
    assert protected_payouts(base, field=50, buy_in=100.0, eligible=True) == [650.0, 350.0, 100.0]
    # ineligible -> unchanged (hard bubble)
    assert protected_payouts(base, field=50, buy_in=100.0, eligible=False) == base
    # tiny field B=0 -> unchanged even if eligible
    assert protected_payouts(base, field=8, buy_in=100.0, eligible=True) == base
    # B=2 field -> two refund ranks
    assert protected_payouts(base, field=200, buy_in=100.0, eligible=True) == [650.0, 350.0, 100.0, 100.0]


def test_protection_flattens_bubble_wider_ranges():
    # 3 left, top-2 paid; hero is the short stack on the bubble vs the covering chip leader.
    stacks = [5000.0, 3000.0, 2000.0]
    base = [650.0, 350.0]
    hero, villain = 2, 0
    hard = hard_bubble_factor(stacks, base, hero, villain)
    prot = protected_bubble_factor(stacks, base, hero, villain, field=50, buy_in=100.0, eligible=True)
    # the refund softens the bust, so the risk premium DROPS -> bubble factor lower -> wider ranges
    assert prot < hard, (prot, hard)
    # ineligible entrant gets the hard bubble unchanged
    inelig = protected_bubble_factor(stacks, base, hero, villain, field=50, buy_in=100.0, eligible=False)
    assert inelig == hard


def test_protection_noop_when_no_bubble_seats():
    stacks = [5000.0, 3000.0, 2000.0]
    base = [650.0, 350.0]
    # field in the 1-9 bracket -> B=0 -> protected == hard
    prot = protected_bubble_factor(stacks, base, 2, 0, field=8, buy_in=100.0, eligible=True)
    hard = hard_bubble_factor(stacks, base, 2, 0)
    assert prot == hard
