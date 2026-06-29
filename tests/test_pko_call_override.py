"""Tests for the PKO bounty-aware live CALL override (live_adapter.pko_call_override).

The certified chip-EV != $-EV serve: a jammer's bounty LOWERS hero's indifference equity (the cash collected
on a win), so a call the prize-only ICM folds becomes +$-EV. The override returns "call" ONLY in the band
[p*_with_bounty, p*_prize_only) -- a marginal hand the bounty flips fold -> call -- and is a conservative
no-op otherwise (no bounty, no jam, multiway, or a hand already in/out of the money on prize alone).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pokerbot.tournament.bounty import min_equity_to_call
from pokerbot.tournament.live_adapter import (
    _DEFAULT_DEEP_JAM,
    _equity_vs_range,
    _hand_label,
    pko_call_override,
)

PAY = [50.0, 30.0, 20.0]


def _card(c):
    return SimpleNamespace(code=c)


def _p(stack, committed, allin, hole=None):
    return SimpleNamespace(stack_bb=stack, committed_total_bb=committed, all_in=allin, hole=hole)


def _state(players):
    return SimpleNamespace(players=players)


def _hero_with_equity_in_band():
    """Find a hand whose equity vs the default jam range sits in the bounty flip band for this spot."""
    stacks = [30, 45, 20, 20, 20, 20]
    p_prize = min_equity_to_call(stacks, PAY, 0, 1, villain_bounty=0.0)
    p_bounty = min_equity_to_call(stacks, PAY, 0, 1, villain_bounty=5.0)
    # scan canonical labels for one whose equity lands strictly inside [p_bounty, p_prize)
    from pokerbot.tournament.pushfold_icm import hand_to_label  # canonical 169 labels
    seen = []
    for r1 in "AKQJT98765432":
        for r2 in "AKQJT98765432":
            for suited in ("s", "o"):
                if r1 == r2 and suited == "o":
                    lab = r1 + r2
                elif r1 == r2:
                    continue
                else:
                    lab = r1 + r2 + suited
                if lab in seen:
                    continue
                seen.append(lab)
                eq = _equity_vs_range(lab, _DEFAULT_DEEP_JAM)
                if p_bounty <= eq < p_prize:
                    return lab, eq, p_bounty, p_prize
    return None


def test_no_bounty_is_noop():
    players = [_p(30, 5, False, (_card("Ah"), _card("Kd"))), _p(0, 45, True)] + [_p(20, 0, False) for _ in range(4)]
    assert pko_call_override(_state(players), 0, PAY, 0.0) is None


def test_no_jam_is_noop():
    players = [_p(30, 5, False, (_card("Ah"), _card("Kd")))] + [_p(25, 0, False) for _ in range(5)]
    assert pko_call_override(_state(players), 0, PAY, 5.0) is None


def test_unknown_payouts_is_noop():
    players = [_p(30, 5, False, (_card("Ah"), _card("Kd"))), _p(0, 45, True)] + [_p(20, 0, False) for _ in range(4)]
    assert pko_call_override(_state(players), 0, None, 5.0) is None


def test_multiway_jam_is_noop():
    players = ([_p(30, 5, False, (_card("Ah"), _card("Kd"))), _p(0, 45, True), _p(0, 40, True)]
              + [_p(20, 0, False) for _ in range(3)])
    assert pko_call_override(_state(players), 0, PAY, 5.0) is None


def test_bounty_flips_marginal_hand_to_call():
    found = _hero_with_equity_in_band()
    assert found is not None, "expected at least one hand in the bounty flip band for this spot"
    lab, eq, p_bounty, p_prize = found
    # build the hole cards for that label
    suit = {"s": ("h", "h"), "o": ("h", "d")}
    if len(lab) == 2:  # pair
        hole = (_card(lab[0] + "h"), _card(lab[1] + "d"))
    else:
        s1, s2 = suit[lab[2]]
        hole = (_card(lab[0] + s1), _card(lab[1] + s2))
    assert _hand_label(hole) == lab
    players = [_p(30, 5, False, hole), _p(0, 45, True)] + [_p(20, 0, False) for _ in range(4)]
    # prize-only ICM would FOLD this hand (eq < p_prize); the $5 bounty flips it to a CALL
    assert pko_call_override(_state(players), 0, PAY, 5.0) == "call"
    # and with NO bounty it stays a fold (no override)
    assert pko_call_override(_state(players), 0, PAY, 0.0) is None
