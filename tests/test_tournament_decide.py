"""Tests for the unified short-stack tournament decision router (pokerbot/tournament/decide.py)."""

from __future__ import annotations

from pokerbot.tournament.decide import decide_pushfold as D

BL = {4: 0.5, 5: 1.0}
STACKS = [10] * 6
PAY = [40, 30, 20, 10]


def test_open_shove_decision():
    assert D(STACKS, PAY, hero=3, hero_hand="AhAs", blinds=BL, behind=[4, 5]).action == "jam"
    assert D(STACKS, PAY, hero=3, hero_hand="7c2d", blinds=BL, behind=[4, 5]).action == "fold"
    # accepts both label and 4-char hand
    assert D(STACKS, PAY, hero=3, hero_hand="AA", blinds=BL, behind=[4, 5]).action == "jam"


def test_call_vs_jam_decision():
    assert D(STACKS, PAY, hero=5, hero_hand="AhAs", blinds=BL, jammer=3).action == "call"
    assert D(STACKS, PAY, hero=5, hero_hand="7c2d", blinds=BL, jammer=3).action == "fold"


def test_deep_defers():
    d = D([40] * 6, PAY, hero=3, hero_hand="AhAs", blinds=BL, behind=[4, 5])
    assert d.action == "defer" and "20bb" in d.note


def test_decision_fields_and_freq():
    d = D(STACKS, PAY, hero=3, hero_hand="AhAs", blinds=BL, behind=[4, 5])
    assert d.source == "open-shove-icm"
    assert 0.0 <= d.freq <= 1.0 and d.freq > 0.99       # AA is a pure jam
    assert d.nash_gap < 1e-3


def test_open_shove_requires_behind():
    import pytest
    with pytest.raises(ValueError):
        D(STACKS, PAY, hero=3, hero_hand="AA", blinds=BL)


def test_decide_deep_takes_plus_ev_gamble_far_from_money():
    # Heads-up WTA (chips $-linear): a +chip-EV gamble should be taken.
    from pokerbot.tournament.decide import decide_deep
    outs = {"shove": [(0.6, [+10, -10]), (0.4, [-10, +10])], "fold": [(1.0, [0, 0])]}
    d = decide_deep([10, 10], [1, 0], hero=0, action_outcomes=outs)
    assert d.action == "shove"


def test_decide_deep_avoids_gamble_on_bubble():
    # Same coinflip on the bubble: the ICM-correct deep choice is to fold the gamble.
    from pokerbot.tournament.decide import decide_deep
    outs = {"shove": [(0.5, [+10, -10, 0, 0]), (0.5, [-10, +10, 0, 0])], "fold": [(1.0, [0, 0, 0, 0])]}
    d = decide_deep([10, 10, 10, 2], [50, 30, 20], hero=0, action_outcomes=outs)
    assert d.action == "fold"


def test_deep_provider_replaces_defer():
    # With a deep provider, a >20bb spot is graded by ICM instead of deferring.
    provider = lambda: {"shove": [(0.6, [+40, -40, 0, 0, 0, 0]), (0.4, [-40, +40, 0, 0, 0, 0])],
                        "fold": [(1.0, [0] * 6)]}
    d = D([40] * 6, PAY, hero=3, hero_hand="AhAs", blinds=BL, behind=[4, 5], deep_provider=provider)
    assert d.action in ("shove", "fold") and d.source == "deep-icm-graded"
    # without a provider it still defers
    assert D([40] * 6, PAY, hero=3, hero_hand="AhAs", blinds=BL, behind=[4, 5]).action == "defer"
