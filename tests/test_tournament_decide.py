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


def test_deal_toggle_off_is_cash_chip_ev():
    """The deal-time ICM toggle-off (deal_struck=True) must produce the CHIP-EV-identical decision to
    the cash path: after a final-table deal the money is split, so play is payout-agnostic chip-EV.

    Construct a bubble spot where ICM-ON and chip-EV DISAGREE (a +chip-EV but -$EV gamble): ICM folds,
    deal-toggle takes it. Then assert the deal-toggle action == argmax of chip_ev_action_value over the
    SAME outcomes (the exact value the cash blueprint/search maximizes), with payouts ignored."""
    from pokerbot.tournament.decide import decide_deep
    from pokerbot.tournament.icm_overlay import chip_ev_action_value
    # +chip-EV gamble for hero (0.5*+12 + 0.5*-10 = +1 chip) but stack-threatening on a flat bubble.
    outs = {"shove": [(0.5, [+12, -12, 0, 0]), (0.5, [-10, +10, 0, 0])], "fold": [(1.0, [0, 0, 0, 0])]}
    stacks, payouts, hero = [10, 10, 10, 2], [50, 30, 20], 0

    icm = decide_deep(stacks, payouts, hero, outs)                 # ICM-graded (deal NOT struck)
    deal = decide_deep(stacks, payouts, hero, outs, deal_struck=True)  # deal struck -> chip-EV

    # ICM and the deal-time chip-EV path DISAGREE here (proves the toggle is live, not a no-op).
    assert icm.action == "fold", "bubble ICM should fold the +chip-EV/-$EV gamble"
    assert deal.action == "shove", "post-deal play is payout-agnostic chip-EV -> takes the +chip gamble"
    assert deal.source == "deal-chip-ev"

    # The deal-toggle action is EXACTLY the cash chip-EV argmax over the same outcomes (payout-agnostic).
    chip_vals = {a: chip_ev_action_value(stacks, o, hero) for a, o in outs.items()}
    assert deal.action == max(chip_vals, key=chip_vals.get)
    # And it is invariant to the payout vector (the defining property of the chip-EV / cash path).
    deal_other_pay = decide_deep(stacks, [99, 1, 0], hero, outs, deal_struck=True)
    assert deal_other_pay.action == deal.action


def test_deep_provider_replaces_defer():
    # With a deep provider, a >20bb spot is graded by ICM instead of deferring.
    provider = lambda: {"shove": [(0.6, [+40, -40, 0, 0, 0, 0]), (0.4, [-40, +40, 0, 0, 0, 0])],
                        "fold": [(1.0, [0] * 6)]}
    d = D([40] * 6, PAY, hero=3, hero_hand="AhAs", blinds=BL, behind=[4, 5], deep_provider=provider)
    assert d.action in ("shove", "fold") and d.source == "deep-icm-graded"
    # without a provider it still defers
    assert D([40] * 6, PAY, hero=3, hero_hand="AhAs", blinds=BL, behind=[4, 5]).action == "defer"
