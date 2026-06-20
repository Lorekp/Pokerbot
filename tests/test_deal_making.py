"""Deal-making ICM toggle-off: post-deal play is payout-agnostic chip-EV (== the cash line)."""

from __future__ import annotations

from pokerbot.tournament.decide import decide_deep
from pokerbot.tournament.icm_overlay import chip_ev_action_value

# A near-the-money final-3 spot. "jam" is a +chip-EV double-or-bust (55/45) but ICM-negative
# (variance is penalized near a pay jump); "fold" is chip-neutral.
_STACKS = [100.0, 100.0, 100.0]
_PAYOUTS = [50.0, 30.0, 20.0]
_JAM = [(0.55, [100.0, -100.0, 0.0]), (0.45, [-100.0, 100.0, 0.0])]
_FOLD = [(1.0, [0.0, 0.0, 0.0])]
_ACTIONS = {"jam": _JAM, "fold": _FOLD}


def test_icm_folds_chipev_jams_near_money():
    icm = decide_deep(_STACKS, _PAYOUTS, 0, _ACTIONS)
    deal = decide_deep(_STACKS, _PAYOUTS, 0, _ACTIONS, deal_struck=True)
    assert icm.action == "fold"          # ICM penalizes the variance -> decline the +chip-EV gamble
    assert deal.action == "jam"          # deal struck -> pure chip-EV takes the +EV gamble
    assert deal.source == "deal-chip-ev"


def test_deal_equals_cash_chip_ev_argmax():
    # the deal-time choice is exactly argmax chip-EV -- the same line the cash engine plays.
    cev = {a: chip_ev_action_value(_STACKS, outs, 0) for a, outs in _ACTIONS.items()}
    cash_choice = max(cev, key=cev.get)
    deal = decide_deep(_STACKS, _PAYOUTS, 0, _ACTIONS, deal_struck=True)
    assert deal.action == cash_choice
    assert cev["jam"] > cev["fold"]      # +chip-EV gamble


def test_deep_icm_equals_chip_ev_no_discontinuity():
    # far from the money chips are $-linear: ICM and the deal toggle pick the SAME action, so the
    # toggle hands off without a discontinuity.
    deep = [100_000.0, 100_000.0, 100_000.0]
    icm = decide_deep(deep, _PAYOUTS, 0, _ACTIONS)
    deal = decide_deep(deep, _PAYOUTS, 0, _ACTIONS, deal_struck=True)
    assert icm.action == deal.action == "jam"
