"""Tests for the ICM overlay (pokerbot/tournament/icm_overlay.py)."""

from __future__ import annotations

import pytest

from pokerbot.tournament.icm import icm_equity
from pokerbot.tournament.icm_overlay import icm_action_value, icm_tilt


def test_action_value_neutral_heads_up():
    # A 50/50 all-in heads-up (chips $-linear) is ICM-neutral: value == fold/current equity.
    outcomes = [(0.5, [+10, -10]), (0.5, [-10, +10])]
    v = icm_action_value([10, 10], outcomes, [60, 40], hero=0)
    assert v == pytest.approx(50.0, abs=1e-6)


def test_action_value_risk_premium_on_bubble():
    # The same 50/50 all-in on the bubble is -$EV vs not gambling (ICM risk premium).
    pre = [10, 10, 2]
    pay = [50, 30, 20]
    base = float(icm_equity(pre, pay)[0])
    outcomes = [(0.5, [+10, -10, 0]), (0.5, [-10, +10, 0])]
    v = icm_action_value(pre, outcomes, pay, hero=0)
    assert v < base                                   # gambling loses $ on the bubble


def test_action_value_reduces_to_chip_ev_winner_take_all():
    # Winner-take-all over 2 players: ICM == chip share, so the ICM action value tracks chip-EV.
    outcomes = [(0.6, [+10, -10]), (0.4, [-10, +10])]
    v = icm_action_value([10, 10], outcomes, [1, 0], hero=0)
    # post-win [20,0]->1.0 ; post-lose [0,20]->0.0 ; value = 0.6
    assert v == pytest.approx(0.6, abs=1e-9)


def test_tilt_identity_when_deep():
    # Heads-up (bubble factor 1): tilt leaves chip values unchanged.
    cv = [0.0, 2.0]
    risk = [0.0, 10.0]
    out = icm_tilt(cv, risk, [10, 10], [60, 40], hero=0, villain=1)
    assert out[0] == pytest.approx(0.0) and out[1] == pytest.approx(2.0)


def test_tilt_penalizes_risk_on_bubble():
    # On the bubble a marginally +chipEV jam becomes -EV after the risk premium.
    cv = [0.0, 2.0]               # fold EV 0, jam chip-EV +2
    risk = [0.0, 10.0]           # jam risks 10 chips
    out = icm_tilt(cv, risk, [10, 10, 2], [50, 30, 20], hero=0, villain=1)
    assert out[1] < out[0]       # fold now beats jam
