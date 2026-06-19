"""HUD-driven value-SIZING escalation overlay (pokerbot/opponent/value_sizing.py).

Verifies the value-extraction lever the population analysis isolated: escalate OUR bet SIZE vs a
sticky/weak-showdown calling station, but ONLY for value hands, DAMPENED by showdown strength,
confidence-gated, and additive (so the mixer firewall still clamps it).
"""

from __future__ import annotations

import numpy as np

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.opponent.stats import OpponentStats
from pokerbot.opponent.value_sizing import (
    _BET_LO,
    _BIG_LO,
    _ALL_IN,
    bluff_suppression_logits,
    river_thin_value_logits,
    station_payoff_score,
    value_sizing_logits,
)

_RIVER = ["Ah", "Kd", "7s", "2c", "9h"]      # 5-card board
_FLOP = ["Ah", "Kd", "7s"]


def _station(**hud) -> OpponentStats:
    """A confirmed loose-passive station (vpip 0.6, af 0.25) optionally carrying server HUD stats."""
    s = OpponentStats(hands=50, vpip=30, pfr=4, aggression_bets_raises=5, aggression_calls=20)
    if hud:
        s.hud = dict(hud)
    return s


def _neutral(**hud) -> OpponentStats:
    """An UNKNOWN-archetype carrier (vpip 0.30, no station/lag/tag/nit floor) so the HUD stats alone
    drive the payoff score — used to test monotonicity without the calling-station floor."""
    s = OpponentStats(hands=50, vpip=15, pfr=8, aggression_bets_raises=10, aggression_calls=10)
    assert s.archetype() == "unknown"
    if hud:
        s.hud = dict(hud)
    return s


def test_air_never_escalates():
    s = _station(wtsd=37, wsd=42, foldToCbet=30)
    adj = value_sizing_logits(s, hand_strength=0.30)  # weak hand
    assert np.allclose(adj, 0.0)


def test_value_hand_escalates_size_vs_station_hud():
    s = _station(wtsd=37, wsd=42, foldToCbet=30)
    adj = value_sizing_logits(s, hand_strength=0.72)
    # over-bet buckets boosted strictly more than mid bets; checking discouraged; nothing on fold
    big = adj[_BIG_LO:_ALL_IN]
    mid = adj[_BET_LO:_BIG_LO]
    assert big.min() > 0 and mid.min() > 0
    assert big.mean() > mid.mean()          # SIZE is escalated, not just frequency
    assert adj[1] < 0                        # don't check back value vs a station
    assert adj[0] == 0 and adj[_ALL_IN] == 0  # fold + all-in untouched (thin value != jam)


def test_high_wsd_dampens_relative_to_low_wsd():
    # Same stickiness, weaker vs stronger showdown range: the weak-showdown player gets MORE escalation.
    # Use neutral-archetype carriers so the calling-station score-floor doesn't mask the WSD signal.
    weak = value_sizing_logits(_neutral(wsd=40, foldToCbet=30), 0.7)
    strong = value_sizing_logits(_neutral(wsd=62, foldToCbet=30), 0.7)
    assert weak[_BIG_LO] > strong[_BIG_LO] >= 0.0


def test_no_data_is_near_zero():
    empty = OpponentStats(hands=0)
    assert np.allclose(value_sizing_logits(empty, 0.8), 0.0)  # conf 0 => no escalation


def test_derived_fallback_without_hud():
    # No server HUD, but count-based fold_to_cbet = 8/40 = 20% (sticky) + saw_flop sample -> escalates.
    s = OpponentStats(hands=60, vpip=33, pfr=5, aggression_bets_raises=6, aggression_calls=24,
                      cbet_faced=40, fold_to_cbet=8, saw_flop=40)
    adj = value_sizing_logits(s, 0.7)
    assert adj[_BIG_LO] > 0.0


def test_env_killswitch(monkeypatch):
    monkeypatch.setenv("POKERBOT_VALUE_SIZING", "0")
    assert np.allclose(value_sizing_logits(_station(wsd=40, foldToCbet=25), 0.8), 0.0)


def test_lag_damped_vs_station():
    # A LAG reg should be escalated far less than a station (raise/fold risk = value-cut).
    lag = OpponentStats(hands=50, vpip=20, pfr=18, aggression_bets_raises=30, aggression_calls=5,
                        cbet_faced=30, fold_to_cbet=6)
    station = OpponentStats(hands=50, vpip=30, pfr=4, aggression_bets_raises=5, aggression_calls=20,
                            cbet_faced=30, fold_to_cbet=6)
    assert value_sizing_logits(lag, 0.7)[_BIG_LO] < value_sizing_logits(station, 0.7)[_BIG_LO]


def test_score_monotone_in_stickiness_and_weakness():
    sticky_weak, _ = station_payoff_score(_neutral(wsd=38, foldToCbet=20))
    loose_strong, _ = station_payoff_score(_neutral(wsd=60, foldToCbet=60))
    assert sticky_weak > loose_strong
    # HUD present => high confidence; blind => low
    _, c_hud = station_payoff_score(_neutral(wsd=40))
    _, c_blind = station_payoff_score(OpponentStats(hands=4))
    assert c_hud > c_blind


def test_overlay_is_additive_and_bounded():
    # The overlay never produces NaN/inf and stays within a sane logit range (alpha-cap clamps further).
    s = _station(wtsd=40, wsd=35, foldToCbet=15, fta=20)
    adj = value_sizing_logits(s, 1.0)
    assert np.isfinite(adj).all()
    assert adj.shape == (NUM_ACTIONS,)
    assert adj.max() <= 2.0 and adj.min() >= -2.0


# ---- bluff-frequency suppression overlay (symmetric, air-only) ----

def test_bluff_suppression_cuts_bets_with_air_vs_station():
    s = _station(wtsd=41, wsd=44, foldToCbet=28, fta=30)
    adj = bluff_suppression_logits(s, hand_strength=0.25)   # air vs a sticky non-folder
    assert adj[_BET_LO:_ALL_IN].max() < 0          # all bet/raise sizes de-prioritized
    assert adj[1] > 0                               # checking/giving up preferred
    assert adj[0] == 0 and adj[_ALL_IN] == 0        # fold + all-in untouched


def test_bluff_suppression_value_hand_is_zero():
    # Value hands never suppress (that's value_sizing's job) — only air.
    s = _station(wsd=42, foldToCbet=28)
    assert np.allclose(bluff_suppression_logits(s, hand_strength=0.80), 0.0)


def test_bluff_suppression_no_data_is_zero():
    assert np.allclose(bluff_suppression_logits(OpponentStats(hands=0), 0.2), 0.0)


def test_bluff_suppression_damped_vs_lag():
    lag = OpponentStats(hands=50, vpip=20, pfr=18, aggression_bets_raises=30, aggression_calls=5,
                        cbet_faced=30, fold_to_cbet=6)
    station = OpponentStats(hands=50, vpip=30, pfr=4, aggression_bets_raises=5, aggression_calls=20,
                            cbet_faced=30, fold_to_cbet=6)
    # A LAG (who folds to bluffs) should be suppressed far less than a station.
    assert abs(bluff_suppression_logits(lag, 0.2)[_BET_LO]) < abs(bluff_suppression_logits(station, 0.2)[_BET_LO])


def test_bluff_suppression_env_killswitch(monkeypatch):
    monkeypatch.setenv("POKERBOT_BLUFF_SUPPRESS", "0")
    s = _station(wsd=42, foldToCbet=28)
    assert np.allclose(bluff_suppression_logits(s, 0.2), 0.0)


# ---- river thin-value expansion (river-only, marginal-band, weak-WSD gated, default OFF) ----

def test_river_thin_value_default_off():
    s = _station(wsd=42, foldToCbet=28)
    assert np.allclose(river_thin_value_logits(s, 0.45, _RIVER), 0.0)  # default OFF


def test_river_thin_value_fires_on_river(monkeypatch):
    monkeypatch.setenv("POKERBOT_RIVER_THIN_VALUE", "1")
    s = _station(wsd=42, foldToCbet=28)          # weak showdown range (wsd<0.50)
    adj = river_thin_value_logits(s, 0.45, _RIVER)  # marginal hand in [0.40, 0.50)
    assert adj[_BET_LO:_BIG_LO].max() > 0          # small/mid value bet boosted
    assert np.allclose(adj[_BIG_LO:_ALL_IN], 0.0)  # NOT the over-bet buckets (don't over-bet 2nd pair)
    assert adj[1] < 0                              # don't check back the thin value hand


def test_river_thin_value_only_on_river(monkeypatch):
    monkeypatch.setenv("POKERBOT_RIVER_THIN_VALUE", "1")
    s = _station(wsd=42, foldToCbet=28)
    assert np.allclose(river_thin_value_logits(s, 0.45, _FLOP), 0.0)   # flop -> no thin value
    assert np.allclose(river_thin_value_logits(s, 0.45, []), 0.0)      # preflop -> no thin value


def test_river_thin_value_band_only(monkeypatch):
    monkeypatch.setenv("POKERBOT_RIVER_THIN_VALUE", "1")
    s = _station(wsd=42, foldToCbet=28)
    assert np.allclose(river_thin_value_logits(s, 0.30, _RIVER), 0.0)  # air below band -> nothing
    assert np.allclose(river_thin_value_logits(s, 0.70, _RIVER), 0.0)  # value above band -> value_sizing's job


def test_river_thin_value_needs_weak_showdown(monkeypatch):
    monkeypatch.setenv("POKERBOT_RIVER_THIN_VALUE", "1")
    # strong showdown range (wsd>=0.50) -> value-cut guard blocks the thin bet
    assert np.allclose(river_thin_value_logits(_station(wsd=58, foldToCbet=28), 0.45, _RIVER), 0.0)
    # no server WSD at all -> no blind thin value
    assert np.allclose(river_thin_value_logits(_station(foldToCbet=28), 0.45, _RIVER), 0.0)
