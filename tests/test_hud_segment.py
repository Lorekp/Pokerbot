"""Tests for pokerbot.opponent.hud_segment (the router guardrail).

Standalone-importable: this module deliberately has no engine deps, so these
tests import it directly.
"""

import math

import pytest

from pokerbot.opponent.hud_segment import (
    SegmentResult,
    classify,
    count_gate,
    _pct,
)


# --- helpers ---------------------------------------------------------------
def _hud(**over):
    """A baseline consistent HUD; override fields per test."""
    h = {
        "vpip": 0.25,
        "pfr": 0.18,
        "threeBet": 0.06,
        "af": 2.5,
        "wsd": 0.45,
        "wtsd": 0.30,
        "_sampleSize": 1000,
        "playingStyle": {"archetype": "tag", "aggression": "aggressive"},
    }
    h.update(over)
    return h


# --- Branch 1: consistency filter -----------------------------------------
def test_hud_none_falls_back_to_gto_untrusted():
    r = classify(None)
    assert isinstance(r, SegmentResult)
    assert r.segment == "gto"
    assert r.trusted is False
    assert "hud_missing" in r.reasons


def test_empty_hud_falls_back():
    r = classify({})
    assert r.segment == "gto" and r.trusted is False
    assert "hud_missing" in r.reasons


def test_missing_af_untrusted():
    h = _hud()
    del h["af"]
    r = classify(h)
    assert r.segment == "gto" and r.trusted is False
    assert "no_af" in r.reasons


def test_low_sample_untrusted():
    r = classify(_hud(_sampleSize=499))
    assert r.segment == "gto" and r.trusted is False
    assert any("sample_lt" in x for x in r.reasons)


def test_sample_500_is_ok_boundary():
    # exactly 500 passes the filter (threshold is < 500)
    r = classify(_hud(_sampleSize=500))
    assert not any("sample_lt" in x for x in r.reasons)


def test_idle_vpip_zero_untrusted():
    r = classify(_hud(vpip=0.0))
    assert r.segment == "gto" and r.trusted is False
    assert "idle_vpip_0" in r.reasons


def test_impossible_3bet_gt_pfr_untrusted():
    r = classify(_hud(threeBet=0.30, pfr=0.10))
    assert r.segment == "gto" and r.trusted is False
    assert "impossible_3bet_gt_pfr" in r.reasons


def test_af_over_20_untrusted():
    r = classify(_hud(af=146.75))
    assert r.segment == "gto" and r.trusted is False
    assert any("af_gt" in x for x in r.reasons)


# --- Branch 2: verified station -------------------------------------------
def test_verified_station():
    # af==0, vpip>0.35, benign (loose-passive) label
    h = _hud(af=0.0, vpip=0.42, pfr=0.05, threeBet=0.0,
             playingStyle={"archetype": "fish", "aggression": "passive"})
    r = classify(h)
    assert r.segment == "station" and r.trusted is True


def test_station_boundary_vpip_must_exceed_035():
    # vpip exactly 0.35 is NOT a station (strict >)
    h = _hud(af=0.0, vpip=0.35, pfr=0.05, threeBet=0.0,
             playingStyle={"archetype": "fish", "aggression": "passive"})
    r = classify(h)
    assert r.segment != "station"


def test_station_demoted_when_label_aggressive():
    # af==0 & vpip>0.35 would be a station, but server says maniac/aggressive
    # -> precision-first veto to gto, untrusted.
    h = _hud(af=0.0, vpip=0.50, pfr=0.30, threeBet=0.10,
             playingStyle={"archetype": "maniac", "aggression": "aggressive"})
    r = classify(h)
    assert r.segment == "gto" and r.trusted is False
    assert any("demote_station" in x for x in r.reasons)


def test_station_not_demoted_for_passive_label():
    h = _hud(af=0.0, vpip=0.50, pfr=0.05, threeBet=0.0,
             playingStyle={"archetype": "caller", "aggression": "passive"})
    r = classify(h)
    assert r.segment == "station"


def test_station_no_playingstyle_block_still_station():
    h = _hud(af=0.0, vpip=0.50, pfr=0.05, threeBet=0.0)
    del h["playingStyle"]
    r = classify(h)
    assert r.segment == "station" and r.trusted is True


# --- Branch 3: rock / nit over-folder -------------------------------------
def test_overfolder_rock():
    h = _hud(af=1.5, vpip=0.15, pfr=0.04, threeBet=0.02,
             playingStyle={"archetype": "rock", "aggression": "passive"})
    r = classify(h)
    assert r.segment == "overfolder" and r.trusted is True


def test_overfolder_boundary_vpip():
    # vpip exactly 0.20 is NOT overfolder (strict <)
    h = _hud(af=1.5, vpip=0.20, pfr=0.04, threeBet=0.02)
    r = classify(h)
    assert r.segment != "overfolder"


# --- Branch 4: aggressor / gto default ------------------------------------
def test_aggressor():
    h = _hud(af=3.0, vpip=0.40, pfr=0.30, threeBet=0.12)
    r = classify(h)
    assert r.segment == "aggressor" and r.trusted is True


def test_aggressor_boundary_af_ge_2():
    h = _hud(af=2.0, vpip=0.30, pfr=0.20, threeBet=0.08)
    r = classify(h)
    assert r.segment == "aggressor"


def test_gto_default_middle():
    # passes filter, not station/overfolder/aggressor -> gto but trusted
    h = _hud(af=1.5, vpip=0.28, pfr=0.15, threeBet=0.06)
    r = classify(h)
    assert r.segment == "gto" and r.trusted is True
    assert "disengage_default" in r.reasons


def test_gto_default_low_af_moderate_vpip():
    # af<2 with vpip in [0.20,0.35] -> neither station/overfolder/aggressor
    h = _hud(af=1.0, vpip=0.25, pfr=0.10, threeBet=0.05)
    r = classify(h)
    assert r.segment == "gto" and r.trusted is True


# --- percent vs fraction auto-detect --------------------------------------
def test_pct_autodetect_fraction():
    assert _pct(0.35) == pytest.approx(0.35)


def test_pct_autodetect_percent():
    assert _pct(35) == pytest.approx(0.35)


def test_pct_none_and_nan():
    assert _pct(None) is None
    assert _pct(float("nan")) is None
    assert _pct("garbage") is None


def test_classify_handles_percent_encoded_hud():
    # same station but stats given as percentages
    h = _hud(af=0.0, vpip=42.0, pfr=5.0, threeBet=0.0,
             playingStyle={"archetype": "fish", "aggression": "passive"})
    r = classify(h)
    assert r.segment == "station" and r.trusted is True


def test_percent_encoded_overfolder():
    h = _hud(af=1.5, vpip=15.0, pfr=4.0, threeBet=2.0,
             playingStyle={"archetype": "rock", "aggression": "passive"})
    r = classify(h)
    assert r.segment == "overfolder"


# --- retired stats must never be read -------------------------------------
def test_bluff_and_fold_to_cbet_never_consulted():
    # A station with pathological bluff/fold_to_cbet values must classify
    # identically to one without them -> proves they are not inputs.
    base = dict(af=0.0, vpip=0.50, pfr=0.05, threeBet=0.0,
                playingStyle={"archetype": "fish", "aggression": "passive"})
    a = classify(_hud(**base))
    b = classify(_hud(bluff=0.99, fold_to_cbet=0.99, **base))
    assert a.segment == b.segment == "station"


# --- count gate ------------------------------------------------------------
@pytest.mark.parametrize("n,expected", [
    (0, True), (1, True), (2, True), (3, True),
    (4, False), (5, False), (9, False),
])
def test_count_gate(n, expected):
    assert count_gate(n) is expected


def test_count_gate_bad_input():
    assert count_gate(None) is False


# --- local param is accepted but inert ------------------------------------
def test_local_param_does_not_change_segment():
    h = _hud(af=0.0, vpip=0.50, pfr=0.05, threeBet=0.0,
             playingStyle={"archetype": "fish", "aggression": "passive"})
    a = classify(h)
    b = classify(h, local={"hands": 999, "vpip": 0})
    assert a.segment == b.segment == "station"
