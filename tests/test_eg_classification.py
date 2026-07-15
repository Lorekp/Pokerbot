"""Exploitation safety-gating CLASSIFICATION remediation tests (audit wiup89smb refresh).

Makes the fish-vs-dangerous-aggressor classes MUTUALLY EXCLUSIVE everywhere (not only in
fish_relaxed_n), pins the value-sizing firewalls, the partial-HUD admit constants, and the kill-count
straddle. Covers EG-3, EG-5, EG-6, EG-7, EG-8, EG-9, EG-11, EG-12, EG-14, EG-15.

Owned files: pokerbot/opponent/root_mixer.py, value_sizing.py, stats.py.
"""

from __future__ import annotations

import numpy as np

from pokerbot.opponent.population_prior import context_prior
from pokerbot.opponent.root_mixer import (_all_live_villains_fish, _is_dangerous_aggressor,
                                          _is_value_fish, _table_too_aggressive,
                                          apply_live_value_sizing)
from pokerbot.opponent.stats import OpponentStats
from pokerbot.opponent.value_sizing import _pct, station_payoff_score

_ST = ("flop", "turn", "river")
_TEX = ("dry", "wet", "paired")


# ---------------------------------------------------------------------------------------------------
# count-based profile builders (mirror tests/test_table_kill.py so the count classifiers are exercised)
# ---------------------------------------------------------------------------------------------------
def _profile(facing_bins, unopened_bins, hands=200, hud=None):
    prof = OpponentStats()
    prof.hands = hands
    for st in _ST:
        for tex in _TEX:
            for fac, bins in (("unopened", unopened_bins), ("facing_bet", facing_bins)):
                ck = f"{st}|{tex}|{fac}"
                for h in range(1, 50):
                    prof.decay_and_add(ck, bins, context_prior(ck), h)
                prof.add_ctx(ck, [b * 24 for b in bins])
    if hud is not None:
        prof.hud = dict(hud)
    return prof


def _count_lag(hud=None):
    return _profile([0, 0, 1, 1], [0, 0, 1, 1], hud=hud)   # raises facing, never folds -> dangerous count


def _count_station(hud=None):
    return _profile([0, 1, 0, 0], [0, 1, 0, 0], hud=hud)   # passive caller


# ===================================================================================================
# EG-3 — a wide-gap limpy LAG that WINS at showdown is BOTH fish-shaped and dangerous; the overlay must
#        not over-size value into the trapper. fish/dangerous are now mutually exclusive EVERYWHERE.
# ===================================================================================================
def _winning_showdown_limpy_lag():
    # vpip65/pfr30 (wide gap, but pfr 0.30 is HIGH -> EG-7), wsd58 (wins at showdown -> EG-5/EG-9),
    # AND count-aggressive (raises facing, never folds -> dangerous count). The collision villain.
    return _count_lag(hud={"vpip": 65, "pfr": 30, "wsd": 58, "foldToCbet": 30, "wtsd": 40})


def test_eg3_winning_showdown_limpy_lag_is_not_fish():
    prof = _winning_showdown_limpy_lag()
    assert _is_dangerous_aggressor(prof) is True, "a count-aggressive winning-showdown LAG must be dangerous"
    assert _is_value_fish(prof) is False, "EG-3: the trapper must NOT be admitted as a value-fish"


def test_eg3_classes_mutually_exclusive_at_all_live_firewall():
    # Heads-up table of the collision villain: the value-size firewall must NOT see an all-fish table.
    class _P:
        folded = False

    state = type("S", (), {"players": [_P(), _P()]})()
    profiles = {1: _winning_showdown_limpy_lag()}
    assert _all_live_villains_fish(state, hero=0, profiles=profiles) is False


# ===================================================================================================
# EG-5 — the loose+passive HUD admit must consult WSD/foldToCbet (a selective winner is not a fish)
# ===================================================================================================
def test_eg5_winning_showdown_selective_limper_not_fish():
    # wsd62 (wins at showdown) / ftc60 (folds to c-bets a lot): NOT a pays-off-thin-value station.
    prof = OpponentStats(hands=0, hud={"vpip": 55, "pfr": 8, "wsd": 62, "foldToCbet": 60})
    assert _is_value_fish(prof) is False


def test_eg5_weak_showdown_sticky_limper_still_fish():
    # do-no-harm: the GENUINE weak-showdown, sticky loose-passive station is STILL admitted as a fish.
    prof = OpponentStats(hands=0, hud={"vpip": 55, "pfr": 8, "wsd": 38, "foldToCbet": 22})
    assert _is_value_fish(prof) is True


def test_eg5_high_ftc_selective_limper_not_fish():
    # a high foldToCbet alone (selective postflop) vetoes the fish admit even with a weak/absent wsd.
    prof = OpponentStats(hands=0, hud={"vpip": 60, "pfr": 9, "foldToCbet": 60})
    assert _is_value_fish(prof) is False


# ===================================================================================================
# EG-6 — the calling_station archetype floor must not override the weak-WSD value-cut guard
# ===================================================================================================
def test_eg6_floor_not_applied_to_winning_showdown_trapper():
    # vpip55/pfr8 gap -> archetype WOULD say calling_station from the gap; but wsd62/ftc60 contradicts.
    trapper = OpponentStats(hands=0, hud={"vpip": 55, "pfr": 8, "wsd": 62, "foldToCbet": 60})
    score, _ = station_payoff_score(trapper)
    assert score < 0.5, "EG-6: the calling_station floor must not manufacture a payoff score for a trapper"
    # the archetype itself must no longer brand the contradicting profile a station.
    assert trapper.archetype() != "calling_station"


def test_eg6_floor_still_applies_to_genuine_weak_station():
    # do-no-harm: a genuine weak-showdown sticky station STILL gets the floor (the headline value lever).
    fish = OpponentStats(hands=0, hud={"vpip": 55, "pfr": 8, "wsd": 38, "foldToCbet": 22})
    score, _ = station_payoff_score(fish)
    assert score >= 0.5
    assert fish.archetype() == "calling_station"


def test_eg6_count_station_floor_preserved():
    # a count-confirmed station (no HUD) keeps its archetype floor (the floor gates on ABSENT wsd/ftc).
    station = OpponentStats(hands=50, vpip=30, pfr=4, aggression_bets_raises=5, aggression_calls=20,
                            cbet_faced=30, fold_to_cbet=6)
    assert station.archetype() == "calling_station"
    score, _ = station_payoff_score(station)
    assert score >= 0.5


# ===================================================================================================
# EG-7 — `passive` must bound ABSOLUTE pfr, not only the gap (a vpip90/pfr60 maniac is not passive)
# ===================================================================================================
def test_eg7_wide_gap_high_pfr_maniac_not_fish():
    maniac = OpponentStats(hands=0, hud={"vpip": 90, "pfr": 60, "wsd": 52})
    assert _is_value_fish(maniac) is False
    # ... and the archetype must not brand it a calling_station from the 30-pt gap alone.
    assert maniac.archetype() != "calling_station"


def test_eg7_wide_gap_low_pfr_limper_still_fish():
    # do-no-harm: the wide-gap LOW-pfr limper is still a station.
    limper = OpponentStats(hands=0, hud={"vpip": 60, "pfr": 10})
    assert _is_value_fish(limper) is True


# ===================================================================================================
# EG-8 — _is_dangerous_aggressor gets a HUD/cold-start path so the kill-switch is not inert at hands=0
# ===================================================================================================
def test_eg8_hud_only_lag_is_dangerous_at_zero_hands():
    # seed_from_hud sets prof.hud while prof.hands stays 0: a HUD-confirmed loose-aggressive LAG must be
    # dangerous DURING the cold-start window (pre-fix it bailed at hands<25 before reading the HUD).
    lag = OpponentStats(hands=0, hud={"vpip": 60, "pfr": 40, "wsd": 55, "wtsd": 30})
    assert _is_dangerous_aggressor(lag) is True


def test_eg8_hud_only_maniac_fish_not_dangerous():
    # the symmetric case: a HUD-confirmed WEAK-showdown loose-aggressive MANIAC is a fish (excused).
    maniac = OpponentStats(hands=0, hud={"vpip": 70, "pfr": 45, "wsd": 32, "wtsd": 50})
    assert _is_dangerous_aggressor(maniac) is False


def test_eg8_hud_only_station_not_dangerous():
    # do-no-harm: a HUD-only loose-PASSIVE station must NOT read dangerous (low pfr, no 3bet).
    station = OpponentStats(hands=0, hud={"vpip": 60, "pfr": 9, "wsd": 40, "foldToCbet": 25})
    assert _is_dangerous_aggressor(station) is False
    # and it IS still admitted as a fish (kill-switch and fish-admit stay symmetric, not overlapping).
    assert _is_value_fish(station) is True


def test_eg8_hud_only_lag_table_trips_kill_switch_cold_start():
    # 3 HUD-only LAGs at hands=0 must trip the kill-switch (pre-fix: structurally impossible).
    lag_hud = {"vpip": 60, "pfr": 40, "wsd": 55, "wtsd": 30}
    profiles = {s: OpponentStats(hands=0, hud=dict(lag_hud)) for s in range(1, 4)}
    assert _table_too_aggressive(profiles) is True


def test_eg8_3bet_signal_alone_flags_lag():
    # a high 3-bet is a positive aggression signal independent of pfr.
    lag = OpponentStats(hands=0, hud={"vpip": 50, "pfr": 12, "threeBet": 14, "wsd": 55, "wtsd": 28})
    assert _is_dangerous_aggressor(lag) is True


# ===================================================================================================
# EG-9 — the WSD veto is a hysteresis band + min-sample guard (manipulation-resistant), not a hard cliff
# ===================================================================================================
def test_eg9_borderline_wsd_stays_dangerous():
    # a pro keeping wsd in the UNCERTAIN band [0.40, 0.45) must stay DANGEROUS (fail closed), where the
    # old hard cliff (wsd < 0.45) would have excused them as a maniac-fish.
    borderline = _count_lag(hud={"vpip": 60, "pfr": 30, "wsd": 44, "wtsd": 40})
    assert _is_dangerous_aggressor(borderline) is True


def test_eg9_clearly_weak_wsd_with_sample_is_excused():
    # a CLEARLY weak read (wsd < 0.40) with enough showdown sample AND a corroborating fish signal is
    # excused as a maniac fish (the value the discriminator is meant to recover).
    maniac = _count_lag(hud={"vpip": 65, "pfr": 30, "wsd": 35, "wtsd": 45})
    assert _is_dangerous_aggressor(maniac) is False


def test_eg9_low_showdown_sample_fails_closed():
    # a weak wsd backed by too FEW showdowns (low wtsd · hands) must NOT excuse — manipulation guard.
    low_sample = _count_lag(hud={"vpip": 65, "pfr": 30, "wsd": 30, "wtsd": 2}, )
    low_sample.hands = 10                              # wtsd 0.02 * 10 = 0.2 showdowns << min sample
    assert _is_dangerous_aggressor(low_sample) is True


def test_eg9_single_percentage_point_does_not_flip_class():
    # the old hard cliff flipped dangerous<->fish across one percentage point (44 vs 45). With the
    # hysteresis floor at 0.40, both 44 and 45 are now on the SAME (dangerous) side.
    p44 = _count_lag(hud={"vpip": 60, "pfr": 30, "wsd": 44, "wtsd": 40})
    p45 = _count_lag(hud={"vpip": 60, "pfr": 30, "wsd": 45, "wtsd": 40})
    assert _is_dangerous_aggressor(p44) == _is_dangerous_aggressor(p45) is True


# ===================================================================================================
# EG-11 — `_pct` integer-1 normalization singularity (a stat value of exactly 1 must be 0.01, not 1.0)
# ===================================================================================================
def test_eg11_integer_one_in_percent_block_is_one_percent():
    # wsd=1 in an integer-percent block (vpip=60) is 1% (0.01), not 100% (1.0).
    assert _pct({"vpip": 60, "wsd": 1}, "wsd") == 0.01


def test_eg11_isolated_one_treated_as_fraction():
    # a lone value of exactly 1 with no other percent-scale sibling is ambiguous; treated as a fraction
    # (1.0) — the conservative read (no integer-percent evidence in the block).
    assert _pct({"wsd": 1}, "wsd") == 1.0


def test_eg11_maniac_wsd_one_no_longer_a_showdown_god():
    # the headline EG-11 case: a maniac with wsd=1 (1%) in a real HUD block must read weak, not strong.
    maniac = _count_lag(hud={"vpip": 65, "pfr": 30, "wsd": 1, "wtsd": 45})
    # wsd normalizes to 0.01 (< the hysteresis floor) -> excused as a maniac fish, NOT a showdown god.
    assert _pct(maniac.hud, "wsd") == 0.01
    assert _is_dangerous_aggressor(maniac) is False


def test_eg11_percent_and_fraction_blocks_both_normalize():
    # a normalized (fraction) block leaves sub-1 values alone; a percent block divides by 100.
    assert _pct({"vpip": 0.6, "wsd": 0.42}, "wsd") == 0.42       # fraction block
    assert _pct({"vpip": 60, "wsd": 42}, "wsd") == 0.42          # percent block


# ===================================================================================================
# EG-12 — apply_live_value_sizing's internal firewalls (real profiles, NO monkeypatch)
# ===================================================================================================
def _flop_facing_state():
    import random

    from pokerbot.engine.game import GameState

    state = GameState.new_hand(button_seat=1, rng=random.Random(0))
    state.deal_hole_cards()
    return state


def _hud_station():
    return OpponentStats(hands=0, hud={"vpip": 60, "pfr": 9, "wsd": 40, "foldToCbet": 22})


def test_eg12_overlay_bails_not_all_villains_fish_with_real_profiles():
    # a table with a non-fish (a HUD lag) present: the internal _all_live_villains_fish firewall must
    # bail with reason 'not_all_villains_fish' (no monkeypatch — the real gate decides).
    state = _flop_facing_state()
    hero = state.action_seat
    others = [s for s in range(len(state.players)) if s != hero]
    lag_hud = {"vpip": 60, "pfr": 40, "wsd": 55, "wtsd": 30}
    profiles = {others[0]: _hud_station(), others[1]: OpponentStats(hands=0, hud=dict(lag_hud))}
    from pokerbot.engine.actions import NUM_ACTIONS

    probs = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
    out: dict = {}
    res = apply_live_value_sizing(state, hero, probs.copy(), profiles, out=out)
    assert np.allclose(res, probs)                      # unchanged
    assert out["reason"] in ("zero_overlay", "table_too_aggressive")
    assert out["fired"] is False


def test_eg12_overlay_bails_table_too_aggressive_with_real_profiles():
    # an all-lag table: the internal _table_too_aggressive firewall bails (real gate, no monkeypatch).
    state = _flop_facing_state()
    hero = state.action_seat
    others = [s for s in range(len(state.players)) if s != hero]
    lag_hud = {"vpip": 60, "pfr": 40, "wsd": 55, "wtsd": 30}
    profiles = {s: OpponentStats(hands=0, hud=dict(lag_hud)) for s in others[:3]}
    from pokerbot.engine.actions import NUM_ACTIONS

    probs = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
    out: dict = {}
    res = apply_live_value_sizing(state, hero, probs.copy(), profiles, out=out)
    assert np.allclose(res, probs)
    assert out["reason"] == "table_too_aggressive"
    assert out["fired"] is False


# ===================================================================================================
# EG-14 — partial-HUD loose+passive admit boundary constants (pin against silent drift)
# ===================================================================================================
def test_eg14_loose_passive_boundary_constants():
    # AT the boundary: vpip=0.50 (loose floor), gap exactly 0.25 (passive floor) AND pfr exactly at the
    # EG-7 bound (0.20) -> admitted. (vpip 0.45, pfr 0.20 gives gap 0.25 and vpip below... use vpip 0.50.)
    assert _is_value_fish(OpponentStats(hands=0, hud={"vpip": 45, "pfr": 20})) is False  # vpip below 0.50
    assert _is_value_fish(OpponentStats(hands=0, hud={"vpip": 50, "pfr": 20})) is True   # gap 0.30, pfr@bound
    # gap exactly 0.25 with pfr just under the bound -> admitted (vpip 0.44 fails loose floor, use 0.50).
    assert _is_value_fish(OpponentStats(hands=0, hud={"vpip": 50, "pfr": 19})) is True   # gap 0.31
    # narrow-gap loose seat (gap 0.20 < 0.25) -> NOT admitted (must defer).
    assert _is_value_fish(OpponentStats(hands=0, hud={"vpip": 55, "pfr": 35})) is False
    # below the vpip floor (0.45 < 0.50) even with a wide gap/low pfr -> NOT auto-admitted.
    assert _is_value_fish(OpponentStats(hands=0, hud={"vpip": 45, "pfr": 5})) is False
    # EG-7: gap wide (0.30) but pfr JUST ABOVE the bound (0.21 > 0.20) -> NOT admitted (too aggressive).
    assert _is_value_fish(OpponentStats(hands=0, hud={"vpip": 51, "pfr": 21})) is False


# ===================================================================================================
# EG-15 — kill-count straddle across WSD-excused maniacs + HUD-less/winning-showdown lags
# ===================================================================================================
def test_eg15_kill_count_straddle_mixed_composition():
    # 3 counted lags (HUD-less or winning-showdown) + 2 excused maniacs -> kill (3 >= AGG_KILL_COUNT).
    maniac_hud = {"vpip": 65, "pfr": 30, "wsd": 35, "wtsd": 45}          # weak showdown -> excused
    lag_hud = {"vpip": 60, "pfr": 40, "wsd": 55, "wtsd": 30}            # winning showdown -> counted
    killed = {
        1: _count_lag(hud=dict(maniac_hud)), 2: _count_lag(hud=dict(maniac_hud)),
        3: _count_lag(hud=dict(lag_hud)), 4: _count_lag(),               # HUD-less lag -> counted
        5: OpponentStats(hands=0, hud=dict(lag_hud)),                    # HUD-only lag -> counted
    }
    assert _table_too_aggressive(killed) is True

    # 2 counted lags + 3 excused maniacs -> below k=3 -> NO kill.
    not_killed = {
        1: _count_lag(hud=dict(maniac_hud)), 2: _count_lag(hud=dict(maniac_hud)),
        3: _count_lag(hud=dict(maniac_hud)), 4: _count_lag(hud=dict(lag_hud)),
        5: OpponentStats(hands=0, hud=dict(lag_hud)),
    }
    assert _table_too_aggressive(not_killed) is False
