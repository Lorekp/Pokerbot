"""NLHE skill-tier leak system (pokerbot.opponent.skill_tiers): the difficulty-preset core.

Pure-function tests — no checkpoint, no engine. Guards leak monotonicity, the flawless/superhuman
identity, continuous-level interpolation, ADAPT gating, and deterministic sampling.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.opponent import skill_tiers as st
from pokerbot.opponent.skill_tiers import (
    ADAPT, DEFAULT_PROFILES, TIER_ORDER, AutoDifficulty, NLHELeakProfile, apply_leaks,
    is_adaptive, profile_for_level, sample_action,
)

ALL_LEGAL = np.ones(NUM_ACTIONS, dtype=bool)


def _uniform():
    return np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float64)


def _facing_bet_mask():
    """Facing a bet: fold/call legal, check illegal; some bets + all-in legal."""
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    m[st.FOLD] = m[st.CALL] = True
    m[st.ALLIN] = True
    for i in st.BETS:
        m[i] = True
    return m


def _open_mask():
    """Unopened/free-check: check legal, call legal-ish, bets + all-in legal, fold legal."""
    m = np.ones(NUM_ACTIONS, dtype=bool)
    return m


# --------------------------------------------------------------------------- shape & registry
def test_tier_order_and_adapt_gating():
    assert TIER_ORDER == ("beginner", "intermediate", "advanced", "expert",
                          "semi_pro", "professional", "superhuman")
    # beginner/intermediate never adapt; advanced+ do.
    assert not is_adaptive("beginner") and not is_adaptive("intermediate")
    for t in ("advanced", "expert", "semi_pro", "professional", "superhuman"):
        assert is_adaptive(t) and t in ADAPT
    # exploit_strength is monotone non-decreasing up the adaptive ladder.
    strengths = [ADAPT[t][0] for t in ("advanced", "expert", "semi_pro", "professional", "superhuman")]
    assert strengths == sorted(strengths)


def test_apply_leaks_returns_legal_distribution():
    out = apply_leaks(_uniform(), _facing_bet_mask(),
                      profile=DEFAULT_PROFILES["beginner"], facing_bet=True, hand_strength=0.3)
    assert out.shape == (NUM_ACTIONS,)
    assert abs(out.sum() - 1.0) < 1e-9
    assert out[st.CHECK] == 0.0          # check illegal facing a bet -> zero mass


# --------------------------------------------------------------------------- flawless identity
def test_superhuman_is_identity():
    p = _uniform()
    out = apply_leaks(p, ALL_LEGAL, profile=DEFAULT_PROFILES["superhuman"],
                      facing_bet=False, hand_strength=0.5)
    np.testing.assert_allclose(out, p, atol=1e-12)
    assert DEFAULT_PROFILES["superhuman"].is_flawless()
    assert DEFAULT_PROFILES["professional"].is_flawless()   # no structural deviation


# --------------------------------------------------------------------------- leak monotonicity
def test_station_increases_call_mass():
    mask = _facing_bet_mask()
    base = _uniform()
    weak = apply_leaks(base, mask, profile=NLHELeakProfile(station=0.2), facing_bet=True, hand_strength=0.3)
    strong = apply_leaks(base, mask, profile=NLHELeakProfile(station=0.8), facing_bet=True, hand_strength=0.3)
    assert strong[st.CALL] > weak[st.CALL] > base[st.CALL] * mask[st.CALL]


def test_overfold_increases_fold_mass_facing_bet():
    mask = _facing_bet_mask()
    base = _uniform()
    light = apply_leaks(base, mask, profile=NLHELeakProfile(overfold=0.1), facing_bet=True, hand_strength=0.3)
    heavy = apply_leaks(base, mask, profile=NLHELeakProfile(overfold=0.9), facing_bet=True, hand_strength=0.3)
    assert heavy[st.FOLD] > light[st.FOLD]


def test_overfold_keeps_premiums():
    """A strong hand (hs high) is folded far less than a weak one for the same overfold knob."""
    mask = _facing_bet_mask()
    base = _uniform()
    weak = apply_leaks(base, mask, profile=NLHELeakProfile(overfold=0.8), facing_bet=True, hand_strength=0.1)
    nuts = apply_leaks(base, mask, profile=NLHELeakProfile(overfold=0.8), facing_bet=True, hand_strength=0.95)
    assert weak[st.FOLD] > nuts[st.FOLD]


def test_passivity_cuts_aggression():
    mask = _open_mask()
    base = _uniform()
    out = apply_leaks(base, mask, profile=NLHELeakProfile(passivity=0.8), facing_bet=False, hand_strength=0.6)
    base_aggr = sum(base[i] for i in st.AGGR)
    out_aggr = sum(out[i] for i in st.AGGR)
    assert out_aggr < base_aggr
    assert out[st.CHECK] > base[st.CHECK]     # aggression sinks into check


def test_underbluff_cuts_weak_hand_aggression_only():
    mask = _open_mask()
    base = _uniform()
    weak = apply_leaks(base, mask, profile=NLHELeakProfile(underbluff=0.9), facing_bet=False, hand_strength=0.1)
    strong = apply_leaks(base, mask, profile=NLHELeakProfile(underbluff=0.9), facing_bet=False, hand_strength=0.9)
    weak_aggr = sum(weak[i] for i in st.AGGR)
    strong_aggr = sum(strong[i] for i in st.AGGR)
    # value hands keep betting; air stops bluffing.
    assert weak_aggr < strong_aggr


def test_loose_preflop_lowers_fold():
    mask = _open_mask()
    base = _uniform()
    out = apply_leaks(base, mask, profile=NLHELeakProfile(loose_preflop=0.9), facing_bet=False,
                      hand_strength=0.5, street="preflop")
    assert out[st.FOLD] < base[st.FOLD]
    # ...and it does nothing postflop (gated on street).
    post = apply_leaks(base, mask, profile=NLHELeakProfile(loose_preflop=0.9), facing_bet=False,
                       hand_strength=0.5, street="flop")
    np.testing.assert_allclose(post, base, atol=1e-12)


def test_sizing_blind_collapses_bet_shape():
    mask = _open_mask()
    # spread mass across the sized-bet buckets
    p = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for i in st.BETS:
        p[i] = 1.0
    p /= p.sum()
    out = apply_leaks(p, mask, profile=NLHELeakProfile(sizing_blind=1.0), facing_bet=False, hand_strength=0.6)
    nonzero_bets = [i for i in st.BETS if out[i] > 1e-9]
    assert len(nonzero_bets) == 1                     # collapsed onto a single bucket
    # total aggression frequency preserved (no all-in mass in input)
    assert abs(sum(out[i] for i in st.BETS) - sum(p[i] for i in st.BETS)) < 1e-9


# --------------------------------------------------------------------------- continuous level
def test_profile_for_level_endpoints():
    cal = {t: DEFAULT_PROFILES[t] for t in TIER_ORDER}
    b = profile_for_level(0.0, cal)
    p = profile_for_level(1.0, cal)
    # level 0 ≈ beginner (loosest), level 1 ≈ professional (near-clean)
    assert b.station > 0.3 and b.loose_preflop > 0.3
    assert p.station == 0.0 and p.loose_preflop == 0.0 and p.overfold == 0.0
    # midpoint is strictly between on a representative knob
    mid = profile_for_level(0.5, cal)
    assert p.passivity <= mid.passivity <= b.passivity


def test_profile_for_level_top_has_no_structural_leaks():
    # level 1.0 maps to professional, never superhuman (superhuman is off the continuous ladder).
    p = profile_for_level(1.0)
    assert p.is_flawless()


# --------------------------------------------------------------------------- sampling determinism
def test_sample_action_deterministic_and_legal():
    mask = _facing_bet_mask()
    p = apply_leaks(_uniform(), mask, profile=DEFAULT_PROFILES["beginner"], facing_bet=True, hand_strength=0.3)
    a = sample_action(p, mask, seed=12345)
    b = sample_action(p, mask, seed=12345)
    assert a == b                                   # same seed -> same draw
    assert mask[a]                                  # never an illegal action
    # different seeds explore different actions over a batch
    draws = {sample_action(p, mask, seed=s) for s in range(200)}
    assert len(draws) >= 2


def test_sample_respects_frequency():
    """A station's call-heavy distribution is sampled call most of the time."""
    mask = _facing_bet_mask()
    p = apply_leaks(_uniform(), mask, profile=NLHELeakProfile(station=0.9), facing_bet=True, hand_strength=0.3)
    counts = np.bincount([sample_action(p, mask, seed=s) for s in range(2000)], minlength=NUM_ACTIONS)
    assert counts.argmax() == st.CALL


# --------------------------------------------------------------------------- automatic
def test_auto_difficulty_raises_level_when_bot_too_weak():
    auto = AutoDifficulty(target_bot_bb_per_100=5.0, level=0.5, gain=0.01)
    # bot losing (−10 bb/100) vs a +5 target => big positive error => level climbs
    for _ in range(50):
        auto.update(-10.0)
    assert auto.level > 0.5
    # ...and bot crushing way past target pulls it back down
    auto2 = AutoDifficulty(target_bot_bb_per_100=0.0, level=0.5, gain=0.01)
    for _ in range(50):
        auto2.update(50.0)
    assert auto2.level < 0.5


def test_sample_action_degenerate_inputs_dont_crash():
    """Robustness: tiny/zero/no-legal-mass inputs must not trip numpy's strict sum-to-1 tolerance."""
    lm = _facing_bet_mask()
    assert lm[sample_action(np.full(NUM_ACTIONS, 1e-14) * lm, lm, seed=1)]   # tiny-but-nonzero mass
    assert lm[sample_action(np.zeros(NUM_ACTIONS), lm, seed=2)]              # all-zero -> uniform legal
    none_legal = np.zeros(NUM_ACTIONS, dtype=bool)
    idx = sample_action(np.ones(NUM_ACTIONS), none_legal, seed=3)            # no legal action -> no crash
    assert 0 <= idx < NUM_ACTIONS


def test_auto_difficulty_smooths_per_hand_noise():
    """Per-hand pnl (high variance) is EMA-smoothed, so zero-mean noise never walks the level off-rail."""
    import random as _r

    auto = AutoDifficulty(target_bot_bb_per_100=0.0, level=0.5, gain=0.0002, half_life=100)
    rng = _r.Random(0)
    lvl = 0.5
    for _ in range(300):
        lvl = auto.update(rng.gauss(0.0, 5.0))
    assert 0.0 <= lvl <= 1.0 and auto._n == 300


def test_auto_from_dict_ignores_unknown_keys():
    auto = AutoDifficulty.from_dict({"target_bot_bb_per_100": 3.0, "level": 0.7, "bogus": 1})
    assert auto.target_bot_bb_per_100 == 3.0 and auto.level == 0.7


def test_tier_profile_unknown_raises():
    with pytest.raises(ValueError):
        st.tier_profile("grandmaster")
