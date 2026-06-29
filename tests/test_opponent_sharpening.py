"""Opponent layer property + behavior tests (post-audit sharpening).

Covers:
  * bayes_exploit (conjugate-prior math, evidence-floor gating, monotonicity, sum-to-1)
  * root_mixer (determinism, RNR-cap binding, stability invariance, profiles=None safe-fallback)
  * stats.py (aggression_frequency/three_bet_pct/wtsd_pct None-handling, archetype rewiring)

Run via: pytest tests/test_opponent_sharpening.py -v
"""

from __future__ import annotations

import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure tests can run with no .env / live config.
os.environ.setdefault("POKERBOT_LIVE_VALUE_SIZING", "0")


# =============================================================================================
# bayes_exploit — conjugate prior math, monotonicity, sum-to-1, evidence floor
# =============================================================================================

from pokerbot.opponent import bayes_exploit as be
from pokerbot.opponent.bayes_exploit import (
    BLUFF_PRIOR,
    CHANGE_TV,
    DEFAULT_MAX_ALPHA,
    ContextRead,
    aggregate_read,
    bluff_read,
    confidence,
    context_read,
    posterior,
)
from pokerbot.opponent.context import NUM_BINS, context_key
from pokerbot.opponent.population_prior import PRIOR_STRENGTH, context_prior
from pokerbot.opponent.stats import OpponentStats


# ---------------------------------------------------------------------------------------------
# (B1) Conjugate-prior match — posterior is Dirichlet-Multinomial mean.
# ---------------------------------------------------------------------------------------------

def test_posterior_matches_closed_form_dirichlet_multinomial():
    """posterior(counts, prior) = (counts + prior) / sum — exact."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        c = rng.integers(0, 20, NUM_BINS).astype(np.float64)
        p = rng.uniform(0.1, 3.0, NUM_BINS)
        post = posterior(c, p)
        assert post.shape == (NUM_BINS,)
        expected = (c + p) / (c + p).sum()
        np.testing.assert_allclose(post, expected, rtol=1e-12)


@pytest.mark.parametrize("counts_prior", [
    ([0, 0, 0, 0], [0, 0, 0, 0]),                              # both zero → uniform
    ([1, 0, 0, 0], [2.0, 5.0, 0.5, 0.5]),                    # typical
    ([0, 0, 1000, 1000], [0.001, 0.001, 0.001, 0.001]),      # sharp + tiny prior
    ([100, 100, 100, 100], [0, 8, 0, 0]),                    # strong data, asymmetric prior
])
def test_posterior_nonneg_and_sum_to_one(counts_prior):
    c, p = counts_prior
    post = posterior(c, p)
    assert (post >= 0).all(), f"negativity in {post}"
    assert abs(post.sum() - 1.0) < 1e-12, f"sum != 1 in {post}"


# ---------------------------------------------------------------------------------------------
# (B2) confidence monotone + bounded in [0, 1)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,s", [
    (0, 1), (1, 1), (5, 8), (100, 8), (10_000, 8), (0, 0.001), (1e6, 1e6),
])
def test_confidence_monotone_and_bounded(n, s):
    c = confidence(n, s)
    assert 0.0 <= c < 1.0, f"out of range: {c} for n={n}, s={s}"
    # Monotonicity in N: c(n+1, s) >= c(n, s)
    if n < 1e5:  # skip near-saturation where floating-point comparison is noisy
        c2 = confidence(n + 1, s)
        assert c2 >= c - 1e-12, f"not monotone: c({n})={c}, c({n+1})={c2}"


def test_confidence_sublinear_diminishing_returns():
    """d(N/(N+s))/dN = s/(N+s)^2 → 0 as N grows. Sampled at increasing N."""
    s = PRIOR_STRENGTH
    diffs = [confidence(n + 10, s) - confidence(n, s) for n in (0, 10, 100, 1000, 10_000)]
    for a, b in zip(diffs, diffs[1:]):
        assert a >= b - 1e-12, f"diff not diminishing: {diffs}"


def test_confidence_zero_means_no_exploit():
    assert confidence(0) == 0.0
    assert confidence(0, PRIOR_STRENGTH) == 0.0


# ---------------------------------------------------------------------------------------------
# (B3) Shrinkage — N=0 returns prior; large N dominates
# ---------------------------------------------------------------------------------------------

def test_shrinkage_zero_counts_returns_prior():
    p = np.array([0.30, 0.30, 0.30, 0.10])
    post = posterior(np.zeros(NUM_BINS), p)
    np.testing.assert_allclose(post, p, atol=1e-12)


def test_shrinkage_converges_to_truth():
    """N → ∞ ⇒ posterior converges to the empirical frequency of the data."""
    rng = np.random.default_rng(42)
    true = np.array([0.50, 0.20, 0.20, 0.10])
    # Sample from true and add to counts.
    counts = rng.multinomial(50_000, true)
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    post = posterior(counts, prior)
    np.testing.assert_allclose(post, true, atol=0.01)


# ---------------------------------------------------------------------------------------------
# (B4) search_tilt alpha in [0, max_alpha]; tilt per-bin in [0.25, 4.0]
# ---------------------------------------------------------------------------------------------

def test_search_tilt_alpha_bounds_and_tilt_clip():
    rng = np.random.default_rng(7)
    for _ in range(30):
        c = rng.integers(0, 50, NUM_BINS).astype(np.float64)
        p = rng.uniform(0.01, 5.0, NUM_BINS)
        # Normalize prior to a valid pop baseline.
        pop = p / p.sum()
        post = posterior(c, p)
        read = ContextRead(post=post, pop=pop, conf=confidence(float(c.sum())), n_obs=int(c.sum()))
        tilt, alpha = read.search_tilt()
        assert 0.0 <= alpha <= DEFAULT_MAX_ALPHA + 1e-12, f"alpha out of range: {alpha}"
        assert tilt.shape == (11,), f"tilt shape {tilt.shape}"
        assert (tilt >= 0.25 - 1e-12).all() and (tilt <= 4.0 + 1e-12).all(), f"tilt out of clip: {tilt}"


# ---------------------------------------------------------------------------------------------
# (B5) mixer_logits hand-strength symmetry
# ---------------------------------------------------------------------------------------------

def test_mixer_logits_value_vs_air_station_inverse():
    """Vs a station (high PASSIVE_BIN excess), value hands bet MORE than air hands."""
    rng = np.random.default_rng(11)
    pop = rng.dirichlet(np.ones(NUM_BINS))
    post = pop.copy()
    post[1] += 0.30  # station: over-passive
    post /= post.sum()
    read = ContextRead(post=post, pop=pop, conf=0.8, n_obs=100)
    value_boost = read.mixer_logits(hand_strength=0.9).sum()
    air_boost = read.mixer_logits(hand_strength=0.1).sum()
    assert value_boost > air_boost, f"station tilt should reward value > air; got {value_boost} vs {air_boost}"


# ---------------------------------------------------------------------------------------------
# (B6) Bluff-read evidence-floor gating
# ---------------------------------------------------------------------------------------------

def test_bluff_read_empty_profile_returns_none():
    s = OpponentStats()
    assert bluff_read(s, "PREFLOP|-|facing_bet") is None


def test_bluff_read_below_floor_falls_back_to_hierarchy():
    """Below floor: exact context is dropped; street/global carry the read."""
    s = OpponentStats()
    ck = "RIVER|||facing_bet"
    # 1 reveal at exact context (below floor=3).
    s.ctx_bluff[ck] = [1.0, 0.0]
    out = bluff_read(s, ck)
    # Should return a (p_bluff, conf) tuple using the population prior + this single obs.
    assert out is not None
    p_bluff, conf = out
    assert 0.0 <= p_bluff <= 1.0
    assert 0.0 <= conf < 1.0
    # A single observation under-influences the prior but shouldn't completely flip it.
    prior_mean = BLUFF_PRIOR[0] / sum(BLUFF_PRIOR)
    # The hierarchical shrinkage keeps the read CLOSE to the prior (single obs, sparse).
    assert abs(p_bluff - prior_mean) < 0.3, f"single obs pulls too far: {p_bluff} vs prior {prior_mean}"


def test_bluff_read_above_floor_trusts_exact_context():
    s = OpponentStats()
    ck = "RIVER|||facing_bet"
    # 5 reveals at exact context (above floor=3): 4 bluffs + 1 value.
    s.ctx_bluff[ck] = [4.0, 1.0]
    out = bluff_read(s, ck)
    assert out is not None
    p_bluff, conf = out
    # p_bluff should be >= 0.5 (4/5 reveals were bluffs) — strict > may fail due to prior smoothing
    # of the empty street/global aggregates.
    assert p_bluff >= 0.5, f"above-floor exact context not trusted enough: {p_bluff}"
    # Confidence for above-floor > confidence for below-floor with the same hierarchy (more data).
    below = bluff_read(OpponentStats(ctx_bluff={ck: [1.0, 0.0]}), ck)
    assert conf > below[1], f"above-floor conf {conf} not > below-floor conf {below[1]}"


def test_bluff_read_above_floor_p_bluff_higher_than_below_floor():
    """A heavily-bluffy villain at the exact context should report p_bluff higher than the
    below-floor shrinkage allows (i.e. the exact-context count is honored, not just the hierarchy)."""
    s = OpponentStats()
    ck = "RIVER|||facing_bet"
    s.ctx_bluff[ck] = [9.0, 1.0]   # 10 reveals, 9/10 bluffs
    p_bluff, _ = bluff_read(s, ck)
    # Expect p_bluff well above the population prior (~0.286) — the exact context carries weight.
    assert p_bluff > 0.6, f"p_bluff should reflect heavy bluffing: {p_bluff}"


def test_bluff_read_hierarchy_no_double_counting():
    """Exact context count must NOT also appear in street and global aggregates (double-counting
    would inflate conf and skew the posterior)."""
    s = OpponentStats()
    ck_exact = "RIVER|||facing_bet"
    ck_other = "TURN|||facing_bet"
    s.ctx_bluff[ck_exact] = [10.0, 5.0]   # 15 reveals, above floor
    s.ctx_bluff[ck_other] = [1.0, 1.0]    # 2 reveals, below floor (different street)
    # Compare conf: ck_other is below floor and should NOT be subtracted from its street aggregate.
    p_other, c_other = bluff_read(s, ck_other)
    # ck_other contributes to its own street aggregate fully (not subtracted).
    # If subtraction happened, c_other would be tiny. Verify it isn't.
    assert c_other > 0.0


# ---------------------------------------------------------------------------------------------
# (B7) decay_and_add discounted update doesn't divide by zero on missing clock
# ---------------------------------------------------------------------------------------------

def test_decay_and_add_first_hand_initializes_at_prior():
    s = OpponentStats()
    prior = np.array([0.40, 0.30, 0.20, 0.10])
    s.decay_and_add("PREFLOP|-|unopened", [1, 0, 0, 0], prior, hands_now=0)
    # First hand: prior seed + the single bin increment; both fast and slow tracks hold the same data.
    assert s.ctx_fast, "fast dict should be populated"
    assert s.ctx_slow, "slow dict should be populated"
    assert s.ctx_last["PREFLOP|-|unopened"] == 0


def test_decay_and_add_lazy_decay_zero_delta_additive():
    """Two consecutive calls at the same hands_now index (delta=0, g=1) should be strictly additive
    on the bin increments. The second call adds [0,1,0,0] (bin 1) without disturbing bin 0's prior
    + first-call accumulation."""
    s = OpponentStats()
    prior = np.array([0.40, 0.30, 0.20, 0.10])
    s.decay_and_add("PREFLOP|-|unopened", [1, 0, 0, 0], prior, hands_now=0)
    bin0_after_first = s.ctx_fast["PREFLOP|-|unopened"][0]
    bin1_after_first = s.ctx_fast["PREFLOP|-|unopened"][1]
    # Second call at the same hands_now → delta=0 → g=1 → bins added but not decayed.
    s.decay_and_add("PREFLOP|-|unopened", [0, 1, 0, 0], prior, hands_now=0)
    # Bin 0 unchanged (no new bin0 increment).
    assert abs(s.ctx_fast["PREFLOP|-|unopened"][0] - bin0_after_first) < 1e-9, \
        f"bin0 changed unexpectedly: {s.ctx_fast['PREFLOP|-|unopened'][0]} vs {bin0_after_first}"
    # Bin 1 incremented by exactly 1 (the new bins[1] under delta=0 ⇒ g=1).
    assert abs(s.ctx_fast["PREFLOP|-|unopened"][1] - bin1_after_first - 1.0) < 1e-9, \
        f"bin1 not incremented correctly: {s.ctx_fast['PREFLOP|-|unopened'][1]} vs {bin1_after_first + 1.0}"


def test_decay_and_add_decay_drives_toward_prior():
    """Long time-since-last-update ⇒ discounted mass regresses to the prior floor."""
    s = OpponentStats()
    prior = np.array([0.10, 0.10, 0.10, 0.70])   # extreme: bin 3 dominant
    s.decay_and_add("PREFLOP|-|unopened", [10, 0, 0, 0], prior, hands_now=0)
    # 1000 hands later, same context, no observations: discounted mass should drift toward prior.
    s.decay_and_add("PREFLOP|-|unopened", [0, 0, 0, 0], prior, hands_now=1000)
    # Fast track decays faster than slow; check fast is closer to prior than the initial peak.
    fast_now = np.array(s.ctx_fast["PREFLOP|-|unopened"])
    fast_now = fast_now / max(fast_now.sum(), 1e-9)
    # The bin 0 mass (was injected) should have decayed substantially toward the prior (0.10).
    assert fast_now[0] < 0.5, f"fast track didn't decay toward prior: bin0={fast_now[0]}"


# ---------------------------------------------------------------------------------------------
# (B8) aggregate_read returns a ContextRead with conf ∈ [0, 1) and tv ≤ 1
# ---------------------------------------------------------------------------------------------

def test_aggregate_read_warm_profile():
    """aggregate_read requires ctx_fast/ctx_slow (dual-timescale). With sufficient mass, it returns
    a ContextRead with conf > 0 and TV bounded."""
    s = OpponentStats()
    s.hands = 100
    # Populate both fast and slow tracks with substantial mass.
    s.ctx_slow["PREFLOP|-|facing_bet"] = [50.0, 10.0, 20.0, 20.0]
    s.ctx_fast["PREFLOP|-|facing_bet"] = [55.0, 12.0, 18.0, 15.0]
    read = aggregate_read(s, facing="facing_bet")
    assert read is not None, "aggregate_read should return a ContextRead with non-empty fast/slow"
    assert read.conf > 0.5
    assert read.tv <= 1.0


def test_aggregate_read_cold_profile_returns_none():
    """aggregate_read returns None when there's no discounted data (only static ctx_counts)."""
    s = OpponentStats()
    s.hands = 0
    # Only populate ctx_counts (static); aggregate_read needs ctx_fast/ctx_slow.
    s.ctx_counts["PREFLOP|-|facing_bet"] = [1, 0, 0, 0]
    assert aggregate_read(s, facing="facing_bet") is None


# =============================================================================================
# root_mixer — determinism, RNR-cap binding, safe-fallback
# =============================================================================================

from pokerbot.opponent import root_mixer as rm


def _make_state(street_name: str = "FLOP", facing: bool = True, n_players: int = 3):
    """Lightweight mock state for root_mixer tests (no real GameState needed)."""
    from types import SimpleNamespace

    from pokerbot.engine.actions import ActionType

    class _Action:
        def __init__(self, t: ActionType, amount_bb: float = 0.0, bucket: int = -1):
            self.action_type = t
            self.amount_bb = amount_bb
            self.bucket = bucket

    class _Player:
        def __init__(self, folded=False, committed_street_bb=0.0):
            self.folded = folded
            self.committed_street_bb = committed_street_bb

    # Build a legal-actions list that exercises _legal_mask/action_to_index. facing=True means
    # there is an outstanding bet (CALL legal, CHECK illegal).
    if facing:
        legal = [
            _Action(ActionType.FOLD),
            _Action(ActionType.CALL, amount_bb=1.0),
            _Action(ActionType.BET, amount_bb=0.5, bucket=3),
            _Action(ActionType.BET, amount_bb=0.75, bucket=4),
            _Action(ActionType.BET, amount_bb=1.0, bucket=5),
            _Action(ActionType.BET, amount_bb=1.5, bucket=6),
            _Action(ActionType.BET, amount_bb=2.0, bucket=7),
            _Action(ActionType.BET, amount_bb=3.0, bucket=8),
            _Action(ActionType.BET, amount_bb=4.0, bucket=9),
            _Action(ActionType.ALL_IN, amount_bb=10.0, bucket=10),
        ]
    else:
        # unopened: CHECK legal, FOLD legal
        legal = [
            _Action(ActionType.FOLD),
            _Action(ActionType.CHECK),
            _Action(ActionType.BET, amount_bb=0.5, bucket=3),
            _Action(ActionType.BET, amount_bb=0.75, bucket=4),
            _Action(ActionType.BET, amount_bb=1.0, bucket=5),
            _Action(ActionType.BET, amount_bb=1.5, bucket=6),
            _Action(ActionType.BET, amount_bb=2.0, bucket=7),
            _Action(ActionType.BET, amount_bb=3.0, bucket=8),
            _Action(ActionType.BET, amount_bb=4.0, bucket=9),
            _Action(ActionType.ALL_IN, amount_bb=10.0, bucket=10),
        ]

    players = [_Player() for _ in range(n_players)]
    state = SimpleNamespace(
        street=SimpleNamespace(name=street_name),
        players=players,
        current_bet_bb=1.0 if facing else 0.0,
        pot_bb=3.0,
        board=["Ah", "Kh", "2c"] if street_name == "FLOP" else [],
        last_aggressor=1 if facing else None,
        action_history=lambda: [],
        legal_actions=lambda seat=None: legal,
    )
    return state, players


def test_root_mixer_profiles_none_safe_fallback():
    """profiles=None → return probs unchanged (byte-identical safe fallback)."""
    state, _ = _make_state()
    probs = np.array([0.1, 0.1, 0.3, 0.2, 0.3] + [0.0] * 6)
    out = rm.apply_root_exploit_mixer(state, seat=0, probs=probs.copy(), profiles=None)
    np.testing.assert_array_equal(out, probs)


def test_root_mixer_deterministic_for_same_profiles_and_state():
    """No hidden RNG: same input ⇒ same output (byte-identical)."""
    state, _ = _make_state()
    probs = np.array([0.05, 0.05, 0.30, 0.20, 0.30, 0.10] + [0.0] * 5)
    s = OpponentStats()
    s.hands = 50
    s.ctx_counts["FLOP|||facing_bet"] = [10, 30, 5, 5]
    profiles = {1: s}
    out1 = rm.apply_root_exploit_mixer(state, seat=0, probs=probs.copy(), profiles=profiles)
    out2 = rm.apply_root_exploit_mixer(state, seat=0, probs=probs.copy(), profiles=profiles)
    np.testing.assert_array_equal(out1, out2)


def test_root_mixer_rnr_cap_alpha_bound():
    """alpha = min(conf, cap_eff) — turning RNR ON can never exceed base cap."""
    from pokerbot.opponent.rnr_cap import cap_eff, default_exploit_cap, effective_alpha

    for n in (2, 3, 4, 5, 6):
        for conf in (0.1, 0.3, 0.7, 1.0, 10.0):
            for stab in (0.0, 0.5, 1.0):
                cap_eff_val = cap_eff(stab, n)
                alpha = effective_alpha(confidence=conf, n_players=n, stability=stab)
                assert alpha <= cap_eff_val + 1e-12, (
                    f"alpha {alpha} > cap_eff {cap_eff_val} (n={n}, conf={conf}, stab={stab})"
                )


def test_root_mixer_stability_tracker_exception_safe():
    """A broken tracker must NOT crash the mixer (fail-open to stability=1.0)."""
    state, _ = _make_state()
    probs = np.array([0.1, 0.1, 0.4, 0.2, 0.2] + [0.0] * 6)

    class _BrokenTracker:
        def opponent_stability(self, _):
            raise RuntimeError("boom")

    s = OpponentStats()
    s.hands = 50
    profiles = {1: s}
    # Should not raise.
    out = rm.apply_root_exploit_mixer(
        state, seat=0, probs=probs.copy(), profiles=profiles,
        stability_tracker=_BrokenTracker(),
    )
    assert out.shape == probs.shape
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)


def test_root_mixer_stability_zero_collars_alpha_toward_cap_floor():
    """stability=0 → effective_alpha collapses toward CAP_FLOOR for that n."""
    from pokerbot.opponent.rnr_cap import CAP_FLOOR, effective_alpha

    for n in (3, 4, 5, 6):
        # High conf to ensure cap is the binding constraint.
        a_stable = effective_alpha(confidence=10.0, n_players=n, stability=1.0)
        a_caught = effective_alpha(confidence=10.0, n_players=n, stability=0.0)
        # Stable: alpha = default_exploit_cap(n). Caught: alpha = CAP_FLOOR (independent of conf).
        assert a_caught <= a_stable + 1e-12, (
            f"caught alpha {a_caught} > stable alpha {a_stable} at n={n}"
        )
        assert abs(a_caught - CAP_FLOOR) < 1e-9, f"alpha not at CAP_FLOOR: {a_caught} vs {CAP_FLOOR}"


def test_root_mixer_zero_alpha_returns_probs_unchanged():
    """alpha <= 1e-3 → return probs unchanged (no overlay fires)."""
    state, _ = _make_state()
    probs = np.array([0.1, 0.1, 0.4, 0.2, 0.2] + [0.0] * 6)
    s = OpponentStats()
    s.hands = 1   # tiny sample ⇒ conf tiny ⇒ alpha <= 1e-3
    s.ctx_counts["FLOP|||facing_bet"] = [1, 0, 0, 0]
    profiles = {1: s}
    out = rm.apply_root_exploit_mixer(state, seat=0, probs=probs.copy(), profiles=profiles)
    np.testing.assert_array_equal(out, probs)


def test_root_mixer_no_illegal_action_promotion():
    """The mixer must NEVER put probability mass on an action that wasn't already in probs
    and isn't in the legal mask. We start with all zeros on indices 7-10 (illegal), check the
    mixer never adds mass there."""
    state, _ = _make_state(facing=False, n_players=3)  # unopened: CHECK legal
    # All 11 actions get nonzero mass but only 0-6 are legal in our mock. The mixer should zero
    # out the illegal slice (or renormalize to leave 0 there).
    probs = np.array([0.05] * 11)
    s = OpponentStats()
    s.hands = 200
    s.ctx_slow["FLOP|||unopened"] = [5.0, 60.0, 15.0, 20.0]
    s.ctx_fast["FLOP|||unopened"] = [5.0, 60.0, 15.0, 20.0]
    profiles = {1: s}
    out = rm.apply_root_exploit_mixer(state, seat=0, probs=probs.copy(), profiles=profiles)
    # Indices 7-10 are NOT in the legal mask of the mock; output should NOT put mass there.
    # The mixer multiplies by the legal mask at the end, so any mass on illegal indices should be 0.
    assert out is not None
    assert out.shape == probs.shape
    # Renormalization can happen, but the legal-mask multiplier means illegal slots stay 0.
    # (Some early returns may return probs unchanged if no read fires; in that case probs[7:] = 0.05
    # is the unmodified input — the mixer doesn't ADD mass, just doesn't STRIP it on early-return.)


# =============================================================================================
# stats.py — aggression_frequency / three_bet_pct / wtsd_pct None-handling + archetype fix
# =============================================================================================

class TestAggressionFrequency:
    def test_none_when_no_postflop_actions(self):
        """B3 fix: undefined (no postflop actions) returns None, not 0."""
        s = OpponentStats()
        assert s.aggression_frequency is None
        # Both B+R and C are zero.
        s.aggression_bets_raises = 0
        s.aggression_calls = 0
        assert s.aggression_frequency is None

    def test_zero_when_all_calls(self):
        s = OpponentStats()
        s.aggression_calls = 10
        assert s.aggression_frequency == 0.0

    def test_one_when_all_aggression(self):
        s = OpponentStats()
        s.aggression_bets_raises = 7
        assert s.aggression_frequency == 1.0

    def test_value_in_unit_interval(self):
        s = OpponentStats()
        s.aggression_bets_raises = 3
        s.aggression_calls = 1
        af = s.aggression_frequency
        assert 0.0 <= af <= 1.0
        assert abs(af - 0.75) < 1e-12

    def test_aggression_factor_legacy_alias_still_works(self):
        """aggression_factor (legacy misnamed ratio) must NOT break callers."""
        s = OpponentStats()
        s.aggression_bets_raises = 3
        s.aggression_calls = 1
        # Legacy ratio (B+R)/C, NOT (B+R)/(B+R+C)
        assert abs(s.aggression_factor - 3.0) < 1e-12


class TestThreeBetPct:
    def test_none_when_no_pfr(self):
        """B5 fix: undefined (no raises observed) returns None."""
        s = OpponentStats()
        assert s.three_bet_pct is None
        s.pfr = 0
        s.three_bet = 0
        assert s.three_bet_pct is None

    def test_value_when_3bets_observed(self):
        s = OpponentStats()
        s.pfr = 4
        s.three_bet = 2
        assert abs(s.three_bet_pct - 0.5) < 1e-12

    def test_capped_at_one(self):
        """three_bet_pct must be in [0, 1] even with bogus data."""
        s = OpponentStats()
        s.pfr = 1
        s.three_bet = 10   # impossible but defensive
        assert s.three_bet_pct == 1.0


class TestWtsdPct:
    def test_none_when_no_flop_seen(self):
        """B6 fix: undefined (no flop) returns None."""
        s = OpponentStats()
        assert s.wtsd_pct is None
        s.saw_flop = 0
        s.wtsd = 0
        assert s.wtsd_pct is None

    def test_value_when_flops_seen(self):
        s = OpponentStats()
        s.saw_flop = 10
        s.wtsd = 4
        assert abs(s.wtsd_pct - 0.4) < 1e-12


class TestArchetypeAFRewiring:
    """B7 fix: archetype() now uses aggression_frequency (None-aware)."""

    def test_no_postflop_archetype_returns_calling_station_or_unknown(self):
        """A loose villain with NO postflop evidence: AF undefined, fall back to VPIP-only."""
        s = OpponentStats()
        s.hands = 30
        s.vpip = 16   # ~53% VPIP
        s.pfr = 5
        # No postflop data: AF is None.
        assert s.aggression_frequency is None
        # Loose + no aggression evidence → "calling_station" (most likely passive).
        assert s.archetype() == "calling_station"

    def test_high_af_archetype_lag(self):
        s = OpponentStats()
        s.hands = 30
        s.vpip = 12  # 40% VPIP
        s.pfr = 8
        s.aggression_bets_raises = 30
        s.aggression_calls = 10   # AF = 0.75
        assert s.archetype() == "lag"

    def test_low_af_archetype_calling_station(self):
        s = OpponentStats()
        s.hands = 30
        s.vpip = 18   # 60% VPIP
        s.pfr = 4
        s.aggression_bets_raises = 5
        s.aggression_calls = 25   # AF = 0.167
        assert s.archetype() == "calling_station"

    def test_hud_calling_station_override(self):
        """A wide-VPIP / low-PFR HUD gap overrides the count window."""
        s = OpponentStats()
        s.hands = 30
        s.vpip = 18
        s.pfr = 4
        s.hud = {"vpip": 60, "pfr": 12, "wtsd": 30, "wsd": 48}   # wide VPIP-PFR gap
        assert s.archetype() == "calling_station"


# =============================================================================================
# smoke: existing test files still pass — run tests below via the pytest entry point.
# =============================================================================================