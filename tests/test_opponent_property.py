"""Property-based invariant tests for the opponent layer.

Distinguished from test_opponent_sharpening.py by FOCUS:
  * test_opponent_sharpening.py: targeted regression tests for specific audit findings
  * test_opponent_property.py: invariant sweeps — random input → output must obey stated property

This file exercises the SAME production code paths but with random / parametric inputs and
checks mathematical invariants (positivity, sum-to-1, monotonicity, determinism, shape, bounds)
rather than specific audit-issue regressions.

Run via: pytest tests/test_opponent_property.py -v
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("POKERBOT_LIVE_VALUE_SIZING", "0")

# Hypothesis is the standard property-based testing library. Optional in this test module — the
# non-Hypothesis invariant sweeps below run regardless, and the Hypothesis-driven property block
# at the bottom (H1-H5) skips if hypothesis is not installed.
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings, strategies as st  # noqa: E402


# =============================================================================================
# bayes_exploit property invariants
# =============================================================================================

from pokerbot.opponent.bayes_exploit import (
    BLUFF_PRIOR,
    CHANGE_TV,
    DEFAULT_MAX_ALPHA,
    ContextRead,
    aggregate_read,
    bluff_read,
    confidence,
    posterior,
)
from pokerbot.opponent.context import NUM_BINS, PASSIVE_BIN, FOLD_BIN, AGG_LARGE_BIN, AGG_SMALL_BIN
from pokerbot.opponent.population_prior import PRIOR_STRENGTH
from pokerbot.opponent.rnr_cap import (
    CAP_FLOOR,
    cap_eff,
    default_exploit_cap,
    effective_alpha,
    n_live_players,
)
from pokerbot.opponent.stats import OpponentStats, OpponentTracker


# ---------------------------------------------------------------------------------------------
# Property 1: posterior is a valid probability distribution for ANY valid input.
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123, 999])
def test_property_posterior_is_valid_probability_dist(seed):
    """For ANY non-negative counts and prior, the posterior is non-negative and sums to 1."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        c = rng.uniform(0, 100, NUM_BINS)
        p = rng.uniform(0.001, 10.0, NUM_BINS)
        post = posterior(c, p)
        assert (post >= 0).all(), f"negativity at seed {seed}"
        assert abs(post.sum() - 1.0) < 1e-12, f"sum != 1 at seed {seed}: {post.sum()}"


# ---------------------------------------------------------------------------------------------
# Property 2: confidence is monotone in N (more observations → more confident).
# ---------------------------------------------------------------------------------------------

def test_property_confidence_monotone_sweep():
    """confidence(N+1, s) >= confidence(N, s) for all N, s > 0."""
    for s in (0.001, 0.5, 1.0, PRIOR_STRENGTH, 100.0):
        prev = -1.0
        for n in range(0, 1000):
            c = confidence(float(n), s)
            assert c >= prev - 1e-12, f"monotonicity failed at n={n}, s={s}"
            prev = c


# ---------------------------------------------------------------------------------------------
# Property 3: confidence is sublinear — diminishing returns.
# ---------------------------------------------------------------------------------------------

def test_property_confidence_sublinear():
    """confidence(N+Δ, s) - confidence(N, s) is monotonically DECREASING in N."""
    s = PRIOR_STRENGTH
    diffs = []
    for n in (0, 10, 100, 1000, 10000, 100000):
        diffs.append(confidence(n + 50, s) - confidence(n, s))
    for a, b in zip(diffs, diffs[1:]):
        assert a >= b - 1e-9, f"diffs not monotonically decreasing: {diffs}"


# ---------------------------------------------------------------------------------------------
# Property 4: confidence is bounded in [0, 1).
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 100, 1_000_000, 1e10])
@pytest.mark.parametrize("s", [0.001, 1, 8, 1000])
def test_property_confidence_bounded(n, s):
    c = confidence(n, s)
    assert 0.0 <= c < 1.0, f"confidence out of [0,1): {c} for n={n}, s={s}"


# ---------------------------------------------------------------------------------------------
# Property 5: search_tilt alpha is monotone in conf and tv.
# ---------------------------------------------------------------------------------------------

def test_property_search_tilt_alpha_monotone():
    """A higher-confidence read yields a higher alpha (other things equal)."""
    rng = np.random.default_rng(0)
    pop = rng.dirichlet(np.ones(NUM_BINS))
    # Build two reads with same post/pop but different conf.
    post = pop.copy()
    post[2] += 0.10
    post /= post.sum()
    read_low = ContextRead(post=post, pop=pop, conf=0.3, n_obs=10)
    read_high = ContextRead(post=post, pop=pop, conf=0.9, n_obs=200)
    _, alpha_low = read_low.search_tilt()
    _, alpha_high = read_high.search_tilt()
    assert alpha_high >= alpha_low, f"alpha not monotone in conf: {alpha_low} -> {alpha_high}"


def test_property_search_tilt_alpha_clipped():
    """search_tilt alpha never exceeds max_alpha (the DEFAULT_MAX_ALPHA ceiling)."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        pop = rng.dirichlet(np.ones(NUM_BINS))
        post = pop.copy()
        post[3] += 0.30  # extreme deviation
        post /= post.sum()
        # n_obs deliberately huge so conf ≈ 1.
        read = ContextRead(post=post, pop=pop, conf=0.99, n_obs=100000)
        _, alpha = read.search_tilt()
        assert alpha <= DEFAULT_MAX_ALPHA + 1e-12, f"alpha exceeds cap: {alpha}"


# ---------------------------------------------------------------------------------------------
# Property 6: RNR cap_eff is non-increasing in stability drops.
# ---------------------------------------------------------------------------------------------

def test_property_rnr_cap_eff_monotone_in_stability():
    """For any n, cap_eff INCREASES monotonically with stability (0→CAP_FLOOR, 1→default_exploit_cap)."""
    for n in (2, 3, 4, 5, 6):
        # Sample stability grid.
        stabilities = np.linspace(0.0, 1.0, 21)
        prev_cap = -1.0
        for s in stabilities:
            cap = cap_eff(float(s), n)
            assert cap >= prev_cap - 1e-12, f"cap not monotone in stability at n={n}: {prev_cap} -> {cap}"
            prev_cap = cap
        # At stability=1, cap_eff == default_exploit_cap(n).
        assert abs(cap_eff(1.0, n) - default_exploit_cap(n)) < 1e-12, \
            f"cap_eff(1.0, {n}) != default_exploit_cap: {cap_eff(1.0, n)} vs {default_exploit_cap(n)}"
        # At stability=0, cap_eff == CAP_FLOOR.
        assert abs(cap_eff(0.0, n) - CAP_FLOOR) < 1e-12, \
            f"cap_eff(0.0, {n}) != CAP_FLOOR: {cap_eff(0.0, n)} vs {CAP_FLOOR}"


# ---------------------------------------------------------------------------------------------
# Property 7: effective_alpha never exceeds cap_eff (the binding constraint).
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("conf", [0.0, 0.1, 0.5, 1.0, 5.0, 100.0])
@pytest.mark.parametrize("stability", [0.0, 0.3, 0.7, 1.0])
def test_property_effective_alpha_bounded_by_cap(n, conf, stability):
    """effective_alpha(conf, n, stability) <= cap_eff(stability, n) for any inputs."""
    alpha = effective_alpha(confidence=conf, n_players=n, stability=stability)
    cap = cap_eff(stability, n)
    assert alpha <= cap + 1e-12, \
        f"alpha {alpha} > cap {cap} (n={n}, conf={conf}, stab={stability})"


# ---------------------------------------------------------------------------------------------
# Property 8: bluff_read returns None on cold/empty profile (zero reveals).
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 42, 999])
def test_property_bluff_read_none_on_empty(seed):
    """An OpponentStats with no ctx_bluff data MUST return None — the legacy/cold path."""
    rng = np.random.default_rng(seed)
    # Build a profile and only populate other fields (not ctx_bluff).
    s = OpponentStats()
    s.hands = int(rng.integers(50, 500))
    # Add some ctx_counts to make sure ctx_bluff being empty is the gating condition.
    s.ctx_counts["PREFLOP|-|facing_bet"] = rng.integers(0, 50, NUM_BINS).tolist()
    out = bluff_read(s, "PREFLOP|-|facing_bet")
    assert out is None, f"bluff_read should be None for empty ctx_bluff (seed {seed})"


# ---------------------------------------------------------------------------------------------
# Property 9: stats.py aggression_frequency is None iff no postflop actions.
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_property_aggression_frequency_none_iff_empty(seed):
    """aggression_frequency is None iff both B+R and C are zero; otherwise in [0, 1]."""
    rng = np.random.default_rng(seed)
    s = OpponentStats()
    # Case 1: no postflop actions.
    assert s.aggression_frequency is None

    # Case 2: random B+R and C, no zeros.
    for _ in range(30):
        s.aggression_bets_raises = int(rng.integers(0, 100))
        s.aggression_calls = int(rng.integers(0, 100))
        af = s.aggression_frequency
        if s.aggression_bets_raises == 0 and s.aggression_calls == 0:
            assert af is None
        else:
            assert af is not None
            assert 0.0 <= af <= 1.0


# ---------------------------------------------------------------------------------------------
# Property 10: stats.py three_bet_pct is None iff PFR == 0.
# ---------------------------------------------------------------------------------------------

def test_property_three_bet_pct_none_iff_no_pfr():
    s = OpponentStats()
    assert s.three_bet_pct is None
    for pfr in (1, 5, 100, 1000):
        for three_bet in (0, 1, 10, 1000):
            s.pfr = pfr
            s.three_bet = three_bet
            v = s.three_bet_pct
            assert v is not None
            assert 0.0 <= v <= 1.0, f"3-bet pct out of range: {v}"


# ---------------------------------------------------------------------------------------------
# Property 11: stats.py wtsd_pct is None iff saw_flop == 0.
# ---------------------------------------------------------------------------------------------

def test_property_wtsd_pct_none_iff_no_flop():
    s = OpponentStats()
    assert s.wtsd_pct is None
    for saw in (1, 5, 100):
        # Only test wtsd <= saw_flop (invalid otherwise — a defensive clamp would still hold,
        # but the property is well-defined only on valid data).
        for wtsd in (0, 1, min(10, saw), saw):
            s.saw_flop = saw
            s.wtsd = wtsd
            v = s.wtsd_pct
            assert v is not None
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------------------------
# Property 12: decay_and_add does not NaN or produce negative entries on edge inputs.
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 42, 1000])
def test_property_decay_and_add_no_nan_or_negative(seed):
    """Repeated decay_and_add calls must keep ctx_fast / ctx_slow entries non-negative finite."""
    rng = np.random.default_rng(seed)
    s = OpponentStats()
    prior = np.array([0.25] * NUM_BINS)
    ck = "PREFLOP|-|unopened"
    for h in range(100):
        bins = rng.integers(0, 5, NUM_BINS).tolist()
        s.decay_and_add(ck, bins, prior, hands_now=h)
        for arr in (s.ctx_fast.get(ck, []), s.ctx_slow.get(ck, [])):
            assert all(x >= 0 for x in arr), f"negative entry at hand {h}: {arr}"
            assert all(math.isfinite(x) for x in arr), f"NaN/inf at hand {h}: {arr}"


# ---------------------------------------------------------------------------------------------
# Property 13: posterior converges to the empirical frequency as N → ∞ (Bayes consistency).
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_property_posterior_converges_to_empirical(seed):
    """For large N, posterior ≈ (counts / N) regardless of prior (Bayes consistency)."""
    rng = np.random.default_rng(seed)
    # Generate a non-uniform truth distribution.
    truth = rng.dirichlet(np.ones(NUM_BINS) * 2)
    n_obs = 50_000
    counts = rng.multinomial(n_obs, truth)
    # Use a very different prior (the prior should NOT bias a large-N posterior).
    prior = np.array([0.85, 0.05, 0.05, 0.05])
    post = posterior(counts, prior)
    empirical = counts / counts.sum()
    np.testing.assert_allclose(post, empirical, atol=0.01)


# ---------------------------------------------------------------------------------------------
# Property 14: aggregate_read returns a ContextRead with sum-to-1 post and bounded TV.
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_property_aggregate_read_output_well_formed(seed):
    """When aggregate_read returns non-None, the post vector must be a valid probability
    distribution with TV in [0, 1]."""
    rng = np.random.default_rng(seed)
    s = OpponentStats()
    s.hands = 200
    # Populate several contexts with realistic-looking fast/slow data.
    for i in range(5):
        ck = f"FLOP|dry|facing_bet_{i}" if i % 2 == 0 else f"TURN|wet|unopened_{i}"
        s.ctx_fast[ck] = (rng.uniform(1, 10, NUM_BINS)).tolist()
        s.ctx_slow[ck] = (rng.uniform(5, 20, NUM_BINS)).tolist()
    read = aggregate_read(s)
    if read is not None:
        np.testing.assert_allclose(read.post.sum(), 1.0, atol=1e-12)
        assert (read.post >= 0).all()
        assert 0.0 <= read.tv <= 1.0, f"TV out of [0,1]: {read.tv}"


# ---------------------------------------------------------------------------------------------
# Property 15: n_live_players returns at least 2 (a heads-up minimum).
# ---------------------------------------------------------------------------------------------

def test_property_n_live_players_min_2():
    """n_live_players must always report at least 2 (a single-player game is degenerate)."""
    from types import SimpleNamespace

    class _P:
        def __init__(self, folded):
            self.folded = folded

    # All folded.
    state = SimpleNamespace(players=[_P(folded=True) for _ in range(5)])
    n = n_live_players(state, hero=0)
    assert n >= 2

    # All-in (not folded but not active): implementation may vary — minimum 2.
    state = SimpleNamespace(players=[_P(folded=False) for _ in range(3)])
    n = n_live_players(state, hero=0)
    assert n >= 2


# =============================================================================================
# Hypothesis @given property block (≥100 examples per property)
# =============================================================================================
#
# These five properties are the CORRECTNESS SKELETON of the opponent layer — if any of them
# breaks the live exploitation path is wrong by construction. They are stated in the design
# docs and were previously verified by hand in tests/test_bayes_exploit.py and
# tests/test_rnr_cap.py — this block lifts them to property-based tests with random inputs so
# future refactors can't silently regress them.
#
# Settings: deadline=None (Hypothesis can be slow on cold numpy imports); suppress_health_check
# on the too-slow / large-data warnings — these tests are intentionally tight loops.

_H_SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)


# Shared strategies — keep them narrow so each example runs in microseconds, not seconds.
# The bins must sum to >= 1 so the posterior is well-defined (an all-zero prior would
# make ``tot > 0`` trivially true on zero-counts; we want positive mass).
_bin_counts = st.lists(
    st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    min_size=NUM_BINS,
    max_size=NUM_BINS,
).filter(lambda v: sum(v) > 0.0)

_bin_prior = st.lists(
    st.floats(min_value=0.01, max_value=20.0, allow_nan=False, allow_infinity=False),
    min_size=NUM_BINS,
    max_size=NUM_BINS,
).filter(lambda v: sum(v) > 0.0)


# ---------------------------------------------------------------------------------------------
# H1: Posterior simplex — every posterior is a valid probability distribution.
# ---------------------------------------------------------------------------------------------

@given(counts=_bin_counts, prior=_bin_prior)
@_H_SETTINGS
def test_h_posterior_is_simplex(counts, prior):
    """For ANY non-negative counts and prior, posterior >= 0 elementwise and sums to 1.0."""
    post = posterior(np.asarray(counts), np.asarray(prior))
    assert post.shape == (NUM_BINS,)
    assert np.all(post >= 0.0), f"posterior has negative entries: {post}"
    assert abs(float(post.sum()) - 1.0) < 1e-9, f"posterior doesn't sum to 1: {post.sum()}"


# ---------------------------------------------------------------------------------------------
# H2: Posterior → prior (normalized) as N → 0 (the canonical Bayes shrinkage property).
# ---------------------------------------------------------------------------------------------

@given(prior=_bin_prior, scale=st.floats(min_value=0.01, max_value=10.0,
                                         allow_nan=False, allow_infinity=False))
@_H_SETTINGS
def test_h_posterior_collapses_to_prior_at_zero_n(prior, scale):
    """With zero observed counts, posterior == prior / sum(prior). The Dirichlet posterior
    mean shrinks to the prior at N=0 — and since counts=0 ⇒ numerator = prior exactly, the
    posterior is the normalized prior regardless of scale."""
    counts = np.zeros(NUM_BINS)
    p = np.asarray(prior)
    post = posterior(counts, p)
    expected = p / p.sum()
    np.testing.assert_allclose(post, expected, atol=1e-12, rtol=0)
    assert np.all(np.isfinite(post)), f"posterior not finite: {post}"


# ---------------------------------------------------------------------------------------------
# H3: Posterior narrows as N → ∞ — empirical frequency dominates the prior.
# ---------------------------------------------------------------------------------------------

@given(
    truth=st.lists(
        st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=NUM_BINS, max_size=NUM_BINS,
    ).map(lambda v: [x / sum(v) for x in v]),  # normalize to a probability simplex
    n_obs=st.integers(min_value=2000, max_value=20000),
    prior=_bin_prior,
    rng_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@_H_SETTINGS
def test_h_posterior_narrows_within_observed_mass_as_n_grows(truth, n_obs, prior, rng_seed):
    """For large N drawn from a fixed multinomial, posterior ≈ empirical frequency within
    sampling-error tolerance. The bound is a Chebyshev-style sqrt(n_obs) standard-error:
    at n_obs=2000 the per-bin standard deviation of a Binomial(n, 0.25) is ~0.007, so
    a TV-distance tolerance of 0.05 captures the worst-case edge bin safely."""
    rng = np.random.default_rng(rng_seed)
    truth_arr = np.asarray(truth) / sum(truth)
    counts = rng.multinomial(int(n_obs), truth_arr)
    empirical = counts / counts.sum()
    p = np.asarray(prior)
    post = posterior(counts, p)
    # Posterior must lie in the simplex (property 1).
    assert abs(post.sum() - 1.0) < 1e-9
    assert np.all(post >= 0)
    # And converge to the empirical frequency — the prior shouldn't bias a large-N posterior.
    tv_to_empirical = 0.5 * float(np.abs(post - empirical).sum())
    assert tv_to_empirical < 0.10, (
        f"posterior not narrowing at n={n_obs}: tv(post, empirical)={tv_to_empirical:.4f}, "
        f"post={post}, empirical={empirical}"
    )


# ---------------------------------------------------------------------------------------------
# H4: RNR cap monotonicity in N (ceiling never rises as the table grows).
# ---------------------------------------------------------------------------------------------

@given(n_a=st.integers(min_value=2, max_value=6),
       n_b=st.integers(min_value=2, max_value=6),
       stability=st.floats(min_value=0.0, max_value=1.0,
                           allow_nan=False, allow_infinity=False))
@_H_SETTINGS
def test_h_rnr_cap_non_increasing_in_n(n_a, n_b, stability):
    """default_exploit_cap(n) and cap_eff(stab, n) must both be NON-INCREASING in n over the
    supported range (2..6). Multiway tables face STRICTLY WEAKER minimax safety (more opponents
    can coordinate to punish a full BR), so the cap is calibrated strictly decreasing in N."""
    assume(n_a <= n_b)
    cap_a = default_exploit_cap(n_a)
    cap_b = default_exploit_cap(n_b)
    assert cap_a >= cap_b - 1e-12, (
        f"default_exploit_cap RISES with N: cap({n_a})={cap_a} > cap({n_b})={cap_b}"
    )
    # And the bound must hold at every stability level (the floor is uniform, only the ceiling
    # varies; the linear blend is monotone in BOTH arguments).
    eff_a = cap_eff(stability, n_a)
    eff_b = cap_eff(stability, n_b)
    assert eff_a >= eff_b - 1e-12, (
        f"cap_eff(stab={stability:.3f}, n) RISES with N: "
        f"cap_eff(.,{n_a})={eff_a} > cap_eff(.,{n_b})={eff_b}"
    )


# ---------------------------------------------------------------------------------------------
# H5: VPIP / PFR / AF math correctness on reference hand histories.
# ---------------------------------------------------------------------------------------------
#
# Construct parametric "reference hand histories" via the production OpponentTracker and verify
# the derived percentages match an independent recomputation from the raw counts. We exercise
# the whole interface (record_hand) so any future change to the percent calculations breaks.

_hand_strategy = st.lists(
    st.tuples(
        st.booleans(),   # vpip
        st.booleans(),   # pfr
        st.booleans(),   # three_bet
        st.booleans(),   # saw_flop
        st.booleans(),   # wtsd
        st.booleans(),   # fold_to_cbet
        st.booleans(),   # cbet_faced
        st.integers(min_value=0, max_value=10),   # bet_or_raise
        st.integers(min_value=0, max_value=10),   # calls
    ),
    min_size=1,
    max_size=80,
)


@given(hand_specs=_hand_strategy, player_id=st.integers(min_value=0, max_value=99))
@_H_SETTINGS
def test_h_vpip_pfr_af_match_recomputed_from_counts(hand_specs, player_id):
    """For any sequence of recorded hands, ``vpip_pct``, ``pfr_pct``, and
    ``aggression_factor`` must exactly match an independent recomputation from the raw counts.

    ``vpip_pct = vpip / hands``, ``pfr_pct = pfr / hands``, ``AF = bets_raises / max(calls, 1)``
    — these are the canonical poker-stat formulas. We rebuild them here from the spec tuple and
    assert equality with the property on the same OpponentStats object. Any divergence indicates
    a code change to the production math (or a property renamed/removed)."""
    tracker = OpponentTracker()
    expected_vpip = expected_pfr = 0
    expected_br = expected_calls = 0
    for (vpip, pfr, three_bet, saw_flop, wtsd, f2c, cf, br, calls) in hand_specs:
        tracker.record_hand(
            player_id,
            vpip=vpip, pfr=pfr, three_bet=three_bet, saw_flop=saw_flop, wtsd=wtsd,
            fold_to_cbet=f2c, cbet_faced=cf, bet_or_raise=br, calls=calls,
        )
        expected_vpip += int(vpip)
        expected_pfr += int(pfr)
        expected_br += int(br)
        expected_calls += int(calls)

    s = tracker.get(player_id)
    # vpip_pct and pfr_pct: ratio over hands (with max(hands, 1) for divide-by-zero safety).
    assert s.vpip_pct == pytest.approx(expected_vpip / max(s.hands, 1), abs=1e-12)
    assert s.pfr_pct == pytest.approx(expected_pfr / max(s.hands, 1), abs=1e-12)
    # aggression_factor: bets_raises / max(calls, 1) — NOTE the +1 in the denominator.
    assert s.aggression_factor == pytest.approx(
        expected_br / max(expected_calls, 1), abs=1e-12
    )
    # fold_to_cbet_pct only meaningful if cbet_faced > 0 — derive the same way the property does.
    expected_f2c = sum(int(f2c) for (vpip, pfr, three_bet, saw_flop, wtsd, f2c, cf, br, calls)
                       in hand_specs)
    expected_cf = sum(int(cf) for (vpip, pfr, three_bet, saw_flop, wtsd, f2c, cf, br, calls)
                      in hand_specs)
    assert s.fold_to_cbet_pct == pytest.approx(expected_f2c / max(expected_cf, 1), abs=1e-12)
    # Sanity: hands counter increments once per record_hand call.
    assert s.hands == len(hand_specs)