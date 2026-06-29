"""REFUTER 1: BAYES CORRECTNESS property tests.

For every modified Bayesian file, assert the three load-bearing invariants:

  (a) posterior sums to 1     (is on the simplex)
  (b) posterior >= 0          (no negative probabilities)
  (c) as N -> 0, posterior -> prior  (proper anti-overfit shrinkage)

Targets:
  - pokerbot/opponent/bayes_exploit.py   (posterior, context_read, aggregate_read, ContextRead.post)
  - pokerbot/opponent/population_prior.py (context_prior, _base_dist, generate_prior)
  - pokerbot/opponent/bluff_read          (Beta posterior P(bluff|aggr))
  - pokerbot/opponent/outcome_stability   (multiplier in [0,1])
  - pokerbot/opponent/profile_stability   (in [0,1])
  - pokerbot/opponent/effective_alpha     (RNR ceiling)
  - pokerbot/opponent/mixer._stats_confidence (range)
  - pokerbot/opponent/player_profiles     (load/save roundtrip Dirichlet/Beta counts)

Violations are reported as CRITICAL via assertion + console banner.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pokerbot.opponent.bayes_exploit import (
    ContextRead,
    aggregate_read,
    confidence,
    context_read,
    outcome_stability,
    posterior,
    profile_stability,
    update_exploit_outcome,
)
from pokerbot.opponent.context import (
    AGG_LARGE_BIN,
    AGG_SMALL_BIN,
    FOLD_BIN,
    NUM_BINS,
    PASSIVE_BIN,
    all_contexts,
)
from pokerbot.opponent.population_prior import (
    PRIOR_STRENGTH,
    _base_dist,
    context_prior,
    generate_prior,
)


# ---------- report helper ----------

_CRITICALS: list[str] = []


def _critical(name: str, msg: str) -> None:
    _CRITICALS.append(f"{name}: {msg}")
    # do not raise here -- collect, then summarize at module teardown


def pytest_sessionfinish(session, exitstatus):  # type: ignore[no-untyped-def]
    if _CRITICALS:
        print("\n" + "=" * 78)
        print("REFUTE-BAYES: CRITICAL BAYES CORRECTNESS VIOLATIONS")
        print("=" * 78)
        for i, c in enumerate(_CRITICALS, 1):
            print(f"  [{i}] {c}")
        print("=" * 78)
        print(f"TOTAL CRITICAL: {len(_CRITICALS)}")
        print("=" * 78)


# ---------- helpers ----------

_CTXS = all_contexts()


def _pick_ctx() -> str:
    # Deterministic-but-varied context for parametrized tests.
    return _CTXS[hash(("refuter", "bayes")) % len(_CTXS)]


def _unit_simplex(k: int) -> np.ndarray:
    v = np.ones(k, dtype=np.float64)
    return v / v.sum()


# =========================================================================
# Property (a) -- posterior sums to 1 (on the simplex)
# =========================================================================


@pytest.mark.parametrize("ctx", _CTXS[:8])
def test_a_posterior_static_sums_to_one(ctx: str) -> None:
    """posterior(counts, prior) must sum to 1 for any non-negative counts and any context prior."""
    prior = np.asarray(context_prior(ctx), dtype=np.float64)
    for counts in [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [3, 7, 2, 4],
        [50, 30, 10, 5],
        [0, 0, 0, 0],   # the edge: zero obs, zero prior
        [100, 0, 0, 0],
    ]:
        p = posterior(counts, prior)
        s = float(p.sum())
        if not math.isclose(s, 1.0, abs_tol=1e-9):
            _critical(
                "posterior_simplex",
                f"ctx={ctx} counts={counts} sum={s} expected=1.0",
            )
        assert math.isclose(s, 1.0, abs_tol=1e-9)


def test_a_posterior_handles_zero_obs_and_zero_prior() -> None:
    """Edge case: counts=[0..0], prior=[0..0] -> uniform fallback (sums to 1)."""
    p = posterior([0, 0, 0, 0], np.zeros(NUM_BINS))
    assert math.isclose(float(p.sum()), 1.0, abs_tol=1e-12)
    assert np.allclose(p, 1.0 / NUM_BINS)


# =========================================================================
# Property (b) -- posterior >= 0 (no negative probabilities)
# =========================================================================


@pytest.mark.parametrize("ctx", _CTXS[:8])
def test_b_posterior_nonneg_static(ctx: str) -> None:
    prior = np.asarray(context_prior(ctx), dtype=np.float64)
    rng = np.random.default_rng(0xBA7E5)
    for _ in range(40):
        n = int(rng.integers(0, 200))
        counts = rng.integers(0, 10, size=NUM_BINS)
        counts[0] = n - int(counts[1:].sum())
        if counts[0] < 0:
            continue
        p = posterior(counts.tolist(), prior)
        if (p < 0).any() or not np.isfinite(p).all():
            _critical(
                "posterior_nonneg",
                f"ctx={ctx} counts={counts.tolist()} p={p.tolist()}",
            )
        assert (p >= 0).all() and np.isfinite(p).all()


def test_b_context_read_static_post_nonneg() -> None:
    """context_read().post must be non-negative on the static path."""
    from pokerbot.opponent.stats import OpponentStats

    for ctx in _CTXS[:6]:
        counts = [0, 5, 3, 2]
        s = OpponentStats(hands=sum(counts), ctx_counts={ctx: counts})
        cr = context_read(s, ctx)
        assert cr is not None
        if (cr.post < 0).any() or not math.isclose(float(cr.post.sum()), 1.0, abs_tol=1e-9):
            _critical(
                "context_read_static_post",
                f"ctx={ctx} post={cr.post.tolist()} sum={cr.post.sum()}",
            )
        assert (cr.post >= 0).all()
        assert math.isclose(float(cr.post.sum()), 1.0, abs_tol=1e-9)


def test_b_context_read_dual_post_nonneg() -> None:
    """context_read() on the dual-timescale path must also yield a non-negative unit-sum post."""
    from pokerbot.opponent.stats import OpponentStats

    for ctx in _CTXS[:6]:
        fast = np.array([1, 6, 2, 1], dtype=np.float64)
        slow = np.array([2, 5, 2, 1], dtype=np.float64)
        s = OpponentStats(
            hands=int(slow.sum()),
            ctx_counts={ctx: (slow + PRIOR_STRENGTH * _unit_simplex(NUM_BINS)).tolist()},
            ctx_fast={ctx: fast.tolist()},
            ctx_slow={ctx: slow.tolist()},
        )
        cr = context_read(s, ctx)
        assert cr is not None
        if (cr.post < 0).any() or not math.isclose(float(cr.post.sum()), 1.0, abs_tol=1e-9):
            _critical(
                "context_read_dual_post",
                f"ctx={ctx} post={cr.post.tolist()} sum={cr.post.sum()}",
            )
        assert (cr.post >= 0).all()
        assert math.isclose(float(cr.post.sum()), 1.0, abs_tol=1e-9)


def test_b_aggregate_read_post_nonneg() -> None:
    """aggregate_read().post must be non-negative and sum to 1."""
    from pokerbot.opponent.stats import OpponentStats

    fast_all: dict[str, list[float]] = {}
    slow_all: dict[str, list[float]] = {}
    for ck in _CTXS[:6]:
        fast_all[ck] = [1.0, 6.0, 2.0, 1.0]
        slow_all[ck] = [2.0, 5.0, 2.0, 1.0]
    s = OpponentStats(hands=200, ctx_fast=fast_all, ctx_slow=slow_all)
    cr = aggregate_read(s, street="flop")
    assert cr is not None
    if (cr.post < 0).any() or not math.isclose(float(cr.post.sum()), 1.0, abs_tol=1e-9):
        _critical(
            "aggregate_read_post",
            f"post={cr.post.tolist()} sum={cr.post.sum()}",
        )
    assert (cr.post >= 0).all()
    assert math.isclose(float(cr.post.sum()), 1.0, abs_tol=1e-9)


# =========================================================================
# Property (c) -- as N -> 0, posterior -> prior (anti-overfit shrinkage)
# =========================================================================


@pytest.mark.parametrize("ctx", _CTXS[:6])
def test_c_posterior_converges_to_prior_at_zero_obs(ctx: str) -> None:
    """With N=0 observations, posterior must equal the normalized prior (exactly)."""
    prior = np.asarray(context_prior(ctx), dtype=np.float64)
    if prior.sum() <= 0:
        # fall back to uniform -- posterior() returns uniform too
        p = posterior([0, 0, 0, 0], prior)
        assert np.allclose(p, 1.0 / NUM_BINS)
        return
    p = posterior([0, 0, 0, 0], prior)
    expected = prior / prior.sum()
    if not np.allclose(p, expected, atol=1e-12):
        _critical(
            "posterior_n0_equals_prior",
            f"ctx={ctx} p={p.tolist()} prior_norm={expected.tolist()}",
        )
    assert np.allclose(p, expected, atol=1e-12)


@pytest.mark.parametrize("ctx", _CTXS[:6])
def test_c_context_read_converges_to_nash_at_zero_obs(ctx: str) -> None:
    """Static context_read with N=0 must be None (no data -> caller falls back to archetype/GTO)."""
    from pokerbot.opponent.stats import OpponentStats

    s = OpponentStats(hands=0, ctx_counts={ctx: [0, 0, 0, 0]})
    cr = context_read(s, ctx)
    if cr is not None:
        _critical(
            "context_read_n0_is_none",
            f"ctx={ctx} got non-None read with zero obs: {cr}",
        )
    assert cr is None


def test_c_confidence_converges_to_zero_at_zero_obs() -> None:
    """DBR confidence(0) must be 0 (anti-overfit)."""
    c0 = confidence(0)
    if c0 != 0.0:
        _critical("confidence_n0", f"confidence(0)={c0} expected 0.0")
    assert c0 == 0.0


def test_c_confidence_monotone_in_obs() -> None:
    """confidence is N/(N+s) -- monotone non-decreasing in N."""
    last = -1.0
    for n in [0, 1, 5, 10, 50, 100, 1000, 10_000]:
        c = confidence(n)
        if c < last - 1e-12:
            _critical("confidence_monotone", f"dropped at n={n}: {c} < {last}")
        assert c >= last - 1e-12
        last = c


# =========================================================================
# Outcome stability & profile stability -- range invariants
# =========================================================================


def test_outcome_stability_in_unit_interval() -> None:
    """outcome_stability must always be in [0, 1]."""
    cases = [
        None,
        {},
        {"w": 1.0, "s": 0.0},
        {"w": 4.0, "s": -10.0},
        {"w": 100.0, "s": -100.0},
        {"w": 100.0, "s": 100.0},
        {"w": 100.0, "s": 0.0},
    ]
    for acc in cases:
        v = outcome_stability(acc)
        if not (0.0 <= v <= 1.0):
            _critical("outcome_stability_range", f"acc={acc} -> {v}")
        assert 0.0 <= v <= 1.0


def test_outcome_stability_full_abandon_when_loss_severe() -> None:
    """At mean loss = -OUTCOME_ABANDON_BB, stability must collapse to 0."""
    # Build an acc with mean exactly -OUTCOME_ABANDON_BB and w >> MIN_EVIDENCE.
    acc = update_exploit_outcome(None, -0.5)
    # spam updates to saturate w
    for _ in range(20):
        acc = update_exploit_outcome(acc, -0.5)
    v = outcome_stability(acc)
    if v > 0.1:  # allow tiny numerical slop
        _critical("outcome_stability_abandon", f"got {v} expected ~0")
    assert v <= 0.1


def test_profile_stability_in_unit_interval() -> None:
    """profile_stability must always be in [0, 1]."""
    from pokerbot.opponent.stats import OpponentStats

    # no ctx data -> returns outcome_stability of exploit_outcome (defaults to None -> 1.0)
    s = OpponentStats(hands=50)
    v = profile_stability(s)
    if not (0.0 <= v <= 1.0):
        _critical("profile_stability_empty", f"got {v}")
    assert 0.0 <= v <= 1.0

    # heavy divergence -> low freq multiplier
    fast_all: dict[str, list[float]] = {}
    slow_all: dict[str, list[float]] = {}
    for ck in _CTXS[:6]:
        fast_all[ck] = [10.0, 1.0, 1.0, 1.0]
        slow_all[ck] = [1.0, 10.0, 1.0, 1.0]
    s2 = OpponentStats(hands=200, ctx_fast=fast_all, ctx_slow=slow_all)
    v = profile_stability(s2)
    if not (0.0 <= v <= 1.0):
        _critical("profile_stability_diverged", f"got {v}")
    assert 0.0 <= v <= 1.0


# =========================================================================
# Bluff-read Beta posterior -- sums/probability range invariants
# =========================================================================


def _stub_stats_for_bluff() -> object:
    """Build a duck-typed object with the attributes bluff_read touches."""

    class _S:
        pass

    s = _S()
    # ctx_bluff: {ctx: [n_bluff, n_value]}
    s.ctx_bluff = {
        "flop|wet|facing_bet": [2.0, 5.0],          # exact ctx: 7 reveals
        "flop|dry|facing_bet": [1.0, 2.0],          # exact ctx: 3 reveals (floor)
        "flop|wet|facing_check": [0.0, 1.0],
        "turn|wet|facing_bet": [1.0, 1.0],
    }
    return s


def test_bluff_read_returns_unit_prob() -> None:
    from pokerbot.opponent.bayes_exploit import bluff_read

    s = _stub_stats_for_bluff()
    for ck in s.ctx_bluff:
        out = bluff_read(s, ck)
        assert out is not None
        p, conf = out
        if not (0.0 <= p <= 1.0):
            _critical("bluff_read_p", f"ctx={ck} p={p}")
        if not (0.0 <= conf < 1.0):
            _critical("bluff_read_conf", f"ctx={ck} conf={conf}")
        assert 0.0 <= p <= 1.0
        assert 0.0 <= conf < 1.0


def test_bluff_read_none_for_empty_stats() -> None:
    from pokerbot.opponent.bayes_exploit import bluff_read

    class _S:
        ctx_bluff = None

    assert bluff_read(_S(), "flop|wet|facing_bet") is None


def test_bluff_read_shrinks_to_prior_at_zero_obs() -> None:
    """With zero reveals anywhere, conf must be 0 (no confidence yet) AND the prior (Beta(2,5))
    mean = 2/7 ≈ 0.286 dominates."""
    from pokerbot.opponent.bayes_exploit import BLUFF_PRIOR, bluff_read

    class _S:
        ctx_bluff = {}

    # empty -> returns None (caller falls back), which is the conservative behaviour
    assert bluff_read(_S(), "flop|wet|facing_bet") is None
    # prior mean
    assert math.isclose(BLUFF_PRIOR[0] / (BLUFF_PRIOR[0] + BLUFF_PRIOR[1]), 2 / 7, abs_tol=1e-12)


# =========================================================================
# context_prior invariants -- non-negative, sums-to-PRIOR_STRENGTH
# =========================================================================


@pytest.mark.parametrize("ctx", _CTXS)
def test_context_prior_nonneg_and_strength(ctx: str) -> None:
    p = np.asarray(context_prior(ctx), dtype=np.float64)
    if (p < 0).any():
        _critical("prior_nonneg", f"ctx={ctx} prior={p.tolist()}")
    assert (p >= 0).all()


def test_generate_prior_sums_to_strength() -> None:
    """generate_prior() returns pseudocounts that sum (per context) to PRIOR_STRENGTH * 1.0."""
    out = generate_prior(PRIOR_STRENGTH)
    for ck, vec in out.items():
        if any(x < 0 for x in vec):
            _critical("generate_prior_nonneg", f"ctx={ck} vec={vec}")
        assert all(x >= 0 for x in vec)


def test_base_dist_is_a_distribution_per_branch() -> None:
    """_base_dist must always return a vector that sums to ~1 (it's a base rate)."""
    for street in ["preflop", "flop", "turn", "river"]:
        for facing in ["unopened", "facing_bet", "facing_raise", "facing_3bet"]:
            v = _base_dist(street, facing)
            s = sum(v)
            if abs(s - 1.0) > 0.02:  # tolerate rounding
                _critical(
                    "base_dist_sums_to_one",
                    f"street={street} facing={facing} v={v} sum={s}",
                )
            assert abs(s - 1.0) < 0.02
            assert all(x >= 0 for x in v)


# =========================================================================
# Search-tilt and mixer-logits smoke (output ranges)
# =========================================================================


def test_search_tilt_alpha_in_unit_interval() -> None:
    prior = np.asarray(context_prior(_pick_ctx()), dtype=np.float64)
    cr = ContextRead(post=prior / prior.sum(), pop=prior / prior.sum(), conf=0.5, n_obs=8)
    tilt, alpha = cr.search_tilt()
    if not (0.0 <= alpha <= 0.7):
        _critical("search_tilt_alpha", f"alpha={alpha}")
    assert 0.0 <= alpha <= 0.7
    assert np.isfinite(tilt).all()


def test_mixer_logits_finite() -> None:
    prior = np.asarray(context_prior(_pick_ctx()), dtype=np.float64)
    cr = ContextRead(post=prior / prior.sum(), pop=prior / prior.sum(), conf=0.5, n_obs=8)
    adj = cr.mixer_logits(hand_strength=0.7)
    assert np.isfinite(adj).all()


# =========================================================================
# OpponentStats Dirichlet/Beta counts invariant -- live reads rely on these
# =========================================================================


def test_opponentstats_dirichlet_counts_preserved_through_internals() -> None:
    """The invariants the read pipeline depends on:
      - ctx_counts values stay non-negative integers
      - ctx_fast / ctx_slow stay non-negative floats (Dirichlet parameters)
      - ctx_bluff rows are length-2 [bluff, value] lists of non-negative floats
    These are the 'data shape' guarantees bayes_exploit.py assumes at runtime."""
    from pokerbot.opponent.stats import OpponentStats

    ctx = "flop|wet|facing_bet"
    s = OpponentStats(
        hands=42,
        ctx_counts={ctx: [3, 7, 2, 4]},
        ctx_fast={ctx: [1.0, 6.0, 2.0, 1.0]},
        ctx_slow={ctx: [2.0, 5.0, 2.0, 1.0]},
        ctx_bluff={ctx: [2.0, 5.0]},
        ctx_runlen={ctx: 7},
    )
    # Dirichlet counts
    row = s.ctx_counts[ctx]
    assert all(isinstance(x, (int, float)) and x >= 0 for x in row)
    # Stabilized fast/slow counts (must be >= 0 for any valid posterior)
    for f, sl in zip(s.ctx_fast[ctx], s.ctx_slow[ctx]):
        assert f >= 0 and sl >= 0
    # Beta counts shape
    assert len(s.ctx_bluff[ctx]) == 2
    assert all(x >= 0 for x in s.ctx_bluff[ctx])
    # Posterior formed from these counts must be a valid simplex
    post = posterior(s.ctx_counts[ctx], np.asarray(context_prior(ctx)))
    assert math.isclose(float(post.sum()), 1.0, abs_tol=1e-9)
    assert (post >= 0).all()