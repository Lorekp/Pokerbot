"""Tests for the 6-max blueprint ranking signal (pokerbot/eval/blueprint_ranking.py).

The headline correctness property is NO FALSE POSITIVES: ranking a blueprint against ITSELF must replay
byte-identically on every CRN deck, so every per-deck (B-A) delta is exactly 0 and the harness can never
fabricate a winner. Plus: paired-delta statistics match a direct numpy recompute, the decide() gate
implements the gate-on-both rule, and the curated pool builds.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from pokerbot.eval.blueprint_ranking import (
    PairedResult,
    TabularBlueprintAgent,
    decide,
    default_pool_roster,
    paired_delta,
    rank_head_to_head,
    rank_vs_pool,
    _combined_bot_builder,
)


# ------------------------------------------------------------------ paired_delta statistics
def test_paired_delta_matches_numpy():
    a = np.array([1.0, 2.0, 3.0, 4.0, 0.5])
    b = np.array([2.0, 4.0, 4.0, 6.0, 1.5])
    r = paired_delta(a, b)
    d = b - a
    exp_mean = d.mean() * 100.0
    exp_se = d.std(ddof=1) / math.sqrt(d.size) * 100.0
    assert r.n == 5
    assert r.delta_bb_per_100 == pytest.approx(exp_mean)
    assert r.se_bb_per_100 == pytest.approx(exp_se)
    assert r.ci95_bb_per_100 == pytest.approx(1.96 * exp_se)
    assert r.a_bb_per_100 == pytest.approx(a.mean() * 100.0)
    assert r.b_bb_per_100 == pytest.approx(b.mean() * 100.0)


def test_paired_delta_identical_arms_is_exactly_zero():
    a = np.array([3.0, -1.0, 2.5, 0.0, 7.0])
    r = paired_delta(a, a.copy())
    assert r.delta_bb_per_100 == 0.0
    assert r.se_bb_per_100 == 0.0
    assert r.ci95_bb_per_100 == 0.0
    assert r.excludes_zero is False
    assert r.b_stronger is False and r.a_stronger is False


def test_paired_delta_variance_reduction_when_correlated():
    # Two highly-correlated arms (shared deck luck) -> the paired diff variance collapses -> VR >> 1.
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 10, size=400)        # the big shared card/board luck
    a = shared + rng.normal(0, 0.5, size=400)
    b = shared + 0.3 + rng.normal(0, 0.5, size=400)   # B is +0.3 bb/hand better, same decks
    r = paired_delta(a, b)
    assert r.var_reduction > 5.0                # pairing must help a lot when arms share variance
    assert r.b_stronger                          # the +0.3 edge is resolved (CI excludes 0)


def test_paired_delta_requires_alignment():
    with pytest.raises(ValueError):
        paired_delta([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        paired_delta([1.0], [2.0])               # < 2 decks


# ------------------------------------------------------------------ decide() gate
def _mk(delta: float, ci: float) -> PairedResult:
    # Only delta_bb_per_100 + ci95_bb_per_100 drive the gate properties; the rest are placeholders.
    return PairedResult(n=100, delta_bb_per_100=delta, se_bb_per_100=ci / 1.96,
                        ci95_bb_per_100=ci, t_stat=delta / (ci / 1.96) if ci else 0.0,
                        a_bb_per_100=0.0, b_bb_per_100=delta, var_reduction=1.0)


def test_decide_b_stronger_needs_both():
    # Both signals clearly B>A -> accept B.
    v, _ = decide(_mk(+5.0, 2.0), _mk(+4.0, 1.0))
    assert "B STRONGER" in v
    # Head-to-head says B, pool inconclusive -> NOT accepted (gate-on-both).
    v, _ = decide(_mk(+5.0, 2.0), _mk(+1.0, 3.0))
    assert "INCONCLUSIVE" in v


def test_decide_a_stronger():
    v, _ = decide(_mk(-5.0, 2.0), _mk(-4.0, 1.0))
    assert "A STRONGER" in v


def test_decide_disagreement_is_flagged():
    # Head-to-head B>A (excl 0), vs-pool A>B (excl 0) -> non-transitivity, must be surfaced.
    v, _ = decide(_mk(+5.0, 2.0), _mk(-4.0, 1.0))
    assert "DISAGREEMENT" in v


def test_decide_inconclusive_when_ci_includes_zero():
    v, _ = decide(_mk(+1.0, 3.0), _mk(+2.0, 5.0))
    assert "INCONCLUSIVE" in v


# ------------------------------------------------------------------ curated pool
def test_default_pool_roster_sums_to_one_and_builds():
    roster = default_pool_roster()
    assert roster, "roster must be non-empty"
    assert sum(w for _n, w in roster) == pytest.approx(1.0, abs=1e-6)
    for name, _w in roster:
        bot = _combined_bot_builder(name, random.Random(0))
        assert callable(bot), f"{name} did not build a callable bot"


# ------------------------------------------------------------------ end-to-end NO FALSE POSITIVE
# A == B must give an EXACTLY zero per-deck delta on every CRN deck (both signals). An empty snapshot
# serves legal-uniform everywhere, which is sufficient to exercise the full rotated harness + agent +
# field bots without any checkpoint dependency.
@pytest.mark.parametrize("snap", [{}])
def test_self_comparison_is_exactly_zero_head_to_head(snap):
    r = rank_head_to_head(dict(snap), dict(snap), decks=6, seed=1)
    assert r.delta_bb_per_100 == 0.0, f"A==B head-to-head must be 0, got {r.delta_bb_per_100}"
    assert r.ci95_bb_per_100 == 0.0
    assert r.excludes_zero is False


def test_self_comparison_is_exactly_zero_vs_pool():
    # Restrict to two light archetypes to keep the integration test fast; A==B must still be exactly 0.
    roster = [("responsive_station", 0.5), ("hs_nit", 0.5)]
    combined, per_arch = rank_vs_pool({}, {}, roster=roster, decks=6, seed=2)
    assert combined.delta_bb_per_100 == 0.0, f"A==B vs-pool must be 0, got {combined.delta_bb_per_100}"
    assert combined.ci95_bb_per_100 == 0.0
    for name, (r, _w) in per_arch.items():
        assert r.delta_bb_per_100 == 0.0, f"A==B vs {name} must be 0, got {r.delta_bb_per_100}"


def test_tabular_agent_implements_hero_contract():
    # The rotated harness calls these on the hero; they must exist and not raise.
    agent = TabularBlueprintAgent({})
    agent.reseed(123)
    agent.set_profile(object())
    agent.set_opponent_profiles({})
    agent.set_exploit_enabled(True)
    agent.reset_hand(0)
    assert isinstance(agent.rng, random.Random)
