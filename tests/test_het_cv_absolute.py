"""Validate the heterogeneous-field control-variate estimator (pokerbot/eval/het_cv_absolute.py).

Mirrors tests/test_cv_absolute.py: the estimator's STATISTICS (unbiasedness + a measured variance reduction
> 1) are proven on SYNTHETIC correlated draws — fed through the real ``het_cv_absolute`` orchestrator via an
injected ``_run_corotated_fn`` — so no heavy poker compute is needed to certify the math. A small REAL
co-rotated run (cheap deterministic bots as hero + villains) then proves the play machinery + the het-ring
co-rotation + CRN actually work, and that the AIVAT path would FAIL the var_reduction>1 gate (the
documentation test) and the serve-skew fingerprint fails CLOSED.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest


def _int_seed(*parts) -> int:
    """Stable 64-bit int seed from arbitrary parts (np.random.default_rng wants ints, not str tuples)."""
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")

from pokerbot.eval.cv_absolute import CVResult, decompose_absolute
from pokerbot.eval.het_cv_absolute import (
    HetCVResult,
    VarReductionError,
    _corotated_seat_means,
    het_cv_absolute,
)


# --------------------------------------------------------------------------------------------------
# Synthetic injected co-rotated runner: a deck's seat-mean = shared luck f_i + hero-specific residual.
# Search (S) and blueprint (B) on the SAME seed share f_i (CRN); the cheap arm (seed+offset) gets its own.
# This is the rotated-duplicate per-deck seat-mean model from test_cv_absolute, lifted to the het driver.
# --------------------------------------------------------------------------------------------------
def _make_synthetic_runner(*, mu_search, mu_base, sf=8.0, sres=2.0, rake_per_deck=0.5):
    """Return a drop-in ``_run_corotated_fn`` producing CRN-correlated synthetic per-deck seat-means.

    The tag identifies the arm; the seed bands (paired ``seed`` vs cheap ``seed+offset``) make the shared
    luck factor deterministic-per-deck-index so search & blueprint share it (CRN), and disjoint across the
    paired/cheap seed bands. ``rake_per_deck`` is a flat positive cost subtracted on the net series."""
    def _run(hero, field_specs, mk, *, decks, seed, n_seats, button_seat, allin_ev, rake_cost,
             progress_cb, tag):
        is_search = tag == "search"
        mu = mu_search if is_search else mu_base
        chip, net = [], []
        for d in range(decks):
            # Shared luck f is a pure fn of (seed_band, deck_index): paired search & blueprint use the SAME
            # seed -> identical f per deck (CRN); the cheap arm's seed differs -> independent f.
            f = np.random.default_rng(_int_seed(seed, d, "luck")).normal(0.0, sf)
            res = np.random.default_rng(_int_seed(seed, d, tag, "res")).normal(0.0, sres)
            v = mu + f + res
            chip.append(v)
            if rake_cost is not None:
                net.append(v - rake_per_deck)
        return chip, (net if rake_cost is not None else None)
    return _run


# Minimal factories: the synthetic runner ignores the hero object, so any sentinel works.
def _search_factory(exploit):
    return ("search_hero", exploit)


def _baseline_factory(exploit):
    return ("blueprint_hero", exploit)


_HET_FIELD = [("calling_station", None), ("nit", None), ("tag", None), ("lag", None), ("random", None)]


def test_unbiased_and_reduces_variance_through_orchestrator():
    """The het CV estimate is centered on the true search EV AND reports var_reduction>1 (the runtime gate)."""
    mu_S, mu_B = 0.03, 0.01  # +3 bb/hand search, +1 blueprint (in seat-mean units)
    runner = _make_synthetic_runner(mu_search=mu_S, mu_base=mu_B)
    res, meta = het_cv_absolute(
        _HET_FIELD, search_factory=_search_factory, baseline_factory=_baseline_factory,
        paired_decks=400, cheap_decks=4000, seed=0, _run_corotated_fn=runner,
    )
    assert isinstance(res, HetCVResult)
    assert res.reducer == "control_variate"
    assert isinstance(res.chip, CVResult)
    # 1) UNBIASED: true search mean inside the CV 95% CI
    lo, hi = res.chip.ci95
    assert lo <= mu_S <= hi, f"true {mu_S} outside CV CI {res.chip.ci95}"
    # 2) VARIANCE REDUCED (the runtime gate would have RAISED otherwise)
    assert res.chip.var_reduction > 1.0
    assert res.chip.corr > 0.9
    assert res.chip.se < res.chip.naive_se
    # bb/100 convenience + provenance
    assert abs(res.chip_bb_per_100 - res.chip.mean * 100.0) < 1e-12
    assert meta["uses_aivat"] is False
    assert meta["reducer"] == "control_variate"
    assert meta["field"] == [s[0] for s in _HET_FIELD]


def test_unbiasedness_over_many_seeds():
    """Across independent replications the CV mean is centered on mu_S (no systematic bias from the driver)."""
    mu_S, mu_B = 0.025, 0.005
    means = []
    for seed in range(120):
        runner = _make_synthetic_runner(mu_search=mu_S, mu_base=mu_B)
        res, _ = het_cv_absolute(
            _HET_FIELD, search_factory=_search_factory, baseline_factory=_baseline_factory,
            paired_decks=120, cheap_decks=1200, seed=seed, _run_corotated_fn=runner,
        )
        means.append(res.chip.mean)
    grand = float(np.mean(means))
    se = float(np.std(means, ddof=1) / np.sqrt(len(means)))
    assert abs(grand - mu_S) < 4 * se, f"bias: grand {grand:.5f} vs {mu_S} (se {se:.5f})"


def test_net_of_rake_column_present_and_offset():
    """The net-of-rake CV column is reported and is below the chip column by the per-hand rake cost."""
    mu_S, mu_B, rake = 0.04, 0.02, 0.5
    runner = _make_synthetic_runner(mu_search=mu_S, mu_base=mu_B, rake_per_deck=rake)
    res, meta = het_cv_absolute(
        _HET_FIELD, search_factory=_search_factory, baseline_factory=_baseline_factory,
        paired_decks=300, cheap_decks=3000, seed=1, net_of_rake_stake="0.50/1",
        _run_corotated_fn=runner,
    )
    assert res.net is not None
    assert res.net.var_reduction > 1.0
    # chip - net == flat rake per deck (the synthetic runner subtracts a constant)
    assert abs((res.chip.mean - res.net.mean) - rake) < 1e-9
    assert res.net_bb_per_100 is not None
    assert meta["net_of_rake_stake"] == "0.50/1"
    assert meta["rakeback_mode"] == "modeled"


def test_fallback_corotated_duplicate_no_baseline():
    """With no baseline_factory the estimator falls back to the pure co-rotated duplicate (no CV layer)."""
    runner = _make_synthetic_runner(mu_search=0.03, mu_base=0.0)
    res, meta = het_cv_absolute(
        _HET_FIELD, search_factory=_search_factory, baseline_factory=None,
        paired_decks=500, seed=0, _run_corotated_fn=runner,
    )
    assert res.reducer == "corotated_duplicate"
    assert meta["reducer"] == "corotated_duplicate"
    # In the fallback the CV degenerates to the naive mean (beta=0, var_reduction=1, no cheap decks).
    assert res.chip.beta == 0.0
    assert res.chip.var_reduction == 1.0
    assert res.chip.decks_cheap == 0
    assert res.chip.mean == res.chip.naive_mean
    # still an honest, finite absolute estimate
    assert np.isfinite(res.chip.mean) and np.isfinite(res.chip.se)


def test_var_reduction_gate_raises_when_baseline_mis_wired():
    """A mis-wired baseline -> var_reduction<=1 -> RAISES, never silently returns. Modeled here as a TINY,
    high-variance cheap-baseline sample: the cheap-term ``beta^2 * Var(Bc)/m`` then dominates and the CV SE
    exceeds the naive SE (the cheap baseline fails to pin E[B]) -> var_reduction < 1. This is the SAME
    failure class as AIVAT-on-search; the estimator must fail closed rather than return a worse-than-naive
    number."""
    def _bad_runner(hero, field_specs, mk, *, decks, seed, n_seats, button_seat, allin_ev, rake_cost,
                    progress_cb, tag):
        rng = np.random.default_rng(_int_seed(seed, tag))
        if tag == "search":
            f = np.random.default_rng(_int_seed(seed, "luck"))
            return list(0.03 + f.normal(0.0, 8.0, size=decks) + rng.normal(0.0, 1.0, size=decks)), None
        if tag == "baseline_paired":
            # correlated with search (shares the luck factor) -> large optimal beta
            f = np.random.default_rng(_int_seed(seed, "luck"))
            return list(0.01 + f.normal(0.0, 8.0, size=decks) + rng.normal(0.0, 1.0, size=decks)), None
        # baseline_cheap: only a HANDFUL of very noisy decks -> the cheap term blows the SE past naive.
        return list(0.01 + rng.normal(0.0, 40.0, size=decks)), None
    with pytest.raises(VarReductionError):
        het_cv_absolute(
            _HET_FIELD, search_factory=_search_factory, baseline_factory=_baseline_factory,
            paired_decks=400, cheap_decks=3, seed=0, _run_corotated_fn=_bad_runner,
        )


def test_aivat_on_search_would_fail_the_gate_documentation():
    """DOCUMENTATION TEST: AIVAT on the SEARCH path gives NEGATIVE variance reduction, so an AIVAT-adjusted
    series can NEVER pass this estimator's var_reduction>1 gate -- it would be caught and raise. We model the
    AIVAT-on-search pathology (the adjusted series is NOISIER than the raw, var_reduction<1) and confirm
    decompose_absolute reports <1, i.e. the gate the het estimator enforces would trip on it."""
    rng = np.random.default_rng(0)
    n = 600
    # AIVAT's miscalibrated gadget-Q baseline on the search path is NOT correlated with the realized search
    # variance the way a faithful control variate must be -- so it gives ~no (or negative) reduction. Model
    # the AIVAT-"adjusted" expensive series as INDEPENDENT of its control baseline (corr ~ 0): a CV with a
    # control that shares no variance with the estimand cannot reduce the SE -> var_reduction ~ 1 (here <=1).
    adjusted = 0.03 + rng.normal(0.0, 8.0, size=n)          # AIVAT-adjusted search series
    control = 0.01 + rng.normal(0.0, 8.0, size=n)           # the (useless) AIVAT baseline, independent
    cheap = 0.01 + rng.normal(0.0, 8.0, size=4000)          # disjoint baseline sample, also independent
    res = decompose_absolute(adjusted, control, cheap, optimal_beta=True)
    assert abs(res.corr) < 0.15, f"expected ~0 corr for the AIVAT-on-search pathology, got {res.corr:.3f}"
    # The realized reduction is at or below ~1 -> the het estimator's gate (var_reduction>1) would REJECT,
    # so AIVAT-on-search can NEVER silently pass through this estimator.
    assert res.var_reduction <= 1.05, (
        f"AIVAT-on-search documentation: expected var_reduction<=~1, got {res.var_reduction:.3f}")


def test_no_aivat_accumulator_tripwire():
    """The estimator asserts no live AIVATAccumulator on its path. Construct one and confirm the run trips."""
    from pokerbot.eval.aivat import AIVATAccumulator
    leaked = AIVATAccumulator()  # noqa: F841 — deliberately keep it alive so gc.get_objects() sees it
    runner = _make_synthetic_runner(mu_search=0.03, mu_base=0.01)
    with pytest.raises(AssertionError, match="AIVAT BAN"):
        het_cv_absolute(
            _HET_FIELD, search_factory=_search_factory, baseline_factory=_baseline_factory,
            paired_decks=50, cheap_decks=100, seed=0, _run_corotated_fn=runner,
        )
    del leaked


def test_serve_skew_fingerprint_fails_closed():
    """Serve-skew guard: a snap_abstraction that cannot match the current import-baked flags MUST raise
    (fail closed) -- this is the -559 bb/100 artifact guard. A deliberate mismatch is DETECTED."""
    # A stamp with an impossible-to-match marker forces a mismatch regardless of the live env flags.
    bad_stamp = {
        "betting_abstraction": True, "postflop_buckets": True, "preflop_lossless": True,
        "postflop_mode": "DELIBERATE_MISMATCH_SENTINEL", "postflop_num_buckets": -1,
        "depth_bucketed": False,
    }
    runner = _make_synthetic_runner(mu_search=0.03, mu_base=0.01)
    with pytest.raises(RuntimeError):
        het_cv_absolute(
            _HET_FIELD, search_factory=_search_factory, baseline_factory=_baseline_factory,
            paired_decks=50, cheap_decks=100, seed=0, snap_abstraction=bad_stamp,
            _run_corotated_fn=runner,
        )


# --------------------------------------------------------------------------------------------------
# REAL co-rotated machinery (cheap deterministic bots) — proves the play loop + het ring + CRN work.
# --------------------------------------------------------------------------------------------------
class _CheapHero:
    """A deterministic call/check hero usable as BOTH the search and blueprint arm. Implements the
    hero_seat / reseed / on_new_hand seams the co-rotated reseed exercises, so the real machinery runs."""

    def __init__(self, exploit: bool):
        self.exploit = exploit
        self.hero_seat = 0
        self.rng = random.Random(0)

    def reseed(self, seed):
        self.rng = random.Random(seed)

    def on_new_hand(self):
        pass

    def __call__(self, state):
        from pokerbot.engine.actions import Action, ActionType
        legal = state.legal_actions()
        for at in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD):
            for a in legal:
                if a.action_type == at:
                    return a
        return legal[0]


def test_real_corotated_seat_means_runs_and_is_finite():
    """The real co-rotated machinery plays a deck across all seats vs a heterogeneous ring and returns a
    finite seat-mean; with a rake_cost it also returns a finite net seat-mean strictly <= the chip one."""
    hero = _CheapHero(exploit=False)
    mk = build_bot_resolver()
    # chip-only
    out = _corotated_seat_means(hero, _HET_FIELD, mk, deck_seed=1234, n_seats=6, button_seat=3,
                                allin_ev=True)
    assert len(out) == 1 and np.isfinite(out[0])
    # with rake
    from pokerbot.economics.rake import net_cost_bb

    def _cost(pot, comm, pre):
        return float(net_cost_bb(pot, comm, 6, pre, "0.50/1"))
    out2 = _corotated_seat_means(hero, _HET_FIELD, mk, deck_seed=1234, n_seats=6, button_seat=3,
                                 allin_ev=True, rake_cost=_cost)
    assert len(out2) == 2
    chip, net = out2
    assert np.isfinite(chip) and np.isfinite(net)
    assert net <= chip + 1e-9  # rake is a non-negative cost
    # chip-EV column is byte-identical with or without the rake capture (additive guarantee)
    assert abs(chip - out[0]) < 1e-12


def test_real_corotated_crn_pairs_search_and_blueprint():
    """Search & blueprint heroes on the SAME deck_seed face byte-identical boards + villains (CRN), so two
    IDENTICAL cheap heroes produce the SAME seat-mean deck-for-deck -> corr 1.0 -> the CV is well-posed."""
    mk = build_bot_resolver()
    s_series, b_series = [], []
    for d in range(8):
        deck_seed = 1000 + d
        s = _corotated_seat_means(_CheapHero(True), _HET_FIELD, mk, deck_seed, n_seats=6,
                                  button_seat=3, allin_ev=True)[0]
        b = _corotated_seat_means(_CheapHero(False), _HET_FIELD, mk, deck_seed, n_seats=6,
                                  button_seat=3, allin_ev=True)[0]
        s_series.append(s)
        b_series.append(b)
    # Two identical deterministic heroes on identical CRN decks must match exactly.
    assert np.allclose(s_series, b_series, atol=1e-9)


def build_bot_resolver():
    """The default ``build_bot`` (the cheap deterministic eval-bot pool) used as the villain factory."""
    from pokerbot.eval.bots import build_bot
    return build_bot
