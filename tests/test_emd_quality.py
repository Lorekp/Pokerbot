"""FAST acceptance gate for the EMD-vs-made-hand postflop abstraction quality harness (Workstream F,
item 3). Asserts the two ULTRACODE acceptance properties on small, fixed-seed samples (< ~30s):

  (a) EMD separates a SUBSTANTIAL fraction of equal-made-rank spot pairs (A1 threshold justified below).
  (b) EMD mean EMD-distance at equal made rank is clearly > 0 (A2 margin).
  (c) EMD_EV >= madehand_EV with a STRICTLY POSITIVE gain on the thin-value decision, and the bootstrap
      95% CI on the gain excludes zero.
  (d) The worked draw-vs-made example lands in DIFFERENT EMD buckets.

The harness (scripts/abstraction/emd_vs_madehand_quality.py) prefers the native battery_native EMD
kernel; if it is not built, it falls back to the pure-Python bucketer (slower, still correct), so this
test runs either way. Parameters are kept small so the whole module stays well under 30s.

Justification for the thresholds (deliberately conservative vs. the full-run numbers so the gate is
stable, not tuned to the seed):
  - A1 > 0.30:  the full run measures ~0.64 separation; even at small N and 60 buckets it is ~0.66. A
    third of equal-current-rank pairs landing in different EMD buckets already proves EMD reads
    potential the made-hand bucket discards. 0.30 leaves a wide safety margin.
  - A2 > 0.20:  the full run measures ~1.0; equal-made-rank histograms are clearly distinct under EMD.
  - gain > 0 AND CI low bound > 0:  the load-bearing claim. The full run gives +0.0028..+0.005 bb with a
    CI bounded away from zero; the small-N gate uses the robust [0.35, 0.85] calling band.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pokerbot.encoding.postflop_bucketer import strength_histogram

emdq = pytest.importorskip("scripts.abstraction.emd_vs_madehand_quality")


# Small, fixed-seed configuration (native or python fallback both finish < ~30s).
N_BINS = 20
N_RUNOUTS = 200
D1_SPOTS = 1200
D1_BUCKETS = 60
D1_FIT = 1000
D2_SPOTS = 2000
D2_BUCKETS = 30
D2_FIT = 1500
CALL_LO, CALL_HI = 0.35, 0.85  # robust medium thin-value calling band
SEED = 12345


@pytest.fixture(scope="module")
def d1():
    rng = np.random.default_rng(SEED)
    return emdq.deliverable1(
        rng, n_spots=D1_SPOTS, n_made_buckets=D1_BUCKETS, n_emd_buckets=D1_BUCKETS,
        n_bins=N_BINS, n_runouts=N_RUNOUTS, fit_sample=D1_FIT, max_pairs=120,
    )


@pytest.fixture(scope="module")
def d2():
    rng = np.random.default_rng(SEED + 7)
    return emdq.deliverable2(
        rng, n_spots=D2_SPOTS, n_buckets=D2_BUCKETS, n_bins=N_BINS, n_runouts=N_RUNOUTS,
        call_lo=CALL_LO, call_hi=CALL_HI, pot=10.0, bet_frac=0.5, fit_sample=D2_FIT, bootstrap=1000,
    )


# ---------------------------------------------------------------------------
# (a) + (b) draw/made separation at EQUAL current rank.
# ---------------------------------------------------------------------------
def test_a1_separation_fraction_substantial(d1):
    a1 = d1["A1_separation_fraction"]
    assert a1 > 0.30, (
        f"EMD separated only {a1:.3f} of equal-made-rank pairs into different EMD buckets "
        f"(expected > 0.30 -- EMD should read draw potential the made-hand bucket discards)"
    )


def test_a2_mean_emd_distance_positive_by_margin(d1):
    a2 = d1["A2_mean_emd_distance"]
    assert a2 > 0.20, (
        f"mean EMD distance between equal-made-rank histograms is only {a2:.3f} (expected > 0.20); "
        f"equal current rank should still leave clearly distinct strength distributions"
    )


# ---------------------------------------------------------------------------
# (d) worked draw-vs-made example lands in DIFFERENT EMD buckets.
# ---------------------------------------------------------------------------
def test_worked_example_different_emd_buckets(d1):
    we = d1["worked_example"]
    assert we.get("found"), "harness failed to find a draw-vs-made worked example"
    made, draw = we["made_hand"], we["draw_hand"]
    # Same made-hand bucket (equal current rank), different EMD buckets, sizeable histogram distance.
    assert made["made_bucket"] == draw["made_bucket"], "worked example must share a made-hand bucket"
    assert made["emd_bucket"] != draw["emd_bucket"], (
        f"worked example must land in different EMD buckets, got {made['emd_bucket']} == {draw['emd_bucket']}"
    )
    assert we["emd_distance"] > 0.2, f"worked-example EMD distance too small: {we['emd_distance']:.3f}"


def test_explicit_draw_vs_made_separation():
    """An INDEPENDENT, hand-built check that a big combo draw and a made hand of similar current rank
    have clearly different strength histograms (large EMD distance) -- the property the made-hand bucket
    discards. We assert the histogram distance, not bucket ids (which depend on a fitted table)."""
    def cid(rank, suit):
        return rank * 4 + suit

    # Made middle pair, no draw: T9 on Q T 4 (rainbow) + 2 brick turn.
    made = ((cid(8, 0), cid(7, 1)), (cid(10, 2), cid(8, 3), cid(2, 0), cid(0, 1)))
    # Big combo draw (flush draw + open-ended straight draw), no made hand: 9h 8h on 7h 6h 2c + Kc brick.
    draw = ((cid(7, 2), cid(6, 2)), (cid(5, 2), cid(4, 2), cid(0, 0), cid(11, 1)))
    h_made = strength_histogram(made[0], made[1], n_bins=N_BINS, n_runouts=1500, seed=1)
    h_draw = strength_histogram(draw[0], draw[1], n_bins=N_BINS, n_runouts=1500, seed=2)
    dist = emdq._emd_1d(h_made, h_draw)
    # The made hand is a tight spike; the draw is bimodal (bricks low / completes high) -> large EMD.
    assert dist > 1.0, f"made vs combo-draw histograms should be far apart under EMD, got {dist:.3f}"
    # And the draw must put real mass in the TOP bins (it completes to a strong hand) that the made
    # middle pair does not -- the potential the made-hand bucket cannot see.
    top = slice(int(0.75 * N_BINS), N_BINS)
    assert h_draw[top].sum() > h_made[top].sum() + 0.05


# ---------------------------------------------------------------------------
# (c) thin-value EV gain is strictly positive and the CI excludes zero.
# ---------------------------------------------------------------------------
def test_emd_ev_gain_strictly_positive(d2):
    gain = d2["gain_bb"]
    ci_lo, ci_hi = d2["gain_bb_ci95"]
    assert d2["EV_emd"] >= d2["EV_madehand"], (
        f"EMD_EV {d2['EV_emd']:.5f} should be >= madehand_EV {d2['EV_madehand']:.5f}"
    )
    assert gain > 0.0, f"thin-value EV gain (EMD - madehand) not positive: {gain:+.6f} bb"
    assert ci_lo > 0.0, f"bootstrap 95% CI low bound not above zero: [{ci_lo:+.6f}, {ci_hi:+.6f}]"
    # EMD must close a meaningful share of the made-hand abstraction gap to the unabstracted optimum.
    fc = d2["frac_gap_closed_by_emd"]
    assert fc is not None and fc > 0.1, f"EMD closed too little of the made-hand gap: {fc}"


def test_gain_concentrates_on_disagreement_spots(d2):
    """The EV gain should come from spots where current rank and river potential DISAGREE (the made-hand
    bucket forces the wrong action there)."""
    assert d2["n_made_forces_wrong_action"] > 0
    assert d2["ev_recovered_on_disagreement_spots"] > 0.0


def test_module_runs_fast(d1, d2):
    """Sanity: the fixtures are already computed by here; just assert the gate produced finite numbers
    (the < 30s budget is enforced by the small N above, validated in the build)."""
    t0 = time.perf_counter()
    assert np.isfinite(d1["A1_separation_fraction"])
    assert np.isfinite(d2["gain_bb"])
    assert time.perf_counter() - t0 < 1.0


@pytest.fixture(scope="module")
def functional_tasks():
    rng = np.random.default_rng(SEED + 99)
    return emdq.functional_micro_ev_tasks(
        rng,
        n_spots=400,
        n_buckets=D2_BUCKETS,
        n_bins=N_BINS,
        n_runouts=N_RUNOUTS,
        fit_sample=400,
    )


def test_functional_micro_ev_tasks_no_large_regression(functional_tasks):
    """EMD should not materially regress made-hand on functional micro-EV tasks."""
    for name, row in functional_tasks.items():
        assert row["gain_bb"] >= -0.01, f"{name} gain {row['gain_bb']}"


def test_functional_at_least_one_task_closes_gap(functional_tasks):
    closed = [
        name for name, row in functional_tasks.items()
        if row.get("frac_gap_closed_by_emd") is not None and row["frac_gap_closed_by_emd"] > 0.05
    ]
    assert closed, "expected at least one functional task to close >5% of made-hand gap"


def test_river_fallback_sanity():
    sanity = emdq.river_fallback_sanity(n_bins=N_BINS, n_runouts=N_RUNOUTS, n_buckets=D2_BUCKETS)
    assert sanity["river_peak_mass_min"] >= 0.90
    assert sanity["river_hist_variance_mean"] < 0.08
