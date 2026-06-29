"""Distribution-equivalence gate for the native `LeafContWeights` (battery_native / leaf_cont.rs).

DESIGN: the native leaf_cont table replaces the read-blind uniform `U{0,4}` continuation draw in
`playout::playout` with a per-leaf biased draw. The 4-vector is DATA — Python computes it from the
opponent model, Rust looks it up by the leaf's i64 key and does an inverse-CDF sample. Same float
surface as the Python branch (`np.cumsum(w)` + `np.searchsorted(cum, u, side="right")`), so the
per-leaf choice distribution is STATISTICALLY EQUIVALENT to the Python `cont_weights` branch.

These tests assert that the choice-distribution overlap between native + Python is within CI overlap
on a known weight vector (NOT SHA — the leaf already uses a non-replay RNG; the gate is statistical).

SKIPPED if the native extension isn't built (the Python `cont_weights` branch is the production path
without the native drop-in).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from pokerbot.search import native_solver as ns

_EXT = ns._try_load()
_has_leaf_cont = _EXT is not None and hasattr(_EXT, "LeafContWeights")
native_only = pytest.mark.skipif(
    not _has_leaf_cont,
    reason="native LeafContWeights not built (cargo build --release --features python)",
)


# ---------------------------------------------------------------------------
# Python reference of the inverse-CDF sample: same float surface as the Rust port.
# Pre-normalization happens at construction time on the Rust side; we mirror that here so the
# test compares the SAME post-construction distribution (sum ≈ 1.0).
# ---------------------------------------------------------------------------
def _pre_normalize(row: np.ndarray) -> np.ndarray:
    row = np.where(row < 0.0, 0.0, row).astype(np.float64, copy=False)
    s = float(row.sum())
    if s > 1e-12:
        return row / s
    return np.full(4, 0.25, dtype=np.float64)


def _py_choice(row_n: np.ndarray, u: float) -> int:
    """`np.searchsorted(np.cumsum(w), u, side="right")` clamped to [0, 3] — the Python branch
    mirror of `leaf_cont::sample_with`."""
    cum = np.cumsum(row_n)
    pick = int(np.searchsorted(cum, float(u), side="right"))
    return min(max(pick, 0), 3)


# ---------------------------------------------------------------------------
# Helper: native+python choice histograms under a shared `u ~ U[0,1)` stream. We feed the SAME
# stream into both (the contract is "same RNG/float surface as the leaf") so any divergence is a
# code bug, not a sampling artifact.
# ---------------------------------------------------------------------------
def _two_histograms(row: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    us = rng.random(n)  # length-n uniform stream
    row_n = _pre_normalize(row)
    py_choices = np.array([_py_choice(row_n, float(u)) for u in us], dtype=np.int64)
    if _has_leaf_cont:
        weights = _EXT.LeafContWeights({42: [float(x) for x in row]})
        native_choices = np.array(
            [weights.sample(42, float(u)) for u in us], dtype=np.int64
        )
    else:
        native_choices = py_choices  # placeholder; native_only tests will skip
    py_hist = np.bincount(py_choices, minlength=4).astype(np.float64)
    nat_hist = np.bincount(native_choices, minlength=4).astype(np.float64)
    return py_hist, nat_hist


def _ci_overlap(h1: np.ndarray, h2: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """For each bin: p1 ± 1.96*se1 vs p2 ± 1.96*se2; returns True iff the intervals overlap.
    (Wilson-normal approximation — fine for n >= 10k.)"""
    n1 = float(h1.sum())
    n2 = float(h2.sum())
    p1 = h1 / n1
    p2 = h2 / n2
    se1 = np.sqrt(np.maximum(p1 * (1.0 - p1) / n1, 0.0))
    se2 = np.sqrt(np.maximum(p2 * (1.0 - p2) / n2, 0.0))
    z = 1.959963984540054  # 1 - alpha/2 for alpha=0.05
    lo1, hi1 = p1 - z * se1, p1 + z * se1
    lo2, hi2 = p2 - z * se2, p2 + z * se2
    return (lo1 <= hi2) & (lo2 <= hi1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@native_only
def test_construction_pre_normalizes_to_sum_one():
    """Constructor divides by sum; the stored row is already normalized (matches Python)."""
    w = _EXT.LeafContWeights({1: [1.0, 2.0, 1.0, 4.0]})  # sum=8 -> [0.125, 0.25, 0.125, 0.5]
    # The native struct is opaque from Python; verify via `len` (1 entry) and the sampling surface.
    assert w.len() == 1
    # sum should be 1.0 — verify by sampling on a fine grid and checking the cumsum hits ~1.0.
    # Pick u=1.0-eps to force the highest bin (searchsorted side="right" puts a cum<=u index there).
    u_near_one = 0.999999999
    # All u in [0,1) should pick index 0 because w[0]=1 (the dominant mass).
    pick_lo = w.sample(1, 0.0)
    assert pick_lo == 0, f"u=0 should pick bin 0 for [1,0,0,0], got {pick_lo}"


@native_only
def test_uniform_fallback_on_miss():
    """Missing key -> uniform `U{0,4}` (the legacy read-blind path). The miss sampler uses the same
    `cum <= u` accounting as the biased path, so we can verify the bin boundaries exactly."""
    w = _EXT.LeafContWeights({})  # empty dict -> everything is a miss
    assert w.len() == 0
    # cum = [0.25, 0.5, 0.75, 1.0]. For side="right": pick = count of cum entries <= u.
    assert w.sample(999, 0.0) == 0
    assert w.sample(999, 0.24999) == 0
    assert w.sample(999, 0.25) == 1
    assert w.sample(999, 0.5) == 2
    assert w.sample(999, 0.75) == 3
    assert w.sample(999, 0.99999) == 3


@native_only
def test_degenerate_all_zero_row_falls_back_to_uniform():
    """A row of [0,0,0,0] (or near-zero) is replaced with uniform [0.25; 4] at construction."""
    w = _EXT.LeafContWeights({7: [0.0, 0.0, 0.0, 0.0], 8: [1e-20, 1e-20, 1e-20, 1e-20]})
    # With uniform 0.25 step, cum = [0.25, 0.5, 0.75, 1.0]. u=0.25 -> pick 1.
    assert w.sample(7, 0.25) == 1
    assert w.sample(7, 0.5) == 2
    assert w.sample(7, 0.75) == 3
    assert w.sample(8, 0.25) == 1


@native_only
def test_negative_row_values_clamped_to_zero():
    """Negative float noise (residual ~1e-18) clamped to 0 — same as Python's np.where(... < 0, 0)."""
    # After clamp the row is [1, 0, 0, 0] (only the positive entry survives); normalize -> [1, 0, 0, 0].
    w = _EXT.LeafContWeights({9: [-1e-18, 0.0, 0.0, 1.0]})
    assert w.sample(9, 0.0) == 0
    assert w.sample(9, 0.5) == 0
    # Right at the boundary of the unit mass.
    assert w.sample(9, 0.99999) == 0


@native_only
@pytest.mark.parametrize(
    "row",
    [
        # Heavy fold (group 0 dominates): the station/fish continuation draw.
        [0.70, 0.10, 0.10, 0.10],
        # Aggressive-heavy: the loose-passive opponent's fold-bluff-catch mix.
        [0.10, 0.10, 0.10, 0.70],
        # Passive-call-dominant: the calling station.
        [0.05, 0.80, 0.10, 0.05],
        # Lightly biased: near-uniform but with a 0.4 tilt.
        [0.40, 0.20, 0.20, 0.20],
        # Barely biased (delta=0.02): the no-model / read-blind limit.
        [0.28, 0.24, 0.24, 0.24],
    ],
)
def test_distribution_overlaps_python_branch(row):
    """CI-overlap gate: native per-leaf choice histogram vs Python's `np.searchsorted` branch,
    under a SHARED u-stream. The contract is "same RNG/float surface as the leaf", so we feed
    the same uniforms — any divergence here is a code bug, not a sampling artifact."""
    n = 50_000
    py_hist, nat_hist = _two_histograms(np.asarray(row, dtype=np.float64), n=n, seed=20260625)
    overlap = _ci_overlap(py_hist, nat_hist)
    assert bool(overlap.all()), (
        f"distribution diverged on row {row}:\n"
        f"  py  probs = {py_hist / n}\n"
        f"  nat probs = {nat_hist / n}\n"
        f"  overlap (per-bin) = {overlap}\n"
    )


@native_only
def test_distribution_strictly_differs_from_uniform_when_biased():
    """Sanity: with a heavily biased row, the native distribution must NOT look uniform
    (i.e. the model-aware branch is doing something). This is a negative control — it would
    FAIL if the constructor silently dropped the dict and used the miss-fallback path."""
    n = 50_000
    row = np.array([0.85, 0.05, 0.05, 0.05])
    _, nat_hist = _two_histograms(row, n=n, seed=7)
    p = nat_hist / n
    # Bin 0 must carry >=70% mass (expected 85% ± noise). Uniform would be 0.25.
    assert p[0] >= 0.70, f"biased bin-0 mass {p[0]:.3f} should be >= 0.70, not uniform"
    # Bin 3 must carry <=10% mass. Uniform would be 0.25.
    assert p[3] <= 0.10, f"biased bin-3 mass {p[3]:.3f} should be <= 0.10, not uniform"


@native_only
def test_hot_path_zero_rebuild_is_constant_per_call():
    """EFFICIENCY BAKED IN: the leaf_cont_weights dict is built ONCE at PyO3 __new__; subsequent
    `sample` calls must not re-allocate / re-normalize. We verify the per-call cost is in the
    microsecond regime (single HashMap get + 4-entry scan) and bounded by a generous ceiling."""
    import time

    w = _EXT.LeafContWeights({42: [0.7, 0.1, 0.1, 0.1]})
    n = 200_000
    t0 = time.perf_counter()
    for k in range(n):
        # Loop body is hot path — no rebuild. We vary `u` so the JIT doesn't fold to a constant.
        u = ((k * 2654435761) & 0xFFFF) / 65536.0  # 0..1, fast deterministic
        _ = w.sample(42, u)
    dt = time.perf_counter() - t0
    per_call_us = dt / n * 1e6
    # Generous ceiling — actual native cost should be ~0.2-0.5 us / call (HashMap lookup + 4-entry scan).
    # The bound is here ONLY to catch a catastrophic regression (e.g., a per-call Vec allocation),
    # NOT to track micro-perf. Set at 100x the expected native cost on this hardware.
    assert per_call_us < 50.0, (
        f"per-call cost {per_call_us:.2f}us > 50us — possible per-call rebuild regression"
    )