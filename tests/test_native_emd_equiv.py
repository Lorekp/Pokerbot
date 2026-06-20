"""Native (Rust) EMD postflop bucketer <-> Python oracle equivalence gate (Workstream F).

The Rust `emd.rs` port speeds up the cloud-scale offline precompute of the potential-aware EMD
postflop abstraction (`pokerbot.encoding.postflop_bucketer`). The cross-check standard is mixed:

  (a) BIT-IDENTITY of the pure-float bucket assignment. `emd_bucket_from_hist` is numpy's
      `argmin(sum(|cumsum(hist) - cumsum(centroid)|))` — a deterministic float reduction over fixed
      inputs, so it MUST match Python exactly (0 mismatches). This is the load-bearing anchor.

  (b) STATISTICAL EQUIVALENCE of the Monte-Carlo histogram. The native path deliberately uses a fast
      xoshiro RNG instead of replaying numpy's PCG64 per-showdown (a proven perf dead-end here), so the
      histograms only converge as `n_runouts` grows. We assert the mean per-spot L1 distance is below a
      documented tolerance at high `n_runouts`, and report the measured mean/max.

  (c) CALIBRATION of the k-means clustering. Native `emd_fit` and Python `EMDBucketer.fit` use
      independent inits, so per-bucket correspondence is meaningless. We instead assert the native
      clustering QUALITY (mean within-cluster EMD inertia, measured on the SAME data) is within ~10% of
      Python's, and report both inertias.

Skips cleanly if the native extension is unavailable (e.g. not built on this box).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pokerbot.encoding import postflop_bucketer as oracle

try:
    from pokerbot.search import native_solver

    _EXT = native_solver._try_load()
except Exception:  # noqa: BLE001
    _EXT = None

pytestmark = pytest.mark.skipif(
    _EXT is None or not hasattr(_EXT, "emd_bucket_from_hist"),
    reason="native battery_native (with emd_bucket_from_hist) not available",
)

N_BINS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _py_bucket(hist: np.ndarray, centroids: np.ndarray) -> int:
    """Python reference: argmin over centroids of L1-of-CDF (first minimum, numpy argmin semantics)."""
    cc = np.cumsum(centroids, axis=1)
    d = np.abs(np.cumsum(hist)[None, :] - cc).sum(axis=1)
    return int(d.argmin())


def _random_normalized(rng: np.random.Generator, n: int, n_bins: int) -> np.ndarray:
    x = rng.random((n, n_bins))
    # ~30% of rows get sparsified (some zero bins) to stress the CDF edges.
    mask = rng.random((n, n_bins)) < 0.3
    x[mask] = 0.0
    s = x.sum(axis=1, keepdims=True)
    s[s == 0.0] = 1.0  # avoid div-by-zero on an all-zero row (then it stays all-zero)
    return x / s


def _random_legal_spot(rng: np.random.Generator):
    """A random legal postflop spot: 2 hole cards + 3 or 4 board cards (flop/turn), all distinct."""
    n_board = int(rng.integers(3, 5))  # 3 (flop) or 4 (turn) -> always has runouts
    cards = rng.choice(52, size=2 + n_board, replace=False)
    hole = [int(cards[0]), int(cards[1])]
    board = [int(c) for c in cards[2:]]
    return hole, board


# ---------------------------------------------------------------------------
# (a) BIT-IDENTITY of bucket assignment — the load-bearing anchor.
# ---------------------------------------------------------------------------
def test_bucket_bit_identical_to_numpy_argmin():
    rng = np.random.default_rng(20260620)
    n_cases = 2000
    mismatches = 0
    first_bad = None
    for _ in range(n_cases):
        n_bins = int(rng.integers(5, 25))
        k = int(rng.integers(2, 30))
        hist = _random_normalized(rng, 1, n_bins)[0]
        centroids = _random_normalized(rng, k, n_bins)
        want = _py_bucket(hist, centroids)
        got = _EXT.emd_bucket_from_hist(hist.tolist(), centroids.tolist())
        if got != want:
            mismatches += 1
            if first_bad is None:
                first_bad = (hist.tolist(), centroids.tolist(), want, got)
    assert mismatches == 0, f"{mismatches}/{n_cases} bucket mismatches; first: {first_bad}"
    print(f"\n[a] bucket bit-identity: {n_cases} cases, {mismatches} mismatches")


# ---------------------------------------------------------------------------
# (b) STATISTICAL EQUIVALENCE of the Monte-Carlo histogram.
# ---------------------------------------------------------------------------
def test_histogram_statistical_equivalence():
    rng = np.random.default_rng(777)
    n_spots = 300
    n_runouts = 2000  # high enough that two uniform-without-replacement samplers converge
    l1s = []
    for i in range(n_spots):
        hole, board = _random_legal_spot(rng)
        py = oracle.strength_histogram(
            tuple(hole), tuple(board), n_bins=N_BINS, n_runouts=n_runouts, seed=i
        )
        nat = np.asarray(
            _EXT.emd_strength_histogram(hole, board, N_BINS, n_runouts, int(i)),
            dtype=np.float64,
        )
        assert nat.shape == py.shape
        l1s.append(float(np.abs(nat - py).sum()))
    l1 = np.asarray(l1s)
    mean_l1 = float(l1.mean())
    max_l1 = float(l1.max())
    print(
        f"\n[b] histogram stat-equiv over {n_spots} spots @ {n_runouts} runouts: "
        f"mean L1 = {mean_l1:.4f}, max L1 = {max_l1:.4f}"
    )
    # Documented tolerance: two independent uniform-without-replacement samplers of the same runout
    # distribution converge in L1 as n_runouts grows; at 2000 runouts the per-spot mean L1 is small.
    assert mean_l1 < 0.08, f"mean L1 {mean_l1:.4f} exceeds tolerance 0.08"


# ---------------------------------------------------------------------------
# (c) CALIBRATION of the clustering (inertia within ~10%).
# ---------------------------------------------------------------------------
def test_clustering_inertia_calibration():
    rng = np.random.default_rng(2024)
    # Build a sample of histograms from real legal spots (the precompute input distribution).
    n_sample = 400
    n_runouts = 400  # modest — clustering quality is robust to the histogram noise floor
    sample = []
    for i in range(n_sample):
        hole, board = _random_legal_spot(rng)
        sample.append(
            oracle.strength_histogram(
                tuple(hole), tuple(board), n_bins=N_BINS, n_runouts=n_runouts, seed=1000 + i
            )
        )
    X = np.asarray(sample, dtype=np.float64)

    num_buckets = 16
    iters = 12

    # Python fit (oracle).
    py_b = oracle.EMDBucketer(num_buckets=num_buckets, n_bins=N_BINS, n_runouts=n_runouts)
    py_b.fit(X, iters=iters, seed=0)
    py_cent = py_b.centroids

    # Native fit.
    nat_cent = np.asarray(
        _EXT.emd_fit(X.tolist(), num_buckets, iters, 0), dtype=np.float64
    )
    assert nat_cent.shape == (num_buckets, N_BINS)

    def _inertia(cent: np.ndarray) -> float:
        cc = np.cumsum(cent, axis=1)
        Xc = np.cumsum(X, axis=1)
        d = np.abs(Xc[:, None, :] - cc[None, :, :]).sum(axis=2)  # (N, K) EMD
        return float(d.min(axis=1).mean())

    py_inertia = _inertia(py_cent)
    nat_inertia = _inertia(nat_cent)
    # Sanity: the native inertia helper agrees with the numpy one on the native centroids.
    nat_inertia_ext = float(_EXT.emd_mean_inertia(X.tolist(), nat_cent.tolist()))

    rel = abs(nat_inertia - py_inertia) / max(py_inertia, 1e-12)
    print(
        f"\n[c] clustering inertia: python = {py_inertia:.5f}, native = {nat_inertia:.5f} "
        f"(native-ext {nat_inertia_ext:.5f}), rel diff = {rel:.3%}"
    )
    assert abs(nat_inertia_ext - nat_inertia) < 1e-9, "native inertia helper disagrees with numpy"
    assert rel < 0.10, f"native clustering inertia {nat_inertia:.5f} not within 10% of python {py_inertia:.5f}"


# ---------------------------------------------------------------------------
# Determinism + speedup report (not a hard gate, but informative).
# ---------------------------------------------------------------------------
def test_batch_determinism_and_speedup_report():
    rng = np.random.default_rng(55)
    n_spots = 200
    n_runouts = 1000
    spots = [_random_legal_spot(rng) for _ in range(n_spots)]

    # Determinism: same seed_base -> identical batch.
    b1 = _EXT.emd_strength_histograms_batch(spots, N_BINS, n_runouts, 7)
    b2 = _EXT.emd_strength_histograms_batch(spots, N_BINS, n_runouts, 7)
    assert b1 == b2, "native batch must be reproducible for a fixed seed_base"

    # Native batch timing.
    t0 = time.perf_counter()
    _EXT.emd_strength_histograms_batch(spots, N_BINS, n_runouts, 7)
    t_native = time.perf_counter() - t0

    # Python timing over the same spots.
    t0 = time.perf_counter()
    for hole, board in spots:
        oracle.strength_histogram(tuple(hole), tuple(board), n_bins=N_BINS, n_runouts=n_runouts)
    t_python = time.perf_counter() - t0

    speedup = t_python / t_native if t_native > 0 else float("inf")
    print(
        f"\n[speed] {n_spots} spots @ {n_runouts} runouts: "
        f"python {t_python:.3f}s, native {t_native:.3f}s -> {speedup:.1f}x"
    )
    assert speedup > 1.0, f"native should be faster, got {speedup:.2f}x"
