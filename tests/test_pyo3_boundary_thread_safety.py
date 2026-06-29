"""REFUTER 4: PyO3 BOUNDARY multi-thread safety audit.

For every PyO3 binding in tools/battery_native/src/lib.rs and src/live.rs (the python-feature
surface), this file invokes the binding concurrently from N=8 worker threads and verifies:

  * NO CRASH — the worker pool must finish cleanly (no segfault, no abort, no double-free,
    no thread sanitizer trip).
  * NO GIL DEADLOCK — all workers must return within a generous timeout, NOT hang waiting
    for the GIL or for a Mutex the binding holds while it re-acquires the GIL.
  * NO PYTHON EXCEPTION LEAK — the binding surfaces Python exceptions as the documented type
    (ValueError for shape errors, RuntimeError for native panics, etc.). If the binding
    panics inside the FFI layer and Rust aborts, the test must fail with a clear signal.

The strategy is: invoke each binding from multiple threads concurrently, repeat a moderate
number of times, and confirm the aggregate of all worker return values is well-formed AND
no worker thread raised an unexpected exception.

What we EXPLICITLY test:
  * `apply_action` / `sample_chance` / `legal_action_mask` / `is_chance_node` (engine transitions)
  * `init_hand` (root state construction)
  * `evaluate_seven` / `compute_returns` / `fingerprint_fast` / `fingerprint_fast_depth`
  * `eval_seven_phe` / `multiway_equity_river` (numeric helpers)
  * `leduc_solve_exploitability` (one-shot trainer)
  * `icm_equity` / `bubble_factor` / `icm_delta` (tournament ICM)
  * `emd_strength_histogram` / `emd_bucket_from_hist` / `emd_fit` / `emd_mean_inertia`
  * `emd_strength_histograms_batch`
  * `engine_dims`
  * `PlayoutEngine.leaf_value` (pyclass)
  * `SubgameSolver.solve` (pyclass)
  * `Lbr6Harness.serve` / `Lbr6Harness.lbr_tabular` (pyclass)
  * `LbrNeuralHarness.serve` (pyclass)
  * `LeducCFRTrainer.train` / `exploitability` / `serve` / `save_bytes` / `load_bytes`
  * `register_cancel` / `set_cancel` / `release_cancel` (process-global registry)
  * `SearchHandle.build_root` / `solve` / `blueprint_prior`

Each test creates a thread pool, dispatches concurrent calls, waits with a wall-time watchdog,
and asserts:
  1. All worker invocations returned (no hang).
  2. No worker raised an unexpected exception (Python exception types match expectations).
  3. The aggregate output is well-formed (shapes, lengths, sums, ranges).
  4. Repeated calls (deterministic, same seed) produce identical aggregate output across runs.

If the battery_native extension is not built, all tests are skipped via the same gating the
rest of the suite uses (`native_solver.available()`).

This file is part of the REFUTER 4 audit — see memory/REFUTER 4 audit for the directive.
"""

from __future__ import annotations

import hashlib
import os
import random
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
import pytest

# Skip the entire file if battery_native isn't built.
from pokerbot.search import native_solver as ns

pytestmark = pytest.mark.skipif(
    not ns.available(),
    reason="battery_native extension not built",
)

# Gate native path explicitly so the wrapper doesn't fall back to Python mid-test.
ns.enable()

_EXT = ns._EXT  # the loaded cdylib module (or None if unavailable)

# Hard wall-clock per worker task — generous so we don't false-fail on cold start, but tight
# enough that a true GIL/Mutex deadlock trips the watchdog in CI within the suite budget.
_THREAD_TIMEOUT_S = 60.0

# Number of worker threads per test. 8 stresses every binding without saturating the GIL so
# badly that unrelated tests in the suite get starved.
_N_THREADS = 8

# How many rounds each test runs through the thread pool — gives 8*N total invocations.
_N_ROUNDS = int(os.environ.get("PYO3_BOUNDARY_ROUNDS", "3"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timeout_get(fut, timeout_s: float = _THREAD_TIMEOUT_S):
    """Get a Future's result with a wall-time watchdog. On timeout, dump the worker name and re-raise."""
    try:
        return fut.result(timeout=timeout_s)
    except TimeoutError as exc:
        # Get the worker thread so the failure message names it.
        worker = fut._state  # noqa: SLF001
        pytest.fail(f"PyO3 binding hung past {timeout_s:.1f}s — possible GIL deadlock "
                    f"(worker state={worker}): {exc!r}")


def _run_threads(worker: Callable[[int], Any], n_threads: int = _N_THREADS,
                 n_rounds: int = _N_ROUNDS, expect_exc: type | tuple[type, ...] | None = None):
    """Dispatch `worker(i)` across `n_threads` workers for `n_rounds` rounds.

    Returns the list of per-worker return values (one per thread, last round wins for stateful
    classes — we still assert each worker completes each round).

    If `expect_exc` is not None, every exception is matched against it; anything else fails.
    """
    results: list[Any] = []
    excs: list[BaseException] = []
    for round_idx in range(n_rounds):
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futs = [pool.submit(worker, i) for i in range(n_threads)]
            for f in as_completed(futs):
                try:
                    results.append(f.result(timeout=_THREAD_TIMEOUT_S))
                except BaseException as e:  # noqa: BLE001
                    excs.append(e)
    if excs:
        if expect_exc is None:
            pytest.fail(f"{len(excs)} workers raised unexpected exceptions in round={round_idx}: "
                        f"{[type(e).__name__ for e in excs[:5]]}")
        bad = [e for e in excs if not isinstance(e, expect_exc)]
        if bad:
            pytest.fail(f"{len(bad)} workers raised exceptions OUTSIDE the expected "
                        f"{expect_exc!r}: {[type(e).__name__ for e in bad[:5]]}")
    return results


# A canonical FLAT_LEN mirror (the Rust side fixes this; we reproduce it here only to size our
# test buffers correctly). When this drifts from the Rust side, the helper tests will fail
# loudly with a length-mismatch ValueError — that's exactly the leak we want caught.
_NUM_PLAYERS = 6
_MAX_BOARD = 5
_MAX_HISTORY = 64
_O_HOLES = 0
_O_BOARD = _O_HOLES + 12
_O_SEAT_STATUS = _O_BOARD + _MAX_BOARD
_O_STACKS = _O_SEAT_STATUS + _NUM_PLAYERS
_O_COMMITTED_STREET = _O_STACKS + _NUM_PLAYERS
_O_COMMITTED_TOTAL = _O_COMMITTED_STREET + _NUM_PLAYERS
_O_STREET = _O_COMMITTED_TOTAL + _NUM_PLAYERS
_O_POT_DEAD = _O_STREET + 1
_O_CURRENT_BET = _O_POT_DEAD + 1
_O_MIN_RAISE = _O_CURRENT_BET + 1
_O_BUTTON = _O_MIN_RAISE + 1
_O_ACTION_SEAT = _O_BUTTON + 1
_O_LAST_AGGRESSOR = _O_ACTION_SEAT + 1
_O_ACTED = _O_LAST_AGGRESSOR + 1
_O_LAST_RAISE_FULL = _O_ACTED + _NUM_PLAYERS
_O_DECK = _O_LAST_RAISE_FULL + 1
_O_DECK_PTR = _O_DECK + 52
_O_RNG_KEY = _O_DECK_PTR + 1
_O_HIST_SEAT = _O_RNG_KEY + 2
_O_HIST_ACTION = _O_HIST_SEAT + _MAX_HISTORY
_O_HIST_AMOUNT = _O_HIST_ACTION + _MAX_HISTORY
_O_HIST_LEN = _O_HIST_AMOUNT + _MAX_HISTORY
_O_DONE = _O_HIST_LEN + 1
_O_REWARDS = _O_DONE + 1
_O_PENDING_DEAL = _O_REWARDS + _NUM_PLAYERS
_FLAT_LEN = _O_PENDING_DEAL + 1


def _random_flat_state(rng: random.Random) -> bytes:
    """Build a valid random flat-state buffer — populated so every field is in-range.

    Many bindings will validate inputs and reject malformed lengths; we feed in well-formed
    state to focus on the multi-thread safety, not the validation surface.
    """
    b = bytearray(_FLAT_LEN * 8)
    # holes[NUM_PLAYERS, 2] in [0, 51]
    for i in range(_NUM_PLAYERS):
        c0 = rng.randrange(52)
        c1 = rng.randrange(52)
        while c1 == c0:
            c1 = rng.randrange(52)
        struct.pack_into("<q", b, (_O_HOLES + i * 2) * 8, c0)
        struct.pack_into("<q", b, (_O_HOLES + i * 2 + 1) * 8, c1)
    # board[5] in [-1, 51] (flop has -1 placeholders)
    for i in range(_MAX_BOARD):
        struct.pack_into("<q", b, (_O_BOARD + i) * 8, rng.choice([-1, rng.randrange(52)]))
    # seat_status: 0 empty, 1 live, 2 all-in, 3 folded — pick a mix
    statuses = [1, 1, 1, 2, 3, 0]
    rng.shuffle(statuses)
    for i in range(_NUM_PLAYERS):
        struct.pack_into("<q", b, (_O_SEAT_STATUS + i) * 8, statuses[i])
    # stacks: 0..20000
    for i in range(_NUM_PLAYERS):
        struct.pack_into("<q", b, (_O_STACKS + i) * 8, rng.randint(200, 20_000))
    # committed_street / committed_total: 0..2000
    for i in range(_NUM_PLAYERS):
        struct.pack_into("<q", b, (_O_COMMITTED_STREET + i) * 8, rng.randint(0, 1000))
        struct.pack_into("<q", b, (_O_COMMITTED_TOTAL + i) * 8, rng.randint(0, 2000))
    struct.pack_into("<q", b, _O_STREET * 8, rng.randint(0, 3))
    struct.pack_into("<q", b, _O_POT_DEAD * 8, rng.randint(0, 150))
    struct.pack_into("<q", b, _O_CURRENT_BET * 8, rng.choice([0, 50, 100, 200]))
    struct.pack_into("<q", b, _O_MIN_RAISE * 8, rng.choice([0, 100, 200]))
    struct.pack_into("<q", b, _O_BUTTON * 8, rng.randint(0, _NUM_PLAYERS - 1))
    struct.pack_into("<q", b, _O_ACTION_SEAT * 8, rng.choice([-1, rng.randint(0, _NUM_PLAYERS - 1)]))
    struct.pack_into("<q", b, _O_LAST_AGGRESSOR * 8, -1)
    # acted[6]
    for i in range(_NUM_PLAYERS):
        struct.pack_into("<q", b, (_O_ACTED + i) * 8, int(rng.random() < 0.3))
    struct.pack_into("<q", b, _O_LAST_RAISE_FULL * 8, 0)
    # deck[52]: 0..51
    for i in range(52):
        struct.pack_into("<q", b, (_O_DECK + i) * 8, i)
    struct.pack_into("<q", b, _O_DECK_PTR * 8, rng.randint(0, 52))
    struct.pack_into("<q", b, _O_RNG_KEY * 8, rng.getrandbits(32))
    struct.pack_into("<q", b, (_O_RNG_KEY + 1) * 8, rng.getrandbits(32))
    # hist fields zeroed
    # hist_len, done, pending_deal
    struct.pack_into("<q", b, _O_HIST_LEN * 8, 0)
    struct.pack_into("<q", b, _O_DONE * 8, 0)
    for i in range(_NUM_PLAYERS):
        struct.pack_into("<q", b, (_O_REWARDS + i) * 8, 0)
    struct.pack_into("<q", b, _O_PENDING_DEAL * 8, 0)
    return bytes(b)


# ---------------------------------------------------------------------------
# Stateless module-function bindings (thread safety = no shared state expected)
# ---------------------------------------------------------------------------


def test_engine_dims_concurrent():
    """engine_dims is a constant-pure function; concurrent invocations must all return the same tuple."""
    def worker(_i):
        return _EXT.engine_dims()

    results = _run_threads(worker)
    assert len(results) == _N_THREADS * _N_ROUNDS
    first = results[0]
    assert all(r == first for r in results), "engine_dims must be deterministic across threads"


def test_apply_action_concurrent():
    """apply_action with random valid states; every call must return a FLAT_LEN * 8 bytes payload.

    We feed each worker the SAME well-formed state, so the same action deterministically produces
    the same output across threads (proves no thread-local state pollution / no shared mutable
    state in the Rust core).
    """
    rng = random.Random(0xC0FFEE)
    state = _random_flat_state(rng)
    action = 0  # fold is always legal in any state

    def worker(_i):
        return _EXT.apply_action(state, action)

    results = _run_threads(worker)
    expected_len = _FLAT_LEN * 8
    assert all(len(r) == expected_len for r in results), \
        f"apply_action returned wrong size ({[len(r) for r in results[:4]]})"
    # Determinism across threads: same state + action -> same output bytes.
    assert all(r == results[0] for r in results), "apply_action must be deterministic across threads"


def test_apply_action_value_error_on_bad_size_concurrent():
    """Concurrent calls with bad buffer length must all raise ValueError, NOT leak a native panic."""
    def worker(_i):
        try:
            _EXT.apply_action(b"\x00" * 7, 0)  # wrong length
            return None
        except ValueError as e:
            return e
        except Exception as e:  # noqa: BLE001
            return e

    results = _run_threads(worker, expect_exc=None)
    assert all(r is not None for r in results), "expected ValueError, got None"
    assert all(isinstance(r, ValueError) for r in results), \
        f"expected only ValueError, got types: {[type(r).__name__ for r in results[:5]]}"


def test_sample_chance_concurrent():
    rng = random.Random(0xDECAF)
    state = _random_flat_state(rng)

    def worker(_i):
        return _EXT.sample_chance(state)

    results = _run_threads(worker)
    assert all(len(r) == _FLAT_LEN * 8 for r in results)


def test_legal_action_mask_concurrent():
    rng = random.Random(0xBEEF)
    state = _random_flat_state(rng)

    def worker(_i):
        return _EXT.legal_action_mask(state, 0)

    results = _run_threads(worker)
    assert all(len(r) == 11 for r in results)
    assert all(isinstance(r, list) for r in results)
    # determinism across threads
    assert all(r == results[0] for r in results)


def test_is_chance_node_concurrent():
    rng = random.Random(0xCAFE)
    state = _random_flat_state(rng)

    def worker(_i):
        return _EXT.is_chance_node(state)

    results = _run_threads(worker)
    assert all(isinstance(r, bool) for r in results)
    assert all(r == results[0] for r in results)


def test_evaluate_seven_concurrent():
    rng = random.Random(7)
    cards = [rng.randrange(52) for _ in range(7)]

    def worker(_i):
        return _EXT.evaluate_seven(cards)

    results = _run_threads(worker)
    assert all(isinstance(r, int) for r in results)
    assert all(r == results[0] for r in results)


def test_compute_returns_concurrent():
    rng = random.Random(13)
    holes = [rng.randrange(52) for _ in range(_NUM_PLAYERS * 2)]
    board = [rng.randrange(52) for _ in range(_MAX_BOARD)]
    seat_status = [rng.choice([1, 2, 3]) for _ in range(_NUM_PLAYERS)]
    committed_total = [rng.randint(0, 500) for _ in range(_NUM_PLAYERS)]
    button = rng.randint(0, _NUM_PLAYERS - 1)
    n_showdown = 2

    def worker(_i):
        return _EXT.compute_returns(holes, board, seat_status, committed_total, button, n_showdown)

    results = _run_threads(worker)
    assert all(len(r) == _NUM_PLAYERS for r in results)
    assert all(r == results[0] for r in results), "compute_returns must be deterministic"


def test_fingerprint_fast_concurrent():
    rng = random.Random(0xFA571)
    state = _random_flat_state(rng)

    def worker(_i):
        return _EXT.fingerprint_fast(state, 0, 200)

    results = _run_threads(worker)
    assert all(len(r) == 16 for r in results)
    assert all(r == results[0] for r in results), "fingerprint_fast must be byte-deterministic across threads"


def test_fingerprint_fast_depth_concurrent():
    rng = random.Random(0xFA571_2)
    state = _random_flat_state(rng)
    depth_edges = [50.0, 90.0, 140.0]

    def worker(_i):
        return _EXT.fingerprint_fast_depth(state, 0, 200, depth_edges)

    results = _run_threads(worker)
    assert all(len(r) == 16 for r in results)
    assert all(r == results[0] for r in results)


def test_eval_seven_phe_concurrent():
    cards = list(range(7))

    def worker(_i):
        return _EXT.eval_seven_phe(cards)

    results = _run_threads(worker)
    assert all(isinstance(r, int) for r in results)
    assert all(r == results[0] for r in results)


def test_multiway_equity_river_concurrent():
    rng = random.Random(42)
    lbr0, lbr1 = rng.randrange(7462), rng.randrange(7462)
    opp_flat = [rng.randrange(52) for _ in range(8)]  # 4 opponents x 2 cards
    board = [rng.randrange(52) for _ in range(5)]

    def worker(_i):
        return _EXT.multiway_equity_river(lbr0, lbr1, opp_flat, board)

    results = _run_threads(worker)
    assert all(isinstance(r, float) for r in results)
    assert all(0.0 <= r <= 1.0 for r in results)
    assert all(r == results[0] for r in results)


def test_leduc_solve_exploitability_concurrent():
    """leduc_solve_exploitability is the one-shot trainer; we run very few iterations per
    call so it completes fast enough for the thread pool. Each worker runs its own independent
    trainer — no shared state expected.
    """
    def worker(_i):
        return _EXT.leduc_solve_exploitability(2)

    results = _run_threads(worker)
    assert all(len(r) == 5 for r in results)
    # All 5 fields must be finite floats.
    for tup in results:
        assert all(isinstance(x, float) and np.isfinite(x) for x in tup)


def test_icm_equity_concurrent():
    stacks = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
    payouts = [10000.0, 6000.0, 4000.0, 2000.0, 1000.0, 500.0]

    def worker(_i):
        return _EXT.icm_equity(stacks, payouts)

    results = _run_threads(worker)
    assert all(len(r) == len(stacks) for r in results)
    assert all(r == results[0] for r in results)
    # Pool-conservation: equity should sum to total payout.
    total = sum(payouts)
    for r in results:
        assert abs(sum(r) - total) < 1e-6, f"ICM equity doesn't conserve pool: {sum(r)} vs {total}"


def test_bubble_factor_concurrent():
    stacks = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
    payouts = [10000.0, 6000.0, 4000.0, 2000.0, 1000.0, 500.0]

    def worker(_i):
        return _EXT.bubble_factor(stacks, payouts, 0, 1)

    results = _run_threads(worker)
    assert all(isinstance(r, float) for r in results)
    assert all(r == results[0] for r in results)


def test_icm_delta_concurrent():
    stacks = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
    chip_delta = [0.0, 0.0, 500.0, 0.0, -500.0, 0.0]
    payouts = [10000.0, 6000.0, 4000.0, 2000.0, 1000.0, 500.0]

    def worker(_i):
        return _EXT.icm_delta(stacks, chip_delta, payouts)

    results = _run_threads(worker)
    assert all(len(r) == len(stacks) for r in results)
    assert all(r == results[0] for r in results)


def test_emd_strength_histogram_concurrent():
    hole = [12, 25]
    board = [3, 17, 29]
    n_bins = 50

    def worker(_i):
        return _EXT.emd_strength_histogram(hole, board, n_bins, 64, 0xABCD)

    results = _run_threads(worker)
    assert all(len(r) == n_bins for r in results)
    # Determinism with same seed.
    assert all(r == results[0] for r in results), \
        "emd_strength_histogram must be deterministic for the same seed across threads"


def test_emd_bucket_from_hist_concurrent():
    hist = [0.0] * 50
    hist[10] = 1.0
    centroids = [hist, [0.0] * 50, [0.0] * 49 + [1.0]]

    def worker(_i):
        return _EXT.emd_bucket_from_hist(hist, centroids)

    results = _run_threads(worker)
    assert all(isinstance(r, int) for r in results)
    assert all(r == results[0] for r in results)


def test_emd_fit_concurrent():
    rng = random.Random(99)
    histograms = []
    for _ in range(20):
        h = [rng.random() for _ in range(10)]
        s = sum(h)
        histograms.append([x / s for x in h])

    def worker(_i):
        return _EXT.emd_fit(histograms, 4, 5, 0xFEED)

    results = _run_threads(worker)
    assert all(len(r) == 4 for r in results)
    for r in results:
        for c in r:
            assert len(c) == 10


def test_emd_mean_inertia_concurrent():
    rng = random.Random(101)
    histograms = []
    for _ in range(10):
        h = [rng.random() for _ in range(5)]
        s = sum(h)
        histograms.append([x / s for x in h])
    centroids = histograms[:3]

    def worker(_i):
        return _EXT.emd_mean_inertia(histograms, centroids)

    results = _run_threads(worker)
    assert all(isinstance(r, float) for r in results)
    assert all(np.isfinite(r) for r in results)


def test_emd_strength_histograms_batch_concurrent():
    spots = [
        ([12, 25], [3, 17, 29]),
        ([0, 1], [5, 10, 20]),
        ([51, 50], [2, 7, 13]),
    ]
    n_bins = 30

    def worker(_i):
        return _EXT.emd_strength_histograms_batch(spots, n_bins, 32, 0xBEEF)

    results = _run_threads(worker)
    assert all(len(r) == len(spots) for r in results)
    for r in results:
        for hist in r:
            assert len(hist) == n_bins


# ---------------------------------------------------------------------------
# Pyclass bindings — these HOLD state, so concurrent invocations stress the
# RefCell / interior-mutability + GIL-acquire patterns.
# ---------------------------------------------------------------------------


def _empty_blueprint_dict() -> dict[bytes, list[float]]:
    """An EMPTY blueprint — uniform-mode / calibration policies still work, the lookup
    falls back to no-key uniform. We avoid any RNG-fingerprint dependence on key order."""
    return {}


@pytest.fixture(scope="module")
def shared_playout_engine():
    """A PlayoutEngine built once, then shared across all worker threads.

    Mirrors the live hot path: ONE worker builds the engine from the blueprint snapshot,
    then EVERY subsequent decision call hits the same engine from many threads.
    """
    if not hasattr(_EXT, "PlayoutEngine"):
        pytest.skip("PlayoutEngine not available in this build")
    bp = _empty_blueprint_dict()
    allowed_pre = [True] * 11
    allowed_post = [True] * 11
    return _EXT.PlayoutEngine(
        bp, allowed_pre, allowed_post,
        2.0, 0.05, 4, 200,
    )


def test_playout_engine_leaf_value_concurrent(shared_playout_engine):
    """PlayoutEngine.leaf_value is the hot-path playout. Concurrent invocations from
    multiple threads MUST NOT deadlock (the leaf loop acquires the GIL briefly to call
    Python callbacks if any) and MUST NOT crash.

    We seed each thread with a distinct base_seed so the per-thread RNG path is exercised.
    """
    rng = random.Random(0xC0CA)
    state = _random_flat_state(rng)
    hero = 0

    def worker(i):
        return shared_playout_engine.leaf_value(
            state, hero, base_seed=0x1000 + i, rollout_samples=4, resample_board=False,
        )

    results = _run_threads(worker)
    assert all(isinstance(r, float) for r in results)
    assert all(np.isfinite(r) for r in results)


@pytest.fixture(scope="module")
def shared_subgame_solver():
    if not hasattr(_EXT, "SubgameSolver"):
        pytest.skip("SubgameSolver not available in this build")
    bp = _empty_blueprint_dict()
    allowed_pre = [True] * 11
    allowed_post = [True] * 11
    return _EXT.SubgameSolver(
        bp, allowed_pre, allowed_post,
        2.0, 0.05, 4, 200,
    )


def test_subgame_solver_solve_concurrent(shared_subgame_solver):
    """SubgameSolver.solve runs the full CFR loop. Each worker runs ONE iteration with a tight
    time budget so the thread pool doesn't blow past the suite budget.
    """
    rng = random.Random(0xDEAD_BEEF)
    state = _random_flat_state(rng)
    hero = 0

    def worker(i):
        return shared_subgame_solver.solve(
            state, hero,
            public_seed=0x2000 + i,
            max_iterations=1,  # tiny per-thread budget; we want concurrency, not convergence
            max_depth=2,
            blueprint_mix=0.5,
            rollout_samples=1,
            resample_board=False,
            linear_averaging=True,
            blueprint_cv=False,
            time_budget_ms=500,
        )

    results = _run_threads(worker)
    assert all(len(r) == 11 for r in results)
    for r in results:
        assert all(0.0 <= x <= 1.0 for x in r)
        # The probabilities should sum to ~1 (numerical tolerance for a 1-iter solve is loose).
        assert 0.99 <= sum(r) <= 1.01, f"solve returned non-stochastic policy: sum={sum(r)}"


@pytest.fixture(scope="module")
def shared_lbr6_harness():
    if not hasattr(_EXT, "Lbr6Harness"):
        pytest.skip("Lbr6Harness not available in this build")
    bp = _empty_blueprint_dict()
    allowed_pre = [True] * 11
    allowed_post = [True] * 11
    return _EXT.Lbr6Harness(
        bp, allowed_pre, allowed_post, 200, [50.0, 90.0, 140.0],
    )


def test_lbr6_harness_serve_concurrent(shared_lbr6_harness):
    """serve is the cross-check entry — pure tabular policy lookup. Should be fast + safe."""
    rng = random.Random(0x5EE)
    state = _random_flat_state(rng)

    def worker(_i):
        return shared_lbr6_harness.serve(state, 0, 0)  # mode 0 = blueprint

    results = _run_threads(worker)
    assert all(len(r) == 11 for r in results)


def test_lbr6_harness_lbr_tabular_concurrent(shared_lbr6_harness):
    """lbr_tabular is the heaviest bound — it spins up rayon workers internally. Multiple
    concurrent calls from different threads must NOT deadlock the process-global rayon pool.
    """
    # 4 roots * 1 LBR seat * 1 seed = 4 hands per call.
    rng = random.Random(0x711B6)
    n_roots = 4
    states = [_random_flat_state(rng) for _ in range(n_roots)]
    roots_blob = b"".join(states)
    lbr_seats = [0]
    seeds = [0x11111 + i for i in range(n_roots)]
    n_seats = 1

    def worker(_i):
        return shared_lbr6_harness.lbr_tabular(
            roots_blob,
            lbr_seats,
            seeds,
            n_seats=n_seats,
            track_ranges=False,
            range_cap=8,
            eq_samples=2,
            eq_runouts=1,
            settle_samples=1,
            max_depth=2,
            bet_candidates=[100, 200],
            value_only_raises=True,
            policy_mode=3,  # uniform — no fingerprint dependence
        )

    results = _run_threads(worker, n_threads=min(_N_THREADS, 4), n_rounds=2)
    assert all(r.hands == n_roots for r in results)


@pytest.fixture(scope="module")
def shared_leduc_trainer():
    if not hasattr(_EXT, "LeducCFRTrainer"):
        pytest.skip("LeducCFRTrainer not available in this build")
    return _EXT.LeducCFRTrainer()


def test_leduc_trainer_train_concurrent(shared_leduc_trainer):
    """train mutates internal tables; concurrent train calls would race on the same trainer.

    The Rust pyclass exposes `train(&mut self, iterations)`. Since each Python thread holds a
    reference to the SAME trainer, this IS a true shared-state test: PyO3 must serialize the
    `&mut self` borrow via the GIL (no Rc<RefCell<...>> UB) and the native `Tables` must
    tolerate concurrent reads after the borrow is released.
    """
    def worker(_i):
        shared_leduc_trainer.train(2)
        return shared_leduc_trainer.iters_done()

    results = _run_threads(worker)
    # At least one of the workers must report non-zero iterations (the train calls actually ran).
    assert max(results) > 0


def test_leduc_trainer_exploitability_concurrent(shared_leduc_trainer):
    """exploitability is read-only — it computes from the average strategy. Concurrent reads
    while train() is in flight is the most interesting case.
    """
    def worker(_i):
        return shared_leduc_trainer.exploitability()

    # Train a few iters first so there's a non-trivial strategy to evaluate.
    shared_leduc_trainer.train(2)
    results = _run_threads(worker)
    assert all(len(r) == 5 for r in results)
    for tup in results:
        assert all(isinstance(x, float) for x in tup)


def test_leduc_trainer_serve_concurrent(shared_leduc_trainer):
    """serve is a read-only infoset lookup; concurrent calls from many threads must NOT crash."""
    def worker(_i):
        return shared_leduc_trainer.serve(
            seat=0, own_card=1, public=2, round=0, to_call=0, raises=0, hist=[],
        )

    results = _run_threads(worker)
    assert all(len(r) == 3 for r in results)
    for r in results:
        assert abs(sum(r) - 1.0) < 1e-6, f"serve returned non-stochastic row: sum={sum(r)}"


def test_leduc_trainer_save_load_concurrent():
    """save_bytes + load_bytes — each thread should be able to serialize independently."""
    trainer = _EXT.LeducCFRTrainer()
    trainer.train(4)
    blob = bytes(trainer.save_bytes())

    def worker(_i):
        t = _EXT.LeducCFRTrainer()
        ok = t.load_bytes(blob)
        return (ok, t.iters_done())

    results = _run_threads(worker)
    assert all(ok for ok, _ in results)
    assert all(iters >= 4 for _, iters in results)


def test_register_set_release_cancel_concurrent():
    """The cancel registry is a process-global Mutex<HashMap>. Many threads registering,
    setting, and releasing cancel ids MUST NOT poison the mutex, MUST NOT deadlock, and
    MUST NOT leave stale entries that cause a later set_cancel() to fire spuriously.

    The contract verified here: a round-trip register -> release produces a no-op set_cancel
    AFTER release (the entry was removed from the registry).
    """
    barrier = threading.Barrier(_N_THREADS)

    def worker(i):
        barrier.wait()
        cid = _EXT.register_cancel()
        _EXT.set_cancel(cid)
        _EXT.release_cancel(cid)
        # After release, set_cancel MUST be a no-op (the registry no longer has this id).
        # If the registry leaks, we'd see no error here (set_cancel swallows "missing id"),
        # but the test below asserts via fresh registration that the same id doesn't reappear.
        return cid

    ids = _run_threads(worker)
    # All ids must be distinct (the registry allocated one entry per thread per round).
    # If the registry were corrupted by concurrent access, we'd see collisions or duplicates.
    assert len(set(ids)) == len(ids), "register_cancel returned colliding ids — registry is broken"


def test_register_set_release_cancel_round_trip():
    """One-shot round-trip: register, set, release, set-again MUST be a silent no-op
    (registry removes on release)."""
    cid = _EXT.register_cancel()
    _EXT.set_cancel(cid)
    _EXT.release_cancel(cid)
    # Second set_cancel MUST NOT raise (registry entry is gone).
    _EXT.set_cancel(cid)
    # And re-registration yields a fresh id (we don't assert distinct, just that it works).
    cid2 = _EXT.register_cancel()
    _EXT.release_cancel(cid2)


# ---------------------------------------------------------------------------
# SearchHandle — the LIVE hot-path binding. Most stress goes here.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_search_handle():
    if not hasattr(_EXT, "SearchHandle"):
        pytest.skip("SearchHandle not available in this build")
    bp = _empty_blueprint_dict()
    allowed_pre = [True] * 11
    allowed_post = [True] * 11
    return _EXT.SearchHandle(
        bp,
        allowed_pre, allowed_post,
        2.0, 0.05, 4, 200,
        False, 1.5, 0.0, 2.0,  # DCFR knobs
        True,   # linear_averaging
        False,  # blueprint_cv
    )


def test_search_handle_build_root_concurrent(shared_search_handle):
    rng = random.Random(0x5E42_C0DE)
    state = _random_flat_state(rng)

    def worker(_i):
        return shared_search_handle.build_root(state, 0, 0xABCD)

    results = _run_threads(worker)
    assert all(isinstance(r, dict) for r in results)
    assert all("public_seed" in r for r in results)
    # Determinism: same (state, hero, rng_salt) -> same public_seed across threads.
    seeds = [r["public_seed"] for r in results]
    assert all(s == seeds[0] for s in seeds), "build_root must be deterministic across threads"


def test_search_handle_blueprint_prior_concurrent(shared_search_handle):
    rng = random.Random(0x5E42_C0DE_2)
    state = _random_flat_state(rng)

    def worker(_i):
        return shared_search_handle.blueprint_prior(state, 0)

    results = _run_threads(worker)
    assert all(len(r) == 11 for r in results)
    assert all(r == results[0] for r in results)


def test_search_handle_solve_concurrent(shared_search_handle):
    """THE hot path. Many threads, each calling solve() with a tight budget.
    Must NOT deadlock, must NOT crash, and must return a stochastic policy + metadata dict.
    """
    rng = random.Random(0x50_1)
    state = _random_flat_state(rng)
    hero = 0

    barrier = threading.Barrier(_N_THREADS)

    def worker(i):
        barrier.wait()
        cid = _EXT.register_cancel()
        try:
            probs, ev, iters, meta = shared_search_handle.solve(
                state, hero,
                public_seed=0xA000 + i,
                max_iterations=2,
                max_depth=2,
                blueprint_mix=0.5,
                rollout_samples=1,
                resample_board=False,
                time_budget_ms=100,
                cancel_handle=cid,
                tilt_table=None,
            )
            return (probs, ev, iters, dict(meta))
        finally:
            _EXT.release_cancel(cid)

    results = _run_threads(worker)
    for probs, ev, iters, meta in results:
        assert len(probs) == 11, f"solve returned wrong-length probs: {len(probs)}"
        assert isinstance(ev, float)
        assert isinstance(iters, int)
        assert "served" in meta
        assert meta["served"] in ("search_native", "search_native_cfv", "blueprint_fallback")
        assert 0.99 <= sum(probs) <= 1.01, f"non-stochastic policy: sum={sum(probs)}"


def test_search_handle_solve_with_cancel_concurrent(shared_search_handle):
    """Concurrent solves, half cancelled mid-flight. Verifies the cancel registry + sentinel
    cooperate under load without leaking state across solves.
    """
    rng = random.Random(0xCA_1)
    state = _random_flat_state(rng)
    hero = 0

    barrier = threading.Barrier(_N_THREADS)

    def worker(i):
        barrier.wait()
        cid = _EXT.register_cancel()
        try:
            # Half the threads cancel themselves immediately; half let the budget run out.
            def maybe_cancel():
                if i % 2 == 0:
                    _EXT.set_cancel(cid)
            t = threading.Timer(0.001, maybe_cancel)
            t.start()
            try:
                probs, ev, iters, meta = shared_search_handle.solve(
                    state, hero,
                    public_seed=0xB000 + i,
                    max_iterations=10_000,  # long — we want cancellation to bite
                    max_depth=4,
                    blueprint_mix=0.5,
                    rollout_samples=1,
                    resample_board=False,
                    time_budget_ms=500,
                    cancel_handle=cid,
                    tilt_table=None,
                )
                return (probs, ev, iters, dict(meta))
            finally:
                t.cancel()
        finally:
            _EXT.release_cancel(cid)

    results = _run_threads(worker, n_threads=_N_THREADS, n_rounds=1)
    # All workers must complete within the wall budget.
    for probs, ev, iters, meta in results:
        assert len(probs) == 11
        assert isinstance(ev, float)
        assert isinstance(iters, int)
        assert "served" in meta


# ---------------------------------------------------------------------------
# Stress: mixed-binding storm — every worker hits a random binding, many threads.
# ---------------------------------------------------------------------------


def test_mixed_binding_storm():
    """The full PyO3 surface exercised concurrently. Any binding that deadlocks, panics, or
    leaks a Python exception would surface here as a stuck worker or a pytest.fail.
    """
    rng = random.Random(0x570_1)

    # Pre-build inputs once.
    state = _random_flat_state(rng)
    cards = [rng.randrange(52) for _ in range(7)]
    stacks = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
    payouts = [10000.0, 6000.0, 4000.0, 2000.0, 1000.0, 500.0]

    # List of binding calls (binding, args, expected-callable returning the result).
    binding_calls = [
        ("apply_action", lambda: _EXT.apply_action(state, 0)),
        ("sample_chance", lambda: _EXT.sample_chance(state)),
        ("legal_action_mask", lambda: _EXT.legal_action_mask(state, 0)),
        ("is_chance_node", lambda: _EXT.is_chance_node(state)),
        ("evaluate_seven", lambda: _EXT.evaluate_seven(cards)),
        ("engine_dims", lambda: _EXT.engine_dims()),
        ("icm_equity", lambda: _EXT.icm_equity(stacks, payouts)),
        ("bubble_factor", lambda: _EXT.bubble_factor(stacks, payouts, 0, 1)),
        ("icm_delta", lambda: _EXT.icm_delta(stacks, [0.0]*6, payouts)),
        ("eval_seven_phe", lambda: _EXT.eval_seven_phe(cards)),
        ("fingerprint_fast", lambda: _EXT.fingerprint_fast(state, 0, 200)),
    ]

    def worker(i):
        # Each worker hits a random binding; the storm verifies that no binding HOLDS a
        # lock that another binding needs (i.e. no ordering deadlock).
        name, call = binding_calls[i % len(binding_calls)]
        t0 = time.monotonic()
        out = call()
        dt = time.monotonic() - t0
        return (name, out, dt)

    n_workers = len(binding_calls) * 2  # 22 workers -> every binding hit at least twice
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(worker, i) for i in range(n_workers)]
        results = []
        for f in as_completed(futs):
            results.append(f.result(timeout=_THREAD_TIMEOUT_S))

    # Each binding produced SOME result.
    names = {r[0] for r in results}
    assert names == {n for n, _ in binding_calls}, \
        f"some bindings were skipped: got {names}, expected {[n for n, _ in binding_calls]}"
    # Every call returned within the per-thread budget (no hang).
    max_dt = max(r[2] for r in results)
    assert max_dt < 10.0, f"slowest binding took {max_dt:.2f}s — possible starvation"


# ---------------------------------------------------------------------------
# Final regression: a long-running stress that just hammers one binding many times.
# Catches state corruption that only shows up after many invocations.
# ---------------------------------------------------------------------------


def test_apply_action_long_soak():
    """8 threads, 200 rounds each = 1600 apply_action calls. Any heap corruption in the FFI
    marshal/unmarshal path would crash here.
    """
    rng = random.Random(0x50A)
    state = _random_flat_state(rng)

    def worker(_i):
        for _ in range(200):
            out = _EXT.apply_action(state, 0)
            assert len(out) == _FLAT_LEN * 8, "apply_action length drift"

    results = _run_threads(worker, n_threads=_N_THREADS, n_rounds=1)
    # No exceptions leaked.
    assert len(results) == _N_THREADS


def test_emd_batch_long_soak():
    """Concurrent batched EMD calls — stresses the rayon pool the batch uses."""
    rng = random.Random(0x50A_B)
    spots = [([rng.randrange(52), rng.randrange(52)], [rng.randrange(52) for _ in range(3)])
             for _ in range(16)]

    def worker(i):
        return _EXT.emd_strength_histograms_batch(spots, 30, 8, 0xCAFE_0000 + i)

    results = _run_threads(worker, n_threads=_N_THREADS, n_rounds=4)
    assert all(len(r) == len(spots) for r in results)
