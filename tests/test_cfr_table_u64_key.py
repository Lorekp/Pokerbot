"""Tests for the cfr_table.rs u64-keyed index switch (search-speed lever).

The `cfr_table` now uses a `HashMap<u64, RowIndex>` index keyed by the FNV-1a 64-bit fold
of the i64 infoset-key vector (see `cfr_table::infoset_key_u64`). The Python `PyCfrTable`
boundary folds the i64 tuple to u64 once per call (matches what `search_session::cfv` does
natively), so the Python-facing contract is unchanged: callers pass `Vec<i64>` keys, the
wrapper folds internally, the underlying storage is scalar.

This test asserts the three contract properties:
  (a) BIT-IDENTITY vs the legacy `HashMap<Vec<i64>, [f64;11]>` path on a fixed scenario.
      Same op stream applied to a fresh `PyCfrTable` and a parallel Python dict shadow must
      produce identical per-key row contents (the storage-layer migration is invisible to
      callers).
  (b) GET/INSERT ROUND-TRIP — every inserted key reads back exactly what was written,
      even after a `clear` + re-population (the generation bump + recycled-groups path).
  (c) THREAD-SAFETY SMOKE — two threads probing the same table concurrently must not crash,
      must not panic on Rust's borrow checker (we hold an immutable snapshot of the table
      via `PyCfrTable` reference-counted handles), and must produce the same final state as
      a single-threaded run with the same op stream. NOTE: PyCfrTable is wrapped behind the
      GIL, so concurrent access is serialised by the GIL — this test exercises the GIL
      re-entrancy + GIL-release patterns, NOT true multi-thread write races.

SKIPPED when the native PyCfrTable is not built (`cargo build --release --features python`).
"""

from __future__ import annotations

import random
import threading
import sys
from typing import Dict, List, Tuple

import pytest

# Native SubgameSolver + PlayoutEngine bindings live in `pokerbot.search.native_solver`. Reuse the
# extension's load path so we get the same DLL across all native tests in the repo.
from pokerbot.search import native_solver as ns

_EXT = ns._try_load()
native_only = pytest.mark.skipif(
    _EXT is None or not hasattr(_EXT, "PyCfrTable"),
    reason="native PyCfrTable not built (cargo build --release --features python)",
)

NUM_ACTIONS = 11  # mirror `core::NUM_ACTIONS`


def _make_op_stream(seed: int, n_ops: int) -> List[Tuple[str, Tuple[int, ...], List[float] | None]]:
    """Generate a deterministic op stream: list of (op_name, key_tuple, payload_or_none)."""
    rng = random.Random(seed)
    ops: List[Tuple[str, Tuple[int, ...], List[float] | None]] = []
    for _i in range(n_ops):
        u = rng.random()
        if u < 0.20:
            op = "clear"
            ops.append((op, (), None))
        elif u < 0.50:
            op = "insert"
            key = tuple(rng.randint(-(1 << 30), 1 << 30) for _ in range(rng.randint(1, 6)))
            row = [rng.random() for _ in range(NUM_ACTIONS)]
            ops.append((op, key, row))
        else:
            op = "add_into"
            if ops and ops[-1][0] in ("insert", "add_into"):
                key = ops[-1][1]
            else:
                key = tuple(rng.randint(-(1 << 30), 1 << 30) for _ in range(rng.randint(1, 6)))
            delta = [rng.uniform(-1e6, 1e6) for _ in range(NUM_ACTIONS)]
            ops.append((op, key, delta))
    return ops


def _apply_legacy(op_stream) -> Dict[Tuple[int, ...], List[float]]:
    """Legacy `HashMap<Vec<i64>, [f64; 11]>` shadow — applies the SAME ops with NO flooring."""
    table: Dict[Tuple[int, ...], List[float]] = {}
    for op, key, payload in op_stream:
        if op == "clear":
            table.clear()
        elif op == "insert":
            table[key] = list(payload)
        elif op == "add_into":
            cur = table.get(key)
            if cur is None:
                cur = list(payload)
                table[key] = cur
            else:
                for i, v in enumerate(payload):
                    cur[i] += v
    return table


def _apply_aosoa(op_stream) -> "object":
    """Native AoSoA PyCfrTable — folds Vec<i64> -> u64 at the wrapper boundary."""
    tab = _EXT.PyCfrTable(0)
    for op, key, payload in op_stream:
        if op == "clear":
            tab.clear()
        elif op == "insert":
            tab.insert(list(key), list(payload))
        elif op == "add_into":
            tab.add_into(list(key), list(payload))
    return tab


# ---- (a) bit-identity vs legacy Vec<i64> shadow ----------------------------------------------

@native_only
def test_u64_keyed_aosoa_matches_legacy_vec_i64_path_bit_for_bit():
    """Every key's row content in the u64-keyed AoSoA PyCfrTable must match the legacy
    Vec<i64>-keyed dict shadow bit-for-bit under the SAME op stream."""
    seed = 0xDEADBEEF
    n_ops = 4_000
    stream = _make_op_stream(seed, n_ops)

    legacy = _apply_legacy(stream)
    aosoa = _apply_aosoa(stream)

    # Live key counts must match.
    assert aosoa.n_rows() == len(legacy), (
        f"key count mismatch: aosoa={aosoa.n_rows()} legacy={len(legacy)}"
    )

    # Per-key row equality (bit-identical: 11-element list compare with `==`).
    mismatches = 0
    for key, want in legacy.items():
        got = aosoa.get_or_zero(list(key))
        if list(got) != list(want):
            mismatches += 1
            if mismatches <= 3:
                print(f"  key={key} want={want} got={got}", file=sys.stderr)
    assert mismatches == 0, f"{mismatches} rows diverged (seed={seed})"


# ---- (b) get/insert round-trip across clear() -------------------------------------------------

@native_only
def test_get_insert_round_trip_with_clear_recycles():
    """Insert keys, verify they read back, `clear()` to bump the generation, then re-populate
    a different key set and verify the recycled group heap serves the new rows correctly."""
    tab = _EXT.PyCfrTable(64)

    # Phase 1: insert 32 keys with a sentinel.
    phase1 = []
    for i in range(32):
        key = (i, i * 7 + 1, i * 13 + 5)
        row = [float(a + i * 100) for a in range(NUM_ACTIONS)]
        tab.insert(list(key), row)
        phase1.append((key, row))

    assert tab.n_rows() == 32

    # Re-read every phase1 key — must match what we wrote (no truncation / no generation churn).
    for key, want in phase1:
        got = tab.get_or_zero(list(key))
        assert list(got) == list(want), f"phase1 key {key} diverged: {got} vs {want}"

    # Phase 2: clear, then populate a NEW set of 16 keys (must NOT collide with phase1 row
    # contents because the generation bump makes the old rows stale).
    tab.clear()
    assert tab.n_rows() == 0
    phase2 = []
    for i in range(16):
        key = (-i, -i * 11 - 1, -i * 17 - 5)
        row = [float(a + i * 1000) for a in range(NUM_ACTIONS)]
        tab.insert(list(key), row)
        phase2.append((key, row))

    assert tab.n_rows() == 16

    # Phase1 keys now MISS (all-zero).
    for key, _ in phase1:
        got = tab.get_or_zero(list(key))
        assert list(got) == [0.0] * NUM_ACTIONS, (
            f"phase1 key {key} should miss post-clear, got {got}"
        )

    # Phase2 keys HIT and match.
    for key, want in phase2:
        got = tab.get_or_zero(list(key))
        assert list(got) == list(want), f"phase2 key {key} diverged: {got} vs {want}"


# ---- (c) thread-safety smoke ----------------------------------------------------------------

@native_only
def test_concurrent_probes_dont_crash_and_final_state_matches_single_threaded():
    """Two threads probe the same pre-populated PyCfrTable concurrently (under the GIL).
    The GIL serialises the Python-side calls, so this is a smoke test for re-entrancy rather
    than a true multi-thread race. The final state must agree with a single-threaded replay."""
    seed = 0xBEEF_C0DE
    rng = random.Random(seed)
    pre_pop_keys = []
    for i in range(2_000):
        k = tuple(rng.randint(-(1 << 30), 1 << 30) for _ in range(rng.randint(1, 5)))
        pre_pop_keys.append(k)

    # Pre-populate ONE table (this is the reference).
    ref = _EXT.PyCfrTable(len(pre_pop_keys))
    for (i, k) in enumerate(pre_pop_keys):
        row = [float(a + i * 11) for a in range(NUM_ACTIONS)]
        ref.insert(list(k), row)

    # Build a CONCURRENT-PROBE table with the same pre-populated keys.
    shared = _EXT.PyCfrTable(len(pre_pop_keys))
    for (i, k) in enumerate(pre_pop_keys):
        row = [float(a + i * 11) for a in range(NUM_ACTIONS)]
        shared.insert(list(k), row)

    n_per_thread = 500
    probe_keys = pre_pop_keys[:200]  # subset for the per-thread probes

    errors: List[str] = []

    def worker():
        try:
            for _ in range(n_per_thread):
                for k in probe_keys:
                    row = shared.get_or_zero(list(k))
                    # Sanity: row[0] must match the pre-populated sentinel.
                    expected_0 = float(probe_keys.index(k) * 11)
                    assert abs(row[0] - expected_0) < 1e-9, (
                        f"row[0] mismatch on key {k}: got {row[0]} want {expected_0}"
                    )
        except Exception as e:
            errors.append(f"{threading.current_thread().name}: {e!r}")

    threads = [threading.Thread(target=worker, name=f"probe-{i}") for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # No thread crashed.
    assert not errors, f"concurrent probe thread errors: {errors}"

    # Final state: every pre-populated key still reads back exactly what we wrote.
    for (i, k) in enumerate(pre_pop_keys):
        want = [float(a + i * 11) for a in range(NUM_ACTIONS)]
        got = shared.get_or_zero(list(k))
        assert list(got) == list(want), (
            f"post-concurrent key {k} diverged: got={got} want={want}"
        )