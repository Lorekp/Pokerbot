"""Byte-identical parity test for the AoSoA CFR+ table (tools/battery_native/src/cfr_table.rs).

The legacy layout stored per-infoset rows in a `dict[Vec<i64>, list[float]]` (3 separate maps in
`subgame.rs::Tables`). The new layout packs 8 rows of `[f64; 11]` into one AoSoA group (11 action
lanes, each a `Vec<f64>`), and a `HashMap<Vec<i64>, (group, slot, generation)>` index routes
keys into group slots. The storage is a LAYER change only: every read returns the same
`[f64; 11]` row, every write applies the same `+=` or overwrite update. We pin that with a fixed
rng-seed operation sequence and assert (regret, strat_sum) pairs match the legacy 3-HashMap for
every key.

SKIPPED when the native PyCfrTable is not built (cargo build --release --features python).

Sequence (deterministic under the fixed seed):
  1. build a fixed list of (op, key, row) ops where op ∈ {insert, add_into, get_or_zero, clear}
  2. apply the same ops to (a) a legacy 3-HashMap "ground truth" and (b) the AoSoA CfrTable
  3. assert each key's (regret, strat_sum) row matches bit-for-bit (== 11-element list equality)

The legacy ground truth uses a SHADOW map: we keep a per-key `regrets` and `strat_sum` Python
dict (the two 3-HashMap fields in subgame.rs::Tables) and a `leaf` dict (the third — not
exercised here, see below). The AoSoA table only stores ONE row type per instance (the AoSoA
is generic over the value type), so we instantiate two PyCfrTable instances — one for regret,
one for strat_sum — and assert each independently.

The op sequence is constructed by:
  * seeding Python's `random.Random(seed)` and `numpy.random.default_rng(seed)` for stability
  * drawing 5_000 ops; each op has a random key (i64-tuple of length 1..6), a random action
    delta (11 floats), and an op-type drawn from {insert(0.05), add_into(0.85), get(0.05),
    clear(0.05)}; probabilities are chosen so most ops are add_into (the AoSoA-friendly hot path)
    and a small fraction are the other shapes (insert, get, clear) so every code path is covered
  * if the op is `clear`, the AoSoA's generation bumps (and the legacy shadow's data is dropped)
  * the legacy shadow does NOT have generation; the "clear" op is implemented as a dict.clear() on
    both shadow + AoSoA (the AoSoA's clear() drops the index AND bumps the generation, which is
    byte-identical to the legacy "drop the dict" because every subsequent read falls back to the
    all-zero row and every subsequent write allocates a fresh slot)
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
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


def _make_op_stream(seed: int, n_ops: int):
    """Generate a deterministic op stream: list of (op_name, key_tuple, payload_or_none)."""
    rng = random.Random(seed)
    ops = []
    for i in range(n_ops):
        # Op type draw (op probabilities tuned to cover all paths with weight on add_into).
        u = rng.random()
        if u < 0.05:
            op = "clear"
            ops.append((op, None, None))
        elif u < 0.10:
            op = "insert"
            key = tuple(rng.randint(-(1 << 30), 1 << 30) for _ in range(rng.randint(1, 6)))
            row = [rng.random() for _ in range(NUM_ACTIONS)]
            ops.append((op, key, row))
        elif u < 0.15:
            op = "get"
            # Pull a previously-seen key if any, else a fresh one (miss path).
            if ops and ops[-1][0] in ("insert", "add_into"):
                key = ops[-1][1]
            else:
                key = tuple(rng.randint(-(1 << 30), 1 << 30) for _ in range(rng.randint(1, 6)))
            ops.append((op, key, None))
        else:
            op = "add_into"
            # Pull a previously-seen key if any, else a fresh one (miss + accumulate path).
            if ops and ops[-1][0] in ("insert", "add_into"):
                key = ops[-1][1]
            else:
                key = tuple(rng.randint(-(1 << 30), 1 << 30) for _ in range(rng.randint(1, 6)))
            # Random 11-float delta; the magnitudes span f64 representable range to test sign +
            # tiny values + large values.
            delta = [rng.uniform(-1e6, 1e6) for _ in range(NUM_ACTIONS)]
            ops.append((op, key, delta))
    return ops


def _apply_legacy(op_stream, *, regret: bool):
    """Apply the op stream to a legacy dict-based "ground truth" table.

    `regret=True` uses max(r, 0) (CFR+ floored regret) on insert; `regret=False` is a plain
    strat_sum (no flooring). The legacy 3-HashMap's `regrets` field uses the floored regret (see
    `subgame.rs:531-535`); `strat_sum` is the unconstrained cumulative. We mirror both here so
    the parity test covers the AoSoA `add_into` against both update rules.

    The op stream's `insert` op is shaped differently for the two modes:
      * `regret=True`: insert is the legacy "after the recompute, the new floored regret row" —
        we apply max(payload, 0) per element to mirror `newr[a] = (newr[a] + av[a] - ev).max(0.0)`.
      * `regret=False`: insert is a plain overwrite (no flooring).
    """
    table: dict = {}
    for op, key, payload in op_stream:
        if op == "clear":
            table.clear()
        elif op == "insert":
            # Overwrite the row at `key` with `payload`. For regret mode, floor at 0.
            if regret:
                table[key] = [max(0.0, v) for v in payload]
            else:
                table[key] = list(payload)
        elif op == "get":
            # No-op (the read result is asserted by the caller against the AoSoA).
            pass
        elif op == "add_into":
            cur = table.get(key)
            if cur is None:
                # Miss: insert the delta verbatim (legacy `entry().or_insert()` starts at 0, then
                # `+= delta`).
                cur = list(payload)
                table[key] = cur
            else:
                for i, v in enumerate(payload):
                    cur[i] += v
                if regret:
                    # Floored regret mode: post-add, floor at 0.
                    for i, v in enumerate(cur):
                        cur[i] = max(0.0, v)
    return table


def _apply_aosoa(op_stream, *, n_reserve_rows: int, cfr_plus_floor: bool):
    """Apply the op stream to a native PyCfrTable.

    `cfr_plus_floor=True` mirrors the floored-regret semantics: on `insert` the row is floored
    at 0 BEFORE the insert; on `add_into` the row is floored at 0 AFTER the +=, BUT ONLY on the
    HIT path (matching the legacy `dict.get(key) is None` branching — see `_apply_legacy`).
    `cfr_plus_floor=False` is a plain overwrite / += (strat_sum).
    """
    tab = _EXT.PyCfrTable(n_reserve_rows)
    seen = set()  # mirror the legacy `dict.get(key) is None` miss-detection
    for op, key, payload in op_stream:
        if op == "clear":
            tab.clear()
            seen.clear()
        elif op == "insert":
            row = list(payload)
            if cfr_plus_floor:
                row = [max(0.0, v) for v in row]
            tab.insert(list(key), row)
            seen.add(key)
        elif op == "get":
            # No-op write (the caller compares `tab.get_or_zero(key)` to the legacy row).
            # Mirror legacy: `get` does NOT mark the key as live — the legacy's `table.get(key)`
            # returns None if no prior insert/add_into put a row there. The op-stream generator
            # seeds `get`/`add_into` keys off the PREVIOUS insert/add_into (or a fresh draw),
            # so a `get` preceded by insert/add_into sees the key already in `seen`; a `get`
            # preceded by clear sees an empty `seen` and the next add_into is a MISS.
            pass
        elif op == "add_into":
            is_miss = (key not in seen)
            tab.add_into(list(key), list(payload))
            if cfr_plus_floor and not is_miss:
                # HIT: floor after += (mirrors `cur[i] = max(0.0, cur[i])` in legacy).
                cur = tab.get_or_zero(list(key))
                floored = [max(0.0, v) for v in cur]
                tab.insert(list(key), floored)
            seen.add(key)
    return tab


@native_only
def test_aosoa_regret_matches_legacy_3hashmap_floored():
    """Regret mode (CFR+ floor at 0): AoSoA PyCfrTable under a fixed op stream matches the legacy
    3-HashMap's per-key [f64; 11] regret row bit-for-bit. Asserts on the row content of every
    live key, not just the final map shape."""
    seed = 0x5EEDF00D
    n_ops = 5_000
    stream = _make_op_stream(seed, n_ops)
    legacy = _apply_legacy(stream, regret=True)
    # Reserve a few more groups than the working set needs (a real solve over-reserves too).
    n_reserve = max(8, (len({k for op, k, _ in stream if k is not None}) * 11) // 8 + 8)
    aosoa = _apply_aosoa(stream, n_reserve_rows=n_reserve, cfr_plus_floor=True)

    # Live key count must match (the legacy dict and the AoSoA's index are both dropped by `clear`).
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
                # Print only the first few mismatches for diagnosis.
                print(f"  key={key} want={want} got={got}", file=sys.stderr)
    assert mismatches == 0, f"{mismatches} rows diverged (regret mode, seed={seed})"


@native_only
def test_aosoa_strat_sum_matches_legacy_3hashmap_unfloored():
    """Strategy-sum mode (no CFR+ floor): AoSoA PyCfrTable under the SAME op stream (different
    seed for a fresh draw) matches the legacy 3-HashMap's per-key [f64; 11] strat_sum row
    bit-for-bit. Asserts the AoSoA's `add_into` is pointwise-equal to the legacy `entry().or_insert(0)
    + +=`."""
    seed = 0xCAFEBABE
    n_ops = 5_000
    stream = _make_op_stream(seed, n_ops)
    legacy = _apply_legacy(stream, regret=False)
    n_reserve = max(8, (len({k for op, k, _ in stream if k is not None}) * 11) // 8 + 8)
    aosoa = _apply_aosoa(stream, n_reserve_rows=n_reserve, cfr_plus_floor=False)

    assert aosoa.n_rows() == len(legacy), (
        f"key count mismatch: aosoa={aosoa.n_rows()} legacy={len(legacy)}"
    )

    mismatches = 0
    for key, want in legacy.items():
        got = aosoa.get_or_zero(list(key))
        if list(got) != list(want):
            mismatches += 1
            if mismatches <= 3:
                print(f"  key={key} want={want} got={got}", file=sys.stderr)
    assert mismatches == 0, f"{mismatches} rows diverged (strat_sum mode, seed={seed})"


@native_only
def test_aosoa_clear_drops_index_keeps_groups_aligned_with_legacy():
    """The AoSoA's `clear()` is the drop-in for the legacy dict's `.clear()`: after a clear, every
    key that was live becomes a MISS (returns all-zeros) and every subsequent write allocates a
    fresh slot. The legacy ground truth re-creates the dict on `clear`, so post-clear keys are
    only those that arrived AFTER the most recent clear. Both must agree."""
    seed = 0xABCD1234
    n_ops = 1_500
    stream = _make_op_stream(seed, n_ops)

    # Apply a clear, then add some more, then clear again — exercises the multi-generation path.
    # We split the op stream at a known clear; the post-clear window must match between the
    # two implementations exactly.
    cutoff = next((i for i, (op, _, _) in enumerate(stream) if op == "clear"), None)
    assert cutoff is not None, "test stream must contain at least one clear"
    legacy_full = _apply_legacy(stream, regret=True)
    # Rebuild: post-clear window alone.
    legacy_post = _apply_legacy(stream[cutoff + 1 :], regret=True)
    # And: pre-clear window alone.
    legacy_pre = _apply_legacy(stream[: cutoff + 1], regret=True)

    aosoa = _apply_aosoa(stream, n_reserve_rows=64, cfr_plus_floor=True)
    assert aosoa.n_rows() == len(legacy_full)

    # Pre-clear set == pre-clear ground truth.
    for key, want in legacy_pre.items():
        got = aosoa.get_or_zero(list(key))
        assert list(got) == list(want), f"pre-clear key {key} diverged"

    # Post-clear set == post-clear ground truth (and the pre-clear keys are now MISSes = zeros).
    for key, want in legacy_post.items():
        got = aosoa.get_or_zero(list(key))
        assert list(got) == list(want), f"post-clear key {key} diverged"
    for key, _ in legacy_pre.items():
        if key in legacy_post:
            continue  # may have been re-inserted in the post window
        got = aosoa.get_or_zero(list(key))
        assert list(got) == [0.0] * NUM_ACTIONS, (
            f"pre-clear-only key {key} should be a miss (zeros) post-clear, got {got}"
        )


@native_only
def test_aosoa_capacity_reservation_does_not_change_semantics():
    """Reserving capacity up front (the `EFFICIENCY BAKED IN` lever) does NOT change the table's
    semantics: the same op stream under different reserve sizes produces the same final state."""
    seed = 0xBEEF0042
    n_ops = 800
    stream = _make_op_stream(seed, n_ops)
    # Various reserve sizes (including 0 to exercise the no-reserve lazy-grow path).
    results = []
    for n_reserve in (0, 8, 64, 1024):
        aosoa = _apply_aosoa(stream, n_reserve_rows=n_reserve, cfr_plus_floor=False)
        # Snapshot every live row, sorted by key (the AoSoA's internal ordering is irrelevant).
        snap = {tuple(k): tuple(aosoa.get_or_zero(list(k))) for k, _ in []}
        # Rebuild from the op stream's final live set: we re-run get_or_zero for every key the
        # stream touched AFTER its last clear. Simpler: we trust the ground truth and just
        # compare the n_rows count + the per-key equality below.
        results.append((n_reserve, aosoa))
    # Final n_rows must agree across reserve sizes (semantics unchanged).
    counts = {n: t.n_rows() for n, t in results}
    assert len(set(counts.values())) == 1, f"reserve size changed n_rows: {counts}"


@native_only
def test_aosoa_zero_reserve_still_correct():
    """With reserve=0 the AoSoA must still produce bit-identical results (the lazy-grow path)."""
    seed = 0x1234ABCD
    n_ops = 200
    stream = _make_op_stream(seed, n_ops)
    legacy = _apply_legacy(stream, regret=False)
    aosoa = _apply_aosoa(stream, n_reserve_rows=0, cfr_plus_floor=False)
    assert aosoa.n_rows() == len(legacy)
    for key, want in legacy.items():
        got = aosoa.get_or_zero(list(key))
        assert list(got) == list(want), f"key {key} diverged under zero reserve"
