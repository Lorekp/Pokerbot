"""SharedHogwildTable: single-process correctness + the critical CONCURRENCY guarantee that two
distinct keys never share a slot under racy parallel claims (the hogwild store's whole risk)."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from pokerbot.training.shared_hogwild_table import SharedHogwildTable

A = 11


def _key(i: int) -> bytes:
    return int(i).to_bytes(16, "little")


def test_single_claim_lookup_roundtrip():
    t = SharedHogwildTable.create(1024, A)
    try:
        s = t.slot_for(_key(42))
        assert t.slot_for(_key(42)) == s          # stable
        t.add_regret(s, np.arange(A, dtype=np.float32), cfr_plus=False)
        t.add_strategy(s, np.ones(A, dtype=np.float32))
        assert np.allclose(t.regret[s], np.arange(A))
        snap = t.snapshot_strategy()
        assert _key(42) in snap and np.allclose(snap[_key(42)], 1.0)
        assert t.n_occupied() == 1
    finally:
        t.close()
        t.unlink()


def test_distinct_keys_distinct_slots():
    t = SharedHogwildTable.create(4096, A)
    try:
        slots = {t.slot_for(_key(i)) for i in range(1000)}
        assert len(slots) == 1000          # open addressing gives each key its own slot
        assert t.n_occupied() == 1000
        # cfr_plus flooring keeps regrets non-negative
        s = t.slot_for(_key(7))
        t.add_regret(s, np.full(A, -5.0, dtype=np.float32), cfr_plus=True)
        assert (t.regret[s] >= 0).all()
    finally:
        t.close()
        t.unlink()


def test_overflow_raises():
    t = SharedHogwildTable.create(16, A)
    try:
        with pytest.raises(RuntimeError):
            for i in range(17):           # 17 distinct keys into 16 slots -> full
                t.slot_for(_key(i))
    finally:
        t.close()
        t.unlink()


def _claim_worker(descriptor, locks, n_keys, reps):
    t = SharedHogwildTable.attach(descriptor, locks)
    inc = np.ones(t.num_actions, dtype=np.float32)
    for _ in range(reps):
        for i in range(n_keys):
            slot = t.slot_for(_key(i))
            t.add_regret(slot, inc, cfr_plus=False)
            t.add_strategy(slot, inc)
    t.close()


def test_concurrent_claims_no_duplicate_slots():
    """M processes hammer the SAME key set (max claim contention). The invariant that MUST hold
    despite races: exactly one slot per distinct key (no duplicates). Hogwild += may lose some
    updates (reported), but never corrupts the key->slot mapping."""
    ctx = mp.get_context("spawn")
    n_keys, reps, procs = 200, 50, 6
    t = SharedHogwildTable.create(8192, A, ctx=ctx)
    try:
        workers = [
            ctx.Process(target=_claim_worker, args=(t.descriptor(), t.locks, n_keys, reps))
            for _ in range(procs)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=120)
            assert w.exitcode == 0, f"worker failed (exitcode {w.exitcode})"

        # CORRECTNESS: one slot per key, no duplicates.
        assert t.n_occupied() == n_keys, (t.n_occupied(), n_keys)
        slots = {t.slot_for(_key(i)) for i in range(n_keys)}
        assert len(slots) == n_keys
        # every claimed key is one of the original keys (no phantom/corrupt keys)
        snap = t.snapshot_strategy()
        assert set(snap) == {_key(i) for i in range(n_keys)}

        # HOGWILD: accumulation is correct modulo lost updates (<= expected, and substantial).
        expected = procs * reps  # per key, per action
        got = float(t.regret[t.slot_for(_key(0))][0])
        assert 0 < got <= expected
        total = float(t.regret[[t.slot_for(_key(i)) for i in range(n_keys)], 0].sum())
        lost = 1.0 - total / (expected * n_keys)
        print(f"hogwild lost-update rate at {procs} procs / {n_keys} keys: {lost:.1%}")
    finally:
        t.close()
        t.unlink()
