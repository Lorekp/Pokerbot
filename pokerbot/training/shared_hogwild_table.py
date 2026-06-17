"""Lock-free (hogwild) shared-memory regret/strategy store for parallel MCCFR.

The key to saturating many cores on one machine (Open Pure CFR / Pluribus): all workers share ONE
regret + strategy-sum table and update it IN PLACE — no per-iteration barrier, merge, or reload. The
table is an open-addressing hash map in ``multiprocessing.shared_memory`` keyed by the 16-byte infoset
fingerprint (stored as two uint64 lo/hi, so snapshots reconstruct keys and false matches are
impossible):

  * Hot path (a known infoset) is LOCK-FREE: hash → linear-probe ``occupied``/key arrays, then a racy
    ``regret[slot] += δ`` (the accepted hogwild noise — provably harmless for external-sampling MCCFR).
  * First-visit CLAIM of an empty slot takes a per-slot lock (pool of N locks, indexed slot % N) and
    re-checks under it, so two distinct keys never share a slot (the lock is a memory barrier: a
    stale lock-free "empty" read just causes a harmless re-check, never a duplicate). After warmup all
    infosets are claimed → no locking → near-linear scaling.

Capacity is FIXED and generous (the bounded abstraction has ~64k infosets); overflow past the load
factor is a hard error (size up), never silent corruption. ~6 MB for 64k×11×2 float32 tables.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

_MIX = 0x9E3779B97F4A7C15  # 64-bit odd constant for mixing lo/hi into the start slot


def dcfr_weights(t: float, alpha: float = 1.5, beta: float = 0.0, gamma: float = 2.0) -> tuple[float, float, float]:
    """Discounted-CFR (Brown & Sandholm 2019) discount factors for iteration ``t`` (>=1).

    Returns ``(pos, neg, strat)``: positive regrets are multiplied by ``t^a/(t^a+1)``, negative by
    ``t^b/(t^b+1)``, and the strategy-sum by ``(t/(t+1))^g``. The empirically-best (alpha=1.5, beta=0,
    gamma=2) discounts old positive regret mildly, halves negative regret each step (beta=0 -> 0.5), and
    down-weights early (bad) strategy contributions — 2-3x faster than CFR+ in NLHE-like domains.
    """
    t = max(float(t), 1.0)
    ta = t ** alpha
    tb = t ** beta
    pos = ta / (ta + 1.0)
    neg = tb / (tb + 1.0)
    strat = (t / (t + 1.0)) ** gamma
    return float(pos), float(neg), float(strat)


class SharedHogwildTable:
    """Open-addressing regret/strategy store in shared memory. Build with ``create()`` in the parent;
    workers re-attach via ``attach(descriptor, locks)``. ``regret``/``strat`` are (capacity, A) numpy
    views over shared buffers; ``slot_for(key16)`` maps a 16-byte fingerprint to its row."""

    def __init__(self, capacity: int, num_actions: int, *, _shms: dict, locks, n_locks: int, owner: bool):
        self.capacity = int(capacity)
        self.num_actions = int(num_actions)
        self.n_locks = int(n_locks)
        self.locks = locks
        self._owner = owner
        self._shms = _shms  # {name: SharedMemory}
        cap, a = self.capacity, self.num_actions
        self.occ = np.ndarray((cap,), np.uint8, buffer=_shms["occ"].buf)
        self.klo = np.ndarray((cap,), np.uint64, buffer=_shms["klo"].buf)
        self.khi = np.ndarray((cap,), np.uint64, buffer=_shms["khi"].buf)
        # float64 (not float32): the strategy-sum grows large under linear weighting and float32 loses
        # the small early increments → corrupts the average (measured: 97 mbb vs <5 on Leduc). 8 bytes
        # × bounded infosets is still tiny (~11 MB for 64k×11×2). Racy 8-byte RMW = accepted hogwild noise.
        self.regret = np.ndarray((cap, a), np.float64, buffer=_shms["reg"].buf)
        self.strat = np.ndarray((cap, a), np.float64, buffer=_shms["str"].buf)
        # Optional VR-MCCFR per-(infoset,action) baseline (Schmid 2019). Allocated only when
        # with_baseline; default path is byte-identical to before (no extra segment).
        self.base = (
            np.ndarray((cap, a), np.float64, buffer=_shms["base"].buf) if "base" in _shms else None
        )

    # ---- construction ----
    @classmethod
    def create(cls, capacity: int, num_actions: int, *, n_locks: int = 256, ctx=None,
               with_baseline: bool = False) -> "SharedHogwildTable":
        ctx = ctx or mp.get_context("spawn")
        cap, a = int(capacity), int(num_actions)
        sizes = {"occ": cap, "klo": cap * 8, "khi": cap * 8, "reg": cap * a * 8, "str": cap * a * 8}
        if with_baseline:
            sizes["base"] = cap * a * 8
        shms = {k: shared_memory.SharedMemory(create=True, size=max(1, sz)) for k, sz in sizes.items()}
        locks = [ctx.Lock() for _ in range(n_locks)]
        t = cls(cap, a, _shms=shms, locks=locks, n_locks=n_locks, owner=True)
        t.occ[:] = 0
        t.klo[:] = 0
        t.khi[:] = 0
        t.regret[:] = 0.0
        t.strat[:] = 0.0
        if t.base is not None:
            t.base[:] = 0.0
        return t

    def descriptor(self) -> dict:
        """Picklable handle for workers: shm names + shape. Pass the lock list separately via the
        worker Process args (mp locks are shareable across spawn that way)."""
        return {
            "names": {k: s.name for k, s in self._shms.items()},
            "capacity": self.capacity,
            "num_actions": self.num_actions,
            "n_locks": self.n_locks,
        }

    @classmethod
    def attach(cls, descriptor: dict, locks) -> "SharedHogwildTable":
        shms = {k: shared_memory.SharedMemory(name=n) for k, n in descriptor["names"].items()}
        return cls(
            descriptor["capacity"], descriptor["num_actions"],
            _shms=shms, locks=locks, n_locks=descriptor["n_locks"], owner=False,
        )

    # ---- core lookup ----
    @staticmethod
    def _split(key16: bytes) -> tuple[int, int]:
        if len(key16) != 16:
            key16 = key16.ljust(16, b"\0")[:16]
        return int.from_bytes(key16[:8], "little"), int.from_bytes(key16[8:], "little")

    def _start(self, lo: int, hi: int) -> int:
        return int((lo ^ ((hi * _MIX) & 0xFFFFFFFFFFFFFFFF)) % self.capacity)

    def slot_for(self, key16: bytes) -> int:
        """Row index for an infoset fingerprint; claims an empty slot on first visit (lock-guarded).
        Raises if the table is full (load factor exceeded)."""
        lo, hi = self._split(key16)
        slot = self._start(lo, hi)
        for _ in range(self.capacity):
            if self.occ[slot]:
                if int(self.klo[slot]) == lo and int(self.khi[slot]) == hi:
                    return slot
                slot = (slot + 1) % self.capacity
                continue
            # Apparently empty — claim under the per-slot lock (re-check: a barrier vs concurrent claims
            # so two distinct keys never share a slot, and a stale "empty" read is corrected here).
            with self.locks[slot % self.n_locks]:
                if not self.occ[slot]:
                    self.klo[slot] = np.uint64(lo)
                    self.khi[slot] = np.uint64(hi)
                    self.occ[slot] = 1  # publish AFTER the key (readers check occ first, then key)
                    return slot
                if int(self.klo[slot]) == lo and int(self.khi[slot]) == hi:
                    return slot
            slot = (slot + 1) % self.capacity
        raise RuntimeError(
            f"SharedHogwildTable full (capacity {self.capacity}); raise capacity for this abstraction."
        )

    # ---- hogwild updates (racy by design) ----
    def add_regret(self, slot: int, delta: np.ndarray, *, cfr_plus: bool = True) -> None:
        row = self.regret[slot]
        row += delta
        if cfr_plus:
            np.maximum(row, 0.0, out=row)  # CFR+ flooring

    def add_strategy(self, slot: int, contribution: np.ndarray) -> None:
        self.strat[slot] += contribution

    def regret_row(self, slot: int) -> np.ndarray:
        return self.regret[slot]

    def base_row(self, slot: int) -> np.ndarray | None:
        """VR-MCCFR baseline row for ``slot`` (None when the table was created without a baseline)."""
        return None if self.base is None else self.base[slot]

    def apply_dcfr_discount(self, t: float, *, alpha: float = 1.5, beta: float = 0.0,
                            gamma: float = 2.0) -> None:
        """Discount the WHOLE accumulated regret/strategy in place by the DCFR factors for iteration ``t``
        (Discounted CFR via periodic discounting — the lock-free analog of per-iteration DCFR, applied by
        the parent at each snapshot tick / round boundary). Positive vs negative regret get separate
        factors; the strategy sum is down-weighted. Racy vs concurrent worker writes = accepted hogwild
        noise (same posture as ``snapshot_*``). With DCFR the per-node ``add_regret`` must run with
        ``cfr_plus=False`` (DCFR keeps signed regret and discounts it, rather than flooring at 0)."""
        if t is None or float(t) <= 0:
            return
        pos, neg, strat = dcfr_weights(t, alpha, beta, gamma)
        reg = self.regret
        if reg is None:
            return
        # Vectorized over the full table: positive*pos, negative*neg (one extra (cap,A) temp — fine at
        # bounded capacity). Only the occupied rows hold nonzero values, so untouched rows stay 0.
        np.multiply(reg, np.where(reg > 0.0, pos, neg), out=reg)
        self.strat *= strat

    # ---- introspection / output ----
    def n_occupied(self) -> int:
        return int(self.occ.sum())

    def load_factor(self) -> float:
        return self.n_occupied() / max(1, self.capacity)

    def _key_bytes(self, slot: int) -> bytes:
        return int(self.klo[slot]).to_bytes(8, "little") + int(self.khi[slot]).to_bytes(8, "little")

    def _occupied_slots(self) -> np.ndarray:
        """Indices of occupied slots from a STABLE COPY of ``occ``. Snapshotting the LIVE shared array
        with ``np.nonzero`` races concurrent slot claims (workers set occ[slot]=1) → numpy raises
        'number of non-zero array elements changed during function execution'. The copy is a single
        memcpy: a concurrent claim just lands in-or-out (fine for a checkpoint). The key (klo/khi) is
        published BEFORE occ=1 in ``slot_for``, so any slot seen occupied here has a valid key."""
        return np.nonzero(np.array(self.occ, copy=True))[0]

    def snapshot_strategy(self) -> dict[bytes, np.ndarray]:
        """{16-byte key -> reach-weighted strategy-sum row} for every occupied slot (for the average
        strategy / checkpoint / exploitability). Copies, so it's a stable point-in-time view. Safe to
        call on a LIVE table (concurrent worker writes give torn float rows = accepted hogwild noise)."""
        out: dict[bytes, np.ndarray] = {}
        for slot in self._occupied_slots():
            out[self._key_bytes(int(slot))] = self.strat[int(slot)].copy()
        return out

    def snapshot_regret(self) -> dict[bytes, np.ndarray]:
        out: dict[bytes, np.ndarray] = {}
        for slot in self._occupied_slots():
            out[self._key_bytes(int(slot))] = self.regret[int(slot)].copy()
        return out

    # ---- lifecycle ----
    def close(self) -> None:
        # Drop numpy views BEFORE closing the buffers (else BufferError on the still-exported memory).
        for attr in ("occ", "klo", "khi", "regret", "strat", "base"):
            setattr(self, attr, None)
        for s in self._shms.values():
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass

    def unlink(self) -> None:
        """Owner-only: free the shared segments. Call after close() once all workers have exited."""
        if not self._owner:
            return
        for s in self._shms.values():
            try:
                s.unlink()
            except Exception:  # noqa: BLE001
                pass
