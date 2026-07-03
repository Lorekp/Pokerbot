"""River OCHS abstraction — the Python serve twin of ``tools/battery_native/src/river_ochs.rs``.

Blueprint-v2 replaces the river's made-hand "mirror fallback" bucket (a SCALAR strength rank
vs a uniform opponent) with OCHS (Opponent-Cluster Hand Strength, Johanson et al. AAMAS 2013):
an 8-vector of win rates against 8 preflop-strength clusters of opponent holdings, assigned to
the nearest of K fitted centroids. The trainer writes river blueprint rows under this bucket,
so the serve path MUST reproduce it exactly.

PARITY CONTRACT (pinned by ``data/abstraction/river_ochs_160.golden.tsv``, tested on BOTH
sides — ``cargo test river_ochs`` and ``tests/test_river_ochs_parity.py``):
  * hand ranks: python ``phevaluator.evaluate_cards`` — the Rust side is a byte-identical port,
    so both see the same integers (1 best .. 7462 worst).
  * OCHS vector: win/tie half-points are INTEGERS (win=2, tie=1) normalized once by an integer
    total — exact in f64, iteration-order independent.
  * nearest centroid: squared-L2 accumulated left->right over the 8 dims in f64, strict ``<``
    keep-first argmin — the same fixed reduction order as the Rust side.

The blob (``ROCHS001``) is tiny (~10 KB): magic(8) | K u32 LE | n_clusters u32 LE |
class_to_cluster[169] u8 | K*8 f64 LE centroids. Buckets are computed on the fly (990
opponent-combo evals, ~1ms in Python) and memoized, so river-node serve cost is one dict probe
after first sight of a (hole, board).

Firewall-safe: pure additive module; only consulted when ``POKERBOT_RIVER_OCHS`` is set.
"""

from __future__ import annotations

import os
import struct
import threading

ROCHS_MAGIC = b"ROCHS001"
N_CLUSTERS = 8
N_PREFLOP_CLASSES = 169


def canonical_preflop_class(c0: int, c1: int) -> int:
    """Twin of ``fingerprint.rs::canonical_preflop_class`` (card id = rank*4 + suit)."""
    r0, s0 = c0 // 4, c0 % 4
    r1, s1 = c1 // 4, c1 % 4
    if r0 == r1:
        return r0
    hr, lr = (r0, r1) if r0 > r1 else (r1, r0)
    tri = hr * (hr - 1) // 2 + lr
    return 13 + tri if s0 == s1 else 13 + 78 + tri


_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _card_str(cid: int) -> str:
    return _RANKS[cid // 4] + _SUITS[cid % 4]


class RiverOchs:
    """Loaded ROCHS001 table + the on-the-fly bucket computation (memoized)."""

    def __init__(self, n_buckets: int, class_to_cluster: list[int], centroids: list[list[float]]):
        if len(class_to_cluster) != N_PREFLOP_CLASSES:
            raise ValueError("class_to_cluster must have 169 entries")
        if len(centroids) != n_buckets or any(len(r) != N_CLUSTERS for r in centroids):
            raise ValueError("centroids must be K x 8")
        self.n_buckets = int(n_buckets)
        self.class_to_cluster = [int(c) for c in class_to_cluster]
        self.centroids = [[float(x) for x in row] for row in centroids]
        # cluster id per (o1, o2) combo — precomputed once (1326 entries).
        self._combo_cluster = {}
        for o1 in range(52):
            for o2 in range(o1 + 1, 52):
                self._combo_cluster[(o1, o2)] = self.class_to_cluster[
                    canonical_preflop_class(o1, o2)
                ]
        self._memo: dict[tuple, int] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ load
    @classmethod
    def from_bytes(cls, b: bytes) -> "RiverOchs":
        if len(b) < 16 + N_PREFLOP_CLASSES or b[:8] != ROCHS_MAGIC:
            raise ValueError("river_ochs: bad blob (magic/short)")
        n_buckets, n_clusters = struct.unpack_from("<II", b, 8)
        if n_clusters != N_CLUSTERS:
            raise ValueError(f"river_ochs: n_clusters {n_clusters} != {N_CLUSTERS}")
        c2c = list(b[16 : 16 + N_PREFLOP_CLASSES])
        expected = 16 + N_PREFLOP_CLASSES + n_buckets * N_CLUSTERS * 8
        if len(b) != expected:
            raise ValueError(f"river_ochs: blob len {len(b)} != expected {expected}")
        cents = []
        p = 16 + N_PREFLOP_CLASSES
        for _ in range(n_buckets):
            cents.append(list(struct.unpack_from(f"<{N_CLUSTERS}d", b, p)))
            p += N_CLUSTERS * 8
        return cls(n_buckets, c2c, cents)

    @classmethod
    def load(cls, path: str) -> "RiverOchs":
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())

    # ----------------------------------------------------------------- bucket
    def ochs_vector(self, hole: tuple[int, int], board5: tuple[int, ...]) -> list[float]:
        """Exhaustive 8-vector of win rates vs the opponent clusters (integer-exact)."""
        from phevaluator import evaluate_cards

        dead = set(hole) | set(board5)
        bstr = [_card_str(c) for c in board5]
        hero_rank = evaluate_cards(_card_str(hole[0]), _card_str(hole[1]), *bstr)
        half = [0] * N_CLUSTERS
        cnt = [0] * N_CLUSTERS
        live = [c for c in range(52) if c not in dead]
        cstr = [_card_str(c) for c in range(52)]
        combo_cluster = self._combo_cluster
        for i, o1 in enumerate(live):
            s1 = cstr[o1]
            for o2 in live[i + 1 :]:
                c = combo_cluster[(o1, o2)]
                opp_rank = evaluate_cards(s1, cstr[o2], *bstr)
                if hero_rank < opp_rank:
                    half[c] += 2
                elif hero_rank == opp_rank:
                    half[c] += 1
                cnt[c] += 1
        return [half[c] / (2 * cnt[c]) if cnt[c] else 0.0 for c in range(N_CLUSTERS)]

    def bucket_of_vector(self, v: list[float]) -> int:
        """Strict-less keep-first argmin over squared-L2, left->right accumulation (parity-pinned)."""
        best = 0
        best_d = float("inf")
        for k, cent in enumerate(self.centroids):
            d = 0.0
            for c in range(N_CLUSTERS):
                x = v[c] - cent[c]
                d += x * x
            if d < best_d:
                best_d = d
                best = k
        return best

    def bucket(self, hole, board) -> int:
        """River bucket (0..K-1) or -1 for a non-river spot / invalid holes."""
        h = [int(c) for c in hole]
        b = [int(c) for c in board if int(c) >= 0]
        if len(h) != 2 or any(c < 0 for c in h) or len(b) != 5:
            return -1
        key = (min(h), max(h)) + tuple(sorted(b))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        v = self.ochs_vector((h[0], h[1]), tuple(b))
        out = self.bucket_of_vector(v)
        with self._lock:
            if len(self._memo) > 2_000_000:  # bound serve-side memory (~100MB worst case)
                self._memo.clear()
            self._memo[key] = out
        return out


# ------------------------------------------------------------------ process-global loader
_GLOBAL: RiverOchs | None = None
_GLOBAL_PATH: str | None = None
_LOAD_LOCK = threading.Lock()


def get_river_ochs() -> RiverOchs | None:
    """The process-global table for ``POKERBOT_RIVER_OCHS`` (None when unset). Reloads only
    when the env path changes (mirrors the EMD frozen-lookup loader)."""
    global _GLOBAL, _GLOBAL_PATH
    path = os.environ.get("POKERBOT_RIVER_OCHS")
    if not path:
        return None
    if _GLOBAL is None or _GLOBAL_PATH != path:
        with _LOAD_LOCK:
            if _GLOBAL is None or _GLOBAL_PATH != path:
                _GLOBAL = RiverOchs.load(path)
                _GLOBAL_PATH = path
    return _GLOBAL
