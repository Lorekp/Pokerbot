"""
Postflop information abstraction (Pluribus-style) for the NLHE infoset key.

The tabular CFR tables key on the exact (hole, board); at convergence scale the card dimension
proliferates and the tables blow up. Pluribus bounds this with ~500 buckets/street of an
equity/potential-aware abstraction. This module maps a postflop (hole, board) to a bounded bucket
id by Expected Hand Strength (HU equity vs a random opponent, completing the board), then binning.

Design:
- EHS is computed via the fast C-backed ``phevaluator`` (same evaluator ``ev_calc`` uses), by Monte
  Carlo over the remaining board cards and one random villain hand. Order-/suit-invariant, so it is
  cached by the sorted card-id tuple.
- v1 bins EHS on fixed edges ([0,1] quantile/equal-width); k-means on equity *distributions*
  (potential-aware, EMD) is the documented Pluribus refinement and a later upgrade.
- Intended to be PRECOMPUTED offline into a lookup (``save``/``load``) so encode-time is O(1) — the
  same pattern as ``WEVAIndex``. Inline computation is ~ms/(hole,board) and only for cold cache.

Card-id convention matches the encoder: ``card_id = rank*4 + suit`` (rank 0-12, suit 0-3).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"

# Number of buckets per street; Pluribus used 500. Tunable (quality vs table size).
DEFAULT_NUM_BUCKETS = 200


def card_id_to_str(card_id: int) -> str:
    """Encoder card id (rank*4 + suit) -> phevaluator string, e.g. 51 -> 'As'."""
    if card_id < 0 or card_id > 51:
        raise ValueError(f"card_id out of range: {card_id}")
    return _RANKS[card_id // 4] + _SUITS[card_id % 4]


def expected_hand_strength(
    hole_ids: tuple[int, ...],
    board_ids: tuple[int, ...],
    *,
    n_iter: int = 2000,
    seed: int = 0,
) -> float:
    """HU equity (win + tie-share) of (hole, board) vs one random hand, completing to 5 board cards.

    Postflop only (>=3 board cards). Returns a float in [0, 1].
    """
    from phevaluator import evaluate_cards

    if len(hole_ids) != 2:
        raise ValueError("hole must be exactly 2 cards")
    if not (3 <= len(board_ids) <= 5):
        raise ValueError("board must have 3-5 cards (postflop)")

    hole = [card_id_to_str(c) for c in hole_ids]
    board = [card_id_to_str(c) for c in board_ids]
    known = set(hole) | set(board)
    deck = [r + s for r in _RANKS for s in _SUITS if (r + s) not in known]
    need = 5 - len(board)
    draws = need + 2  # remaining board + 2 villain cards
    if draws > len(deck):
        raise ValueError("not enough cards left in deck")

    rng = np.random.default_rng(seed)
    deck_arr = np.array(deck, dtype=object)
    score = 0.0
    for _ in range(n_iter):
        pick = deck_arr[rng.choice(len(deck), size=draws, replace=False)]
        full_board = board + list(pick[:need])
        hero = evaluate_cards(*(hole + full_board))      # lower = stronger
        villain = evaluate_cards(*(list(pick[need:]) + full_board))
        if hero < villain:
            score += 1.0
        elif hero == villain:
            score += 0.5
    return score / n_iter


def _canonical_key(hole_ids, board_ids) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (tuple(sorted(int(c) for c in hole_ids)), tuple(sorted(int(c) for c in board_ids)))


# Fast inline postflop bucket: ONE made-hand evaluation per (hole, board) (~2 us via the C-backed
# phevaluator), binned by the 7462-class hand rank. This is the per-node-affordable card abstraction
# for the CFR traversal (the EHS Monte-Carlo bucketer above is ~ms/cold-key — meant for offline
# precompute, far too slow inline). Coarser than potential-aware EHS (ignores draw equity), but it
# bounds the postflop card dimension so the tables stop growing, and it's deterministic ⇒ the
# fingerprint stays deterministic ⇒ pool/single shard merge stays bit-identical. Refine later with a
# draw flag or the offline EHS precompute. Cached by suit-sorted (hole, board) for repeat nodes.
_FAST_BUCKET_CACHE: dict[tuple, int] = {}


def fast_made_hand_bucket(hole_ids, board_ids, num_buckets: int = DEFAULT_NUM_BUCKETS) -> int:
    """Bin (hole, board) into ``num_buckets`` by current made-hand rank. Postflop only (>=3 board)."""
    h = tuple(sorted(int(c) for c in hole_ids))
    b = tuple(sorted(int(c) for c in board_ids if int(c) >= 0))
    if len(b) < 3 or len(h) != 2:
        return -1
    key = (h, b, int(num_buckets))
    hit = _FAST_BUCKET_CACHE.get(key)
    if hit is not None:
        return hit
    from phevaluator import evaluate_cards

    cards = [card_id_to_str(c) for c in h] + [card_id_to_str(c) for c in b]
    rank = evaluate_cards(*cards)  # 1 (best) .. 7462 (worst), suit-isomorphic except flushes
    bucket = min(num_buckets - 1, (rank - 1) * num_buckets // 7462)
    _FAST_BUCKET_CACHE[key] = bucket
    return bucket


def strength_histogram(hole_ids, board_ids, *, n_bins: int = 20, n_runouts: int = 200,
                       seed: int = 0) -> np.ndarray:
    """Potential-aware HAND-STRENGTH DISTRIBUTION of (hole, board): a normalized histogram (length
    ``n_bins``) of the hero's made-hand-strength percentile over sampled board completions to the river.

    This is the signal a made-hand-rank bucket THROWS AWAY: a flush/straight DRAW is bimodal (high
    percentile when it completes, low when it bricks), while a made top pair is unimodal and high. Two
    hands of identical CURRENT rank but different draw potential get different histograms => different
    EMD buckets. At the river (5-card board) there are no runouts, so the histogram is a single spike at
    the made-hand percentile (correctly collapsing to made-hand bucketing — no potential left).
    Percentile uses the 7462-class rank (1 best .. 7462 worst): pct = 1 - (rank-1)/7461."""
    from phevaluator import evaluate_cards

    h = tuple(sorted(int(c) for c in hole_ids))
    b = tuple(sorted(int(c) for c in board_ids if int(c) >= 0))
    hist = np.zeros(int(n_bins), dtype=np.float64)
    if len(b) < 3 or len(h) != 2:
        return hist
    hole = [card_id_to_str(c) for c in h]
    board = [card_id_to_str(c) for c in b]
    need = 5 - len(board)
    if need <= 0:  # river: single deterministic strength -> one spike
        rank = evaluate_cards(*(hole + board))
        pct = 1.0 - (rank - 1) / 7461.0
        hist[min(int(n_bins) - 1, int(pct * n_bins))] += 1.0
        return hist
    known = set(hole) | set(board)
    deck = [r + s for r in _RANKS for s in _SUITS if (r + s) not in known]
    rng = np.random.default_rng(seed)
    deck_arr = np.array(deck, dtype=object)
    for _ in range(int(n_runouts)):
        pick = deck_arr[rng.choice(len(deck), size=need, replace=False)]
        rank = evaluate_cards(*(hole + board + list(pick)))
        pct = 1.0 - (rank - 1) / 7461.0
        hist[min(int(n_bins) - 1, int(pct * n_bins))] += 1.0
    s = hist.sum()
    return hist / s if s > 0 else hist


def _emd_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Earth Mover's Distance between two normalized histograms = L1 of their CDFs."""
    return float(np.abs(np.cumsum(a) - np.cumsum(b)).sum())


class EMDBucketer:
    """Potential-aware postflop abstraction (the Pluribus refinement): clusters hands by their
    hand-strength DISTRIBUTION under Earth Mover's Distance, so draws and made hands of equal current
    rank land in different buckets. Centroids are fit offline over a sample of (hole, board) histograms
    (k-means under EMD) and serialized; ``bucket()`` is then an O(K) nearest-centroid lookup. Intended
    as the ``POKERBOT_POSTFLOP_MODE=emd`` table for the next blueprint retrain."""

    def __init__(self, num_buckets: int = DEFAULT_NUM_BUCKETS, *, n_bins: int = 20,
                 n_runouts: int = 200, centroids: np.ndarray | None = None) -> None:
        self.num_buckets = int(num_buckets)
        self.n_bins = int(n_bins)
        # CASH-ENGINE-AUDIT F-ENC-BUCKETER-3: n_runouts=0 silently breaks the abstraction
        # (every query buckets to centroid 0 via the all-zero histogram path). Refuse loudly.
        if int(n_runouts) <= 0:
            raise ValueError(
                f"EMDBucketer.n_runouts must be > 0 (got {n_runouts!r}) — 0 silently collapses "
                f"every histogram to centroid 0."
            )
        self.n_runouts = int(n_runouts)
        self.centroids = None if centroids is None else np.asarray(centroids, dtype=np.float64)
        self._cache: dict[tuple, int] = {}
        # Lock for the per-instance cache (the module-level _FAST_BUCKET_CACHE has its own lock).
        # Multiple threads invoking bucket() concurrently on the same bucketer need serialization
        # on the cache write — read is lock-free (``dict.get`` is GIL-safe on existing keys).
        self._cache_lock = threading.Lock()

    def fit(self, histograms, *, iters: int = 12, seed: int = 0) -> "EMDBucketer":
        """k-means under EMD over a (N, n_bins) array of pre-computed strength histograms. Centroid
        update uses the mean histogram of each cluster (a standard, stable EMD-barycenter surrogate)."""
        X = np.asarray(histograms, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < self.num_buckets:
            raise ValueError(f"need >= num_buckets ({self.num_buckets}) sample histograms, got {X.shape}")
        rng = np.random.default_rng(seed)
        cent = X[rng.choice(X.shape[0], size=self.num_buckets, replace=False)].copy()
        Xc = np.cumsum(X, axis=1)
        for _ in range(int(iters)):
            Cc = np.cumsum(cent, axis=1)
            # EMD(x, c) = L1(CDF_x - CDF_c); assign each x to nearest centroid
            d = np.abs(Xc[:, None, :] - Cc[None, :, :]).sum(axis=2)
            assign = d.argmin(axis=1)
            for k in range(self.num_buckets):
                members = X[assign == k]
                if len(members):
                    cent[k] = members.mean(axis=0)
        self.centroids = cent
        return self

    def bucket(self, hole_ids, board_ids) -> int:
        if self.centroids is None:
            raise RuntimeError("EMDBucketer not fitted/loaded — call fit() or load() first")
        key = _canonical_key(hole_ids, board_ids)
        # Lock-free fast path on cache hit (GIL-protected dict.get on existing key).
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        with self._cache_lock:
            # Re-check under lock to avoid duplicate histogram computation under concurrent misses.
            hit = self._cache.get(key)
            if hit is not None:
                return hit
            hist = strength_histogram(key[0], key[1], n_bins=self.n_bins, n_runouts=self.n_runouts)
            cc = np.cumsum(self.centroids, axis=1)
            d = np.abs(np.cumsum(hist)[None, :] - cc).sum(axis=1)
            b = int(d.argmin())
            self._cache[key] = b
            return b

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "kind": "emd", "num_buckets": self.num_buckets, "n_bins": self.n_bins,
            "n_runouts": self.n_runouts,
            "centroids": (self.centroids.tolist() if self.centroids is not None else None),
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EMDBucketer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(num_buckets=int(d["num_buckets"]), n_bins=int(d.get("n_bins", 20)),
                   n_runouts=int(d.get("n_runouts", 200)),
                   centroids=(np.asarray(d["centroids"], dtype=np.float64)
                              if d.get("centroids") is not None else None))


class PostflopBucketer:
    """Maps (hole, board) -> bounded bucket id by EHS. Cache + precompute-to-disk."""

    def __init__(
        self,
        num_buckets: int = DEFAULT_NUM_BUCKETS,
        *,
        edges: np.ndarray | None = None,
        n_iter: int = 2000,
    ) -> None:
        self.num_buckets = int(num_buckets)
        # Bin edges in [0,1]; default equal-width. ``edges`` has num_buckets-1 interior cut points.
        self.edges = (
            np.asarray(edges, dtype=np.float64)
            if edges is not None
            else np.linspace(0.0, 1.0, self.num_buckets + 1)[1:-1]
        )
        self.n_iter = int(n_iter)
        self._cache: dict[tuple, int] = {}

    def ehs(self, hole_ids, board_ids) -> float:
        return expected_hand_strength(
            tuple(int(c) for c in hole_ids),
            tuple(int(c) for c in board_ids),
            n_iter=self.n_iter,
        )

    def bucket(self, hole_ids, board_ids) -> int:
        key = _canonical_key(hole_ids, board_ids)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        e = expected_hand_strength(key[0], key[1], n_iter=self.n_iter)
        b = int(np.searchsorted(self.edges, e, side="right"))
        b = max(0, min(self.num_buckets - 1, b))
        self._cache[key] = b
        return b

    def save(self, path: str | Path) -> None:
        """Persist edges + the precomputed (hole,board)->bucket cache (the lookup table)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_buckets": self.num_buckets,
            "edges": self.edges.tolist(),
            "n_iter": self.n_iter,
            # JSON-safe: encode the tuple key as a string "h0,h1|b0,b1,b2"
            "cache": {
                ",".join(map(str, k[0])) + "|" + ",".join(map(str, k[1])): v
                for k, v in self._cache.items()
            },
        }
        p.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PostflopBucketer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        b = cls(
            num_buckets=int(data["num_buckets"]),
            edges=np.asarray(data["edges"], dtype=np.float64),
            n_iter=int(data.get("n_iter", 2000)),
        )
        for ks, v in data.get("cache", {}).items():
            hs, bs = ks.split("|")
            hole = tuple(int(x) for x in hs.split(",") if x != "")
            board = tuple(int(x) for x in bs.split(",") if x != "")
            b._cache[(hole, board)] = int(v)
        return b
