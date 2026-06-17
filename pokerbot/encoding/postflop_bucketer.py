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
