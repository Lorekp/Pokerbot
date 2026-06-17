"""Pure JAX 7-card hand evaluation for NLHE showdowns."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

_STRAIGHT_WINDOWS = jnp.array(
    [
        [14, 13, 12, 11, 10],
        [13, 12, 11, 10, 9],
        [12, 11, 10, 9, 8],
        [11, 10, 9, 8, 7],
        [10, 9, 8, 7, 6],
        [9, 8, 7, 6, 5],
        [8, 7, 6, 5, 4],
        [7, 6, 5, 4, 3],
        [6, 5, 4, 3, 2],
        [14, 5, 4, 3, 2],
    ],
    dtype=jnp.int32,
)

_COMBOS = jnp.array(
    [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 5],
        [0, 1, 2, 3, 6],
        [0, 1, 2, 4, 5],
        [0, 1, 2, 4, 6],
        [0, 1, 2, 5, 6],
        [0, 1, 3, 4, 5],
        [0, 1, 3, 4, 6],
        [0, 1, 3, 5, 6],
        [0, 1, 4, 5, 6],
        [0, 2, 3, 4, 5],
        [0, 2, 3, 4, 6],
        [0, 2, 3, 5, 6],
        [0, 2, 4, 5, 6],
        [0, 3, 4, 5, 6],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 6],
        [1, 2, 3, 5, 6],
        [1, 2, 4, 5, 6],
        [1, 3, 4, 5, 6],
        [2, 3, 4, 5, 6],
    ],
    dtype=jnp.int32,
)


def card_rank(card: Array) -> Array:
    return card // 4 + 2


def card_suit(card: Array) -> Array:
    return card % 4


def straight_high(ranks: Array) -> Array:
    def window_high(window: Array) -> Array:
        hits = jnp.array([jnp.any(ranks == window[i]) for i in range(5)])
        matched = jnp.all(hits)
        high = jnp.where(window[0] == 14, jnp.where(window[1] == 5, 5, 14), window[0])
        return jnp.where(matched, high, 0)

    highs = jax.vmap(window_high)(_STRAIGHT_WINDOWS)
    return jnp.max(highs)


def score_five(ranks: Array, suits: Array) -> Array:
    ranks = jnp.sort(ranks)[::-1]
    counts = jnp.bincount(ranks, length=15)
    is_flush = jnp.all(suits == suits[0])
    s_high = straight_high(ranks)
    is_straight = s_high > 0

    count_vals = jnp.sort(counts)[::-1]

    cat = jnp.int32(0)
    cat = jnp.where(is_flush & is_straight, 8, cat)
    cat = jnp.where((count_vals[0] == 4) & (cat < 8), 7, cat)
    cat = jnp.where((count_vals[0] == 3) & (count_vals[1] >= 2) & (cat < 7), 6, cat)
    cat = jnp.where(is_flush & ~is_straight & (cat < 6), 5, cat)
    cat = jnp.where(is_straight & ~is_flush & (cat < 5), 4, cat)
    cat = jnp.where((count_vals[0] == 3) & (count_vals[1] < 2) & (cat < 4), 3, cat)
    cat = jnp.where((count_vals[0] == 2) & (count_vals[1] == 2) & (cat < 3), 2, cat)
    cat = jnp.where((count_vals[0] == 2) & (count_vals[1] < 2) & (cat < 2), 1, cat)

    # within-category positional code: order the 5 cards by (count of that rank desc, then rank
    # desc) and read off base-15 digits, so pairs/trips/quads sort ahead of kickers and one formula
    # tiebreaks every category. Straights/straight-flushes use s_high (handles the wheel A-2-3-4-5).
    # FIX 2026-06-08: the old `(cat << 32) + primary` dropped the category under x64-OFF (int32
    # `<< 32` == 0) and used mis-scaled tiebreakers; this repacks correctly within int32.
    # cat<=8 => cat*15**5 <= 6.08e6; within in [0, 15**5-1] => no overlap, fits int32.
    key = counts[ranks] * 15 + ranks
    ordered = ranks[jnp.argsort(key)[::-1]]
    hist = jnp.sum(ordered * jnp.array([15**4, 15**3, 15**2, 15, 1], dtype=jnp.int32))
    within = jnp.where((cat == 4) | (cat == 8), s_high, hist)
    return (cat * jnp.int32(15**5) + within).astype(jnp.int32)


def evaluate_seven(cards: Array) -> Array:
    def score_combo(idx: Array) -> Array:
        chosen = cards[idx]
        ranks = card_rank(chosen)
        suits = card_suit(chosen)
        return score_five(ranks, suits)

    scores = jax.vmap(score_combo)(_COMBOS)
    return jnp.max(scores)


def player_showdown_score(holes: Array, board: Array, board_valid: Array) -> Array:
    seven = jnp.concatenate([holes, jnp.where(board_valid, board, 0)])
    valid_mask = jnp.concatenate([jnp.ones(2, bool), board_valid])
    seven = jnp.where(valid_mask, seven, 0)
    return evaluate_seven(seven)
