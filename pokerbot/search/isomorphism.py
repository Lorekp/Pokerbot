"""Suit-isomorphic card canonicalization for value network inputs."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Sequence

import numpy as np

from pokerbot.engine.cards import Card
from pokerbot.engine.jax_engine import HISTORY_FEATURE_DIM

NUM_SUITS = 4


def _card_index(card: Card) -> int:
    return card.index


def build_suit_permutation(cards: Sequence[Card]) -> dict[int, int]:
    """Map original suit ids to canonical 0..k-1 by first-seen order."""
    suit_map: dict[int, int] = {}
    next_suit = 0
    for card in cards:
        s = int(card.suit)
        if s not in suit_map:
            suit_map[s] = next_suit
            next_suit += 1
    for s in range(NUM_SUITS):
        if s not in suit_map:
            suit_map[s] = next_suit
            next_suit += 1
    return suit_map


def permute_card_index(idx: int, suit_map: dict[int, int]) -> int:
    rank = idx // 4
    suit = idx % 4
    return rank * 4 + suit_map[suit]


def canonicalize_cards(
    hole: tuple[Card, Card] | None,
    board: Sequence[Card],
) -> tuple[tuple[int, int] | None, tuple[int, ...], dict[int, int]]:
    """Return canonical card indices and suit permutation."""
    ordered: list[Card] = list(board)
    if hole:
        ordered.extend(hole)
    suit_map = build_suit_permutation(ordered)
    canon_board = tuple(permute_card_index(c.index, suit_map) for c in board)
    canon_hole = None
    if hole:
        canon_hole = (
            permute_card_index(hole[0].index, suit_map),
            permute_card_index(hole[1].index, suit_map),
        )
    return canon_hole, canon_board, suit_map


def canonicalize_history_encoding(
    raw_history: np.ndarray,
    *,
    pot_bb: float | None = None,
    spr_bucket: int | None = None,
) -> np.ndarray:
    """
    Pass-through with dtype/shape normalization and optional chip-isomorphism buckets.

    When pot_bb/spr_bucket are provided, embeds bucket indices in the last two feature dims.
    """
    out = np.asarray(raw_history, dtype=np.float32)
    if out.shape[0] < HISTORY_FEATURE_DIM:
        pad = np.zeros(HISTORY_FEATURE_DIM, dtype=np.float32)
        pad[: out.shape[0]] = out
        out = pad
    out = out[:HISTORY_FEATURE_DIM].copy()
    if pot_bb is not None:
        out[-2] = float(spr_bucket if spr_bucket is not None else int(pot_bb * 2) % 16)
        out[-1] = float(min(pot_bb, 512.0))
    return out


def history_cache_key(
    canon_board: tuple[int, ...],
    street: int,
    action_seq: tuple[int, ...],
) -> str:
    """LRU cache key for isomorphic subgame roots."""
    payload = f"{canon_board}|{street}|{action_seq}"
    return hashlib.md5(payload.encode()).hexdigest()


@lru_cache(maxsize=4096)
def cached_canonicalize_key(
    canon_board: tuple[int, ...],
    street: int,
    action_seq: tuple[int, ...],
) -> str:
    return history_cache_key(canon_board, street, action_seq)
