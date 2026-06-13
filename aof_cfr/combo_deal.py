"""52-card deal utilities for multiway MCCFR."""

from __future__ import annotations

import random

from aof_cfr.buckets import LABELS_169, combo_to_bucket

RANKS = "23456789TJQKA"
SUITS = "cdhs"

FULL_DECK: tuple[str, ...] = tuple(f"{r}{s}" for r in RANKS for s in SUITS)
_CARD_ID_CACHE: dict[str, int] = {}


def _ensure_card_ids() -> None:
    if _CARD_ID_CACHE:
        return
    from aof_equity_grids import _card_str_to_id

    for c in FULL_DECK:
        _CARD_ID_CACHE[c] = int(_card_str_to_id(c))


def full_deck() -> list[str]:
    return list(FULL_DECK)


def card_id(card: str) -> int:
    _ensure_card_ids()
    return _CARD_ID_CACHE[card]


def id_to_card(cid: int) -> str:
    return card_from_id(cid)   # reuse the cached reverse map (was an O(52) scan per call)


def all_card_ids() -> tuple[int, ...]:
    _ensure_card_ids()
    return tuple(_CARD_ID_CACHE[c] for c in FULL_DECK)


_ID_TO_CARD: dict[int, str] | None = None


def card_from_id(cid: int) -> str:
    global _ID_TO_CARD
    _ensure_card_ids()
    if _ID_TO_CARD is None:
        _ID_TO_CARD = {v: k for k, v in _CARD_ID_CACHE.items()}
    return _ID_TO_CARD[int(cid)]


def combo_to_bucket_id(c0: str, c1: str) -> str:
    return combo_to_bucket(c0, c1)


def bucket_from_card_ids(c0: int, c1: int) -> str:
    return combo_to_bucket(card_from_id(c0), card_from_id(c1))


class ComboDealer:
    """Deal ``2 * num_players`` cards without replacement."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def deal_holes(self, num_players: int) -> list[tuple[str, str]]:
        deck = full_deck()
        self._rng.shuffle(deck)
        need = 2 * num_players
        if need > len(deck):
            raise ValueError(f"need {need} cards for {num_players} players")
        picked = deck[:need]
        return [(picked[2 * i], picked[2 * i + 1]) for i in range(num_players)]

    def remaining_outcomes(self, used: set[str]) -> list[str]:
        return [c for c in FULL_DECK if c not in used]

    def chance_outcome_prob(self, n_remaining: int) -> float:
        return 1.0 / n_remaining


_ensure_card_ids()
