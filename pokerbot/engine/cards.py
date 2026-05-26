"""Playing card primitives."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


SUIT_CHARS = "cdhs"
RANK_CHARS = "23456789TJQKA"


@dataclass(frozen=True, slots=True, order=True)
class Card:
    rank: Rank
    suit: Suit

    @classmethod
    def from_str(cls, text: str) -> Card:
        text = text.strip()
        if len(text) != 2:
            raise ValueError(f"Invalid card: {text!r}")
        rank_char, suit_char = text[0].upper(), text[1].lower()
        if rank_char not in RANK_CHARS or suit_char not in SUIT_CHARS:
            raise ValueError(f"Invalid card: {text!r}")
        return cls(Rank(RANK_CHARS.index(rank_char) + 2), Suit(SUIT_CHARS.index(suit_char)))

    def to_str(self) -> str:
        return f"{RANK_CHARS[self.rank - 2]}{SUIT_CHARS[self.suit]}"

    @property
    def index(self) -> int:
        """Unique index 0-51."""
        return (self.rank - 2) * 4 + int(self.suit)


class Deck:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self._cards = [Card(Rank(r), Suit(s)) for r in range(2, 15) for s in range(4)]
        self._rng.shuffle(self._cards)
        self._idx = 0

    def deal(self, n: int = 1) -> list[Card]:
        if self._idx + n > len(self._cards):
            raise RuntimeError("Deck exhausted")
        out = self._cards[self._idx : self._idx + n]
        self._idx += n
        return out


def parse_cards(text: str) -> list[Card]:
    """Parse board like 'Ks7h2d' or 'Ks 7h 2d'."""
    text = text.replace(" ", "").strip()
    if not text:
        return []
    if len(text) % 2 != 0:
        raise ValueError(f"Invalid card string: {text!r}")
    return [Card.from_str(text[i : i + 2]) for i in range(0, len(text), 2)]


def parse_holding(text: str) -> tuple[Card, Card]:
    cards = parse_cards(text)
    if len(cards) != 2:
        raise ValueError(f"Expected 2 hole cards, got {text!r}")
    return cards[0], cards[1]
