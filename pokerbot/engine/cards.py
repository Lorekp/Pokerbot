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
        # CASH-ENGINE-AUDIT #16 — accept both 2-char ("Ah") and 3-char ("10h") rank forms. The 3-char
        # form is rare but legal — rejecting it silently would mangle any solver / OCR stream that
        # outputs the natural decimal-rank form for ten. Cap at 3 chars to refuse garbage.
        if not (2 <= len(text) <= 3):
            raise ValueError(f"Invalid card: {text!r}")
        if len(text) == 3:
            # CASH-ENGINE-AUDIT CRIT-3 fix: "10h" must map to Ten of Hearts, not raise. We special-case
            # the "10" rank prefix (which isn't in RANK_CHARS) and otherwise require the 2-char prefix
            # to be a valid single-rank char. Any other 3-char input is rejected as garbage.
            if text.startswith("10"):
                rank_str, suit_char = "10", text[2].lower()
            else:
                raise ValueError(f"Invalid card: {text!r}")
        else:
            rank_str, suit_char = text[0].upper(), text[1].lower()
        if rank_str not in RANK_CHARS and rank_str != "10":
            raise ValueError(f"Invalid card: {text!r}")
        if suit_char not in SUIT_CHARS:
            raise ValueError(f"Invalid card: {text!r}")
        rank_idx = 8 if rank_str == "10" else RANK_CHARS.index(rank_str)
        return cls(Rank(rank_idx + 2), Suit(SUIT_CHARS.index(suit_char)))

    def to_str(self) -> str:
        return f"{RANK_CHARS[self.rank - 2]}{SUIT_CHARS[self.suit]}"

    # str(card) == 'Jh' (the canonical parseable form). Without this, str() fell through to the
    # dataclass repr, which board_texture() cannot parse — so every site that built a context key
    # via str(card) (agent, heuristic battery hero) silently collapsed postflop textures to '-',
    # splitting a player's history across keys incompatible with the context_key_rs sites
    # (train/serve skew). repr() stays the verbose dataclass form.
    def __str__(self) -> str:
        return self.to_str()

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
    """Parse board like 'Ks7h2d' or 'Ks 7h 2d'. Also accepts 3-char '10h' forms mixed with 2-char
    forms, e.g. '10h10d10s10c' (a board of four tens). Tokenizes on suit-character boundaries so
    each card is either 2 or 3 chars long."""
    text = text.replace(" ", "").strip()
    if not text:
        return []
    out: list[Card] = []
    i = 0
    while i < len(text):
        if i + 2 < len(text) and text[i:i+2] == "10" and text[i+2] in SUIT_CHARS:
            out.append(Card.from_str(text[i:i+3]))
            i += 3
        elif i + 1 < len(text) and text[i+1] in SUIT_CHARS:
            out.append(Card.from_str(text[i:i+2]))
            i += 2
        else:
            raise ValueError(f"Invalid card string: {text!r} (parse error at index {i})")
    return out


def parse_holding(text: str) -> tuple[Card, Card]:
    cards = parse_cards(text)
    if len(cards) != 2:
        raise ValueError(f"Expected 2 hole cards, got {text!r}")
    return cards[0], cards[1]
