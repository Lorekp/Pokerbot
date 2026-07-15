"""Shared, dependency-light preflop hand-strength primitives."""

from __future__ import annotations

from collections.abc import Sequence
import math

_RANK_VAL = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
             "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
_CHEN_HIGH = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}
_RANK_CHARS = "23456789TJQKA"
_SUIT_CHARS = "cdhs"


def card_index_to_code(index: int) -> str:
    """Convert the engine's rank-major 0..51 card index to canonical text."""
    i = int(index)
    if not 0 <= i < 52:
        raise ValueError(f"card index out of range: {index}")
    return _RANK_CHARS[i // 4] + _SUIT_CHARS[i % 4]


def chen_score(hole: Sequence[str]) -> float | None:
    """Bill Chen preflop strength (~-1 for 72o to 20 for AA)."""
    if len(hole) < 2:
        return None
    try:
        r0, s0 = _RANK_VAL[hole[0][0].upper()], hole[0][1].lower()
        r1, s1 = _RANK_VAL[hole[1][0].upper()], hole[1][1].lower()
    except (KeyError, IndexError):
        return None
    hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
    base = _CHEN_HIGH.get(hi, hi / 2.0)
    if r0 == r1:
        return max(base * 2.0, 5.0)
    score = base + (2.0 if s0 == s1 else 0.0)
    gap = hi - lo - 1
    score -= {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}.get(gap, 5.0)
    if gap <= 1 and hi < 12:
        score += 1.0
    return math.ceil(score)
