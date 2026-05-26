"""Seven-card hand evaluation for Texas Hold'em."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable

from pokerbot.engine.cards import Card, Rank


HAND_CATEGORY = {
    "high_card": 0,
    "pair": 1,
    "two_pair": 2,
    "three_kind": 3,
    "straight": 4,
    "flush": 5,
    "full_house": 6,
    "four_kind": 7,
    "straight_flush": 8,
}


def _straight_high(ranks: list[int]) -> int | None:
    unique = sorted(set(ranks), reverse=True)
    if Rank.ACE in unique:
        unique.append(1)
    unique = sorted(set(unique), reverse=True)
    for i in range(len(unique) - 4):
        window = unique[i : i + 5]
        if window[0] - window[4] == 4 and len(window) == 5:
            return window[0]
    return None


def _score_five(cards: tuple[Card, Card, Card, Card, Card]) -> tuple[int, ...]:
    ranks = sorted((int(c.rank) for c in cards), reverse=True)
    suits = [int(c.suit) for c in cards]
    rank_counts = Counter(ranks)
    counts_sorted = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    if is_flush and straight_high:
        return (HAND_CATEGORY["straight_flush"], straight_high)
    if counts_sorted[0][1] == 4:
        quad = counts_sorted[0][0]
        kicker = max(r for r in ranks if r != quad)
        return (HAND_CATEGORY["four_kind"], quad, kicker)
    if counts_sorted[0][1] == 3 and counts_sorted[1][1] >= 2:
        trips = counts_sorted[0][0]
        pair = counts_sorted[1][0]
        return (HAND_CATEGORY["full_house"], trips, pair)
    if is_flush:
        return (HAND_CATEGORY["flush"], *ranks)
    if straight_high:
        return (HAND_CATEGORY["straight"], straight_high)
    if counts_sorted[0][1] == 3:
        trips = counts_sorted[0][0]
        kickers = sorted((r for r in ranks if r != trips), reverse=True)
        return (HAND_CATEGORY["three_kind"], trips, *kickers)
    if counts_sorted[0][1] == 2 and counts_sorted[1][1] == 2:
        high_pair, low_pair = sorted((counts_sorted[0][0], counts_sorted[1][0]), reverse=True)
        kicker = max(r for r in ranks if r not in (high_pair, low_pair))
        return (HAND_CATEGORY["two_pair"], high_pair, low_pair, kicker)
    if counts_sorted[0][1] == 2:
        pair = counts_sorted[0][0]
        kickers = sorted((r for r in ranks if r != pair), reverse=True)
        return (HAND_CATEGORY["pair"], pair, *kickers)
    return (HAND_CATEGORY["high_card"], *ranks)


def evaluate_hand(cards: Iterable[Card]) -> tuple[int, ...]:
    """Return comparable score tuple for best 5-card hand from up to 7 cards."""
    card_list = list(cards)
    if len(card_list) < 5:
        raise ValueError("Need at least 5 cards to evaluate")
    best: tuple[int, ...] | None = None
    for combo in combinations(card_list, 5):
        score = _score_five(combo)  # type: ignore[arg-type]
        if best is None or score > best:
            best = score
    assert best is not None
    return best


def compare_hands(hands: list[list[Card]]) -> list[int]:
    """Return indices of winners (may tie)."""
    scores = [evaluate_hand(h) for h in hands]
    best = max(scores)
    return [i for i, s in enumerate(scores) if s == best]
