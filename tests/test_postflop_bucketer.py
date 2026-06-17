"""Postflop EHS bucketer: sensible ordering, bounded ids, determinism, save/load."""

from __future__ import annotations

import pytest

from pokerbot.encoding.postflop_bucketer import (
    PostflopBucketer,
    card_id_to_str,
    expected_hand_strength,
)


def _cid(rank: int, suit: int) -> int:
    """rank 0..12 (2..A), suit 0..3 (c,d,h,s) -> encoder card id."""
    return rank * 4 + suit


# Hole/board fixtures (card ids).
ACES = (_cid(12, 0), _cid(12, 1))            # Ac Ad
DRY_LOW_BOARD = (_cid(0, 0), _cid(5, 1), _cid(7, 2))   # 2c 7d 9h
TRASH = (_cid(0, 0), _cid(5, 1))             # 2c 7d
HIGH_BOARD = (_cid(12, 2), _cid(11, 3), _cid(10, 2))  # Ah Ks Qh


def test_card_id_to_str_roundtrip() -> None:
    assert card_id_to_str(_cid(12, 3)) == "As"
    assert card_id_to_str(_cid(0, 0)) == "2c"
    assert card_id_to_str(_cid(8, 1)) == "Td"  # rank 8 -> 'T'


def test_ehs_strong_beats_weak() -> None:
    strong = expected_hand_strength(ACES, DRY_LOW_BOARD, n_iter=1500)
    weak = expected_hand_strength(TRASH, HIGH_BOARD, n_iter=1500)
    assert 0.0 <= weak <= 1.0 and 0.0 <= strong <= 1.0
    assert strong > 0.7, strong          # overpair on a dry low board is very strong HU
    assert weak < 0.35, weak             # 2-7 on A-K-Q is very weak
    assert strong > weak


def test_bucket_ordering_and_bounds() -> None:
    b = PostflopBucketer(num_buckets=50, n_iter=1500)
    bs = b.bucket(ACES, DRY_LOW_BOARD)
    bw = b.bucket(TRASH, HIGH_BOARD)
    assert 0 <= bw < 50 and 0 <= bs < 50
    assert bs > bw
    # Deterministic + cached.
    assert b.bucket(ACES, DRY_LOW_BOARD) == bs
    # Order-invariance of hole/board input.
    assert b.bucket((ACES[1], ACES[0]), DRY_LOW_BOARD[::-1]) == bs


def test_save_load_roundtrip(tmp_path) -> None:
    b = PostflopBucketer(num_buckets=50, n_iter=1000)
    bs = b.bucket(ACES, DRY_LOW_BOARD)
    bw = b.bucket(TRASH, HIGH_BOARD)
    p = tmp_path / "buckets.json"
    b.save(p)
    b2 = PostflopBucketer.load(p)
    assert b2.num_buckets == 50
    # Cached entries restored exactly (no recompute needed).
    assert b2._cache[(tuple(sorted(ACES)), tuple(sorted(DRY_LOW_BOARD)))] == bs
    assert b2._cache[(tuple(sorted(TRASH)), tuple(sorted(HIGH_BOARD)))] == bw


def test_preflop_rejected() -> None:
    with pytest.raises(ValueError):
        expected_hand_strength(ACES, (), n_iter=10)
