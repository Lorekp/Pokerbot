"""Canonical key decode + fixture loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

EXPECTED_KEY_SPEC_HASH = "3748b5f9ca94412b"
FIXTURES_DIR = Path(__file__).resolve().parent / "datasets" / "fixtures"


def _check_key_spec() -> None:
    from aof_equity_showdown import key_spec_hash

    h = key_spec_hash()
    if h != EXPECTED_KEY_SPEC_HASH:
        raise ValueError(f"key_spec_hash mismatch: {h} != {EXPECTED_KEY_SPEC_HASH}")


def unpack_key(key: int) -> list[tuple[int, int]]:
    cards: list[tuple[int, int]] = []
    for i in range(4):
        part = (int(key) >> (16 * i)) & 0xFFFF
        rank = (part >> 2) & 0xF
        suit = part & 0x3
        cards.append((rank, suit))
    return cards


def card_id(rank: int, suit: int) -> int:
    return rank * 4 + suit


def representative_from_key(key: int) -> tuple[int, int, int, int]:
    c = unpack_key(key)
    h0 = card_id(c[0][0], c[0][1])
    h1 = card_id(c[1][0], c[1][1])
    v0 = card_id(c[2][0], c[2][1])
    v1 = card_id(c[3][0], c[3][1])
    return (h0, h1, v0, v1)


def load_keys(name: str) -> np.ndarray:
    _check_key_spec()
    path = FIXTURES_DIR / f"{name}_keys.uint64.npy"
    if not path.is_file():
        raise FileNotFoundError(f"fixture missing: {path}")
    return np.load(path)


def fixtures_dir() -> Path:
    return FIXTURES_DIR
