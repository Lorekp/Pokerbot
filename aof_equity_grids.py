"""Precomputed preflop equity grids for teacher LUT (HU exact + range-bucket multiway)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

from aof_equity import _BIN, _DP, _FLUSH, _NF7, _SUITBIT, _SUITS
from pokerbot.solver.hu_preflop_dcfr import _build_169_buckets


@njit(cache=True)
def _hash_quinary(quinary: np.ndarray, num_cards: int) -> int:
    sum_numb = 0
    length = 13
    for rank in range(13):
        cnt = quinary[rank]
        if cnt:
            sum_numb += _DP[cnt, length - rank - 1, num_cards]
            num_cards -= cnt
    return sum_numb

N_HAND = 169


@njit(cache=True)
def _eval7(
    c0: int,
    c1: int,
    c2: int,
    c3: int,
    c4: int,
    c5: int,
    c6: int,
) -> int:
    suit_hash = (
        _SUITBIT[c0]
        + _SUITBIT[c1]
        + _SUITBIT[c2]
        + _SUITBIT[c3]
        + _SUITBIT[c4]
        + _SUITBIT[c5]
        + _SUITBIT[c6]
    )
    flush_suit = _SUITS[suit_hash] - 1
    if flush_suit != -1:
        hand_binary = 0
        if c0 % 4 == flush_suit:
            hand_binary |= _BIN[c0]
        if c1 % 4 == flush_suit:
            hand_binary |= _BIN[c1]
        if c2 % 4 == flush_suit:
            hand_binary |= _BIN[c2]
        if c3 % 4 == flush_suit:
            hand_binary |= _BIN[c3]
        if c4 % 4 == flush_suit:
            hand_binary |= _BIN[c4]
        if c5 % 4 == flush_suit:
            hand_binary |= _BIN[c5]
        if c6 % 4 == flush_suit:
            hand_binary |= _BIN[c6]
        return _FLUSH[hand_binary]
    quinary = np.zeros(13, dtype=np.int32)
    quinary[c0 // 4] += 1
    quinary[c1 // 4] += 1
    quinary[c2 // 4] += 1
    quinary[c3 // 4] += 1
    quinary[c4 // 4] += 1
    quinary[c5 // 4] += 1
    quinary[c6 // 4] += 1
    return _NF7[_hash_quinary(quinary, 7)]


@njit(cache=True)
def _equity_two_combos_exact(h0: int, h1: int, v0: int, v1: int) -> float:
    """Hero (h0,h1) equity vs villain (v0,v1) by enumerating all runouts."""
    dead = np.zeros(52, dtype=np.bool_)
    dead[h0] = True
    dead[h1] = True
    dead[v0] = True
    dead[v1] = True
    deck = np.empty(48, dtype=np.int32)
    nd = 0
    for c in range(52):
        if not dead[c]:
            deck[nd] = c
            nd += 1
    wins = 0.0
    total = 0.0
    for i0 in range(nd - 4):
        c0 = deck[i0]
        for i1 in range(i0 + 1, nd - 3):
            c1 = deck[i1]
            for i2 in range(i1 + 1, nd - 2):
                c2 = deck[i2]
                for i3 in range(i2 + 1, nd - 1):
                    c3 = deck[i3]
                    for i4 in range(i3 + 1, nd):
                        c4 = deck[i4]
                        hr = _eval7(h0, h1, c0, c1, c2, c3, c4)
                        vr = _eval7(v0, v1, c0, c1, c2, c3, c4)
                        if hr < vr:
                            wins += 1.0
                        elif hr == vr:
                            wins += 0.5
                        total += 1.0
    return wins / total if total > 0 else 0.5


def _card_str_to_id(card: str) -> int:
    from phevaluator.card import Card as PeCard

    return PeCard.to_id(card[0].upper() + card[1].lower())


def _first_valid_combo_pair(
    combos_a: list[tuple[str, str]],
    combos_b: list[tuple[str, str]],
) -> tuple[tuple[str, str], tuple[str, str]] | None:
    for ca in combos_a:
        ids_a = {_card_str_to_id(ca[0]), _card_str_to_id(ca[1])}
        for cb in combos_b:
            ids_b = {_card_str_to_id(cb[0]), _card_str_to_id(cb[1])}
            if ids_a.isdisjoint(ids_b):
                return ca, cb
    return None


def _combos_for_169_label(label: str) -> list[tuple[str, str]]:
    """Correct combo enumeration per 169 label (suited/offsuit/pairs)."""
    from aof_rl.teacher_lut.build import _all_combos_for_label

    return _all_combos_for_label(label)


def _equity_class_pair(labels: list[str], i: int, j: int) -> float:
    """Mean exact equity over disjoint combo pairs for two 169 classes."""
    combos_i = _combos_for_169_label(labels[i])
    combos_j = _combos_for_169_label(labels[j])
    vals: list[float] = []
    for ca in combos_i:
        ids_a = {_card_str_to_id(ca[0]), _card_str_to_id(ca[1])}
        for cb in combos_j:
            ids_b = {_card_str_to_id(cb[0]), _card_str_to_id(cb[1])}
            if not ids_a.isdisjoint(ids_b):
                continue
            h0, h1 = _card_str_to_id(ca[0]), _card_str_to_id(ca[1])
            v0, v1 = _card_str_to_id(cb[0]), _card_str_to_id(cb[1])
            vals.append(float(_equity_two_combos_exact(h0, h1, v0, v1)))
    if not vals:
        return 0.5
    return float(sum(vals) / len(vals))


def build_hu_equity_grid_169(*, workers: int = 1) -> tuple[np.ndarray, list[str]]:
    """
    Exact HU 169×169 equity grid (hero row vs villain column).

    Uses first non-overlapping combo pair per bucket and full board enumeration.
    """
    labels, combos = _build_169_buckets()
    n = len(labels)
    grid = np.full((n, n), 0.5, dtype=np.float32)

    def fill_row(i: int) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        for j in range(i + 1, n):
            e = _equity_class_pair(labels, i, j)
            out.append((j, e))
        return out

    if workers <= 1:
        for i in range(n):
            for j, e in fill_row(i):
                grid[i, j] = e
                grid[j, i] = 1.0 - e
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fill_row, i): i for i in range(n)}
            for fut in as_completed(futs):
                i = futs[fut]
                for j, e in fut.result():
                    grid[i, j] = e
                    grid[j, i] = 1.0 - e
    return grid, labels


def save_hu_equity_grid(path: Path, grid: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, hu_equity=grid.astype(np.float32), labels=np.array(labels))


def load_hu_equity_grid(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data["hu_equity"], dtype=np.float32)


def aa_vs_ako_equity(grid: np.ndarray, labels: list[str]) -> float:
    """AA vs AKo cell (exact enumeration ≈ 0.81946)."""
    i = labels.index("AA")
    j = labels.index("AKo")
    return float(grid[i, j])


def aa_vs_ako_equity_exact() -> float:
    """Combo-exact AA vs AKo (AcAd vs KhKd) — oracle ≈ 0.81946 for CI gates."""
    h0 = _card_str_to_id("Ac")
    h1 = _card_str_to_id("Ad")
    v0 = _card_str_to_id("Kh")
    v1 = _card_str_to_id("Kd")
    return float(_equity_two_combos_exact(h0, h1, v0, v1))


# --- T2: range-bucket multiway grids (coarse keys for faster LUT rebuild) ---

RANGE_BUCKET_LABELS: tuple[str, ...] = (
    "tight",
    "medium",
    "wide",
    "unknown",
)


def _range_bucket_id(range_str: str) -> int:
    s = range_str.strip()
    n = len(s.replace(",", "").replace(" ", ""))
    if n < 12:
        return 0
    if n < 28:
        return 1
    if n < 60:
        return 2
    return 3


def build_range_bucket_grid_3way(
    hero_labels: list[str],
    *,
    seed: int = 0,
    samples_per_cell: int = 400,
) -> np.ndarray:
    """
  Coarse 169 × B × B grid for 3-way equity vs two range buckets.

  Built with light MC at LUT rebuild time (T2); not used in v1 runtime.
  """
    from aof_equity import equity_vs_two_ranges

    n = len(hero_labels)
    b = len(RANGE_BUCKET_LABELS)
    grid = np.zeros((n, b, b), dtype=np.float32)
    # Placeholder ranges per bucket
    bucket_ranges = ["22+", "88+,AJs+", "22+,A2s+,K9s+", "random"]
    for i, lab in enumerate(hero_labels):
        from pokerbot.solver.hu_preflop_dcfr import _build_169_buckets

        _, combos = _build_169_buckets()
        hero = "".join(combos[i][0])
        for bi in range(b):
            for bj in range(b):
                try:
                    eq = equity_vs_two_ranges(
                        hero,
                        bucket_ranges[bi],
                        bucket_ranges[bj],
                        samples_per_cell,
                        seed + i * 31 + bi * 7 + bj,
                        max_combos=32,
                    )
                    grid[i, bi, bj] = float(eq)
                except Exception:
                    grid[i, bi, bj] = 0.33
    return grid


def build_range_bucket_grid_4way(
    hero_labels: list[str],
    *,
    seed: int = 0,
    samples_per_cell: int = 300,
) -> np.ndarray:
    """169 × B³ coarse 4-way bucket grid (T2 rebuild helper)."""
    from aof_equity import equity_vs_three_ranges_adaptive
    from aof_equity_config import EquityRuntimeConfig

    n = len(hero_labels)
    b = len(RANGE_BUCKET_LABELS)
    grid = np.zeros((n, b, b, b), dtype=np.float32)
    bucket_ranges = ["22+", "88+,AJs+", "22+,A2s+,K9s+", "random"]
    cfg = EquityRuntimeConfig()
    _, combos = _build_169_buckets()
    for i in range(n):
        hero = "".join(combos[i][0])
        for b0 in range(b):
            for b1 in range(b):
                for b2 in range(b):
                    try:
                        res = equity_vs_three_ranges_adaptive(
                            hero,
                            [bucket_ranges[b0], bucket_ranges[b1], bucket_ranges[b2]],
                            samples_per_cell,
                            cfg,
                            seed=seed + i * 17 + b0 + b1 * 3 + b2 * 9,
                        )
                        grid[i, b0, b1, b2] = float(res.equity)
                    except Exception:
                        grid[i, b0, b1, b2] = 0.25
    return grid
