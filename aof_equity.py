"""Vectorized preflop equity (numpy sampling + numba JIT phevaluator tables)."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange
from phevaluator.card import Card as PeCard
from phevaluator.tables import (
    BINARIES_BY_ID,
    DP,
    FLUSH,
    NO_FLUSH_7,
    SUITBIT_BY_ID,
    SUITS,
)
from pokerkit import Card as PkCard, parse_range

if TYPE_CHECKING:
    pass

# Lookup tables as contiguous arrays for numba.
_BIN = np.array(BINARIES_BY_ID, dtype=np.uint64)
_SUITBIT = np.array(SUITBIT_BY_ID, dtype=np.int32)
_SUITS = np.array(SUITS, dtype=np.int32)
_FLUSH = np.array(FLUSH, dtype=np.int32)
_NF7 = np.array(NO_FLUSH_7, dtype=np.int32)
_DP = np.array(DP, dtype=np.int32)


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


@njit(parallel=True, cache=True)
def _mc_hu_from_boards(
    h0: int,
    h1: int,
    v0: int,
    v1: int,
    boards: np.ndarray,
) -> float:
    n = boards.shape[0]
    wins = 0.0
    for i in prange(n):
        b0 = boards[i, 0]
        b1 = boards[i, 1]
        b2 = boards[i, 2]
        b3 = boards[i, 3]
        b4 = boards[i, 4]
        hr = _eval7(h0, h1, b0, b1, b2, b3, b4)
        vr = _eval7(v0, v1, b0, b1, b2, b3, b4)
        if hr < vr:
            wins += 1.0
        elif hr == vr:
            wins += 0.5
    return wins / n


def _sample_boards(
    h0: int,
    h1: int,
    v0: int,
    v1: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dead = np.zeros(52, dtype=np.bool_)
    dead[h0] = dead[h1] = dead[v0] = dead[v1] = True
    deck = np.nonzero(~dead)[0].astype(np.uint8)
    noise = rng.random((n, deck.size))
    pick = np.argsort(noise, axis=1)[:, :5]
    return deck[pick]


def _pk_card_id(c: PkCard) -> int:
    return PeCard.to_id(f"{c.rank.value}{c.suit.value}")


@functools.lru_cache(maxsize=1024)
def _range_combo_ids(range_str: str, max_combos: int, seed: int) -> np.ndarray:
    combos = list(parse_range(range_str))
    if not combos:
        return np.empty((0, 2), dtype=np.uint8)
    pairs = np.empty((len(combos), 2), dtype=np.uint8)
    for i, combo in enumerate(combos):
        cs = list(combo)
        pairs[i, 0] = _pk_card_id(cs[0])
        pairs[i, 1] = _pk_card_id(cs[1])
    if len(pairs) <= max_combos:
        return pairs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=max_combos, replace=False)
    return pairs[idx]


@njit(cache=True)
def _mc_equity_one_combo(
    h0: int,
    h1: int,
    v0: int,
    v1: int,
    n_samples: int,
    seed: int,
) -> float:
    """Sample n boards in-numba and return hero equity vs one villain combo."""
    if n_samples <= 0:
        return 0.5
    dead = np.zeros(52, dtype=np.bool_)
    dead[h0] = True
    dead[h1] = True
    dead[v0] = True
    dead[v1] = True
    deck = np.empty(48, dtype=np.int32)
    n_deck = 0
    for c in range(52):
        if not dead[c]:
            deck[n_deck] = c
            n_deck += 1
    wins = 0.0
    state = np.uint64(seed & 0xFFFFFFFF)
    for _ in range(n_samples):
        used = np.zeros(52, dtype=np.bool_)
        used[h0] = True
        used[h1] = True
        used[v0] = True
        used[v1] = True
        board = np.empty(5, dtype=np.int32)
        for bi in range(5):
            state = (state * np.uint64(6364136223846793005) + np.uint64(1)) & np.uint64(
                0xFFFFFFFFFFFFFFFF
            )
            pick = int(state % np.uint64(n_deck))
            card = deck[pick]
            tries = 0
            while used[card] and tries < n_deck:
                state = (state * np.uint64(6364136223846793005) + np.uint64(1)) & np.uint64(
                    0xFFFFFFFFFFFFFFFF
                )
                pick = int(state % np.uint64(n_deck))
                card = deck[pick]
                tries += 1
            used[card] = True
            board[bi] = card
        hr = _eval7(h0, h1, board[0], board[1], board[2], board[3], board[4])
        vr = _eval7(v0, v1, board[0], board[1], board[2], board[3], board[4])
        if hr < vr:
            wins += 1.0
        elif hr == vr:
            wins += 0.5
    return wins / n_samples


@njit(parallel=True, cache=True)
def _equity_vs_range_numba(
    h0: int,
    h1: int,
    v0_arr: np.ndarray,
    v1_arr: np.ndarray,
    n_per_combo: np.ndarray,
    base_seed: int,
) -> float:
    n_valid = v0_arr.shape[0]
    eqs = np.empty(n_valid, dtype=np.float64)
    for k in prange(n_valid):
        n = int(n_per_combo[k])
        if n <= 0:
            eqs[k] = 0.5
        else:
            eqs[k] = _mc_equity_one_combo(
                h0, h1, int(v0_arr[k]), int(v1_arr[k]), n, base_seed + k * 9973
            )
    wins_weighted = 0.0
    total_n = 0
    for k in range(n_valid):
        n = int(n_per_combo[k])
        if n <= 0:
            continue
        wins_weighted += eqs[k] * n
        total_n += n
    if total_n <= 0:
        return 0.5
    return wins_weighted / total_n


def equity_vs_range_fast(
    hero_key: str,
    range_str: str,
    samples: int,
    seed: int,
    max_combos: int = 64,
) -> float:
    """Hero equity vs range; numba parallel MC over villain combos."""
    from aof_equity_cache import get_equity_cache

    cache = get_equity_cache()
    cached = cache.get(hero_key, range_str, samples)
    if cached is not None:
        return cached

    h0 = PeCard.to_id(hero_key[:2])
    h1 = PeCard.to_id(hero_key[2:])
    v_pairs = _range_combo_ids(range_str, max_combos, seed ^ 0x9E3779B1)
    if v_pairs.size == 0:
        return 0.5

    v0_list: list[int] = []
    v1_list: list[int] = []
    for k in range(v_pairs.shape[0]):
        v0 = int(v_pairs[k, 0])
        v1 = int(v_pairs[k, 1])
        if v0 in (h0, h1) or v1 in (h0, h1):
            continue
        v0_list.append(v0)
        v1_list.append(v1)
    n_valid = len(v0_list)
    if n_valid == 0:
        return 0.5

    v0_arr = np.asarray(v0_list, dtype=np.int32)
    v1_arr = np.asarray(v1_list, dtype=np.int32)
    base, extra = divmod(samples, n_valid)
    n_per = np.empty(n_valid, dtype=np.int32)
    for k in range(n_valid):
        n_per[k] = base + (1 if k < extra else 0)

    result = float(_equity_vs_range_numba(h0, h1, v0_arr, v1_arr, n_per, seed))
    cache.put(hero_key, range_str, samples, result)
    return result


@functools.lru_cache(maxsize=512)
def equity_vs_range(
    hero_key: str,
    range_str: str,
    samples: int,
    seed: int,
    max_combos: int,
) -> float:
    """Hero equity vs villain calling range (Monte Carlo, vectorized)."""
    return equity_vs_range_fast(
        hero_key, range_str, samples, seed, max_combos
    )


@njit(cache=True)
def _share_one_board_3(
    h0: int,
    h1: int,
    v0: int,
    v1: int,
    w0: int,
    w1: int,
    b0: int,
    b1: int,
    b2: int,
    b3: int,
    b4: int,
) -> tuple:
    hr = _eval7(h0, h1, b0, b1, b2, b3, b4)
    vr = _eval7(v0, v1, b0, b1, b2, b3, b4)
    wr = _eval7(w0, w1, b0, b1, b2, b3, b4)
    best = hr
    if vr < best:
        best = vr
    if wr < best:
        best = wr
    n_top = 0
    if hr == best:
        n_top += 1
    if vr == best:
        n_top += 1
    if wr == best:
        n_top += 1
    s0 = 1.0 / n_top if hr == best else 0.0
    s1 = 1.0 / n_top if vr == best else 0.0
    s2 = 1.0 / n_top if wr == best else 0.0
    return s0, s1, s2


@njit(parallel=True, cache=True)
def _mc_3way_shares_all(
    h0: int,
    h1: int,
    v0: int,
    v1: int,
    w0: int,
    w1: int,
    boards: np.ndarray,
) -> np.ndarray:
    """Return normalized pot shares for players 0,1,2 (same hole order as args)."""
    n = boards.shape[0]
    if n <= 0:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    p0 = np.zeros(n, dtype=np.float64)
    p1 = np.zeros(n, dtype=np.float64)
    p2 = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        s0, s1, s2 = _share_one_board_3(
            h0,
            h1,
            v0,
            v1,
            w0,
            w1,
            boards[i, 0],
            boards[i, 1],
            boards[i, 2],
            boards[i, 3],
            boards[i, 4],
        )
        p0[i] = s0
        p1[i] = s1
        p2[i] = s2
    acc = np.array([p0.sum(), p1.sum(), p2.sum()], dtype=np.float64) / n
    total = acc[0] + acc[1] + acc[2]
    if total > 0:
        acc /= total
    return acc


@njit(parallel=True, cache=True)
def _mc_3way_from_boards(
    h0: int,
    h1: int,
    v0: int,
    v1: int,
    w0: int,
    w1: int,
    boards: np.ndarray,
) -> float:
    n = boards.shape[0]
    wins = 0.0
    for i in prange(n):
        b0 = boards[i, 0]
        b1 = boards[i, 1]
        b2 = boards[i, 2]
        b3 = boards[i, 3]
        b4 = boards[i, 4]
        hr = _eval7(h0, h1, b0, b1, b2, b3, b4)
        vr = _eval7(v0, v1, b0, b1, b2, b3, b4)
        wr = _eval7(w0, w1, b0, b1, b2, b3, b4)
        if hr < vr and hr < wr:
            wins += 1.0
        elif hr == vr and hr == wr:
            wins += 1.0 / 3.0
        elif (hr == vr and hr < wr) or (hr == wr and hr < vr):
            wins += 0.5
    return wins / n


def _sample_boards_3way(
    h0: int,
    h1: int,
    v0: int,
    v1: int,
    w0: int,
    w1: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dead = np.zeros(52, dtype=np.bool_)
    for c in (h0, h1, v0, v1, w0, w1):
        dead[c] = True
    deck = np.nonzero(~dead)[0].astype(np.uint8)
    noise = rng.random((n, deck.size))
    pick = np.argsort(noise, axis=1)[:, :5]
    return deck[pick]


# Hero share in 3-way pots uses _mc_3way_from_boards / _share_one_board_3.
# RL terminal payoffs use _mc_3way_shares_all in aof_rl.payoff (chip splits, known holes).

EXACT_PAIR_CAP = 256
FACING2_MAX_COMBOS = 96


@functools.lru_cache(maxsize=256)
def equity_vs_two_ranges(
    hero_key: str,
    range_a: str,
    range_b: str,
    samples: int,
    seed: int,
    max_combos: int,
) -> float:
    """Hero equity vs two independent villain ranges (3-way MC)."""
    h0 = PeCard.to_id(hero_key[:2])
    h1 = PeCard.to_id(hero_key[2:])
    pairs_a = _range_combo_ids(range_a, max_combos, seed ^ 0x9E3779B1)
    pairs_b = _range_combo_ids(range_b, max_combos, seed ^ 0x85EBCA6B)
    if pairs_a.size == 0 or pairs_b.size == 0:
        return 0.5

    rng = np.random.default_rng(seed)
    n_pairs = pairs_a.shape[0] * pairs_b.shape[0]
    base, extra = divmod(samples, n_pairs)
    total_wins = 0.0
    total_n = 0
    idx = 0

    for ka in range(pairs_a.shape[0]):
        v0, v1 = int(pairs_a[ka, 0]), int(pairs_a[ka, 1])
        for kb in range(pairs_b.shape[0]):
            w0, w1 = int(pairs_b[kb, 0]), int(pairs_b[kb, 1])
            dead = {h0, h1, v0, v1, w0, w1}
            if len(dead) < 6:
                idx += 1
                continue
            n = base + (1 if idx < extra else 0)
            idx += 1
            if n <= 0:
                continue
            boards = _sample_boards_3way(h0, h1, v0, v1, w0, w1, n, rng)
            eq = _mc_3way_from_boards(h0, h1, v0, v1, w0, w1, boards)
            total_wins += eq * n
            total_n += n

    if total_n == 0:
        return 0.5
    return total_wins / total_n


def equity_vs_two_ranges_adaptive(
    hero_key: str,
    range_a: str,
    range_b: str,
    samples: int,
    seed: int,
    *,
    max_combos: int = FACING2_MAX_COMBOS,
    return_meta: bool = False,
) -> float | tuple[float, dict]:
    """
    Facing-2 path: exact enumeration when few valid combo pairs, else seeded MC.

    Card blocking is enforced (six distinct hole cards required per sample).
    """
    h0 = PeCard.to_id(hero_key[:2])
    h1 = PeCard.to_id(hero_key[2:])
    cap = max(max_combos, 512)
    pairs_a = _range_combo_ids(range_a, cap, seed ^ 0x9E3779B1)
    pairs_b = _range_combo_ids(range_b, cap, seed ^ 0x85EBCA6B)
    if pairs_a.size == 0 or pairs_b.size == 0:
        return 0.5

    valid: list[tuple[int, int, int, int]] = []
    for ka in range(pairs_a.shape[0]):
        v0, v1 = int(pairs_a[ka, 0]), int(pairs_a[ka, 1])
        for kb in range(pairs_b.shape[0]):
            w0, w1 = int(pairs_b[kb, 0]), int(pairs_b[kb, 1])
            if len({h0, h1, v0, v1, w0, w1}) >= 6:
                valid.append((v0, v1, w0, w1))

    if not valid:
        out = 0.5
        meta = {
            "method": "mc",
            "samples_used": 0,
            "valid_products": 0,
            "warning": "no valid combo pairs",
        }
        return (out, meta) if return_meta else out
    if len(valid) > EXACT_PAIR_CAP:
        out = equity_vs_two_ranges(
            hero_key, range_a, range_b, samples, seed, max_combos
        )
        meta = {
            "method": "mc",
            "samples_used": samples,
            "valid_products": len(valid),
            "warning": "combo product over cap; subsampled MC",
        }
        return (out, meta) if return_meta else out

    rng = np.random.default_rng(seed)
    base, extra = divmod(samples, len(valid))
    total_wins = 0.0
    total_n = 0
    for idx, (v0, v1, w0, w1) in enumerate(valid):
        n = base + (1 if idx < extra else 0)
        if n <= 0:
            continue
        boards = _sample_boards_3way(h0, h1, v0, v1, w0, w1, n, rng)
        eq = _mc_3way_from_boards(h0, h1, v0, v1, w0, w1, boards)
        total_wins += eq * n
        total_n += n
    if total_n == 0:
        out = 0.5
    else:
        out = total_wins / total_n
    meta = {
        "method": "exact",
        "samples_used": int(total_n),
        "valid_products": len(valid),
        "warning": None,
    }
    return (out, meta) if return_meta else out


def _hero_hole_ids(hero_key: str) -> tuple[int, int]:
    h0 = PeCard.to_id(hero_key[:2])
    h1 = PeCard.to_id(hero_key[2:])
    return h0, h1


def _valid_combo_tuples(
    h0: int,
    h1: int,
    range_strs: list[str],
    max_combos: int,
    seed: int,
) -> list[tuple[int, ...]]:
    """Enumerate villain hole id tuples (2 cards per range) with no overlap with hero."""
    pair_lists: list[np.ndarray] = []
    for i, rs in enumerate(range_strs):
        pairs = _range_combo_ids(rs, max_combos, seed ^ (0x9E3779B1 + i * 0x85EBCA6B))
        if pairs.size == 0:
            return []
        pair_lists.append(pairs)

    def rec(depth: int, chosen: list[int]) -> None:
        if depth == len(pair_lists):
            dead = {h0, h1, *chosen}
            if len(dead) == 2 + len(chosen):
                valid.append(tuple(chosen))
            return
        for row in range(pair_lists[depth].shape[0]):
            c0, c1 = int(pair_lists[depth][row, 0]), int(pair_lists[depth][row, 1])
            if c0 in (h0, h1) or c1 in (h0, h1):
                continue
            if any(c in chosen for c in (c0, c1)):
                continue
            rec(depth + 1, chosen + [c0, c1])

    valid: list[tuple[int, ...]] = []
    rec(0, [])
    return valid


def equity_vs_three_ranges_adaptive(
    hero_key: str,
    range_strs: list[str],
    samples: int,
    seed: int,
    *,
    max_combos: int = FACING2_MAX_COMBOS,
    config=None,
    return_meta: bool = False,
) -> float | tuple[float, dict]:
    """Hero equity vs three independent villain ranges (4-way pot)."""
    from aof_equity_config import EquityRuntimeConfig, default_equity_config

    cfg = config or default_equity_config()
    cap = cfg.exact_product_cap_3way
    h0, h1 = _hero_hole_ids(hero_key)
    valid = _valid_combo_tuples(h0, h1, range_strs[:3], max_combos, seed)
    if not valid:
        out = 0.5
        meta = {"method": "mc", "samples_used": 0, "valid_products": 0, "warning": None}
        return (out, meta) if return_meta else out

    if len(valid) > cap:
        out = _equity_vs_n_ranges_mc(
            h0, h1, range_strs[:3], samples, seed, max_combos, n_villains=3
        )
        meta = {
            "method": "mc",
            "samples_used": samples,
            "valid_products": len(valid),
            "warning": "combo product over cap; subsampled MC",
        }
        return (out, meta) if return_meta else out

    rng = np.random.default_rng(seed)
    base, extra = divmod(samples, len(valid))
    total_wins = 0.0
    total_n = 0
    for idx, combo in enumerate(valid):
        v0, v1, w0, w1, x0, x1 = combo
        n = base + (1 if idx < extra else 0)
        if n <= 0:
            continue
        boards = _sample_boards_4way(
            np.array([h0, h1, v0, v1, w0, w1, x0, x1], dtype=np.int32), n, rng
        )
        eq = _mc_4way_hero_share(h0, h1, v0, v1, w0, w1, x0, x1, boards)
        total_wins += eq * n
        total_n += n
    out = total_wins / total_n if total_n else 0.5
    meta = {
        "method": "exact",
        "samples_used": int(total_n),
        "valid_products": len(valid),
        "warning": None,
    }
    return (out, meta) if return_meta else out


def equity_vs_four_ranges_adaptive(
    hero_key: str,
    range_strs: list[str],
    samples: int,
    seed: int,
    *,
    max_combos: int = FACING2_MAX_COMBOS,
    config=None,
    return_meta: bool = False,
) -> float | tuple[float, dict]:
    """Hero equity vs four villain ranges (5-way pot)."""
    from aof_equity_config import EquityRuntimeConfig, default_equity_config

    cfg = config or default_equity_config()
    cap = cfg.exact_product_cap_4way
    h0, h1 = _hero_hole_ids(hero_key)
    valid = _valid_combo_tuples(h0, h1, range_strs[:4], max_combos, seed)
    if not valid:
        out = 0.5
        meta = {"method": "mc", "samples_used": 0, "valid_products": 0, "warning": None}
        return (out, meta) if return_meta else out

    if len(valid) > cap:
        out = _equity_vs_n_ranges_mc(
            h0, h1, range_strs[:4], samples, seed, max_combos, n_villains=4
        )
        meta = {
            "method": "mc",
            "samples_used": samples,
            "valid_products": len(valid),
            "warning": "combo product over cap; subsampled MC",
        }
        return (out, meta) if return_meta else out

    rng = np.random.default_rng(seed)
    base, extra = divmod(samples, len(valid))
    total_wins = 0.0
    total_n = 0
    for idx, combo in enumerate(valid):
        ids = [h0, h1, *combo]
        n = base + (1 if idx < extra else 0)
        if n <= 0:
            continue
        hole_arr = np.array(ids, dtype=np.int32)
        boards = _sample_boards_4way(hole_arr, n, rng)
        eq = _mc_4way_hero_share(
            ids[0],
            ids[1],
            ids[2],
            ids[3],
            ids[4],
            ids[5],
            ids[6],
            ids[7],
            boards,
        )
        total_wins += eq * n
        total_n += n
    out = total_wins / total_n if total_n else 0.5
    meta = {
        "method": "exact",
        "samples_used": int(total_n),
        "valid_products": len(valid),
        "warning": None,
    }
    return (out, meta) if return_meta else out


def _equity_vs_n_ranges_mc(
    h0: int,
    h1: int,
    range_strs: list[str],
    samples: int,
    seed: int,
    max_combos: int,
    *,
    n_villains: int,
) -> float:
    """Subsampled MC when combo product is too large for exact enumeration."""
    rng = np.random.default_rng(seed)
    pair_lists = [
        _range_combo_ids(rs, max_combos, seed ^ (0x9E3779B1 + i * 9973))
        for i, rs in enumerate(range_strs[:n_villains])
    ]
    if any(p.size == 0 for p in pair_lists):
        return 0.5
    wins = 0.0
    used = 0
    for _ in range(max(samples, 1)):
        chosen: list[int] = []
        ok = True
        for pairs in pair_lists:
            row = int(rng.integers(0, pairs.shape[0]))
            c0, c1 = int(pairs[row, 0]), int(pairs[row, 1])
            if c0 in (h0, h1) or c1 in (h0, h1) or c0 in chosen or c1 in chosen:
                ok = False
                break
            chosen.extend([c0, c1])
        if not ok or len({h0, h1, *chosen}) < 2 + 2 * n_villains:
            continue
        if n_villains == 3:
            v0, v1, w0, w1, x0, x1 = chosen
            boards = _sample_boards_4way(
                np.array([h0, h1, v0, v1, w0, w1, x0, x1], dtype=np.int32), 1, rng
            )
            wins += _mc_4way_hero_share(h0, h1, v0, v1, w0, w1, x0, x1, boards)
        else:
            ids = [h0, h1, *chosen]
            boards = _sample_boards_4way(np.array(ids, dtype=np.int32), 1, rng)
            wins += _mc_4way_hero_share(
                ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], boards
            )
        used += 1
    return wins / used if used else 0.5


def _card_id_str(card: str) -> int:
    return PeCard.to_id(card[0].upper() + card[1].lower())


def equity_hu_exact(
    cards_a: tuple[str, str],
    cards_b: tuple[str, str],
    samples: int,
    seed: int,
) -> float:
    """P(player A wins) at showdown vs known B hole (numba MC)."""
    a0, a1 = _card_id_str(cards_a[0]), _card_id_str(cards_a[1])
    b0, b1 = _card_id_str(cards_b[0]), _card_id_str(cards_b[1])
    rng = np.random.default_rng(seed)
    boards = _sample_boards(a0, a1, b0, b1, max(samples, 1), rng)
    return float(_mc_hu_from_boards(a0, a1, b0, b1, boards))


@njit(cache=True)
def _share_one_board_4(
    h0: int,
    h1: int,
    v0: int,
    v1: int,
    w0: int,
    w1: int,
    x0: int,
    x1: int,
    b0: int,
    b1: int,
    b2: int,
    b3: int,
    b4: int,
) -> tuple:
    hr = _eval7(h0, h1, b0, b1, b2, b3, b4)
    vr = _eval7(v0, v1, b0, b1, b2, b3, b4)
    wr = _eval7(w0, w1, b0, b1, b2, b3, b4)
    xr = _eval7(x0, x1, b0, b1, b2, b3, b4)
    best = hr
    if vr < best:
        best = vr
    if wr < best:
        best = wr
    if xr < best:
        best = xr
    n_top = 0
    if hr == best:
        n_top += 1
    if vr == best:
        n_top += 1
    if wr == best:
        n_top += 1
    if xr == best:
        n_top += 1
    s0 = 1.0 / n_top if hr == best else 0.0
    s1 = 1.0 / n_top if vr == best else 0.0
    s2 = 1.0 / n_top if wr == best else 0.0
    s3 = 1.0 / n_top if xr == best else 0.0
    return s0, s1, s2, s3


@njit(parallel=True, cache=True)
def _mc_4way_shares_all(
    h0: int,
    h1: int,
    v0: int,
    v1: int,
    w0: int,
    w1: int,
    x0: int,
    x1: int,
    boards: np.ndarray,
) -> np.ndarray:
    """Return normalized pot shares for four players (hole order matches args)."""
    n = boards.shape[0]
    if n <= 0:
        return np.array([0.25, 0.25, 0.25, 0.25])
    p0 = np.zeros(n, dtype=np.float64)
    p1 = np.zeros(n, dtype=np.float64)
    p2 = np.zeros(n, dtype=np.float64)
    p3 = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        s0, s1, s2, s3 = _share_one_board_4(
            h0,
            h1,
            v0,
            v1,
            w0,
            w1,
            x0,
            x1,
            boards[i, 0],
            boards[i, 1],
            boards[i, 2],
            boards[i, 3],
            boards[i, 4],
        )
        p0[i] = s0
        p1[i] = s1
        p2[i] = s2
        p3[i] = s3
    acc = np.array([p0.sum(), p1.sum(), p2.sum(), p3.sum()], dtype=np.float64) / n
    total = acc[0] + acc[1] + acc[2] + acc[3]
    if total > 0:
        acc /= total
    return acc


@njit(parallel=True, cache=True)
def _mc_4way_hero_share(
    h0: int,
    h1: int,
    v0: int,
    v1: int,
    w0: int,
    w1: int,
    x0: int,
    x1: int,
    boards: np.ndarray,
) -> float:
    """Hero fraction of pot at showdown (4 players, known holes)."""
    n = boards.shape[0]
    share = 0.0
    for i in prange(n):
        b0 = boards[i, 0]
        b1 = boards[i, 1]
        b2 = boards[i, 2]
        b3 = boards[i, 3]
        b4 = boards[i, 4]
        hr = _eval7(h0, h1, b0, b1, b2, b3, b4)
        vr = _eval7(v0, v1, b0, b1, b2, b3, b4)
        wr = _eval7(w0, w1, b0, b1, b2, b3, b4)
        xr = _eval7(x0, x1, b0, b1, b2, b3, b4)
        best = hr
        if vr < best:
            best = vr
        if wr < best:
            best = wr
        if xr < best:
            best = xr
        n_top = 0
        if hr == best:
            n_top += 1
        if vr == best:
            n_top += 1
        if wr == best:
            n_top += 1
        if xr == best:
            n_top += 1
        if hr == best and n_top > 0:
            share += 1.0 / n_top
    return share / n


def _sample_boards_4way(
    hole_ids: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample boards given 8 known card ids (4 players x 2)."""
    dead = np.zeros(52, dtype=np.bool_)
    for c in hole_ids:
        dead[int(c)] = True
    deck = np.nonzero(~dead)[0].astype(np.uint8)
    noise = rng.random((n, deck.size))
    pick = np.argsort(noise, axis=1)[:, :5]
    return deck[pick]


def warmup() -> None:
    """JIT-compile numba kernels (call once at startup)."""
    boards = np.array([[0, 4, 8, 12, 16]], dtype=np.uint8)
    _mc_hu_from_boards(48, 49, 0, 1, boards)
    _mc_3way_from_boards(48, 49, 0, 1, 2, 3, boards)
    _mc_4way_hero_share(48, 49, 0, 1, 2, 3, 4, 5, boards)
    v0 = np.array([0], dtype=np.int32)
    v1 = np.array([1], dtype=np.int32)
    n_per = np.array([1], dtype=np.int32)
    _equity_vs_range_numba(48, 49, v0, v1, n_per, 42)
    _mc_3way_shares_all(48, 49, 0, 1, 2, 3, boards)
    _mc_4way_shares_all(48, 49, 0, 1, 2, 3, 4, 5, boards)
