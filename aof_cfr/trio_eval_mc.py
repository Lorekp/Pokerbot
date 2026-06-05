"""Monte Carlo equity kernels for 3-max bucket tensor (OMPEval native or numba fallback)."""

from __future__ import annotations

import math
import random

import numpy as np
from numba import njit

from aof_cfr.nway_eval_orientation import (
    eval7_backend_name,
    eval7_seat_rank,
    native_eval7_available,
    pair_equity_counters_3way,
    seat_share_from_ranks,
)
from aof_equity_grids import _eval7, _exact_six_card_combo_metrics

__all__ = [
    "build_live_deck",
    "eval7_backend_name",
    "mc_eval_backend_name",
    "mc_nway_shares",
    "mc_nway_shares_adaptive",
]


def mc_eval_backend_name() -> str:
    return eval7_backend_name()


def build_live_deck(hole_ids_flat: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Build live deck after removing hole cards; assert size/distinct/disjoint invariants."""
    dead = np.zeros(52, dtype=np.bool_)
    seen: set[int] = set()
    for k in range(2 * n):
        cid = int(hole_ids_flat[k])
        if cid in seen:
            raise ValueError(f"duplicate hole card id {cid} in deal")
        seen.add(cid)
        if cid < 0 or cid >= 52:
            raise ValueError(f"hole card id out of range: {cid}")
        dead[cid] = True
    deck = np.empty(52, dtype=np.int32)
    nd = 0
    for c in range(52):
        if not dead[c]:
            deck[nd] = c
            nd += 1
    expected = 52 - len(seen)
    if nd != expected:
        raise AssertionError(f"live deck size {nd} != expected {expected}")
    if nd > 0 and len(set(deck[:nd].tolist())) != nd:
        raise AssertionError("live deck contains duplicate cards")
    for c in deck[:nd]:
        if int(c) in seen:
            raise AssertionError(f"live deck card {c} overlaps hole cards")
    return deck, nd


@njit(cache=True)
def _build_live_deck_numba(hole_ids_flat: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    dead = np.zeros(52, dtype=np.bool_)
    for k in range(2 * n):
        dead[hole_ids_flat[k]] = True
    deck = np.empty(52, dtype=np.int32)
    nd = 0
    for c in range(52):
        if not dead[c]:
            deck[nd] = c
            nd += 1
    return deck, nd


@njit(cache=True)
def _mc_nway_shares_numba(hole_ids_flat: np.ndarray, n: int, n_boards: int, seed: int) -> np.ndarray:
    shares, _ = _mc_nway_shares_and_pair_eq_numba(hole_ids_flat, n, n_boards, seed)
    return shares


@njit(cache=True)
def _mc_nway_shares_and_pair_eq_numba(
    hole_ids_flat: np.ndarray, n: int, n_boards: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    deck, nd = _build_live_deck_numba(hole_ids_flat, n)
    if nd < 5:
        out = np.empty(n, dtype=np.float64)
        for pi in range(n):
            out[pi] = 1.0 / n
        pair_eq = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        return out, pair_eq
    wins = np.zeros(n, dtype=np.float64)
    btn_bb_w = 0.0
    btn_sb_w = 0.0
    sb_bb_w = 0.0
    work = np.empty(nd, dtype=np.int32)
    board = np.empty(5, dtype=np.int32)
    np.random.seed(seed)
    for _ in range(n_boards):
        for di in range(nd):
            work[di] = deck[di]
        for bi in range(5):
            j = bi + np.random.randint(0, nd - bi)
            tmp = work[bi]
            work[bi] = work[j]
            work[j] = tmp
        for bi in range(5):
            board[bi] = work[bi]
        ranks = np.empty(n, dtype=np.int32)
        best = 1_000_000
        nwin = 0
        for pi in range(n):
            h0 = hole_ids_flat[2 * pi]
            h1 = hole_ids_flat[2 * pi + 1]
            rank = _eval7(h0, h1, board[0], board[1], board[2], board[3], board[4])
            ranks[pi] = rank
            if rank < best:
                best = rank
                nwin = 1
            elif rank == best:
                nwin += 1
        share = 1.0 / nwin
        for pi in range(n):
            if ranks[pi] == best:
                wins[pi] += share
        if n == 3:
            hr_btn = ranks[0]
            hr_sb = ranks[1]
            hr_bb = ranks[2]
            if hr_btn < hr_bb:
                btn_bb_w += 1.0
            elif hr_btn == hr_bb:
                btn_bb_w += 0.5
            if hr_btn < hr_sb:
                btn_sb_w += 1.0
            elif hr_btn == hr_sb:
                btn_sb_w += 0.5
            if hr_sb < hr_bb:
                sb_bb_w += 1.0
            elif hr_sb == hr_bb:
                sb_bb_w += 0.5
    inv = 1.0 / n_boards
    pair_eq = np.array([btn_bb_w * inv, btn_sb_w * inv, sb_bb_w * inv], dtype=np.float64)
    return wins * inv, pair_eq


def _pick_board_fisher_yates(deck: list[int], rng: random.Random) -> list[int]:
    work = list(deck)
    nd = len(work)
    for bi in range(5):
        j = bi + rng.randrange(nd - bi)
        work[bi], work[j] = work[j], work[bi]
    return work[:5]


def _board_share_vector(
    hole_ids_flat: np.ndarray,
    n: int,
    board: list[int],
    *,
    force_numba: bool = False,
) -> np.ndarray:
    ranks: list[int] = []
    native_omp = False
    for pi in range(n):
        h0 = int(hole_ids_flat[2 * pi])
        h1 = int(hole_ids_flat[2 * pi + 1])
        rank, native_omp = eval7_seat_rank(
            h0, h1, board[0], board[1], board[2], board[3], board[4], force_numba=force_numba
        )
        ranks.append(rank)
    return seat_share_from_ranks(ranks, native_omp=native_omp)


def _mc_nway_shares_native(
    hole_ids_flat: np.ndarray,
    n: int,
    n_boards: int,
    seed: int,
) -> np.ndarray:
    shares, _ = _mc_nway_shares_and_pair_eq_native(hole_ids_flat, n, n_boards, seed)
    return shares


def _mc_nway_shares_and_pair_eq_native(
    hole_ids_flat: np.ndarray,
    n: int,
    n_boards: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Native OMPEval eval7 + Fisher-Yates boards; optional 3-way pair eq from same samples."""
    deck_arr, nd = build_live_deck(hole_ids_flat, n)
    if nd < 5:
        return np.full(n, 1.0 / n, dtype=np.float64), np.array([0.5, 0.5, 0.5], dtype=np.float64)
    deck = [int(deck_arr[i]) for i in range(nd)]

    wins = np.zeros(n, dtype=np.float64)
    btn_bb_w = 0.0
    btn_sb_w = 0.0
    sb_bb_w = 0.0
    rng = random.Random(seed & 0xFFFFFFFF)
    for _ in range(n_boards):
        board = _pick_board_fisher_yates(deck, rng)
        ranks: list[int] = []
        native_omp = False
        for pi in range(n):
            h0 = int(hole_ids_flat[2 * pi])
            h1 = int(hole_ids_flat[2 * pi + 1])
            rank, native_omp = eval7_seat_rank(
                h0, h1, board[0], board[1], board[2], board[3], board[4]
            )
            ranks.append(rank)
        wins += seat_share_from_ranks(ranks, native_omp=native_omp)
        if n == 3:
            bb, bs, sb = pair_equity_counters_3way(ranks, native_omp=native_omp)
            btn_bb_w += bb
            btn_sb_w += bs
            sb_bb_w += sb
    inv = 1.0 / n_boards
    pair_eq = np.array([btn_bb_w * inv, btn_sb_w * inv, sb_bb_w * inv], dtype=np.float64)
    return wins * inv, pair_eq


def _exact_nway_shares(hole_ids_flat: np.ndarray, n: int) -> np.ndarray:
    from aof_cfr.terminal_ev import _exact_nway_shares_numba

    return _exact_nway_shares_numba(hole_ids_flat, n)


def _exact_nway_shares_and_pair_eq(
    hole_ids_flat: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    if n == 3 and hole_ids_flat.shape[0] >= 6:
        h = hole_ids_flat
        s0, s1, s2, bb, bs, sb = _exact_six_card_combo_metrics(
            int(h[0]), int(h[1]), int(h[2]), int(h[3]), int(h[4]), int(h[5])
        )
        return (
            np.array([s0, s1, s2], dtype=np.float64),
            np.array([bb, bs, sb], dtype=np.float64),
        )
    shares = _exact_nway_shares(hole_ids_flat, n)
    return shares, np.array([0.5, 0.5, 0.5], dtype=np.float64)


def mc_nway_shares(hole_ids_flat: np.ndarray, n: int, n_boards: int, seed: int) -> np.ndarray:
    """Monte Carlo pot shares; native eval7 when available (orientation-pinned)."""
    shares, _ = mc_nway_shares_and_pair_eq(hole_ids_flat, n, n_boards, seed)
    return shares


def mc_nway_shares_and_pair_eq(
    hole_ids_flat: np.ndarray, n: int, n_boards: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo pot shares + 3-way pair equities from the same board samples."""
    _, nd = build_live_deck(hole_ids_flat, n)
    if nd >= 5:
        n_runouts = math.comb(nd, 5)
        if n_boards >= n_runouts:
            return _exact_nway_shares_and_pair_eq(hole_ids_flat, n)
    if native_eval7_available():
        return _mc_nway_shares_and_pair_eq_native(hole_ids_flat, n, n_boards, seed)
    return _mc_nway_shares_and_pair_eq_numba(hole_ids_flat, n, n_boards, seed)


def mc_nway_shares_adaptive(
    hole_ids_flat: np.ndarray,
    n: int,
    seed: int,
    *,
    stderr_gate: float,
    max_boards: int,
    min_boards: int = 16,
    batch_boards: int = 64,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Sample boards until stderr gate or budget cap; return shares + pair_eq."""
    _, nd = build_live_deck(hole_ids_flat, n)
    if nd < 5:
        shares = np.full(n, 1.0 / n, dtype=np.float64)
        return shares, np.array([0.5, 0.5, 0.5], dtype=np.float64), 0, "degenerate"
    n_runouts = math.comb(nd, 5)
    if n_runouts <= max_boards:
        shares, pair_eq = _exact_nway_shares_and_pair_eq(hole_ids_flat, n)
        return shares, pair_eq, int(n_runouts), "exact_enum"

    total_boards = 0
    mean = np.zeros(n, dtype=np.float64)
    pair_mean = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    boards = 0
    step = max(min_boards, batch_boards)

    while boards < max_boards:
        add = min(step, max_boards - boards)
        chunk, pair_chunk = mc_nway_shares_and_pair_eq(hole_ids_flat, n, add, seed + boards)
        boards += add
        if total_boards == 0:
            mean = chunk.copy()
            pair_mean = pair_chunk.copy()
        else:
            mean = (mean * total_boards + chunk * add) / boards
            pair_mean = (pair_mean * total_boards + pair_chunk * add) / boards
        total_boards = boards
        stderr_bound = 0.5 / (2.0 * math.sqrt(total_boards))
        if total_boards >= min_boards and stderr_bound <= stderr_gate:
            return mean.copy(), pair_mean.copy(), total_boards, "mc"
        step = min(step * 2, 512)

    if total_boards >= n_runouts:
        shares, pair_eq = _exact_nway_shares_and_pair_eq(hole_ids_flat, n)
        return shares, pair_eq, int(n_runouts), "exact_enum_cap"
    return mean.copy(), pair_mean.copy(), total_boards, "mc_cap"
