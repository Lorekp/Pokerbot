"""Exact terminal chip-EV payoffs for AoF push/fold."""

from __future__ import annotations

import numpy as np
from numba import njit

from aof_equity_grids import _eval7

_SHARE_CACHE: dict[tuple[tuple[int, int], ...], tuple[float, ...]] = {}


@njit(cache=True)
def _exact_nway_shares_numba(hole_ids_flat: np.ndarray, n: int) -> np.ndarray:
    """Hero pot share for each seat by full runout enumeration."""
    dead = np.zeros(52, dtype=np.bool_)
    for k in range(2 * n):
        dead[hole_ids_flat[k]] = True
    deck = np.empty(52, dtype=np.int32)
    nd = 0
    for c in range(52):
        if not dead[c]:
            deck[nd] = c
            nd += 1
    wins = np.zeros(n, dtype=np.float64)
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
                        ranks = np.empty(n, dtype=np.int32)
                        best = 1_000_000
                        nwin = 0
                        for pi in range(n):
                            h0 = hole_ids_flat[2 * pi]
                            h1 = hole_ids_flat[2 * pi + 1]
                            rank = _eval7(h0, h1, c0, c1, c2, c3, c4)
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
                        total += 1.0
    if total <= 0.0:
        out = np.empty(n, dtype=np.float64)
        for pi in range(n):
            out[pi] = 1.0 / n
        return out
    return wins / total


def _post_blind(position_index: int, seats: tuple[str, ...], stack_bb: float) -> float:
    seat = seats[position_index]
    if seat == "SB":
        return min(0.5, stack_bb)
    if seat == "BB":
        return min(1.0, stack_bb)
    return 0.0


def blind_vector(seats: tuple[str, ...], stack_bb: float) -> np.ndarray:
    return np.array([_post_blind(i, seats, stack_bb) for i in range(len(seats))])


def fold_terminal_returns(
    folded: set[int],
    alive: list[int],
    invested: np.ndarray,
) -> list[float]:
    """Net bb profit per seat when only one player remains."""
    if len(alive) != 1:
        raise ValueError("fold_terminal requires exactly one survivor")
    pot = float(invested.sum())
    winner = alive[0]
    out = (-invested).astype(np.float64).tolist()
    out[winner] = pot - invested[winner]
    return out


def _exact_2way_share(h0: int, h1: int, v0: int, v1: int) -> float:
    from aof_equity_grids import _equity_two_combos_exact

    return float(_equity_two_combos_exact(h0, h1, v0, v1))


def _exact_nway_shares(hole_ids: list[tuple[int, int]]) -> list[float]:
    """Hero share of pot for each seat by full runout enumeration."""
    key = tuple(hole_ids)
    cached = _SHARE_CACHE.get(key)
    if cached is not None:
        return list(cached)
    n = len(hole_ids)
    flat = np.empty(2 * n, dtype=np.int32)
    for i, (a, b) in enumerate(hole_ids):
        flat[2 * i] = a
        flat[2 * i + 1] = b
    out = tuple(float(x) for x in _exact_nway_shares_numba(flat, n))
    _SHARE_CACHE[key] = out
    return list(out)


def showdown_terminal_returns(
    hole_ids: list[tuple[int, int]],
    invested: np.ndarray,
) -> list[float]:
    """Net bb profit per seat at all-in showdown (zero-sum)."""
    pot = float(invested.sum())
    shares = _exact_nway_shares(hole_ids)
    out = []
    for i, sh in enumerate(shares):
        out.append(sh * pot - float(invested[i]))
    return out


def hu_showdown_sb_utility(
    sb_bucket: int,
    bb_bucket: int,
    eq: np.ndarray,
    *,
    stack_bb: float = 8.0,
    sb_posted: float = 0.5,
    bb_posted: float = 1.0,
    game_model: int | None = None,
) -> float:
    """SB chip-EV when both players jam (matches solver_hu_dcfr; game-model aware)."""
    from aof_cfr.pushfold_payoffs import contested_pot_bb
    pot_called = contested_pot_bb(2, 0.0, stack_bb, game_model=game_model)
    return float(eq[sb_bucket, bb_bucket] * pot_called - stack_bb)


def hu_fold_sb_utility(*, sb_posted: float = 0.5) -> float:
    return -sb_posted


def hu_push_fold_sb_utility(*, sb_posted: float = 0.5, bb_posted: float = 1.0) -> float:
    return sb_posted + bb_posted - sb_posted
