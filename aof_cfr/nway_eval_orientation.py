"""Pin N-way showdown rank orientation — OMPEval higher-is-better vs numba lower-is-better.

All N-way winner selection must go through ``pick_nway_winners`` so comparison direction
cannot be re-inverted at individual call sites (same pattern as HU ``native_outcome``).
"""

from __future__ import annotations

import numpy as np

from aof_equity_grids import _eval7

_eval7_native = None


def _load_native_eval7():
    global _eval7_native
    if _eval7_native is not None:
        return _eval7_native
    try:
        from tools.ompeval_cli import native as omp_native

        if omp_native.native_backend_available():
            mod = omp_native.load_native_module()
            if mod is not None and hasattr(mod, "eval7"):
                _eval7_native = mod.eval7
                return _eval7_native
    except Exception:
        pass
    _eval7_native = False
    return False


def native_eval7_available() -> bool:
    return bool(_load_native_eval7())


def eval7_backend_name() -> str:
    return "hu_table_native" if native_eval7_available() else "numba_eval7"


def eval7_seat_rank(
    h0: int,
    h1: int,
    b0: int,
    b1: int,
    b2: int,
    b3: int,
    b4: int,
    *,
    force_numba: bool = False,
) -> tuple[int, bool]:
    """Return (rank, used_native). Rank is in backend-native order (not yet compared)."""
    if not force_numba:
        native = _load_native_eval7()
        if native:
            return int(native(h0, h1, b0, b1, b2, b3, b4)), True
    return int(_eval7(h0, h1, b0, b1, b2, b3, b4)), False


def rank_beats(rank: int, best: int, *, native_omp: bool) -> bool:
    """True if ``rank`` strictly beats ``best`` under the active backend."""
    if native_omp:
        return rank > best
    return rank < best


def pick_nway_winners(ranks: list[int] | np.ndarray, *, native_omp: bool) -> tuple[int, int]:
    """Return (best_rank, n_winners). Comparison direction pinned here only."""
    if len(ranks) == 0:
        raise ValueError("pick_nway_winners requires at least one rank")
    best = int(ranks[0])
    nwin = 1
    for r in ranks[1:]:
        ri = int(r)
        if rank_beats(ri, best, native_omp=native_omp):
            best = ri
            nwin = 1
        elif ri == best:
            nwin += 1
    return best, nwin


def seat_share_from_ranks(ranks: list[int] | np.ndarray, *, native_omp: bool) -> np.ndarray:
    """Pot shares (sum to 1) for one board given per-seat ranks."""
    ranks_arr = np.asarray(ranks, dtype=np.int32)
    n = ranks_arr.shape[0]
    best, nwin = pick_nway_winners(ranks_arr, native_omp=native_omp)
    out = np.zeros(n, dtype=np.float64)
    share = 1.0 / nwin
    for i in range(n):
        if int(ranks_arr[i]) == best:
            out[i] = share
    return out


def pair_equity_counters_3way(
    ranks: list[int] | np.ndarray, *, native_omp: bool
) -> tuple[float, float, float]:
    """btn-BB, btn-SB, sb-BB win shares on one board (3 players)."""
    hr_btn = int(ranks[0])
    hr_sb = int(ranks[1])
    hr_bb = int(ranks[2])

    def win_share(a: int, b: int) -> float:
        if rank_beats(a, b, native_omp=native_omp):
            return 1.0
        if a == b:
            return 0.5
        return 0.0

    return win_share(hr_btn, hr_bb), win_share(hr_btn, hr_sb), win_share(hr_sb, hr_bb)
