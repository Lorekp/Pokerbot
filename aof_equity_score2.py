"""Integer score2 reference oracle for HU showdown equity (Phase 2 native rewrite)."""

from __future__ import annotations

import numpy as np
from numba import njit

from aof_equity_grids import _eval7
from bench._keys import representative_from_key

DENOM2 = 3_424_608  # 2 * C(48, 5)

# Derived float32 parity gate (replaces split 1e-9 / 1e-7 thresholds).
GATE_B_FLOAT32_ATOL = 0.0


@njit(cache=True)
def _score2_two_combos_exact(h0: int, h1: int, v0: int, v1: int) -> int:
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
    score2 = 0
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
                            score2 += 2
                        elif hr == vr:
                            score2 += 1
    return score2


def score2_two_combos_exact(h0: int, h1: int, v0: int, v1: int) -> int:
    """Hero (h0,h1) score2 vs villain (v0,v1) by enumerating all runouts."""
    return int(_score2_two_combos_exact(h0, h1, v0, v1))


def score2_from_key(key: int) -> int:
    h0, h1, v0, v1 = representative_from_key(key)
    return score2_two_combos_exact(h0, h1, v0, v1)


def equity_from_score2(score2: int) -> float:
    return score2 / DENOM2


def equity_float32_from_score2(score2: int | np.ndarray) -> np.float32 | np.ndarray:
    """Bit-identical legacy float bridge: float32(score2 / DENOM2)."""
    return (np.asarray(score2, dtype=np.float64) / np.float64(DENOM2)).astype(np.float32)
