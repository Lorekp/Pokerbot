"""ICM-aware all-in EV -- the bridge from AoF showdown equity to the ICM primitive.

Why a dedicated module instead of reusing ``aof_cfr.terminal_ev.showdown_terminal_returns``:
that returns *chip-EV* via the **expected pot share**. ICM is **non-linear**, so
``E[ICM(outcome)] != ICM(E[chips])`` -- you cannot feed an expected share into ICM. The correct
ICM EV of an all-in enumerates the *discrete* outcomes (hero wins / chops / loses the showdown),
maps each to post-hand stacks, and averages the ICM equity over them.

So this module re-uses only the trusted card evaluator ``aof_equity_grids._eval7`` (read-only) to
count win/chop/lose probabilities by exact runout enumeration, then weights
:func:`pokerbot.tournament.icm.icm_equity` over the discrete outcomes.

Used by the ICM short-stack push/fold solver (``pushfold_icm.py``) as the contested-pot terminal.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numba import njit

from aof_equity_grids import _card_str_to_id, _eval7
from pokerbot.tournament.icm import icm_equity


@njit(cache=True)
def _hu_outcome_probs(h0: int, h1: int, v0: int, v1: int):
    """Exact (P(hero wins), P(chop), P(hero loses)) over all C(48,5) runouts for a heads-up
    all-in. Mirrors the enumeration in ``aof_cfr.terminal_ev`` but returns the discrete-outcome
    split that ICM needs (not just the expected share)."""
    dead = np.zeros(52, dtype=np.bool_)
    dead[h0] = True
    dead[h1] = True
    dead[v0] = True
    dead[v1] = True
    deck = np.empty(52, dtype=np.int32)
    nd = 0
    for c in range(52):
        if not dead[c]:
            deck[nd] = c
            nd += 1
    win = 0.0
    chop = 0.0
    lose = 0.0
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
                        if hr < vr:           # lower rank == stronger hand (AoF convention)
                            win += 1.0
                        elif hr == vr:
                            chop += 1.0
                        else:
                            lose += 1.0
                        total += 1.0
    return win / total, chop / total, lose / total


@njit(cache=True)
def _three_way_best_probs(a0, a1, b0, b1, c0, c1):
    """Exact P(player A wins outright), for a 3-way all-in, over all C(46,5) runouts (ties split but
    counted toward A only on a clean A-win; used to validate the product-pairwise multiway approx)."""
    dead = np.zeros(52, dtype=np.bool_)
    for x in (a0, a1, b0, b1, c0, c1):
        dead[x] = True
    deck = np.empty(52, dtype=np.int32)
    nd = 0
    for cc in range(52):
        if not dead[cc]:
            deck[nd] = cc
            nd += 1
    a_share = 0.0
    total = 0.0
    for i0 in range(nd - 4):
        d0 = deck[i0]
        for i1 in range(i0 + 1, nd - 3):
            d1 = deck[i1]
            for i2 in range(i1 + 1, nd - 2):
                d2 = deck[i2]
                for i3 in range(i2 + 1, nd - 1):
                    d3 = deck[i3]
                    for i4 in range(i3 + 1, nd):
                        d4 = deck[i4]
                        ra = _eval7(a0, a1, d0, d1, d2, d3, d4)
                        rb = _eval7(b0, b1, d0, d1, d2, d3, d4)
                        rc = _eval7(c0, c1, d0, d1, d2, d3, d4)
                        best = min(ra, rb, rc)
                        nwin = (ra == best) + (rb == best) + (rc == best)
                        if ra == best:
                            a_share += 1.0 / nwin
                        total += 1.0
    return a_share / total


def three_way_equity(hero, v1, v2) -> float:
    """Hero's exact 3-way pot equity (share) vs two villains. Inputs are (id,id) or 4-char strings."""
    h = parse_hole(hero) if isinstance(hero, str) else tuple(hero)
    x = parse_hole(v1) if isinstance(v1, str) else tuple(v1)
    y = parse_hole(v2) if isinstance(v2, str) else tuple(v2)
    return float(_three_way_best_probs(int(h[0]), int(h[1]), int(x[0]), int(x[1]), int(y[0]), int(y[1])))


def parse_hole(s: str) -> tuple[int, int]:
    """'AsKd' / '2c2d' -> (card_id, card_id) using the AoF card encoding."""
    s = s.strip()
    if len(s) != 4:
        raise ValueError(f"expected a 4-char two-card string like 'AsKd', got {s!r}")
    return _card_str_to_id(s[:2]), _card_str_to_id(s[2:])


def hu_outcome_probs(hero_hole, villain_hole) -> tuple[float, float, float]:
    """(P(hero wins), P(chop), P(hero loses)) for a heads-up all-in to showdown.

    ``hero_hole`` / ``villain_hole`` are ``(card_id, card_id)`` tuples or 4-char strings.
    ``P(win) + 0.5*P(chop)`` equals hero's pot equity (validated against AoF in tests).
    """
    h = parse_hole(hero_hole) if isinstance(hero_hole, str) else tuple(hero_hole)
    v = parse_hole(villain_hole) if isinstance(villain_hole, str) else tuple(villain_hole)
    pw, pc, pl = _hu_outcome_probs(int(h[0]), int(h[1]), int(v[0]), int(v[1]))
    return float(pw), float(pc), float(pl)


def hu_allin_icm_ev(hero_hole, villain_hole, stacks: Sequence[float],
                    payouts: Sequence[float], hero: int, villain: int) -> np.ndarray:
    """Per-seat ICM $-EV of a heads-up all-in (``hero`` vs ``villain`` for the smaller stack)
    taken to showdown. Bystander seats are valued through the outcome too (their ICM equity moves
    when a rival busts). Returns an array over all seats; conserves the prize pool.

    This is the contested terminal the ICM push/fold solver prices a jam/call against. Contrast
    with chip-EV: near the money this is *lower* than the chip-EV-equivalent for the at-risk player
    (the ICM risk premium), which is exactly why correct push/fold ranges tighten on the bubble.
    """
    pw, pc, pl = hu_outcome_probs(hero_hole, villain_hole)
    s = [float(x) for x in stacks]
    h, v = s[hero], s[villain]
    risked = min(h, v)                                   # an all-in is for the smaller stack
    win = list(s);  win[hero] = h + risked;  win[villain] = v - risked
    lose = list(s); lose[hero] = h - risked; lose[villain] = v + risked
    # chop returns each player's contribution -> stacks unchanged (HU, equal contributions)
    eq_win = icm_equity(win, payouts)
    eq_lose = icm_equity(lose, payouts)
    eq_chop = icm_equity(s, payouts)
    return pw * eq_win + pc * eq_chop + pl * eq_lose
