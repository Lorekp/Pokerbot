"""ICM primitive -- the shared core of the tournament layer.

The Independent Chip Model (ICM) converts a *snapshot* of tournament chip stacks into real-money
($) equity, given the remaining payout structure. It is the bridge that makes a tournament a
**$-EV** game rather than the **chip-EV** game the cash engine solves.

Three public functions, all unit-agnostic in the stack/payout currency (bb, chips, or $ -- only
ratios matter for equity, and payouts set the $ scale):

  * :func:`icm_equity(stacks, payouts)` -- Malmuth-Harville finish-probability ICM. Returns the
    $-equity of each seat. Conserves the prize pool (``sum(equity) == sum(payouts[:n])``).
  * :func:`bubble_factor(stacks, payouts, hero, villain)` -- the pairwise risk premium: how many
    $ hero risks per $ hero can gain in an all-in vs ``villain``. ``1.0`` = chip-EV-neutral (deep /
    far from the money); ``>1`` = ICM pressure (tightens calling). The cheap live-usable reduction.
  * :func:`icm_delta(pre_stacks, chip_delta, payouts)` -- **THE wrap** every consumer calls: maps a
    hand's chip result to a $-equity delta (post-hand stacks vs pre, busts clamped to 0).

Design notes:
  * Malmuth-Harville assigns ``P(player finishes first) = stack / total``, then recurses on the
    remaining players for the remaining payout ranks. Exact; ``O(n!/(n-k)!)`` for ``k`` paid ranks,
    which is sub-second for a final table (``n<=9``). For hot paths use :func:`bubble_factor` (one
    pair of equity calls) rather than re-solving.
  * Busted seats (stack ``<= 0``) are treated as already-eliminated and take the *worst* available
    ranks (split equally on a simultaneous bust). This is what makes :func:`icm_delta` correct when
    a hand busts a player: the buster realizes the bottom payout, survivors split the rest.

This module is intentionally dependency-light (numpy + stdlib only) and imports nothing from the
engine, so it is certifiable in isolation. See ``tests/test_icm.py``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np


@lru_cache(maxsize=None)
def _mh(stk: tuple[float, ...], pay: tuple[float, ...]) -> tuple[float, ...]:
    """Malmuth-Harville equity for ``m`` players with strictly-positive ``stk``, competing for the
    ``m`` payout ranks ``pay`` (``pay[0]`` = best). Returns equities parallel to ``stk``.

    Memoized on ``(stk, pay)`` so identical remaining-field subproblems are solved once. The result
    is positional, so two real player groups with the same stack multiset + payouts share the cache.
    """
    m = len(stk)
    if m == 0:
        return ()
    total = sum(stk)
    if total <= 0.0:  # defensive: caller passes positives, but never divide by zero
        s = (sum(pay) / m) if m else 0.0
        return tuple(s for _ in range(m))
    out = [0.0] * m
    rank0 = pay[0]
    deeper = m > 1 and len(pay) > 1
    for k in range(m):
        p_first = stk[k] / total          # P(player k finishes first among the remaining field)
        out[k] += p_first * rank0
        if deeper:
            sub = _mh(stk[:k] + stk[k + 1:], pay[1:])   # conditional on k first, rest compete for pay[1:]
            si = 0
            for j in range(m):
                if j == k:
                    continue
                out[j] += p_first * sub[si]
                si += 1
    return tuple(out)


def _normalize_payouts(payouts: Sequence[float], n: int) -> list[float]:
    """Pay vector padded/truncated to length ``n`` (unpaid finishes -> 0.0)."""
    pay = [float(p) for p in payouts][:n]
    if len(pay) < n:
        pay = pay + [0.0] * (n - len(pay))
    return pay


def icm_equity(stacks: Sequence[float], payouts: Sequence[float]) -> np.ndarray:
    """$-equity of each seat under Malmuth-Harville ICM.

    Args:
        stacks: per-seat chip counts (any unit). Seats with ``<= 0`` are treated as already out.
        payouts: prize for finishing 1st, 2nd, ... (best first). Shorter than ``len(stacks)`` is
            padded with zeros; only the top entries need be non-zero.

    Returns:
        ``np.ndarray`` of $-equity per seat; ``sum`` equals ``sum(payouts[:n])`` (prize conserved).
    """
    stacks = [float(s) for s in stacks]
    n = len(stacks)
    if n == 0:
        return np.zeros(0)
    pay = _normalize_payouts(payouts, n)
    eq = [0.0] * n

    pos = [i for i in range(n) if stacks[i] > 0.0]
    zero = [i for i in range(n) if stacks[i] <= 0.0]
    z = len(zero)
    if z:
        bottom = pay[n - z:]                  # the worst z ranks go to the eliminated seats
        share = sum(bottom) / z               # simultaneous-bust convention: split equally
        for i in zero:
            eq[i] = share

    top_pay = tuple(pay[: n - z])             # surviving field competes for the top ranks
    pos_stk = tuple(stacks[i] for i in pos)
    sub = _mh(pos_stk, top_pay)
    for r, i in enumerate(pos):
        eq[i] = sub[r]
    return np.asarray(eq, dtype=float)


def bubble_factor(stacks: Sequence[float], payouts: Sequence[float], hero: int, villain: int) -> float:
    """Pairwise ICM risk premium for ``hero`` stacking off against ``villain`` for the smaller stack.

    ``BF = (equity hero risks by busting) / (equity hero gains by doubling)``. ``1.0`` means chips
    are $-linear here (deep / far from the money -> play chip-EV); ``> 1`` means busting costs more
    than doubling gains (ICM pressure -> demand more equity to call). ``inf`` if hero cannot gain
    (already locked). Monotone increasing as the field approaches a pay jump / the bubble.
    """
    stacks = [float(s) for s in stacks]
    h, v = stacks[hero], stacks[villain]
    risked = min(h, v)                        # an all-in is for the smaller stack
    base = float(icm_equity(stacks, payouts)[hero])

    win = list(stacks);  win[hero] = h + risked;  win[villain] = v - risked
    lose = list(stacks); lose[hero] = h - risked; lose[villain] = v + risked
    eq_win = float(icm_equity(win, payouts)[hero])
    eq_lose = float(icm_equity(lose, payouts)[hero])

    reward = eq_win - base
    risk = base - eq_lose
    if reward <= 1e-12:
        return float("inf")
    return risk / reward


def icm_delta(pre_stacks: Sequence[float], chip_delta: Sequence[float],
              payouts: Sequence[float]) -> np.ndarray:
    """$-equity change from a hand: ``ICM(pre + chip_delta) - ICM(pre)``.

    This is the single wrap point that turns a chip-EV terminal (the engine's ``compute_returns``)
    into the tournament's $-EV signal. ``chip_delta`` is the per-seat net chips (sum ~= 0); post-hand
    stacks are clamped at 0 so a bust realizes the bottom payout.
    """
    pre = [float(s) for s in pre_stacks]
    if len(chip_delta) != len(pre):
        raise ValueError(f"chip_delta len {len(chip_delta)} != stacks len {len(pre)}")
    post = [max(0.0, pre[i] + float(chip_delta[i])) for i in range(len(pre))]
    return icm_equity(post, payouts) - icm_equity(pre, payouts)
