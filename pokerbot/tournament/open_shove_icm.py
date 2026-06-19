"""ICM open-shove (first-in jam) solver for any position -- multiway short-stack play.

The dominant short-stack decision is the *unopened* shove: it folds to hero, hero jams or folds,
and each player behind can call. This solves that game to an approximate equilibrium under ICM,
generalizing the exact heads-up solver in :mod:`pokerbot.tournament.pushfold_icm`.

Model (a self-consistent fixed point between hero's jam range and each caller's call range):
  * Hero's jam EV is computed by enumerating every *subset* of callers ``S`` -- exact for the action
    structure -- with ``P(S calls)`` from the callers' marginal call frequencies.
  * Terminal chips use **equal-stack, no-side-pot** ICM (the standard Nash push/fold chart
    assumption): the pot is won outright by the best hand. Multiway hero equity vs a subset uses the
    **product-of-pairwise** approximation ``eq_multi = prod_o E_j[eq(i, range_o)]`` (a standard,
    slightly conservative multiway estimate); on a loss the pot is split among the callers (a
    second-order effect on hero's own ladder equity).
  * Each caller's call range is the ICM best response vs hero's jam range, modeled heads-up (other
    behind players folding) -- exact 2-way ICM via the equity grid.

Validation (see tests): with a single player behind (``hero=SB, behind=[BB]``) it **reduces to the
exact HU solver** (~58% at 10bb), and jam frequency is **monotone in position** (SB widest ->
UTG tightest), as it must be. ``nash_gap`` reports the measured exploitability (multiplayer games
have no Nash-uniqueness guarantee, so this is a measured residual, not a theorem).

Honest scope: this is exact for fold-equity + single-caller branches (the bulk of first-in EV) and
approximate for the rarer multi-caller branches; it assumes ~equal effective stacks (no side pots).
Unequal-stack side pots and exact >=3-way equity are refinements tracked for the full solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from pokerbot.tournament.icm import icm_equity
from pokerbot.tournament.pushfold_icm import _hu_grid


def _regret_match(reg: np.ndarray) -> np.ndarray:
    pos = np.clip(reg, 0.0, None)
    s = pos.sum(axis=1, keepdims=True)
    return np.where(s > 0, pos / np.where(s > 0, s, 1.0), 0.5)


def _hero_fold_value(stacks, payouts, hero: int, blinds: dict[int, float], ante: float) -> float:
    """Hero's ICM equity when it folds first-in: hero forfeits its committed chips (ante + its own
    posted blind), credited to the big-blind seat (the nominal walk winner) so chips are conserved.
    The behind players' sub-game for the blinds is a second-order ladder effect on hero's equity."""
    hero_committed = ante + blinds.get(hero, 0.0)
    post = list(float(s) for s in stacks)
    post[hero] -= hero_committed
    if hero_committed > 0:
        bb_seat = max(blinds, key=lambda s: blinds[s]) if blinds else None
        bb_seat = bb_seat if (bb_seat is not None and bb_seat != hero) else (
            min(range(len(stacks)), key=lambda s: (s == hero)))   # fall back to any other seat
        if bb_seat != hero:
            post[bb_seat] += hero_committed
    return float(icm_equity(post, payouts)[hero])


@dataclass
class OpenShoveSolution:
    hero_jam: np.ndarray            # P(jam) per 169 bucket
    call: dict[int, np.ndarray]     # seat -> P(call) per 169 bucket
    labels: tuple[str, ...]
    weights: np.ndarray
    hero_jam_pct: float
    nash_gap: float

    def jam_freq(self, hand: str) -> float:
        return float(self.hero_jam[list(self.labels).index(hand)])


def solve_open_shove_icm(stacks: Sequence[float], payouts: Sequence[float], hero: int,
                         behind: Sequence[int], *, blinds: dict[int, float] | None = None,
                         ante: float = 0.0, iters: int = 3000) -> OpenShoveSolution:
    """Solve hero's first-in jam/fold vs the players ``behind`` (each may call), under ICM.

    ``stacks`` is the full remaining field; ``hero`` and ``behind`` index seats (``behind`` in action
    order). ``blinds`` maps seat -> posted blind (e.g. ``{sb_seat: 0.5, bb_seat: 1.0}``). ``ante`` is
    the per-player ante (every dealt seat posts it). A player's *committed* chips = ante + its blind;
    folded players' committed chips become dead money the winner collects, so antes widen ranges.
    """
    eq, labels, w = _hu_grid()
    n = len(labels)
    stacks = [float(s) for s in stacks]
    blinds = dict(blinds or {})
    pj = w / w.sum()
    behind = list(behind)

    def committed(s: int) -> float:
        return ante + blinds.get(s, 0.0)

    def icm_post(deltas: dict[int, float]) -> np.ndarray:
        post = list(stacks)
        for s, d in deltas.items():
            post[s] += d
        return icm_equity(post, payouts)

    # Effective at-risk amount (equal-stack assumption: min of the involved stacks).
    def R(a: int, b: int) -> float:
        return min(stacks[a], stacks[b])

    hero_post = stacks[hero]

    all_seats = list(range(len(stacks)))

    # --- caller best-response constants (HU hero-jam vs caller o), per caller seat ---
    # caller o calling vs hero bucket i: win prob for o = 1 - eq[i, j]  (eq is hero-vs-villain).
    # All terminals conserve chips: every non-contesting seat is debited its committed (ante+blind).
    def caller_affine(o: int):
        r = R(hero, o)
        folded = [s for s in all_seats if s != hero and s != o]
        dead = sum(committed(s) for s in folded)
        fd = {f: -committed(f) for f in folded}
        owin = icm_post({hero: -r, o: +r + dead, **fd})[o]            # o wins showdown
        olose = icm_post({hero: +r + dead, o: -r, **fd})[o]           # o loses showdown
        # o folds -> hero scoops the pot uncontested (collects everyone's committed)
        others = [s for s in all_seats if s != hero]
        ofold = icm_post({hero: +sum(committed(s) for s in others),
                          **{s: -committed(s) for s in others}})[o]
        return owin, olose, ofold

    caller_const = {o: caller_affine(o) for o in behind}

    # regrets
    rh = np.zeros((n, 2))                       # hero [fold, jam]
    rc = {o: np.zeros((n, 2)) for o in behind}  # caller [fold, call]
    acc_h = np.zeros(n)
    acc_c = {o: np.zeros(n) for o in behind}
    acc_c_w = {o: 0.0 for o in behind}
    tsum = 0.0

    # precompute subsets of callers (could be up to 2^len(behind))
    subsets = []
    for k in range(len(behind) + 1):
        for S in combinations(behind, k):
            subsets.append(S)

    def hero_jam_value(call_freqs: dict[int, np.ndarray]) -> np.ndarray:
        """EV_jam per hero bucket, enumerating caller subsets."""
        cbar = {o: float((pj * call_freqs[o]).sum()) for o in behind}      # marginal call prob
        # per-caller average equity for hero bucket i vs o's *calling* range (normalized)
        avgeq = {}
        for o in behind:
            cw = pj * call_freqs[o]
            m = cw.sum()
            avgeq[o] = (eq @ cw) / m if m > 1e-12 else np.full(n, 0.5)     # vector over i
        V = np.zeros(n)
        for S in subsets:
            p = 1.0
            for o in behind:
                p *= cbar[o] if o in S else (1.0 - cbar[o])
            if p <= 0:
                continue
            if not S:                                                      # everyone folds -> steal
                others = [s for s in all_seats if s != hero]
                val = icm_post({hero: +sum(committed(s) for s in others),
                                **{s: -committed(s) for s in others}})[hero]
                V += p * val
                continue
            r = min([stacks[hero]] + [stacks[o] for o in S])
            folded = [s for s in all_seats if s != hero and s not in S]
            dead = sum(committed(f) for f in folded)
            fd = {f: -committed(f) for f in folded}
            win = icm_post({hero: +len(S) * r + dead, **{o: -r for o in S}, **fd})[hero]  # hero scoops
            # hero loses: split the pot among the callers, hero busts the at-risk r
            pot = (len(S) + 1) * r + dead
            lose_deltas = {hero: -r, **fd}
            for o in S:
                lose_deltas[o] = pot / len(S) - r                          # each caller's net if they split
            lose = icm_post(lose_deltas)[hero]
            eq_multi = np.ones(n)
            for o in S:
                eq_multi = eq_multi * avgeq[o]
            V += p * (eq_multi * win + (1.0 - eq_multi) * lose)
        return V

    for t in range(1, iters + 1):
        sig_h = _regret_match(rh)
        jam = sig_h[:, 1]
        call_freqs = {o: _regret_match(rc[o])[:, 1] for o in behind}

        # hero update
        V_jam = hero_jam_value(call_freqs)
        V_fold = np.full(n, _hero_fold_value(stacks, payouts, hero, blinds, ante))
        V_h = jam * V_jam + (1 - jam) * V_fold
        rh[:, 0] += V_fold - V_h
        rh[:, 1] += V_jam - V_h
        rh = np.clip(rh, 0, None)
        acc_h += t * jam

        # caller updates (HU vs hero's jam range)
        jam_w = pj * jam
        jam_mass = float(jam_w.sum())
        eq_vs_hero = (jam_w @ eq) / jam_mass if jam_mass > 1e-12 else np.full(n, 0.5)  # over caller bucket j
        for o in behind:
            owin, olose, ofold = caller_const[o]
            # o's call EV per bucket j: o wins w.p (1 - eq_vs_hero[j])
            V_call = (1.0 - eq_vs_hero) * owin + eq_vs_hero * olose
            V_cfold = np.full(n, ofold)
            cf = call_freqs[o]
            V_o = cf * V_call + (1 - cf) * V_cfold
            rc[o][:, 0] += V_cfold - V_o
            rc[o][:, 1] += V_call - V_o
            rc[o] = np.clip(rc[o], 0, None)
            acc_c[o] += t * jam_mass * cf
            acc_c_w[o] += t * jam_mass
        tsum += t

    avg_jam = acc_h / tsum
    avg_call = {o: (acc_c[o] / acc_c_w[o] if acc_c_w[o] > 0 else np.zeros(n)) for o in behind}

    # exploitability: hero BR vs avg callers + each caller BR vs avg hero
    Vj = hero_jam_value(avg_call)
    Vf = _hero_fold_value(stacks, payouts, hero, blinds, ante)
    br_h = float((pj * np.maximum(Vj, Vf)).sum())
    cur_h = float((pj * (avg_jam * Vj + (1 - avg_jam) * Vf)).sum())
    gap = br_h - cur_h
    jam_w = pj * avg_jam
    jam_mass = float(jam_w.sum())
    eq_vs_hero = (jam_w @ eq) / jam_mass if jam_mass > 1e-12 else np.full(n, 0.5)
    for o in behind:
        owin, olose, ofold = caller_const[o]
        V_call = (1.0 - eq_vs_hero) * owin + eq_vs_hero * olose
        br_o = float((pj * np.maximum(V_call, ofold)).sum())
        cur_o = float((pj * (avg_call[o] * V_call + (1 - avg_call[o]) * ofold)).sum())
        gap += br_o - cur_o

    return OpenShoveSolution(
        hero_jam=avg_jam, call=avg_call, labels=labels, weights=w,
        hero_jam_pct=float((avg_jam * w).sum() / w.sum()), nash_gap=gap,
    )
