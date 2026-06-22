"""Validation: the certified push/fold ROOT PRIOR vs the uniform-over-legal fallback (7-handed shallow).

The 7-handed tournament search has NO trained blueprint, so its root prior at N>6 is uniform-over-legal.
This harness measures, head-to-head, whether serving the CERTIFIED ante-aware ICM push/fold range
(pokerbot.search.pushfold_prior, reusing pokerbot.tournament.open_shove_icm) as the root prior plays
STRONGER short-stack poker than that uniform fallback.

Game (the exact-game push/fold node the prior serves): a single 7-handed shallow preflop orbit. Each seat
acts once in order from UTG: first-in jams-or-folds; facing a jam calls-or-folds (the push/fold
abstraction). Cards are dealt from a real deck and all-ins resolved by a real board eval with side pots
(reusing sim._settle_allin). Hero (seat 0, rotated through every seat to remove positional bias) plays
either the PRIOR ranges or the UNIFORM-random {jam/fold}, {call/fold} policy; every other seat plays the
PRIOR ranges (a competent field). We report hero's chip-EV in bb/100 with a 95% CI; the prior arm should
beat the uniform arm by a wide, CI-excluded margin.

This is a strength measurement of the PRIOR ITSELF (the lever the task adds), isolated from the rest of
the search machinery, which is the honest unit: the prior is exactly what the search root serves in the
push/fold band.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Repo root on the path so the AoF card evaluator (aof_equity_grids, used by sim) imports standalone.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pokerbot.tournament.sim as sim
from pokerbot.tournament.open_shove_icm import solve_open_shove_icm
from pokerbot.tournament.pushfold_icm import _hu_grid

EQ, LABELS, W = _hu_grid()
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
WTA = [1.0, 0.0]


def _jam_range(stacks_bb, hero, behind, blinds, ante, cache, iters):
    key = ("jam", round(stacks_bb[hero], 2), tuple(sorted(behind)), tuple(sorted(blinds.items())), ante)
    if key in cache:
        return cache[key]
    sol = solve_open_shove_icm(stacks_bb, WTA, hero, list(behind), blinds=blinds, ante=ante, iters=iters)
    cache[key] = sol.hero_jam
    return sol.hero_jam


def _call_range(stacks_bb, hero, jammer, behind, blinds, ante, cache, iters):
    jam_behind = tuple(sorted(set(behind) | {hero}))
    key = ("call", round(stacks_bb[jammer], 2), jammer, jam_behind, tuple(sorted(blinds.items())), ante)
    if key in cache:
        sol_labels, call = cache[key]
    else:
        sol = solve_open_shove_icm(stacks_bb, WTA, jammer, list(jam_behind), blinds=blinds,
                                   ante=ante, iters=iters)
        cache[key] = (sol.labels, sol.call)
        sol_labels, call = sol.labels, sol.call
    return call[hero]


def _hero_delta(n, hero, uniform, stacks_bb, ante, rng, cache, iters):
    """One 7-handed shallow push/fold orbit. Hero net = final stack - starting stack (bb)."""
    deck = list(range(52)); rng.shuffle(deck)
    holes = {s: (deck[2 * s], deck[2 * s + 1]) for s in range(n)}
    board = deck[2 * n:2 * n + 5]
    sb_seat, bb_seat = n - 2, n - 1
    blinds = {sb_seat: 0.5, bb_seat: 1.0}
    committed = {s: ante + blinds.get(s, 0.0) for s in range(n)}
    stacks = {s: stacks_bb[s] - committed[s] for s in range(n)}
    order = list(range(n)); jammers, folded = [], set()
    for s in order:
        b = LABEL_TO_IDX[sim._LABELS[sim._bucket(*holes[s])]]
        full = [stacks[i] + committed[i] for i in range(n)]
        if not jammers:
            behind = [o for o in order if o != s and o not in folded and o > s]
            if not behind:
                folded.add(s); continue
            jam = (rng.random() < 0.5) if (s == hero and uniform) else \
                _jam_range(full, s, behind, blinds, ante, cache, iters)[b] > 0.5
            if jam:
                committed[s] += stacks[s]; stacks[s] = 0.0; jammers.append(s)
            else:
                folded.add(s)
        else:
            j = jammers[0]
            behind = [o for o in order if o != s and o not in folded and o > s]
            call = (rng.random() < 0.5) if (s == hero and uniform) else \
                _call_range(full, s, j, behind, blinds, ante, cache, iters)[b] > 0.5
            if call:
                committed[s] += stacks[s]; stacks[s] = 0.0; jammers.append(s)
            else:
                folded.add(s)
    if len(jammers) <= 1:
        winner = jammers[0] if jammers else sim._last_unfolded(order, folded)
        stacks[winner] += sum(committed.values())
    else:
        dead = sum(committed[s] for s in range(n) if s not in jammers)
        won = sim._settle_allin(jammers, {s: committed[s] for s in jammers}, holes, board, dead)
        for s, amt in won.items():
            stacks[s] += amt
    return stacks[hero] - stacks_bb[hero]


def run(n=7, depth=10.0, ante=0.0, hands=4000, iters=600, seed=0):
    cache = {}
    stacks_bb = [depth] * n
    out = {}
    for arm, uniform in (("uniform", True), ("prior", False)):
        rng = np.random.default_rng(seed)
        deltas = []
        for h in range(hands):
            hero = h % n  # rotate hero through every seat (no positional bias)
            deltas.append(_hero_delta(n, hero, uniform, stacks_bb, ante, rng, cache, iters))
        d = np.asarray(deltas)
        bb100 = d.mean() * 100.0
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) * 100.0
        out[arm] = (bb100, ci)
        print(f"[{arm:7s}] {bb100:+8.2f} bb/100  CI +/-{ci:5.2f}  (n={len(d)})")
    diff = out["prior"][0] - out["uniform"][0]
    dci = np.hypot(out["prior"][1], out["uniform"][1])
    print(f"[prior-uniform] {diff:+8.2f} bb/100  CI +/-{dci:5.2f}  "
          f"=> prior {'STRONGER (CI-excluded)' if diff - dci > 0 else 'not separated'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--depth", type=float, default=10.0)
    ap.add_argument("--ante", type=float, default=0.0)
    ap.add_argument("--hands", type=int, default=4000)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(n=a.n, depth=a.depth, ante=a.ante, hands=a.hands, iters=a.iters, seed=a.seed)
