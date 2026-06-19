"""Quantify the tournament engine's PROFIT edge over an ICM-naive (chip-EV) player.

Most of the field plays chip-EV (ignores ICM): on the bubble they call jams too wide because the
chips look +EV, bleeding real-money equity. This measures, on a canonical bubble call spot, the
$-equity (in prize-fraction units) the ICM engine gains over a chip-EV player by making the
ICM-correct call/fold decision on every hand -- the engine's super-human edge, made concrete.

The call EV is exact: a call's post-stacks depend only on win/lose (not the villain's exact cards),
so ICM(call) = eq*ICM(win) + (1-eq)*ICM(lose) with eq = hero's equity vs the jamming range.

Run:  python scripts/tournament/icm_edge_report.py
"""

from __future__ import annotations

import sys

import numpy as np

from pokerbot.tournament.icm import icm_equity
from pokerbot.tournament.open_shove_icm import solve_open_shove_icm
from pokerbot.tournament.pushfold_icm import _hu_grid


def main() -> int:
    eq, labels, w = _hu_grid()
    pj = w / w.sum()

    # Bubble: 4 left, 3 paid. Two 12bb stacks, one 8bb (jammer), one 3bb (short about to bust).
    # Hero (seat 0, 12bb) faces an open-jam from seat 1 (8bb). Seats 2 (12bb), 3 (3bb) folded.
    stacks = [12.0, 8.0, 12.0, 3.0]
    pay = [50.0, 30.0, 20.0]                      # 3 paid (4th = 0): the bubble
    hero, jammer = 0, 1
    R = min(stacks[hero], stacks[jammer])         # 8bb all-in

    # Jammer's range = ICM-naive chip-EV open-shove (winner-take-all): wide.
    chip_field = solve_open_shove_icm(stacks, [1, 0, 0, 0], hero=jammer, behind=[2, 3, 0],
                                      blinds={2: 0.0, 3: 0.0, 0: 0.5}, iters=2500)
    jam_range = chip_field.hero_jam
    jr = pj * jam_range
    jr = jr / jr.sum()                            # jammer's normalized hand distribution

    # Hero's equity vs that jamming range, per hero bucket j: eq_bb = sum_i jr[i]*(1-eq[i,j])
    hero_eq = 1.0 - (jr @ eq)                     # vector over hero bucket j

    # ICM terminal equities for hero (call -> win/lose R; fold -> stay).
    win = icm_equity([stacks[hero] + R, stacks[jammer] - R, stacks[2], stacks[3]], pay)[hero]
    lose = icm_equity([stacks[hero] - R, stacks[jammer] + R, stacks[2], stacks[3]], pay)[hero]
    fold = icm_equity([stacks[hero] - 0.5, stacks[jammer], stacks[2], stacks[3] + 0.5], pay)[hero]

    ev_call = hero_eq * win + (1.0 - hero_eq) * lose          # $-equity of calling, per hand
    ev_fold = fold

    # chip-EV of calling (per hand): equity * pot - amount risked (dead blinds ~0 here)
    pot = 2 * R + 0.5
    chip_call = hero_eq * pot - R

    icm_calls = ev_call > ev_fold                # the ICM engine calls when it's $-correct
    chip_calls = chip_call > 0.0                 # the chip-EV player calls when chips look +EV

    # $-equity each player realizes per hand, and the engine's edge
    ev_icm = np.where(icm_calls, ev_call, ev_fold)
    ev_chip = np.where(chip_calls, ev_call, ev_fold)
    edge = ev_icm - ev_chip                       # >= 0 (engine plays the $-max action)

    icm_pct = float((icm_calls * w).sum() / w.sum())
    chip_pct = float((chip_calls * w).sum() / w.sum())
    total_edge = float((edge * pj).sum())         # avg $ (prize fraction) gained per hand dealt
    # express as bb-equivalent: prize fraction * total prize / (a rough bb scale) -- report raw + %
    leak_hands = float(((chip_calls & ~icm_calls) * w).sum() / w.sum())

    print("=" * 66)
    print("ICM ENGINE PROFIT EDGE vs a chip-EV player (bubble call spot)")
    print("  stacks (bb):", stacks, " payouts:", pay, " (4 left, 3 paid)")
    print("  hero faces an 8bb open-jam from an ICM-naive (chip-EV) jammer")
    print("-" * 66)
    print("  call range:  ICM engine %.1f%%   chip-EV player %.1f%%" % (100 * icm_pct, 100 * chip_pct))
    print("  hands the chip-EV player MIS-calls (ICM folds): %.1f%% of range" % (100 * leak_hands))
    print("  engine $-equity edge: %.4f prize-fraction per dealt hand" % total_edge)
    print("  (i.e. the chip-EV player bleeds this much real-money equity by ignoring ICM)")
    ok = total_edge > 0 and icm_pct < chip_pct
    print("-" * 66)
    print("VERDICT: %s  (engine calls tighter on the bubble and gains $; edge %s 0)"
          % ("EDGE CONFIRMED" if ok else "REVIEW", ">" if total_edge > 0 else "<="))
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
