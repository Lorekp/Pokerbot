"""Certification harness for the tournament short-stack ICM engine.

Produces the evidence that the push/fold engine plays at (certified) Nash / super-human level:
  1. HU push/fold vs PUBLISHED Nash charts across a depth grid (chip-EV via winner-take-all).
  2. Open-shove ranges by position (10bb), positional monotonicity + small exploitability.
  3. ICM dynamics: bubble tightens calls AND widens shoves (the bubble bully).
  4. Headline: the maximum measured Nash gap (exploitability) across the whole grid.

Run:  python scripts/tournament/certify_pushfold.py
Exit code 0 = all checks pass.
"""

from __future__ import annotations

import sys

from pokerbot.tournament.open_shove_icm import solve_open_shove_icm as openjam
from pokerbot.tournament.pushfold_icm import solve_hu_icm_pushfold as husolve

# Published heads-up Nash push/fold (winner-take-all, chip-EV) reference points: SB shove% / BB call%.
# Source: standard Nash push/fold charts (HoldemResources / SAGE), effective stack in bb.
PUBLISHED_HU = {
    5:  (0.71, 0.62),
    8:  (0.63, 0.46),
    10: (0.58, 0.37),
    12: (0.53, 0.33),
    15: (0.46, 0.27),
}
HU_TOL = 0.05            # absolute tolerance on range %
GAP_TOL = 5e-3           # max acceptable exploitability (prize-fraction units)


def main() -> int:
    checks = []
    max_gap = 0.0

    print("=" * 70)
    print("1) HEADS-UP PUSH/FOLD vs PUBLISHED NASH (winner-take-all == chip-EV)")
    print("   depth   solved SB/BB     published SB/BB    dev      gap")
    for depth, (psb, pbb) in PUBLISHED_HU.items():
        s = husolve([depth, depth], [1, 0], sb=0, bb=1, iters=6000)
        dev = max(abs(s.sb_jam_pct - psb), abs(s.bb_call_pct - pbb))
        max_gap = max(max_gap, s.nash_gap)
        ok = dev <= HU_TOL and s.nash_gap <= GAP_TOL
        checks.append(ok)
        print("   %4dbb   %.2f / %.2f      %.2f / %.2f       %.3f   %.1e  %s"
              % (depth, s.sb_jam_pct, s.bb_call_pct, psb, pbb, dev, s.nash_gap,
                 "OK" if ok else "FAIL"))

    print("\n2) OPEN-SHOVE BY POSITION (6-max, 10bb, ICM payouts [40,30,20,10])")
    stacks = [10] * 6
    pay = [40, 30, 20, 10]
    bl = {4: 0.5, 5: 1.0}
    order = [("UTG", 0, [1, 2, 3, 4, 5]), ("HJ", 1, [2, 3, 4, 5]), ("CO", 2, [3, 4, 5]),
             ("BTN", 3, [4, 5]), ("SB", 4, [5])]
    pcts = []
    print("   pos    jam%    gap")
    for name, seat, behind in order:
        s = openjam(stacks, pay, hero=seat, behind=behind, blinds=bl, iters=3000)
        pcts.append(s.hero_jam_pct)
        max_gap = max(max_gap, s.nash_gap)
        print("   %-4s   %.1f%%   %.1e" % (name, 100 * s.hero_jam_pct, s.nash_gap))
    mono = all(pcts[i] <= pcts[i + 1] + 1e-9 for i in range(len(pcts) - 1))
    checks.append(mono)
    print("   positional monotonicity (UTG<=...<=SB): %s" % ("OK" if mono else "FAIL"))

    print("\n3) ICM BUBBLE DYNAMICS (HU-neutral vs 2bb-bystander bubble)")
    hu = husolve([10, 10], [60, 40], sb=0, bb=1, iters=6000)
    bub = husolve([10, 10, 2], [50, 30, 20], sb=0, bb=1, iters=6000)
    tighter = bub.bb_call_pct < hu.bb_call_pct - 0.10
    bully = bub.sb_jam_pct > hu.sb_jam_pct + 0.10
    checks.append(tighter and bully)
    print("   BB call: %.1f%% (HU) -> %.1f%% (bubble)   %s"
          % (100 * hu.bb_call_pct, 100 * bub.bb_call_pct, "tighter OK" if tighter else "FAIL"))
    print("   SB jam : %.1f%% (HU) -> %.1f%% (bubble)   %s"
          % (100 * hu.sb_jam_pct, 100 * bub.sb_jam_pct, "bully  OK" if bully else "FAIL"))
    max_gap = max(max_gap, hu.nash_gap, bub.nash_gap)

    print("\n" + "=" * 70)
    passed = all(checks)
    print("MAX EXPLOITABILITY (Nash gap) across grid: %.2e  (tol %.0e)" % (max_gap, GAP_TOL))
    print("VERDICT: %s  (%d/%d checks passed)"
          % ("CERTIFIED" if passed and max_gap <= GAP_TOL else "REVIEW",
             sum(checks), len(checks)))
    return 0 if (passed and max_gap <= GAP_TOL) else 1


if __name__ == "__main__":
    sys.exit(main())
