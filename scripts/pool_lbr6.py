"""Pool independent LBR6 seed-runs (scripts/lbr6_audit.py --out a.json b.json ...) into ONE estimate.

Per-hand variance from stack-off pots means a single local run has a wide CI; running many independent
seeds (different machines / cloud spot) and pooling shrinks it exactly like a single large-N run. Combines
by hand-weighted mean + total-variance decomposition (within-run + between-run).

  python scripts/pool_lbr6.py runs/lbr6_*.json
"""

from __future__ import annotations

import glob
import json
import sys


def main(paths: list[str]) -> None:
    files = [p for pat in paths for p in glob.glob(pat)]
    runs = []
    for p in files:
        with open(p) as f:
            runs.append(json.load(f))
    if not runs:
        print("no JSON runs matched")
        return

    # consistency check: all runs should target the same thing
    tgt = {(r["label"], r["n_seats"]) for r in runs}
    N = sum(r["hands"] for r in runs)
    M = sum(r["hands"] * r["ev_bb_per_hand"] for r in runs) / N
    # total per-hand variance = within-run + between-run (about the pooled mean)
    ss = sum(r["hands"] * (r["std_bb_per_hand"] ** 2) + r["hands"] * (r["ev_bb_per_hand"] - M) ** 2
             for r in runs)
    per_hand_var = ss / N
    stderr = (per_hand_var / N) ** 0.5
    lo, hi = (M - 1.96 * stderr) * 100, (M + 1.96 * stderr) * 100

    print(f"pooled {len(runs)} runs  (targets={tgt})")
    print(f"  total hands : {N}")
    print(f"  LBR edge    : {M*100:+.2f} bb/100   (95% CI [{lo:+.2f}, {hi:+.2f}])")
    print(f"  stderr      : {stderr*100:.2f} bb/100")
    verdict = ("BEATABLE: best-responder profits (CI excludes 0)" if lo > 0
               else "robust at this CI: LBR cannot prove a positive exploit")
    print(f"  verdict     : {verdict}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/lbr6_*.json"])
