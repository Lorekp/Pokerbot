#!/usr/bin/env python3
"""G3 prior A/B: shipped hardcoded-medians prior vs the IRC-FITTED per-context prior
(irc_population_prior_fitted.json), same seeds, heuristic hero (the shrinkage/cold-start
machinery is hero-independent and ~100x cheaper than search solves).

GATE (per plan): cold-start windows (uplift_curve 0-100) >= old prior on the weak archetypes,
full-run uplift no worse anywhere, hs_tag control unchanged. Neutral steady-state + better
cold-start = PASS (promotion of the fitted file to the default path happens only after ALL
batteries that compare against Phase-0 baselines have finished).

Waits for runs/sweep/sweep_DONE.marker (don't fight earlier stages for cores).
Usage: python scripts/run_prior_ab.py [--seeds 18] [--hands 3000] [--par 8] [--no-wait]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WAIT_MARKER = REPO / "runs" / "sweep" / "sweep_DONE.marker"
OUT_DIR = REPO / "runs" / "prior_ab"
FITTED = REPO / "data" / "player_profiles" / "irc_population_prior_fitted.json"
ARCHES = "hs_station,hs_nit,hs_lag,hs_tag"


def run_one(arm: str, seed: int, hands: int) -> None:
    out = OUT_DIR / f"pab_{arm}_s{seed}.json"
    if out.exists():
        return
    env = {**os.environ, "POKERBOT_BETTING_ABSTRACTION": "1", "PYTHONIOENCODING": "utf-8"}
    if arm == "fitted":
        env["POKERBOT_PRIOR_PATH"] = str(FITTED)
    cmd = [sys.executable, "scripts/eval_exploitation_battery.py",
           "--hero", "heuristic", "--hands", str(hands), "--seed", str(seed), "--allin-ev",
           "--archetypes", ARCHES, "--no-gear-changer", "--out", str(out)]
    with open(OUT_DIR / f"pab_{arm}_s{seed}.log", "w", encoding="utf-8") as lf:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)


def aggregate() -> None:
    import numpy as np

    print(f"\n{'archetype':12s} {'arm':>7s} {'k':>3s} {'pooled':>8s} {'cold 0-100':>11s} {'200+':>8s}")
    for arch in ARCHES.split(","):
        for arm in ("old", "fitted"):
            ups, cold, tail = [], [], []
            for f in sorted(OUT_DIR.glob(f"pab_{arm}_s*.json")):
                try:
                    r = json.loads(f.read_text(encoding="utf-8"))["results"].get(arch)
                except (ValueError, KeyError):
                    continue
                if not r:
                    continue
                ups.append(r["uplift_bb_per_hand"])
                c = r.get("uplift_curve") or {}
                cw = [c[w] for w in ("0-10", "10-25", "25-50", "50-100") if w in c]
                if cw:
                    cold.append(float(np.mean(cw)))
                if "200-3000" in c:
                    tail.append(c["200-3000"])
            if ups:
                print(f"{arch:12s} {arm:>7s} {len(ups):>3d} {np.mean(ups):>+8.3f} "
                      f"{np.mean(cold) if cold else 0:>+11.3f} {np.mean(tail) if tail else 0:>+8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=18)
    ap.add_argument("--hands", type=int, default=3000)
    ap.add_argument("--par", type=int, default=8)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate()
        return
    if not FITTED.exists():
        raise FileNotFoundError(f"{FITTED} missing — run scripts/analyze_irc_population.py first")
    if not args.no_wait:
        while not WAIT_MARKER.exists():
            print("waiting for the adaptation sweep to finish...", flush=True)
            time.sleep(300)

    jobs = [(arm, seed) for arm in ("old", "fitted") for seed in range(args.seeds)]
    print(f"{len(jobs)} runs (par={args.par})")
    with ThreadPoolExecutor(max_workers=args.par) as ex:
        list(ex.map(lambda j: run_one(j[0], j[1], args.hands), jobs))
    (OUT_DIR / "prior_ab_DONE.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S"),
                                                  encoding="utf-8")
    aggregate()


if __name__ == "__main__":
    main()
