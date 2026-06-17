#!/usr/bin/env python3
"""G4 adaptation tuning sweep: PRIOR_STRENGTH x K_RAMP x GAMMA_FAST (3x3x3), CRN same-seed.

Vehicle = the HEURISTIC hero (the read layer's constants are hero-independent and ExploitHero is
~100x cheaper than 10-iter search solves) on hs_station (steady-state value) + hs_tag (control,
must not degrade) + the gear-changer hs_station->hs_lag (recovery latency). The winner — if any
config beats the default on post-switch uplift with same-seed delta CI>0 AND non-degraded
steady-state/control — gets the expensive 18-replicate confirm on the production config.

The env overrides are import-time, so every config runs as a SUBPROCESS with its own environment.
Waits for runs/rootmixer_fair_DONE.marker first (don't fight the Phase-0 battery for cores).

Usage: python scripts/run_adaptation_sweep.py [--seeds 6] [--hands 3000] [--par 8] [--no-wait]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARKER = REPO / "runs" / "rootmixer_fair_DONE.marker"
OUT_DIR = REPO / "runs" / "sweep"

GRID = {
    "POKERBOT_PRIOR_STRENGTH": ["4.0", "8.0", "16.0"],
    "POKERBOT_K_RAMP": ["2.0", "4.0", "8.0"],
    "POKERBOT_GAMMA_FAST": ["0.85", "0.90", "0.95"],
}


def run_one(cfg: dict[str, str], seed: int, hands: int) -> Path:
    tag = f"P{cfg['POKERBOT_PRIOR_STRENGTH']}_K{cfg['POKERBOT_K_RAMP']}_G{cfg['POKERBOT_GAMMA_FAST']}"
    out = OUT_DIR / f"sw_{tag}_s{seed}.json"
    if out.exists():
        return out  # restart-proof: completed configs are skipped
    env = {**os.environ, **cfg, "POKERBOT_BETTING_ABSTRACTION": "1", "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "scripts/eval_exploitation_battery.py",
           "--hero", "heuristic", "--hands", str(hands), "--seed", str(seed), "--allin-ev",
           "--archetypes", "hs_station,hs_tag",
           "--gear-before", "hs_station", "--gear-after", "hs_lag",
           "--out", str(out)]
    log = OUT_DIR / f"sw_{tag}_s{seed}.log"
    with open(log, "w", encoding="utf-8") as lf:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
    return out


def aggregate() -> None:
    import numpy as np

    rows = []
    for f in sorted(OUT_DIR.glob("sw_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        tag = f.stem.rsplit("_s", 1)[0]
        rows.append((tag, d.get("results") or {}))
    by_tag: dict[str, list[dict]] = {}
    for tag, res in rows:
        by_tag.setdefault(tag, []).append(res)
    print(f"\n{'config':24s} {'k':>2s} {'station':>9s} {'tag(ctl)':>9s} {'post_sw':>9s} {'rec25':>7s}")
    ranking = []
    for tag, runs in sorted(by_tag.items()):
        st = [r["hs_station"]["uplift_bb_per_hand"] for r in runs if "hs_station" in r]
        tg = [r["hs_tag"]["uplift_bb_per_hand"] for r in runs if "hs_tag" in r]
        ps = [r["gear_changer"]["uplift_post_switch"] for r in runs if "gear_changer" in r]
        rec = [r["gear_changer"].get("post_switch_curve", {}).get("0-25")
               for r in runs if "gear_changer" in r]
        rec = [x for x in rec if x is not None]
        if not st:
            continue
        row = (tag, len(st), float(np.mean(st)), float(np.mean(tg)) if tg else 0.0,
               float(np.mean(ps)) if ps else 0.0, float(np.mean(rec)) if rec else 0.0)
        ranking.append(row)
        print(f"{row[0]:24s} {row[1]:>2d} {row[2]:>+9.3f} {row[3]:>+9.3f} {row[4]:>+9.3f} {row[5]:>+7.3f}")
    # Score: post-switch uplift + early-recovery window, with the control as a hard veto.
    ok = [r for r in ranking if r[3] > -0.10]
    if ok:
        best = max(ok, key=lambda r: r[4] + 0.5 * r[5])
        print(f"\nbest (control-clean): {best[0]}  -> confirm at 18 reps on the production config "
              f"before changing any default.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--hands", type=int, default=3000)
    ap.add_argument("--par", type=int, default=8)
    ap.add_argument("--no-wait", action="store_true", help="don't wait for the Phase-0 marker")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate()
        return
    if not args.no_wait:
        while not MARKER.exists():
            print(f"waiting for {MARKER.name} (Phase-0 battery still running)...", flush=True)
            time.sleep(300)

    configs = [dict(zip(GRID.keys(), vals)) for vals in itertools.product(*GRID.values())]
    jobs = [(cfg, seed) for cfg in configs for seed in range(args.seeds)]
    print(f"{len(configs)} configs x {args.seeds} seeds = {len(jobs)} runs (par={args.par})")
    with ThreadPoolExecutor(max_workers=args.par) as ex:
        for i, _ in enumerate(ex.map(lambda j: run_one(j[0], j[1], args.hands), jobs)):
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(jobs)} done", flush=True)
    (OUT_DIR / "sweep_DONE.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    aggregate()


if __name__ == "__main__":
    main()
