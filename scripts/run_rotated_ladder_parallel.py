#!/usr/bin/env python3
"""Parallel seat-rotated skill-ladder: fan (field x seed) across cores, then POOL.

Each (field, seed) is a fully INDEPENDENT rotated-duplicate eval (disjoint deck-seed range, its own
online-learning state), so they are embarrassingly parallel with no shared state — pure process
isolation, bit-identical to running each by hand. Pooling the per-seed estimates across seeds buys
PRECISION from idle cores (tighter CI) rather than just finishing one run faster, and sidesteps the
ON-arm's sequential learning bottleneck entirely (it is per-process here, not cross-deck).

Reuses the tested CLI (scripts/eval_exploitation_battery.py --rotate-seats); adds NO new eval code.
Run:  python scripts/run_rotated_ladder_parallel.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SNAP = r"checkpoints/bp6max_lossless2_iter200.pkl"


def _pool(rows):
    """Pool independent (n, mean, std) estimates -> (N, pooled_mean, pooled_se) (standard pooled var)."""
    n_tot = sum(n for n, _, _ in rows)
    if n_tot == 0:
        return 0, 0.0, 0.0
    mean = sum(n * m for n, m, _ in rows) / n_tot
    # within-group SS + between-group SS (both about the pooled mean) / (N-1)
    ss = sum((n - 1) * s * s for n, _, s in rows if n > 1)
    ss += sum(n * (m - mean) ** 2 for n, m, _ in rows)
    var = ss / (n_tot - 1) if n_tot > 1 else 0.0
    return n_tot, mean, math.sqrt(var / n_tot)


def _run_one(field: str, seed: int, decks: int, iters: int, out: Path) -> tuple[str, int, int]:
    env = dict(os.environ)
    env["POKERBOT_PREFLOP_LOSSLESS"] = "1"
    env["POKERBOT_SEARCH_PREFLOP"] = "1"
    cmd = [sys.executable, str(_ROOT / "scripts" / "eval_exploitation_battery.py"),
           "--hero", "pluribus", "--blueprint-snapshot", _SNAP,
           "--field", field, "--seed", str(seed), "--hands", str(decks),
           "--rotate-seats", "--root-mixer", "--root-mixer-unopened", "--allin-ev",
           "--iterations", str(iters), "--no-aivat-leafq", "--out", str(out)]
    p = subprocess.run(cmd, env=env, cwd=str(_ROOT), capture_output=True, text=True)
    return field, seed, p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="real_recreational_table,real_solid_table,real_pro_table")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--decks", type=int, default=160, help="decks per (field, seed) process")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--workers", type=int, default=9, help="concurrent subprocesses")
    ap.add_argument("--outdir", default="runs/rot_ladder")
    args = ap.parse_args()

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = [(f, s, outdir / f"{f}_s{s}.json") for f in fields for s in seeds]
    print(f"launching {len(jobs)} jobs ({len(fields)} fields x {len(seeds)} seeds x {args.decks} decks) "
          f"on {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, f, s, args.decks, args.iters, o): (f, s) for f, s, o in jobs}
        for fut in as_completed(futs):
            f, s, rc = fut.result()
            print(f"  done: {f} seed={s} rc={rc}", flush=True)

    print("\n=== POOLED seat-rotated win-rate (bb/100, 95% CI) ===", flush=True)
    hdr = f"{'field':24s} {'base bb/100':>14s} {'95% CI':>20s} {'+exploit':>10s} {'uplift':>9s} {'decks':>6s}"
    print(hdr, flush=True)
    summary = {}
    for f in fields:
        off_rows, on_rows, up_rows = [], [], []
        for s in seeds:
            jp = outdir / f"{f}_s{s}.json"
            if not jp.exists():
                continue
            r = json.loads(jp.read_text(encoding="utf-8")).get("results", {}).get(f)
            if not r:
                continue
            b, e = r["off_base"], r["on_exploit"]
            off_rows.append((b["decks"], b["mean_bb"], b["std"]))
            on_rows.append((e["decks"], e["mean_bb"], e["std"]))
            up_rows.append((r["decks"], r["uplift_bb_per_hand"], r["uplift_se"] * math.sqrt(r["decks"])))
        if not off_rows:
            print(f"{f:24s}  (no results)", flush=True)
            continue
        nN, om, ose = _pool(off_rows)
        _, em, _ = _pool(on_rows)
        _, um, use = _pool(up_rows)
        lo, hi = 100 * (om - 1.96 * ose), 100 * (om + 1.96 * ose)
        print(f"{f:24s} {100 * om:>+14.1f} {('['+format(lo,'+.1f')+', '+format(hi,'+.1f')+']'):>20s} "
              f"{100 * em:>+10.1f} {100 * um:>+9.1f} {nN:>6d}", flush=True)
        summary[f] = {"base_bb100": 100 * om, "base_ci95": [lo, hi], "exploit_bb100": 100 * em,
                      "uplift_bb100": 100 * um, "uplift_se_bb100": 100 * use, "decks": nN}
    (outdir / "pooled_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwrote {outdir / 'pooled_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
