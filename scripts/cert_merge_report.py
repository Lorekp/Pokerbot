#!/usr/bin/env python3
"""Merge cert1 box results into the final strength-cert table.

Inputs (downloaded from s3://.../cert_pluribus/cert1/results/ by the babysitter):
  h2h_seed<SEED>_shard<SH>.json  — engine_ab_h2h NEW-vs-INC per-deck nets (disjoint deck sets)
  pool_new.json / pool_inc.json  — pool_absolute_one_arm per-archetype per-deck means (CRN seeds)

Outputs: pooled NEW-INC chips/deck + 95% CI (paired by deck, shards/seeds pooled), per-seed
breakdown, pool absolute bb/100 per archetype per arm + CRN-paired NEW-INC delta per archetype.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

BB_CHIPS = 100.0  # core.rs::BB_CHIPS


def _ci(d):
    d = np.asarray(d, float)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else 0.0
    return m, 1.96 * se, d.size


def main() -> int:
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/cert1_results"

    print("=== H2H: NEW vs INCUMBENT (CRN-paired, seat-rotated duplicate) ===")
    all_nets = []
    per_seed = {}
    for f in sorted(glob.glob(os.path.join(res_dir, "h2h_seed*_shard*.json"))):
        d = json.load(open(f))
        nets = d.get("per_deck_nets", [])
        seed = d.get("seed")
        per_seed.setdefault(seed, []).extend(nets)
        all_nets.extend(nets)
        m, ci, n = _ci(nets) if nets else (0, 0, 0)
        print(f"  {os.path.basename(f)}: n={n} decks  {m:+.1f} ±{ci:.1f} chips/deck "
              f"(settled={d.get('showdown_money_settled')})")
    for seed, nets in sorted(per_seed.items()):
        m, ci, n = _ci(nets)
        print(f"  seed {seed}: n={n}  NEW-INC {m:+.2f} ±{ci:.2f} chips/deck "
              f"= {m/BB_CHIPS*100/2:+.1f} ±{ci/BB_CHIPS*100/2:.1f} bb/100"
              f"  (per deck = 2 rotations ≈ 2 hands)")
    if all_nets:
        m, ci, n = _ci(all_nets)
        lo, hi = m - ci, m + ci
        print(f"\n  POOLED: n={n} decks  NEW-INC = {m:+.2f} ±{ci:.2f} chips/deck  "
              f"CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"          = {m/BB_CHIPS*100/2:+.2f} ±{ci/BB_CHIPS*100/2:.2f} bb/100 "
              f"(bb={BB_CHIPS} chips, 2 hands/deck)")
        verdict = ("NEW STRONGER (CI excludes 0)" if lo > 0 else
                   "INCUMBENT STRONGER (CI excludes 0)" if hi < 0 else
                   "NOT DISTINGUISHABLE (CI includes 0)")
        print(f"  H2H VERDICT: {verdict}")

    print("\n=== POOL: absolute vs CoinPoker-mirror roster (CRN deck seeds) ===")
    pools = {}
    for arm in ("new", "inc"):
        f = os.path.join(res_dir, f"pool_{arm}.json")
        if os.path.exists(f):
            pools[arm] = json.load(open(f))
    if "new" in pools and "inc" in pools:
        combo = {}
        for arm, d in pools.items():
            print(f"  [{arm}] combined {d.get('combined_bb100'):+.1f} bb/100  "
                  f"(decks={d.get('decks')}, final={d.get('final')})")
        for arch in pools["new"]["per_archetype"]:
            a = pools["new"]["per_archetype"][arch]
            b = pools["inc"]["per_archetype"].get(arch)
            if not b:
                continue
            da = np.asarray(a["deck_means"], float)
            db = np.asarray(b["deck_means"], float)
            n = min(da.size, db.size)
            m, ci, _ = _ci((da[:n] - db[:n]) * 100.0)
            print(f"    {arch:>20}: NEW {a['bb100']:+8.1f} | INC {b['bb100']:+8.1f} | "
                  f"paired NEW-INC {m:+7.1f} ±{ci:.1f} bb/100 (n={n})")
            combo[arch] = (m, ci)
    else:
        print("  (pool results incomplete)", pools.keys())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
