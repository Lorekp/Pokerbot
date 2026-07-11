#!/usr/bin/env python3
"""Merge cert battery shard outputs into ONE self-contained VERDICT.json.

Inputs (in --out-dir): h2h_seed*_shard*.json (engine_ab_h2h per-shard results, chips/deck,
CRN-paired NEW-vs-INCUMBENT), pool_new.json / pool_inc.json (pool_absolute_one_arm, bb/100,
CRN deck seeds => cross-process paired diff is exact).

Output: VERDICT.json with (a) pooled H2H paired diff + 95% CI over ALL decks, (b) pool
absolute bb/100 per arm + CRN-paired NEW-INC diff per archetype and combined, (c) a plain
verdict string. Exit 0 even on partial inputs (writes what it can + lists gaps) — the marker
for hard failure stays the launcher's FAILED.txt path.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/opt/pokerbot/cert_out")
    ap.add_argument("--run-id", default="cert")
    args = ap.parse_args()
    od = args.out_dir
    gaps = []

    # ---- H2H shards (chips/deck, paired by deck) ----
    nets, meta = [], []
    for f in sorted(glob.glob(os.path.join(od, "h2h_seed*_shard*.json"))):
        try:
            d = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            gaps.append(f"{os.path.basename(f)}: {e}")
            continue
        pd = d.get("per_deck_nets", [])
        nets += list(pd)
        meta.append({
            "file": os.path.basename(f),
            "decks": len(pd),
            "final": "verdict" in d,   # incremental checkpoints lack the final fields
            "showdown_money_settled": d.get("showdown_money_settled"),
            "seed": d.get("seed"),
            "git_rev": d.get("git_rev"),
        })
    h2h = {"n_decks": len(nets), "shards": meta}
    if len(nets) > 1:
        a = np.asarray(nets, dtype=float)
        mean = float(a.mean())
        se = float(a.std(ddof=1) / math.sqrt(len(a)))
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        h2h.update({"new_minus_inc_per_deck": mean, "ci95": [lo, hi], "se": se})
        verdict = ("NEW >= INCUMBENT: H2H CI excludes 0, NEW ahead" if lo > 0 else
                   ("INCUMBENT STRONGER: H2H CI excludes 0" if hi < 0 else
                    "INCONCLUSIVE: H2H CI includes 0"))
    else:
        verdict = "NO H2H DATA"
        gaps.append("no h2h per-deck nets found")

    # ---- Pool absolutes + CRN-paired per-archetype diff (bb/100) ----
    pool = {}
    for arm in ("pool_new", "pool_inc"):
        p = os.path.join(od, f"{arm}.json")
        try:
            pool[arm] = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            pool[arm] = None
            gaps.append(f"{arm}.json: {e}")
    pool_diff = {}
    if pool.get("pool_new") and pool.get("pool_inc"):
        pn, pi = pool["pool_new"], pool["pool_inc"]
        for name in (pn.get("per_archetype") or {}):
            try:
                x = np.asarray(pn["per_archetype"][name]["deck_means"], dtype=float)
                y = np.asarray(pi["per_archetype"][name]["deck_means"], dtype=float)
                n = min(x.size, y.size)
                d = (x[:n] - y[:n]) * 100.0
                m = float(d.mean())
                s = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
                pool_diff[name] = {
                    "new_bb100": pn["per_archetype"][name]["bb100"],
                    "inc_bb100": pi["per_archetype"][name]["bb100"],
                    "paired_diff_bb100": m,
                    "ci95": [m - 1.96 * s, m + 1.96 * s],
                    "n_decks": int(n),
                }
            except Exception as e:  # noqa: BLE001
                pool_diff[name] = {"error": str(e)}

    out = {
        "run_id": args.run_id,
        "methodology": ("RAW blueprint vs RAW blueprint (search:false both arms), CRN-paired "
                        "seat-rotated duplicate engine_ab_h2h (chips/deck) + CRN pool absolutes "
                        "(bb/100); showdown money settled (harness fix default-ON); closure ON "
                        "both arms"),
        "h2h_new_vs_incumbent": h2h,
        "pool_combined_bb100": {
            "new": (pool.get("pool_new") or {}).get("combined_bb100"),
            "incumbent": (pool.get("pool_inc") or {}).get("combined_bb100"),
        },
        "pool_per_archetype": pool_diff,
        "gaps": gaps,
        "verdict": verdict,
    }
    with open(os.path.join(od, "VERDICT.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"verdict": verdict, "h2h": {k: v for k, v in h2h.items() if k != "shards"},
                      "pool_combined_bb100": out["pool_combined_bb100"], "gaps": gaps}, indent=1),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
