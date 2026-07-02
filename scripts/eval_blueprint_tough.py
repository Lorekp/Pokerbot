#!/usr/bin/env python3
"""Production-readiness eval of a 6-max blueprint TABLE: absolute bb/100 (seat-rotated duplicate) vs a
SOFT anchor and a panel of TOUGH opponents, plus a SELF do-no-harm test (the blueprint vs a field of
itself, which must be ~0). Pure-tabular (no search); parallel OFF arm via POKERBOT_ROTATED_WORKERS.

MUST be a real script file (not a stdin heredoc): the rotated OFF arm uses a Windows ProcessPoolExecutor
(spawn), whose workers RE-IMPORT __main__ -- impossible from stdin, so heredoc + workers hangs with no CPU
load. Hence the if-__name__ guard + module-top serve-env setup (so re-importing workers bake the same
encoder flags before importing pokerbot).

  python scripts/eval_blueprint_tough.py --blueprint checkpoints/serve_blueprint_6max.pkl --decks 2000 --workers 10
"""

from __future__ import annotations

import os

# Encoder serve env -- set BEFORE importing pokerbot (the encoder bakes these at import). Matches the
# iter62000 / deployed stamp (betting abstraction + postflop fast buckets + preflop lossless, 200 buckets).
# Worker processes re-import this module, so this runs in them too, before their pokerbot import.
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "emd")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "160")
# EMD-K160 deployed regime (promoted 2026-07-02, certified +32.89 bb/100 vs made-hand iter62000):
# frozen flop/turn EMD lookup + train-mirrored made-hand fallback on the river (the river is not
# in the blob; without the mirror flag the river frozen-miss serves a UNIFORM policy).
os.environ.setdefault("POKERBOT_POSTFLOP_EMD_FROZEN", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")))
os.environ.setdefault("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "1")

import argparse  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(path: str) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    inner = obj["strat_sum"] if (isinstance(obj, dict) and "strat_sum" in obj) else obj
    return {bytes(k): np.asarray(v, dtype=np.float64) for k, v in inner.items()}


# SOFT anchor + TOUGH panel. Each is a homogeneous 5-villain field (the rotated estimator requires it).
# 'self' is special-cased to the blueprint-vs-itself peer test (villain_snapshot).
SOFT = [("responsive_station", "soft: the dominant calling-station field (value target)")]
TOUGH = [
    ("pro_balanced", "tough: near-equilibrium balanced reg (MDF defense, polarized)"),
    ("pro_aggro", "tough: disciplined aggressive reg (light 3bets/barrels)"),
    ("__self__", "tough: a field of THE BLUEPRINT ITSELF (peer / do-no-harm; expect ~0)"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", default="checkpoints/serve_blueprint_6max.pkl")
    ap.add_argument("--decks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="artifacts/eval_blueprint_tough.json")
    args = ap.parse_args()

    os.environ["POKERBOT_ROTATED_WORKERS"] = str(int(args.workers))
    os.environ.setdefault("POKERBOT_BP_COVERAGE", "1")

    import json

    import pokerbot.eval.blueprint_ranking as br

    snap = _load(args.blueprint)
    cov = br.serve_coverage_probe(snap)
    print(f"[tough] blueprint {args.blueprint}: {len(snap):,} infosets   "
          f"serve coverage {round((cov or 0)*100,1)}% postflop   "
          f"decks={args.decks}  workers={args.workers}", flush=True)
    if cov is not None and cov < 0.5:
        raise SystemExit(f"serve coverage {cov*100:.0f}% < 50% -- abstraction mismatch, aborting")

    def absbb(*, villain_name=None, villain_snapshot=None, decks=None):
        a = br._rotated_seat_means(snap, villain_name=villain_name, villain_snapshot=villain_snapshot,
                                   decks=decks or args.decks, seed=args.seed)
        m = a.mean() * 100.0
        se = a.std(ddof=1) / math.sqrt(len(a)) * 100.0
        return m, 1.96 * se, len(a)

    results = {}
    t0 = time.time()
    print("\niter62000 pure-tabular ABSOLUTE bb/100 (seat-rotated duplicate):", flush=True)
    for name, desc in SOFT + TOUGH:
        t = time.time()
        if name == "__self__":
            m, ci, n = absbb(villain_snapshot=snap)
            label = "5x SELF (iter62000)"
        else:
            m, ci, n = absbb(villain_name=name)
            label = f"5x {name}"
        lo, hi = m - ci, m + ci
        verdict = ("WINS (CI>0)" if lo > 0 else ("LOSES (CI<0)" if hi < 0 else "~breakeven (CI spans 0)"))
        results[name] = {"bb_per_100": m, "ci95": ci, "n": n, "lo": lo, "hi": hi, "verdict": verdict, "desc": desc}
        print(f"  vs {label:22s}: {m:+8.2f} +- {ci:6.2f} bb/100  [{verdict:22s}]  ({time.time()-t:.0f}s)",
              flush=True)
        print(f"        {desc}", flush=True)
    print(f"\n[tough] total {time.time()-t0:.0f}s with {args.workers} workers", flush=True)

    # Production-readiness summary
    soft_ok = results["responsive_station"]["lo"] > 0
    tough_loses = [n for n in ("pro_balanced", "pro_aggro") if results[n]["hi"] < 0]
    self_r = results["__self__"]
    self_ok = abs(self_r["bb_per_100"]) <= self_r["ci95"] * 1.5  # within ~1.5x CI of 0
    print("\n=== PRODUCTION-READINESS (blueprint TABLE) ===", flush=True)
    print(f"  soft extraction (stations): {'STRONG +' if soft_ok else 'weak'} "
          f"({results['responsive_station']['bb_per_100']:+.0f} bb/100)", flush=True)
    print(f"  tough do-no-harm: {'PASS (no fixed reg beats it)' if not tough_loses else 'FAILS vs '+','.join(tough_loses)}",
          flush=True)
    print(f"  self do-no-harm (vs itself ~0): {'PASS' if self_ok else 'IMBALANCE'} "
          f"({self_r['bb_per_100']:+.0f} +- {self_r['ci95']:.0f})", flush=True)
    out = {"blueprint": args.blueprint, "infosets": len(snap), "decks": args.decks,
           "coverage": cov, "results": results,
           "production_ready_table": bool(soft_ok and not tough_loses and self_ok)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[tough] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
