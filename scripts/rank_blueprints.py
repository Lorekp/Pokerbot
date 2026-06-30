#!/usr/bin/env python3
"""Rank two 6-max NLHE BLUEPRINT checkpoints by win-rate (bb/100) with a PAIRED (B-A) CI.

The trustworthy 6-max strength signal (LBR cannot rank 3+ player blueprints -- no Nash, and our LBR
RISES as the strategy sharpens). Mirrors the Pluribus recipe: mbb/game vs a FIXED realistic field +
head-to-head, SEAT-ROTATED duplicate dealing, CRN-paired, never a bare point estimate. Compares the raw
TABLES via pure-tabular play (no search, no broken AIVAT), so it directly answers "did this training
change (DCFR vs CFR+, more iters, a richer abstraction) produce a STRONGER blueprint?".

  python scripts/rank_blueprints.py --a OLD.nlcfr --b NEW.nlcfr --decks 600 --seed 0 --out rank.json
  python scripts/rank_blueprints.py --a cfrplus.pkl --b dcfr.pkl --decks 1500 --signals both

Accepts a native NLCFR001 checkpoint (.nlcfr/.bin) or a snapshot/trainer pickle (.pkl). A and B must
share the SAME card abstraction. CRITICAL: a blueprint that misses its training abstraction plays UNIFORM
(silent uniform-vs-uniform => fake "no difference"), so this CLI sets the encoder flags
(POKERBOT_BETTING_ABSTRACTION / POKERBOT_POSTFLOP_BUCKETS / POKERBOT_PREFLOP_LOSSLESS) from the snapshot's
training stamp BEFORE importing pokerbot (the encoder bakes them at import) AND enforces a serve-coverage
floor -- a misconfigured serve FAILS LOUD instead of returning meaningless noise.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

# scripts/ + repo root on the path so the native NLCFR loader imports for a .nlcfr input.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---- loaders that DO NOT import pokerbot (so we can set encoder env vars before the pokerbot import) ----
def _load_raw(path: str):
    """Return ``(raw_obj, abstraction_stamp_or_None)`` without importing pokerbot. For a native NLCFR
    checkpoint the raw_obj is the already-converted ``{bytes: row}`` table; for a pickle it is the loaded
    object (stamped trainer dict or raw snapshot)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nlcfr", ".bin"):
        from f32_regret_strategy_delta import avg_strategy, load_checkpoint  # type: ignore

        ckpt = load_checkpoint(path)
        table = {bytes(k): np.asarray(avg_strategy(row), dtype=np.float64) for k, row in ckpt["strat"].items()}
        return ("native", table), None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    stamp = obj.get("abstraction") if isinstance(obj, dict) else None
    return ("pickle", obj), stamp


def _to_table(raw) -> dict:
    """Convert a loaded raw object into the ``{fingerprint(bytes) -> strat row}`` table the adapter serves."""
    kind, obj = raw
    if kind == "native":
        return obj
    if isinstance(obj, dict) and "strat_sum" in obj:
        return {bytes(k): np.asarray(v, dtype=np.float64) for k, v in obj["strat_sum"].items()}
    return {bytes(k): np.asarray(v, dtype=np.float64) for k, v in obj.items()}


def _configure_serve_env(stamp: dict | None, args) -> dict:
    """Set the encoder env vars (baked at pokerbot import) so blueprint serve keys HIT the trained table.
    Defaults to the native-trainer / deployed serve config (betting abstraction + postflop buckets +
    preflop lossless, 200 buckets, mode 'fast'); a stamped snapshot overrides from its own training stamp,
    and explicit CLI flags override everything (needed for the EMD K160 production path -- mode 'emd' with
    a frozen lookup / centroid table -- which a 'fast' default would silently mis-serve)."""
    cfg = {"betting_abstraction": True, "postflop_buckets": True, "preflop_lossless": True,
           "postflop_num_buckets": args.num_buckets, "postflop_mode": "fast",
           "emd_table": None, "emd_frozen": None}
    if stamp:
        for k in ("betting_abstraction", "postflop_buckets", "preflop_lossless",
                  "postflop_num_buckets", "postflop_mode"):
            if k in stamp:
                cfg[k] = stamp[k]
        # propagate an EMD frozen/centroid path from the stamp ONLY if it exists locally (cloud paths won't)
        for sk, ck in (("postflop_emd_frozen_path", "emd_frozen"), ("postflop_emd_table", "emd_table")):
            p = stamp.get(sk)
            if p and os.path.exists(p):
                cfg[ck] = p
    # explicit CLI flags win (and are REQUIRED for emd when the stamp's path isn't local)
    if args.postflop_mode:
        cfg["postflop_mode"] = args.postflop_mode
    if args.emd_table:
        cfg["emd_table"] = args.emd_table
    if args.emd_frozen:
        cfg["emd_frozen"] = args.emd_frozen

    os.environ["POKERBOT_BETTING_ABSTRACTION"] = "1" if cfg["betting_abstraction"] else "0"
    os.environ["POKERBOT_POSTFLOP_BUCKETS"] = "1" if cfg["postflop_buckets"] else "0"
    os.environ["POKERBOT_PREFLOP_LOSSLESS"] = "1" if cfg["preflop_lossless"] else "0"
    os.environ["POKERBOT_POSTFLOP_NUM_BUCKETS"] = str(int(cfg["postflop_num_buckets"]))
    os.environ["POKERBOT_POSTFLOP_MODE"] = str(cfg["postflop_mode"])
    if cfg["emd_frozen"]:
        os.environ["POKERBOT_POSTFLOP_EMD_FROZEN"] = str(cfg["emd_frozen"])
    if cfg["emd_table"]:
        os.environ["POKERBOT_POSTFLOP_EMD_TABLE"] = str(cfg["emd_table"])
    os.environ["POKERBOT_BP_COVERAGE"] = "1"          # enable the serve-coverage guard
    if str(cfg["postflop_mode"]).lower() == "emd" and not (cfg["emd_frozen"] or cfg["emd_table"]):
        print("[rank] WARNING: postflop_mode=emd but no --emd-frozen/--emd-table (and none in the stamp "
              "exists locally); serve will FAIL or mis-key. Pass the EMD blob explicitly.", flush=True)
    return cfg


class _Progress:
    """Live per-arm progress with a time-throttled heartbeat: %, decks/min, current-arm ETA, overall ETA
    (the standing rule -- a non-trivial job must show stuck-vs-slow up front)."""

    def __init__(self, total_decks: int, every_s: float = 5.0):
        self.total = max(1, total_decks)
        self.every_s = every_s
        self.t0 = time.time()
        self.done_before = 0
        self._cur_tag = None
        self._cur_arm_total = 1
        self._cur_t0 = self.t0
        self._last_print = 0.0

    def __call__(self, done: int, arm_total: int, tag: str):
        now = time.time()
        if tag != self._cur_tag:
            if self._cur_tag is not None:
                self.done_before += self._cur_arm_total
            self._cur_tag, self._cur_arm_total, self._cur_t0, self._last_print = tag, arm_total, now, 0.0
        if now - self._last_print < self.every_s and done + 1 < arm_total:
            return
        self._last_print = now
        overall_done = self.done_before + done
        arm_rate = max(1e-9, done / max(1e-9, now - self._cur_t0))
        overall_rate = max(1e-9, overall_done / max(1e-9, now - self.t0))
        print(f"[rank] {tag:24s} {done:>5d}/{arm_total} ({100.0*done/max(1,arm_total):3.0f}%)  "
              f"{arm_rate*60:5.0f} decks/min  arm-ETA {_fmt((arm_total-done)/arm_rate)}  | "
              f"overall {overall_done}/{self.total} ({100.0*overall_done/self.total:3.0f}%)  "
              f"ETA {_fmt((self.total-overall_done)/overall_rate)}", flush=True)


def _fmt(sec: float) -> str:
    sec = max(0.0, float(sec))
    return f"{sec:.0f}s" if sec < 90 else (f"{sec/60:.0f}m" if sec < 5400 else f"{sec/3600:.1f}h")


def _line(r) -> str:
    flag = "B>A" if r.b_stronger else ("A>B" if r.a_stronger else "~0 (CI incl 0)")
    return (f"  delta(B-A) {r.delta_bb_per_100:+8.2f} +- {r.ci95_bb_per_100:6.2f} bb/100   "
            f"[A {r.a_bb_per_100:+7.2f} | B {r.b_bb_per_100:+7.2f}]   "
            f"t={r.t_stat:+5.2f}  VR={r.var_reduction:4.1f}x  n={r.n}   -> {flag}")


def _result_dict(r) -> dict:
    return {"delta_bb_per_100": r.delta_bb_per_100, "ci95_bb_per_100": r.ci95_bb_per_100,
            "se_bb_per_100": r.se_bb_per_100, "t_stat": r.t_stat,
            "a_bb_per_100": r.a_bb_per_100, "b_bb_per_100": r.b_bb_per_100,
            "var_reduction": r.var_reduction, "n": r.n,
            "b_stronger": r.b_stronger, "a_stronger": r.a_stronger, "excludes_zero": r.excludes_zero}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline blueprint (.nlcfr/.bin/.pkl) -- the OLD / control arm")
    ap.add_argument("--b", required=True, help="candidate blueprint (.nlcfr/.bin/.pkl) -- the NEW / test arm")
    ap.add_argument("--decks", type=int, default=600, help="distinct deals per arm (each = n_seats rotations)")
    ap.add_argument("--seed", type=int, default=0, help="CRN seed (same seed across arms => paired decks)")
    ap.add_argument("--signals", choices=("both", "head_to_head", "vs_pool"), default="both")
    ap.add_argument("--num-buckets", type=int, default=200, help="postflop bucket count for the serve env")
    ap.add_argument("--postflop-mode", default=None, choices=("fast", "emd", "ehs"),
                    help="serve abstraction mode; default = stamp's mode, else 'fast'. Use 'emd' for the "
                         "EMD K160 production blueprint (then pass --emd-frozen/--emd-table).")
    ap.add_argument("--emd-frozen", default=None, help="path to the frozen EMD lookup (.emdfrzn) for emd mode")
    ap.add_argument("--emd-table", default=None, help="path to the EMD centroids (.json) for emd mode")
    ap.add_argument("--coverage-floor", type=float, default=0.5,
                    help="min postflop serve coverage; below this the run FAILS LOUD (abstraction mismatch)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default=None, help="write the full result JSON here")
    args = ap.parse_args()

    # 1) Load raw + peek the abstraction stamp BEFORE importing pokerbot.
    print(f"[rank] loading A={args.a}", flush=True)
    raw_a, stamp_a = _load_raw(args.a)
    print(f"[rank] loading B={args.b}", flush=True)
    raw_b, stamp_b = _load_raw(args.b)

    # 2) Configure the encoder serve env from the stamp (or native defaults) BEFORE the pokerbot import.
    cfg = _configure_serve_env(stamp_a or stamp_b, args)
    print(f"[rank] serve env: betting_abstraction={cfg['betting_abstraction']} "
          f"postflop_buckets={cfg['postflop_buckets']} preflop_lossless={cfg['preflop_lossless']} "
          f"num_buckets={cfg['postflop_num_buckets']} mode={cfg['postflop_mode']} "
          f"(stamped={'yes' if (stamp_a or stamp_b) else 'no/native'})", flush=True)
    if not (stamp_a or stamp_b):
        print("[rank] NOTE: native/unstamped checkpoints -> the serve env is NOT auto-verified against a "
              "training stamp. --num-buckets / --postflop-mode MUST match how A and B were TRAINED, or the "
              "serve-coverage guard will (correctly) fail the run.", flush=True)

    # 3) NOW import pokerbot (encoder constants are baked here).
    from pokerbot.eval.blueprint_ranking import decide, default_pool_roster, rank_head_to_head, rank_vs_pool
    from pokerbot.eval.lbr import check_blueprint_serve_env

    snap_a, snap_b = _to_table(raw_a), _to_table(raw_b)
    print(f"[rank] A: {len(snap_a):,} infosets   B: {len(snap_b):,} infosets   "
          f"decks={args.decks}  seed={args.seed}  signals={args.signals}", flush=True)

    # 4) Strict up-front serve check for STAMPED snapshots (the authoritative guard); the coverage floor
    #    in the rank functions is the backstop for unstamped native checkpoints.
    for stamp, label in ((stamp_a, args.label_a), (stamp_b, args.label_b)):
        if stamp:
            check_blueprint_serve_env(expected=stamp, strict=True)
            print(f"[rank] serve-env stamp check OK for {label}", flush=True)
    if args.a == args.b:
        print("[rank] NOTE: A and B are the SAME file -> every paired delta must be exactly 0 (self-check)",
              flush=True)

    roster = default_pool_roster()
    n_h2h = 2 if args.signals in ("both", "head_to_head") else 0
    n_pool = 2 * len(roster) if args.signals in ("both", "vs_pool") else 0
    prog = _Progress((n_h2h + n_pool) * args.decks)
    floor = args.coverage_floor

    out: dict = {"a": args.a, "b": args.b, "decks": args.decks, "seed": args.seed,
                 "a_infosets": len(snap_a), "b_infosets": len(snap_b), "serve_env": cfg,
                 "roster": [(n, round(w, 4)) for n, w in roster]}

    h2h = pool = None
    if args.signals in ("both", "head_to_head"):
        print("\n[rank] === HEAD-TO-HEAD (A vs field-of-B, B vs field-of-A; CRN-paired) ===", flush=True)
        t = time.time()
        h2h = rank_head_to_head(snap_a, snap_b, decks=args.decks, seed=args.seed,
                                coverage_floor=floor, progress_cb=prog)
        print(_line(h2h) + f"   ({_fmt(time.time()-t)})", flush=True)
        out["head_to_head"] = _result_dict(h2h)

    if args.signals in ("both", "vs_pool"):
        print("\n[rank] === VS FIXED POOL (CoinPoker-mirror field; population-weighted; CRN-paired) ===",
              flush=True)
        t = time.time()
        pool, per_arch = rank_vs_pool(snap_a, snap_b, roster=roster, decks=args.decks, seed=args.seed,
                                      coverage_floor=floor, progress_cb=prog)
        print(f"  {'archetype':20s} {'weight':>7s}   {'delta(B-A) bb/100':>22s}   verdict", flush=True)
        for name, (r, w) in per_arch.items():
            flag = "B>A" if r.b_stronger else ("A>B" if r.a_stronger else "~0")
            print(f"  {name:20s} {w:7.3f}   {r.delta_bb_per_100:+9.2f} +- {r.ci95_bb_per_100:7.2f}   {flag}",
                  flush=True)
        print("  " + "-" * 66, flush=True)
        print("  COMBINED (field-weighted):", flush=True)
        print(_line(pool) + f"   ({_fmt(time.time()-t)})", flush=True)
        out["vs_pool"] = _result_dict(pool)
        out["vs_pool_per_archetype"] = {n: {**_result_dict(r), "weight": w} for n, (r, w) in per_arch.items()}

    print("\n[rank] === VERDICT ===", flush=True)
    if h2h is not None and pool is not None:
        verdict, meta = decide(h2h, pool, label_a=args.label_a, label_b=args.label_b)
        print(f"  {verdict}", flush=True)
        print(f"    head-to-head: {h2h.summary()}", flush=True)
        print(f"    vs-pool:      {pool.summary()}", flush=True)
        out["verdict"] = verdict
        out["verdict_meta"] = meta
    else:
        r = h2h or pool
        which = "head-to-head" if h2h is not None else "vs-pool"
        single = (f"{args.label_b} stronger" if r.b_stronger else
                  (f"{args.label_a} stronger" if r.a_stronger else "inconclusive (CI includes 0)"))
        print(f"  [{which} only] {single}: {r.summary()}", flush=True)
        out["verdict"] = f"{which}: {single}"

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n[rank] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
