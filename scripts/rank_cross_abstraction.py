#!/usr/bin/env python3
"""Rank two blueprints with DIFFERENT serve abstractions by win-rate (bb/100) with a paired (B-A) CI.

The encoder bakes ONE postflop abstraction per process at import, so an EMD-K160 blueprint (mode=emd) and a
made-hand blueprint (mode=fast) CANNOT be served in the same process. So each arm runs in its OWN subprocess
with its OWN serve env, computing per-deck seat-means vs the SAME fixed CoinPoker-mirror field on the SAME
deck seeds. The field bots are abstraction-INDEPENDENT (HUD/heuristic, not blueprint-served) and the deck
seed is a pure function of (seed, deck_index), so the two arms are CRN-PAIRED deck-for-deck -- the parent
pairs the per-deck means -> paired (B-A) delta + CI. This is the vs-pool signal across abstractions
(head-to-head can't cross abstractions: both blueprints at one table would need both serve envs at once).

  python scripts/rank_cross_abstraction.py \
    --a checkpoints/serve_blueprint_6max.pkl --a-mode fast --a-num-buckets 200 \
    --b runs/native_cfr/bp_emd_k160_1.nlcfr  --b-mode emd  --b-num-buckets 160 \
    --b-emd-frozen data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn \
    --decks 2000 --seed 0 --workers 10 --label-a iter62000 --label-b EMD_K160 --out-json artifacts/rank_emd_vs_iter62000.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nlcfr", ".bin"):
        from f32_regret_strategy_delta import avg_strategy, load_checkpoint  # type: ignore

        ck = load_checkpoint(path)
        return {bytes(k): np.asarray(avg_strategy(row), dtype=np.float64) for k, row in ck["strat"].items()}
    import pickle

    obj = pickle.load(open(path, "rb"))
    inner = obj["strat_sum"] if (isinstance(obj, dict) and "strat_sum" in obj) else obj
    return {bytes(k): np.asarray(v, dtype=np.float64) for k, v in inner.items()}


def _arm_env(base, *, mode, num_buckets, emd_frozen, workers, river_ochs="",
             preflop_hist_key=False) -> dict:
    """Serve env for one arm (set by the parent for the subprocess; baked at the child's pokerbot import)."""
    env = dict(base)
    env["POKERBOT_BETTING_ABSTRACTION"] = "1"
    env["POKERBOT_POSTFLOP_BUCKETS"] = "1"
    env["POKERBOT_PREFLOP_LOSSLESS"] = "1"
    env["POKERBOT_POSTFLOP_NUM_BUCKETS"] = str(int(num_buckets))
    env["POKERBOT_POSTFLOP_MODE"] = str(mode)
    env["POKERBOT_BP_COVERAGE"] = "1"
    env["POKERBOT_ROTATED_WORKERS"] = str(int(workers))
    if str(mode).lower() == "emd" and emd_frozen:
        env["POKERBOT_POSTFLOP_EMD_FROZEN"] = os.path.abspath(emd_frozen)
        # river is keyed made-hand by the trainer (not in the flop/turn blob) -> serve must mirror or the
        # river plays UNIFORM and the EMD arm ranks falsely low (verified river hit 0.000 -> 0.601).
        env["POKERBOT_EMD_FROZEN_MIRROR_FALLBACK"] = "1"
    if river_ochs:
        # blueprint-v2 river OCHS: the arm's river keys come from the OCHS table the arm's
        # blueprint trained under (a skew would HIT wrong-semantics river rows, not miss).
        env["POKERBOT_RIVER_OCHS"] = os.path.abspath(river_ochs)
    else:
        env.pop("POKERBOT_RIVER_OCHS", None)  # never leak the parent's env into a non-OCHS arm
    if preflop_hist_key:
        # blueprint-v3 preflop history key: the arm's PREFLOP keys carry the b"H" bytes its
        # blueprint trained under (skew in either direction = 100% preflop miss -> uniform, and the
        # arm's <50% coverage gate fails loud).
        env["POKERBOT_PREFLOP_HIST_KEY"] = "1"
    else:
        env.pop("POKERBOT_PREFLOP_HIST_KEY", None)  # arm env is fully determined by the arm's args
    return env


def _arm_main(args) -> int:
    # The parent already set this process's serve env -> importing pokerbot bakes the correct abstraction.
    from pokerbot.eval import blueprint_ranking as br

    snap = _load(args.blueprint)
    cov = br.serve_coverage_probe(snap)
    covpct = round((cov or 0) * 100, 1)
    print(f"[arm] {args.label}: {len(snap):,} infosets  serve coverage {covpct}% postflop "
          f"(mode={os.environ.get('POKERBOT_POSTFLOP_MODE')} K={os.environ.get('POKERBOT_POSTFLOP_NUM_BUCKETS')})",
          flush=True)
    if cov is not None and cov < 0.5:
        print(f"[arm] FATAL {args.label} coverage {covpct}% < 50% -- abstraction mis-served (uniform)", flush=True)
        return 3
    roster = br.default_pool_roster()
    per = {}
    for name, w in roster:
        a = br._rotated_seat_means(snap, villain_name=name, decks=args.decks, seed=args.seed)
        per[name] = {"weight": float(w), "means": [float(x) for x in a]}
        print(f"[arm] {args.label} vs {name}: mean {float(np.mean(a))*100:+.1f} bb/100 ({len(a)} decks)", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "infosets": len(snap), "coverage": cov, "per": per}, f)
    return 0


def _run_arm(blueprint, *, mode, num_buckets, emd_frozen, decks, seed, workers, label,
             river_ochs="", preflop_hist_key=False) -> dict:
    out = tempfile.mktemp(suffix=f"_{label}.json")
    cmd = [sys.executable, os.path.abspath(__file__), "--arm", "--blueprint", blueprint,
           "--decks", str(decks), "--seed", str(seed), "--out", out, "--label", label]
    env = _arm_env(os.environ, mode=mode, num_buckets=num_buckets, emd_frozen=emd_frozen,
                   workers=workers, river_ochs=river_ochs, preflop_hist_key=preflop_hist_key)
    print(f"[xrank] launching arm {label} ({mode}/K{num_buckets}) ...", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("[arm]"):
            print("   " + line, flush=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + "\n" + r.stderr[-2000:] + "\n")
        raise SystemExit(f"arm {label} failed (rc={r.returncode})")
    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    os.remove(out)
    return data


def _paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); d = b - a; n = d.size
    mean = float(d.mean()); se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mean * 100.0, 1.96 * se * 100.0, float(a.mean()) * 100.0, float(b.mean()) * 100.0, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="store_true", help="(internal) run one abstraction arm in this process")
    ap.add_argument("--blueprint"); ap.add_argument("--out"); ap.add_argument("--label", default="")
    ap.add_argument("--decks", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--a"); ap.add_argument("--a-mode", default="fast"); ap.add_argument("--a-num-buckets", type=int, default=200); ap.add_argument("--a-emd-frozen", default="")
    ap.add_argument("--b"); ap.add_argument("--b-mode", default="emd"); ap.add_argument("--b-num-buckets", type=int, default=160); ap.add_argument("--b-emd-frozen", default="")
    ap.add_argument("--a-river-ochs", default="", help="ROCHS table for arm A (blueprint-v2 river OCHS)")
    ap.add_argument("--b-river-ochs", default="", help="ROCHS table for arm B (blueprint-v2 river OCHS)")
    ap.add_argument("--a-preflop-hist-key", action="store_true",
                    help="arm A trained with POKERBOT_PREFLOP_HIST_KEY=1 (blueprint-v3)")
    ap.add_argument("--b-preflop-hist-key", action="store_true",
                    help="arm B trained with POKERBOT_PREFLOP_HIST_KEY=1 (blueprint-v3, e.g. bp3_prefhist_1)")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.arm:
        return _arm_main(args)

    print(f"[xrank] {args.label_a} ({args.a_mode}/K{args.a_num_buckets}) vs {args.label_b} "
          f"({args.b_mode}/K{args.b_num_buckets})  decks={args.decks} seed={args.seed} workers={args.workers}",
          flush=True)
    da = _run_arm(args.a, mode=args.a_mode, num_buckets=args.a_num_buckets, emd_frozen=args.a_emd_frozen,
                  decks=args.decks, seed=args.seed, workers=args.workers, label=args.label_a,
                  river_ochs=args.a_river_ochs, preflop_hist_key=args.a_preflop_hist_key)
    db = _run_arm(args.b, mode=args.b_mode, num_buckets=args.b_num_buckets, emd_frozen=args.b_emd_frozen,
                  decks=args.decks, seed=args.seed, workers=args.workers, label=args.label_b,
                  river_ochs=args.b_river_ochs, preflop_hist_key=args.b_preflop_hist_key)

    # both arms used the SAME roster (default_pool_roster is deterministic) -> pair by archetype.
    if list(da["per"]) != list(db["per"]):
        raise SystemExit("arms used different rosters -- cannot pair")
    print(f"\n[xrank] paired (B-A) by archetype   "
          f"[{args.label_a}: {da['infosets']:,}if cov{(da['coverage'] or 0)*100:.0f}%  |  "
          f"{args.label_b}: {db['infosets']:,}if cov{(db['coverage'] or 0)*100:.0f}%]", flush=True)
    comb_a = comb_b = None
    per_out = {}
    for name in da["per"]:
        w = da["per"][name]["weight"]
        a = np.asarray(da["per"][name]["means"]); b = np.asarray(db["per"][name]["means"])
        delta, ci, am, bm, n = _paired(a, b)
        flag = f"{args.label_b}>{args.label_a}" if (delta - ci) > 0 else (f"{args.label_a}>{args.label_b}" if (delta + ci) < 0 else "~0")
        print(f"  vs 5x {name:20s} (w={w:.3f}): delta(B-A) {delta:+8.2f} +- {ci:6.2f} bb/100  "
              f"[A {am:+7.2f} | B {bm:+7.2f}]  {flag}", flush=True)
        comb_a = (w * a) if comb_a is None else comb_a + w * a
        comb_b = (w * b) if comb_b is None else comb_b + w * b
        per_out[name] = {"weight": w, "delta_bb_per_100": delta, "ci95": ci, "a": am, "b": bm}

    delta, ci, am, bm, n = _paired(comb_a, comb_b)
    b_stronger = (delta - ci) > 0; a_stronger = (delta + ci) < 0
    verdict = (f"{args.label_b} STRONGER (95% CI excludes 0)" if b_stronger else
               (f"{args.label_a} STRONGER (95% CI excludes 0)" if a_stronger else
                "INCONCLUSIVE (95% CI includes 0)"))
    print("  " + "-" * 70, flush=True)
    print(f"  COMBINED (field-weighted): delta(B-A) {delta:+.2f} +- {ci:.2f} bb/100  "
          f"[A {am:+.2f} | B {bm:+.2f}]  n={n}", flush=True)
    print(f"\n[xrank] VERDICT: {verdict}", flush=True)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"a": args.a, "b": args.b, "label_a": args.label_a, "label_b": args.label_b,
                       "decks": args.decks, "seed": args.seed,
                       "a_infosets": da["infosets"], "b_infosets": db["infosets"],
                       "a_coverage": da["coverage"], "b_coverage": db["coverage"],
                       "combined": {"delta_bb_per_100": delta, "ci95": ci, "a": am, "b": bm, "n": n},
                       "per_archetype": per_out, "verdict": verdict}, f, indent=2)
        print(f"[xrank] wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
