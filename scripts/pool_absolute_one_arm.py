#!/usr/bin/env python3
"""ONE-ARM absolute pool eval (CRN deck seeds) — the per-arm-env twin of track2_s3c_pool.py.

track2_s3c_pool runs its two arms (search vs raw) in ONE process, which cannot host two different
KEYING regimes (import/OnceLock latch). This driver runs a SINGLE hero system — spec + blueprint —
vs the CoinPoker-mirror default_pool_roster, seat-rotated duplicate on CRN deck seeds
(deck seed = pure fn of (seed, deck_index), identical across processes). Run it once per system in
a process whose POKERBOT_* env IS that system's serve regime; per-deck seat-mean arrays are stored
in the JSON so a cross-process paired diff (same --seed/--decks) is exact CRN pairing.

Usage (NEW system example — env set BEFORE python):
  POKERBOT_FULLHIST_KEY=1 ... python scripts/pool_absolute_one_arm.py \
      --blueprint checkpoints/bp_pluribus_finemenu_iter50k.pkl \
      --spec '{"settle_showdown_leaf": true}' --decks 600 --seed 0 --out artifacts/pool_new.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import engine_ab_h2h as _eab  # noqa: E402  (env setdefaults; pinned env always wins)
import numpy as np  # noqa: E402
from track2_s3c_pool import SearchHeroAgent, _run_arm  # noqa: E402  (proven harness pieces)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--spec", default='{"settle_showdown_leaf": true}',
                    help="engine_ab_h2h arm spec JSON (worker_env NOT supported here — set the "
                         "process env instead; that is the point of this driver)")
    ap.add_argument("--decks", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--archetypes", default="")
    ap.add_argument("--label", default="one_arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()

    spec = json.loads(args.spec)
    assert spec.get("worker_env") is None, "set the process env instead of worker_env here"

    from pokerbot.search import native_solver as _ns
    if not _ns.native_subgame_available():
        _ns.enable()
    print(f"[pool1] native={_ns.available()} subgame={_ns.native_subgame_available()} "
          f"rayon={_ns._EXT.rayon_num_threads()}", flush=True)
    print("[pool1] keying env: " + " ".join(
        f"{k}={os.environ.get(k, '')}" for k in (
            "POKERBOT_FULLHIST_KEY", "POKERBOT_PRESUM_KEY", "POKERBOT_MRR",
            "POKERBOT_CLOSE_CALLED_STREET", "POKERBOT_PRE_KEEP", "POKERBOT_PRE_RR_KEEP",
            "POKERBOT_RIVER_OCHS", "POKERBOT_POSTFLOP_MODE")), flush=True)

    from pokerbot.eval.blueprint_ranking import default_pool_roster
    roster = default_pool_roster()
    if args.archetypes:
        want = {x.strip() for x in args.archetypes.split(",") if x.strip()}
        roster = [(n, w) for n, w in roster if n in want]
        wsum = sum(w for _n, w in roster) or 1.0
        roster = [(n, w / wsum) for n, w in roster]
    print(f"[pool1] roster: {[(n, round(w, 3)) for n, w in roster]}", flush=True)

    snap = _eab.load_snapshot(args.blueprint)
    print(f"[pool1] blueprint {args.blueprint}: {len(snap):,} infosets", flush=True)

    pf = _eab._build_policy(spec, snap, args.seed, _ROOT)

    def factory(*, exploit: bool):
        return SearchHeroAgent(pf, random.Random(0))

    per_arch = {}
    comb = None
    t0 = time.perf_counter()
    for name, w in roster:
        print(f"\n[pool1] === archetype {name} (w={w:.3f}) ===", flush=True)
        means, wall = _run_arm(factory, name, decks=args.decks, seed=args.seed,
                               workers=args.workers, tag=args.label,
                               progress_every=args.progress_every)
        per_arch[name] = {
            "weight": w, "n": int(means.size),
            "bb100": float(means.mean()) * 100.0,
            "wall_s": wall,
            "deck_means": [float(x) for x in means],
        }
        comb = (w * means) if comb is None else comb + w * means
        _write(args, spec, roster, per_arch, comb, t0, final=False)
    _write(args, spec, roster, per_arch, comb, t0, final=True)
    return 0


def _write(args, spec, roster, per_arch, comb, t0, *, final):
    result = {
        "driver": "pool_absolute_one_arm",
        "label": args.label,
        "blueprint": args.blueprint,
        "spec": spec,
        "decks": args.decks,
        "seed": args.seed,
        "roster": [(n, w) for n, w in roster],
        "per_archetype": per_arch,
        "combined_bb100": (float(np.mean(comb)) * 100.0) if comb is not None else None,
        "keying_env": {k: os.environ.get(k, "") for k in (
            "POKERBOT_FULLHIST_KEY", "POKERBOT_PRESUM_KEY", "POKERBOT_MRR",
            "POKERBOT_CLOSE_CALLED_STREET", "POKERBOT_PRE_KEEP", "POKERBOT_PRE_RR_KEEP",
            "POKERBOT_RIVER_OCHS", "POKERBOT_POSTFLOP_MODE", "POKERBOT_POSTFLOP_EMD_FROZEN")},
        "wall_s": time.perf_counter() - t0,
        "final": final,
    }
    # Cert6 lookup-miss instrument: per-street blueprint coverage (POKERBOT_BP_COVERAGE=1 opt-in;
    # {} when disabled). Counts masked-to-zero rows as misses -- the honest serve-quality surface.
    try:
        from pokerbot.eval.lbr import bp_coverage_report
        result["bp_coverage"] = bp_coverage_report()
    except Exception:  # noqa: BLE001
        result["bp_coverage"] = {}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    if final:
        print(f"[pool1] wrote {args.out}  combined={result['combined_bb100']:+.1f} bb/100",
              flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
