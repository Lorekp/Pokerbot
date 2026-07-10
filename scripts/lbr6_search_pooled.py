"""In-process pooled 6-max LBR vs the DEPLOYED blueprint+search serve (converged iter10).

ONE warm process (no per-cell JAX reimport, no multiprocess oversubscription) that pools many small seeds
into a tight CI per menu. The seat ACTS the full depth-limited Pluribus search (iter10 uniform-leaf = the
deployed/basestrength config); the LBR best-responds, range-tracking against the cheap blueprint MODEL (the
two-policy harness, pokerbot/eval/lbr6.py:603). value_only_raises (search fold-probs un-modellable).

  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 \
    python scripts/lbr6_search_pooled.py --hands 400 --seeds 20 --menus fc,fcpa
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS",
                      "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1")
os.environ.setdefault("POKERBOT_SEARCH_BLUEPRINT_BLEND", "0")  # converged solve never blends; keep inert


def _pool(triples):
    triples = [t for t in triples if t[2]]
    N = sum(n for _, _, n in triples)
    M = sum(m * n for m, _, n in triples) / N
    ss = sum(n * (sd ** 2) + n * (m - M) ** 2 for m, sd, n in triples)
    se = math.sqrt((ss / N) / N)
    return {"bb_per_100": round(M * 100, 1), "ci_lo": round((M - 1.96 * se) * 100, 1),
            "ci_hi": round((M + 1.96 * se) * 100, 1), "hands": N, "k": len(triples)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--menus", default="fc,fcpa")
    ap.add_argument("--n-seats", type=int, default=6)
    ap.add_argument("--blueprint-snapshot",
                    default=r"checkpoints/bp6max_dcfrvr_iter200.pkl")
    ap.add_argument("--out", default="runs/lbr6_search_pooled.json")
    ap.add_argument("--raw", action="store_true",
                    help="serve the RAW blueprint (no search) through the SAME estimator/pooling -- the "
                         "matched no-search baseline for the served-with-search numbers (iters ignored).")
    ap.add_argument("--two-sided", action="store_true",
                    help="match the DEPLOYED serve: two-sided subgame solve (POKERBOT_LIVE_TWO_SIDED=1). "
                         "One-sided (the default) is the 'highly exploitable over-aggression' config.")
    ap.add_argument("--rnr-restrict", action="store_true",
                    help="match the DEPLOYED serve: RNR-restricted solver-internal BR (POKERBOT_RNR_SOLVER=1).")
    args = ap.parse_args()

    import pickle

    from pokerbot.eval import lbr6
    from pokerbot.eval.lbr import make_blueprint_policy
    from pokerbot.eval.solver_exploitability import make_search_policy_fn

    with open(args.blueprint_snapshot, "rb") as f:
        snap_full = pickle.load(f)
    snap = (snap_full["strat_sum"]
            if isinstance(snap_full, dict) and "strat_sum" in snap_full else snap_full)
    print(f"blueprint snapshot: {len(snap)} infosets", flush=True)
    model = make_blueprint_policy(snap)
    _MENUS = {"fc": (), "fcpa": (lbr6._POT_BET_INDEX, lbr6.ALL_IN),
              "full": (lbr6._HALF_POT_INDEX, lbr6._POT_BET_INDEX, lbr6.ALL_IN)}
    menus = [m for m in args.menus.split(",") if m in _MENUS]

    results = {}
    t0 = time.time()
    for menu in menus:
        trip = []
        for seed in range(args.seeds):
            # New search policy per seed (its own deterministic RNG); converged iter10 (blend inert).
            # --raw: serve the bare blueprint model through the identical estimator (matched baseline).
            act = model if args.raw else make_search_policy_fn(
                snap, leaf_select="uniform", seed=seed, iterations=args.iters)
            cfg = lbr6.LBR6Config(hands=args.hands, seed=seed, n_seats=args.n_seats,
                                  track_ranges=True, paired=False, value_only_raises=True)
            cfg.bet_candidates = _MENUS[menu]
            r = lbr6.lbr6_exploitability(act, cfg, model_fn=model)
            trip.append((r.ev_bb_per_hand, r.stderr_bb_per_hand * (r.hands ** 0.5), r.hands))
            p = _pool(trip)
            print(f"  [{menu}] seed {seed:2d} done | running pool: {p['bb_per_100']:+.1f} "
                  f"[{p['ci_lo']:+.1f},{p['ci_hi']:+.1f}] ({p['hands']}h, {(time.time()-t0)/60:.1f}min)",
                  flush=True)
            results[menu] = p
            with open(args.out, "w") as wf:
                json.dump({"target": "blueprint-search", "iters": args.iters, "menus": results},
                          wf, indent=1)
    print("\n=== DEPLOYED blueprint+search (converged iter10) 6-max LBR exploitability ===", flush=True)
    for menu in menus:
        p = results[menu]
        print(f"  {menu:5s}: {p['bb_per_100']:+.1f} bb/100  [{p['ci_lo']:+.1f}, {p['ci_hi']:+.1f}]  "
              f"({p['hands']}h, {p['k']} seeds)", flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
