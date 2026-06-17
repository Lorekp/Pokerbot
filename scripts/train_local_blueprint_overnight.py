#!/usr/bin/env python3
"""Overnight LOCAL 6-max blueprint training — fix the postflop-coverage hole.

Why: the cloud-trained snapshot (176k infosets) saw only ~4k unique deals in 9h (pathological
throughput, cause uninvestigated) and has ~0% POSTFLOP serve-time coverage — the solver-internal
exploitation path (bias the modeled opponent, CFR best-responds) had no postflop prior to deviate
from, which is what pushed us onto the hand-calibrated root-mixer bridge. Local hogwild throughput
is healthy (~105k infosets in 20s), so a multi-hour local run dwarfs what the broken cloud run
actually accomplished.

Design: timed rounds of run_nlhe_hogwild with H5 cross-round CONTINUATION (init_strat/init_regret
from the previous round — correct CFR accumulation, NOT fresh-round summing), CFR+ (fewest moving
parts unattended), client_bet_sizing=True (the SAME action abstraction as serving — blueprint keys
must hit). After each round the accumulated strat is pickled (crash-safe: a kill loses at most one
round) in the {"strat_sum": {fingerprint -> row}} shape the battery/adapter already consume.

Usage: POKERBOT_BETTING_ABSTRACTION=1 python scripts/train_local_blueprint_overnight.py \
          [--hours 10] [--workers 8] [--traversals 4000] [--out path.pkl]
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
# CRITICAL key-compat: the SERVE side (eval/lbr.py, imported by the adapter) fingerprints postflop
# states with the 200-bucket made-hand abstraction (it sets these env defaults at import). The
# trainer must fingerprint IDENTICALLY or every postflop row is unreachable at serve time —
# measured: a run without these had 100% preflop / 0.00% postflop serve-time hit rate.
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.0, help="wall-clock training budget")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--traversals", type=int, default=4000,
                    help="deals per worker per round (each deal traversed once per live seat)")
    ap.add_argument("--capacity", type=int, default=1 << 22,
                    help="shared-table slots (4.2M ~ 0.8GB; the cloud snapshot used 176k infosets)")
    ap.add_argument("--out", default=r"checkpoints/bp6max_local_v1.pkl")
    ap.add_argument("--resume", action="store_true",
                    help="continue from --out if it exists (strat always; regret too when the "
                         "snapshot has it — older snapshots lacked regret, so the first resumed "
                         "round restarts regrets while the strategy average keeps accumulating)")
    args = ap.parse_args()

    from pokerbot.training.sd_cfr_nlhe_hogwild import run_nlhe_hogwild

    deadline = time.time() + args.hours * 3600.0
    strat = None
    regret = None
    rnd = 0
    if args.resume and os.path.exists(args.out):
        with open(args.out, "rb") as f:
            snap = pickle.load(f)
        strat = snap.get("strat_sum")
        regret = snap.get("regret_sum")  # None in pre-resume-format snapshots
        rnd = int(snap.get("round", 0))
        print(f"RESUME from {args.out}: round {rnd}, {len(strat or {}):,} infosets, "
              f"regret={'yes' if regret else 'NO (strategy-only continuation)'}", flush=True)
    print(f"local blueprint training: workers={args.workers} traversals/round={args.traversals} "
          f"capacity={args.capacity} budget={args.hours}h -> {args.out}", flush=True)
    while time.time() < deadline:
        rnd += 1
        t0 = time.time()
        res = run_nlhe_hogwild(
            traversals=args.traversals,
            num_workers=args.workers,
            capacity=args.capacity,
            occupancy=(True,) * 6,
            client_bet_sizing=True,
            cfr_plus=True,
            seed=1000 + rnd,             # fresh deal stream every round; regret CONTINUES via init_*
            init_strat=strat,
            init_regret=regret,
        )
        strat, regret = res["strat"], res["regret"]
        dt = time.time() - t0
        # Crash-safe snapshot: write to a temp file then replace, so a kill never corrupts the pkl.
        tmp = args.out + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"strat_sum": strat, "regret_sum": regret, "round": rnd,
                         "n_infosets": res["n_infosets"]}, f)
        os.replace(tmp, args.out)
        remaining = (deadline - time.time()) / 3600.0
        print(f"[round {rnd:3d}] {dt:6.1f}s  infosets={res['n_infosets']:>9,}  "
              f"load={res['load_factor']:.3f}  remaining={remaining:.2f}h", flush=True)
        if res["load_factor"] > 0.85:
            print("table load > 0.85 — stopping before hash-table degradation; raise --capacity "
                  "for longer runs", flush=True)
            break
    print(f"DONE: {rnd} rounds, snapshot at {args.out}", flush=True)


if __name__ == "__main__":
    main()
