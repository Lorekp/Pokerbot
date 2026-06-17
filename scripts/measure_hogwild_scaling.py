#!/usr/bin/env python3
"""H5.3 — hogwild NLHE scaling-curve harness.

Measures THROUGHPUT (deals/s, traversals/s) and PARALLEL EFFICIENCY vs worker count for the
shared-memory lock-free trainer. Run on the 64-vCPU box (or locally with a small budget) to produce
the honest {8,16,32,60} curve. Research expectation: ~linear to ~16 workers, then 50-70% efficiency at
32-64 as L3 bandwidth saturates (Graviton3 is single-NUMA, so there is no remote-NUMA penalty) — the
sweet spot may be ~40-48. We REPORT the real curve; we do not assume linearity.

Each worker runs ``--per-worker`` deals (held fixed), so ideal-linear throughput grows ∝ workers; the
efficiency column = throughput(W)/W normalized to the smallest W (1.00 = perfectly linear from there).

Usage (box):  python scripts/measure_hogwild_scaling.py --per-worker 200 --workers 8 16 32 60
Usage (local smoke): python scripts/measure_hogwild_scaling.py --per-worker 6 --workers 1 2 4
"""

from __future__ import annotations

import argparse
import json
import os
import time

# Bounded (deployment-aligned) fingerprint — must be set before the encoder is imported.
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-worker", type=int, default=200, help="deals per worker (held fixed across W)")
    p.add_argument("--workers", type=int, nargs="+", default=[8, 16, 32, 60])
    p.add_argument("--capacity", type=int, default=1 << 22)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-client-bet-sizing", action="store_true",
                   help="run full-action (≈10x slower) instead of the abstracted (deployment) sizing")
    p.add_argument("--out", default=None, help="optional JSON path for the curve")
    args = p.parse_args()

    from pokerbot.engine.jax_engine import NUM_PLAYERS  # noqa: F401  (ensures engine import works)
    from pokerbot.training.sd_cfr_nlhe_hogwild import _DEFAULT_OCCUPANCY, run_nlhe_hogwild

    n_seats = sum(_DEFAULT_OCCUPANCY)
    client = not args.no_client_bet_sizing
    print(f"Hogwild NLHE scaling curve — per-worker={args.per_worker} deals, seats/deal={n_seats}, "
          f"client_bet_sizing={client}, capacity={args.capacity}", flush=True)
    print(f"{'workers':>7} {'wall_s':>9} {'deals/s':>10} {'trav/s':>10} {'efficiency':>11} "
          f"{'infosets':>9} {'load':>6}", flush=True)

    rows = []
    base_tps_per_worker = None
    for w in sorted(args.workers):
        t0 = time.perf_counter()
        res = run_nlhe_hogwild(
            traversals=args.per_worker, num_workers=w, capacity=args.capacity,
            client_bet_sizing=client, seed=args.seed,
        )
        wall = time.perf_counter() - t0
        deals = w * args.per_worker
        deals_s = deals / wall
        trav_s = deals_s * n_seats
        tps_per_worker = deals_s / w
        if base_tps_per_worker is None:
            base_tps_per_worker = tps_per_worker
        eff = tps_per_worker / base_tps_per_worker if base_tps_per_worker else 0.0
        rows.append({
            "workers": w, "wall_s": wall, "deals_per_s": deals_s, "traversals_per_s": trav_s,
            "efficiency": eff, "n_infosets": res["n_infosets"], "load_factor": res["load_factor"],
        })
        print(f"{w:>7} {wall:>9.1f} {deals_s:>10.2f} {trav_s:>10.1f} {eff:>10.0%}  "
              f"{res['n_infosets']:>9} {res['load_factor']:>6.3f}", flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"per_worker": args.per_worker, "n_seats": n_seats,
                       "client_bet_sizing": client, "curve": rows}, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
