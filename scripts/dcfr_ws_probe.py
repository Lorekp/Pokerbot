"""ONE work_steal-scaling data point: deals/s for a single (work_steal, RAYON_NUM_THREADS) config, measured
at a FIXED warm training state so the node-growth (nodes/deal grows ~8x over training) can't confound the
core-scaling comparison. The on-box loop (dcfr_curve_onbox.sh) invokes this once per (WS, threads) with
RAYON_NUM_THREADS already exported, since rayon fixes its global pool from that env at first parallel call.

Prints ONE json line to stdout: {threads, ws, n_deals, warm_iters, window_iters, wall_s, deals_per_s,
iters_per_s, infosets}. This is the long-owed measurement of work_steal's load-imbalance payoff at 32/64/
128 cores (local 12t was +9-15%; the real win is at scale). Holds CFR+/made-hand fixed; only the scheduler
(static par_iter().fold vs atomic-broadcast work-steal) and thread count vary.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("PYD_DIR", "."))
import battery_native as bn  # noqa: E402

NUM_BUCKETS = int(os.environ.get("DCFR_NUM_BUCKETS", "200"))
N_DEALS = int(os.environ.get("PROBE_N_DEALS", "1024"))
WS = os.environ.get("WS", "0") == "1"
WARM_ITERS = int(os.environ.get("WARM_ITERS", "2000"))      # fixed warm state -> same node-growth for all
WINDOW_ITERS = int(os.environ.get("WINDOW_ITERS", "1500"))  # timed window (fixed iters, not secs)
THREADS = os.environ.get("RAYON_NUM_THREADS", "(default)")
HOG_CAP = int(os.environ.get("DCFR_HOG_CAP", str(1 << 21)))


def main():
    # CFR+ arm (dcfr=False), work_steal toggled per WS; everything else held at the train-arm config.
    t = bn.NlheCFRTrainer(NUM_BUCKETS, [], N_DEALS, 6, float("-inf"), False, None, 0,
                          False, HOG_CAP, WS, False, -300000.0, 0.05)
    # warm to a fixed iter count so deals/s is measured at the SAME training state across configs
    t.train(WARM_ITERS)
    t0 = time.time()
    t.train(WINDOW_ITERS)
    wall = time.time() - t0
    deals = WINDOW_ITERS * N_DEALS
    rec = {"threads": THREADS, "ws": int(WS), "n_deals": N_DEALS, "warm_iters": WARM_ITERS,
           "window_iters": WINDOW_ITERS, "wall_s": round(wall, 3),
           "deals_per_s": round(deals / wall, 1) if wall else 0.0,
           "iters_per_s": round(WINDOW_ITERS / wall, 3) if wall else 0.0,
           "infosets": t.n_infosets()}
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
