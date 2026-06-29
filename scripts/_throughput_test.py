#!/usr/bin/env python3
"""HONEST throughput harness: native CFR+ trainer, single timed train() run.

Usage:
  python scripts/_throughput_test.py VARIANT_LABEL NUM_BUCKETS N_DEALS ITERS OUT_BLOB OUT_JSON

Times the PRODUCTION train() path (releases GIL, no per-phase profiling overhead).
Writes the trained-state blob and a JSON metrics record.

This is a LOCAL relative comparison (AMD Ryzen 5 7600X, 12 logical / 6 physical),
fixed RAYON_NUM_THREADS for fairness. NOT a 128-vCPU cloud measurement.

Warmup policy: NONE. We time the FULL train() from a fresh table for both
variants. The warmup transient is identical for both, so the comparison is fair.
"""
import sys
import os
import time
import json
import hashlib

# Ensure the repo-root battery_native.pyd is importable regardless of the
# script's own directory (python puts the script dir first on sys.path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import battery_native


def main():
    if len(sys.argv) != 7:
        print("usage: VARIANT NUM_BUCKETS N_DEALS ITERS OUT_BLOB OUT_JSON", file=sys.stderr)
        sys.exit(2)

    variant = sys.argv[1]
    num_buckets = int(sys.argv[2])
    n_deals = int(sys.argv[3])
    iters = int(sys.argv[4])
    out_blob = sys.argv[5]
    out_json = sys.argv[6]

    rayon_threads = os.environ.get("RAYON_NUM_THREADS", "<unset>")

    # Fresh table, production config:
    #   NlheCFRTrainer(num_buckets, depth_edges=[], n_deals, n_seats=6,
    #                  prune_threshold=-inf, fast_merge=False, emd_frozen_path=None)
    t = battery_native.NlheCFRTrainer(
        num_buckets, [], n_deals, 6, float("-inf"), False, None
    )

    # Time the whole train() call. train() loops internally over `iters`.
    t0 = time.perf_counter()
    t.train(iters)
    t1 = time.perf_counter()
    wall = t1 - t0

    blob = bytes(t.save_bytes())
    with open(out_blob, "wb") as f:
        f.write(blob)
    blob_sha = hashlib.sha256(blob).hexdigest()

    iters_done = t.iters_done()
    n_infosets = t.n_infosets()

    rec = {
        "variant": variant,
        "num_buckets": num_buckets,
        "n_deals": n_deals,
        "iters": iters,
        "iters_done": iters_done,
        "wall_sec": wall,
        "iters_per_sec": iters / wall if wall > 0 else None,
        "deals_per_sec": (iters * n_deals) / wall if wall > 0 else None,
        "n_infosets": n_infosets,
        "blob_sha256": blob_sha,
        "blob_path": os.path.abspath(out_blob),
        "rayon_threads": rayon_threads,
    }
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=2)

    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
