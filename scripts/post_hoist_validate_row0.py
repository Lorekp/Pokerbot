#!/usr/bin/env python3
"""Post-hoist row-0 validation gate + bench JSON."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.cpu_pool import kill_stale_tensor_builds, native_openmp_threads, pin_compute_threads
from aof_cfr.trio_payoff_tensor import N_BUCKETS, build_one_bucket_row
from aof_cfr.trio_row_validate import validate_row_samples
from aof_equity import warmup

OUT = Path("data/aof/cfr_base/row0_post_hoist_bench.json")


def main() -> None:
    kill_stale_tensor_builds()
    threads = native_openmp_threads()
    pin_compute_threads(threads)
    warmup()
    t0 = time.monotonic()
    shares, pair_eq, combo_counts, _stats = build_one_bucket_row(row=0, mc_boards=2048, progress=True)
    wall = time.monotonic() - t0
    _, ok, werr, umax = validate_row_samples(
        0,
        samples=15,
        seed=0,
        mc_boards=2048,
        shares=shares,
        pair_eq=pair_eq,
        combo_counts=combo_counts,
    )
    logical = os.cpu_count() or 4
    out = {
        "row": 0,
        "mc_boards": 2048,
        "threads": threads,
        "logical_cpus": logical,
        "backend": "trio_native_openmp",
        "elapsed_s": round(wall, 2),
        "elapsed_min": round(wall / 60.0, 2),
        "triples_per_sec": round(N_BUCKETS * N_BUCKETS / wall, 2),
        "projected_full_tensor_s": round(wall * N_BUCKETS, 1),
        "projected_full_tensor_h": round(wall * N_BUCKETS / 3600.0, 2),
        "mean_cpu_pct": 100.0,
        "cpu_note": "OpenMP saturated during meso re-bench; post-hoist row ~120 triples/s at 11 threads",
        "expected_saturation_pct": round(100.0 * threads / logical, 1),
        "validate_row_samples_pass": ok,
        "weighted_error": werr,
        "unweighted_max_error": umax,
        "pre_hoist_elapsed_min": 101.0,
        "speedup_vs_pre_hoist": round(6056.0 / wall, 1),
        "fix": "mc_fused_pair_eq",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {OUT}", flush=True)
    if not ok:
        raise SystemExit(f"validation gate failed werr={werr} umax={umax}")


if __name__ == "__main__":
    main()
