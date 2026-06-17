#!/usr/bin/env python3
"""Meso benchmark for native trio row builder (ETA projection)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.cpu_pool import native_openmp_threads, pin_compute_threads
from aof_cfr.trio_payoff_tensor import (
    DEFAULT_TRIPLE_SAMPLE_BUDGET,
    N_BUCKETS,
    _native_build_row_columns,
    tensor_build_backend_name,
)
from aof_equity import warmup


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark native trio row builder")
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--mc-boards", type=int, default=DEFAULT_TRIPLE_SAMPLE_BUDGET)
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args()

    from tools.ompeval_cli import native as omp_native

    if not omp_native.native_trio_available():
        raise SystemExit("native trio backend not available; build tools/ompeval_cli first")

    warmup()
    threads = args.threads if args.threads is not None else native_openmp_threads()
    pin_compute_threads(threads)
    print(
        f"meso_trio_native: row={args.row} mc_boards={args.mc_boards} "
        f"threads={threads} backend={tensor_build_backend_name()}",
        flush=True,
    )
    n = N_BUCKETS
    shares = np.zeros((n, n, 3), dtype=np.float64)
    pair_eq = np.zeros((n, n, 3), dtype=np.float64)
    combo_counts = np.zeros((n, n), dtype=np.int32)

    t0 = time.monotonic()
    _native_build_row_columns(
        args.row,
        shares,
        pair_eq,
        combo_counts,
        mc_boards=args.mc_boards,
        n=n,
        omp_threads=threads,
        progress=True,
        started=t0,
    )
    elapsed = time.monotonic() - t0
    triples = n * n
    projected_full_s = elapsed * N_BUCKETS
    stats = {
        "row": args.row,
        "mc_boards": args.mc_boards,
        "threads": threads,
        "backend": tensor_build_backend_name(),
        "triples": triples,
        "elapsed_s": round(elapsed, 2),
        "triples_per_sec": round(triples / elapsed, 2) if elapsed > 0 else 0.0,
        "projected_full_tensor_s": round(projected_full_s, 1),
        "projected_full_tensor_h": round(projected_full_s / 3600.0, 2),
    }
    out_path = Path("data/aof/cfr_base/phase0_meso_bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
