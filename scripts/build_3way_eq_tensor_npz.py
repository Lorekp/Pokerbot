#!/usr/bin/env python3
"""One-shot build of 3-max bucket showdown tensor (combo-exact + MC boards)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.cpu_pool import (
    assert_no_oversubscription,
    default_pool_workers,
    describe_parallel_layout,
    kill_stale_tensor_builds,
    native_openmp_threads,
    native_row_pool_workers,
    omp_threads_per_row_worker,
    pin_compute_threads,
    pool_layout,
)
from aof_cfr.run_logger import init_run_log, log_step
from aof_cfr.trio_payoff_tensor import (
    DEFAULT_BOARDS_PER_COMBO,
    DEFAULT_TENSOR,
    DEFAULT_TENSOR_CHECKPOINT_DIR,
    N_BUCKETS,
    _use_native_trio,
    benchmark_one_bucket_row,
    build_one_bucket_row,
    build_trio_equity_tensor,
    require_native_trio_build,
    save_trio_row_slice,
    save_trio_tensor,
    tensor_build_backend_name,
)
from aof_cfr.trio_row_validate import validate_row_samples
from aof_equity import warmup


def main() -> None:
    p = argparse.ArgumentParser(description="Build 3-way bucket equity tensor")
    p.add_argument("--out", default=str(DEFAULT_TENSOR))
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"process pool size (default: {default_pool_workers()} physical-core heuristic)",
    )
    p.add_argument("--chunk-rows", type=int, default=1)
    p.add_argument(
        "--mc-boards",
        type=int,
        default=DEFAULT_BOARDS_PER_COMBO,
        help=f"boards per combo for share MC (default {DEFAULT_BOARDS_PER_COMBO})",
    )
    p.add_argument(
        "--row-workers",
        type=int,
        default=None,
        help="row-parallel process count for full build (default: physical-core heuristic)",
    )
    p.add_argument(
        "--row-parallel",
        action="store_true",
        help="Build rows in parallel processes (default: serial rows, full OpenMP per row — better on single socket)",
    )
    p.add_argument(
        "--no-row-parallel",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--benchmark-row", type=int, default=None)
    p.add_argument(
        "--only-row",
        type=int,
        default=None,
        help="Build and save one bucket row slice (e.g. 0) instead of full tensor",
    )
    p.add_argument(
        "--validate-samples",
        type=int,
        default=0,
        help="After --only-row, spot-check N triples vs exact oracle",
    )
    p.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="Pin OpenMP/BLAS threads (serial build: one process, this many threads)",
    )
    p.add_argument(
        "--kill-stale",
        action="store_true",
        help="Kill conflicting tensor/gate builds before starting",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_TENSOR_CHECKPOINT_DIR),
        help="Per-row checkpoint directory (resume on restart)",
    )
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable per-row checkpoint writes",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing row checkpoints",
    )
    p.add_argument(
        "--no-seed-gate-slices",
        action="store_true",
        help="Do not copy gate_row_slices into checkpoint dir",
    )
    args = p.parse_args()

    log_path = init_run_log("build_3way_eq_tensor")

    if args.kill_stale or args.only_row is not None or args.benchmark_row is not None:
        kill_stale_tensor_builds(include_pair_eq=False)

    warmup()
    require_native_trio_build(context="build_3way_eq_tensor_npz")
    native = True
    row_par = bool(args.row_parallel and not args.no_row_parallel)
    row_workers = args.row_workers
    if row_par and row_workers is None:
        row_workers = native_row_pool_workers()
    if row_par:
        omp_per = (
            args.omp_threads
            if args.omp_threads is not None
            else omp_threads_per_row_worker(row_workers)
        )
        assert_no_oversubscription(row_workers, omp_per, label="full tensor")
    else:
        row_workers = 1
        omp_per = args.omp_threads if args.omp_threads is not None else native_openmp_threads()
        assert_no_oversubscription(1, omp_per, label="full tensor")
    pin_compute_threads(omp_per)
    par_info = describe_parallel_layout(row_workers if row_par else 1, omp_per)
    layout = pool_layout(
        row_workers if row_par else 1,
        native_trio=True,
        openmp_threads=omp_per,
        row_parallel=row_par,
    )
    log_step(
        f"Tensor build start: boards_per_combo={args.mc_boards} "
        f"backend={tensor_build_backend_name()} layout={layout} "
        f"parallel={par_info} row_parallel={row_par} log={log_path}"
    )
    if args.benchmark_row is not None:
        stats = benchmark_one_bucket_row(
            row=args.benchmark_row,
            mc_boards=args.mc_boards,
            workers=args.workers,
        )
        print(json.dumps(stats, indent=2))
        return

    if args.only_row is not None:
        shares, pair_eq, combo_counts, stats = build_one_bucket_row(
            row=args.only_row,
            mc_boards=args.mc_boards,
            workers=args.workers,
        )
        out = Path(args.out)
        save_trio_row_slice(
            out,
            args.only_row,
            shares,
            pair_eq,
            combo_counts,
            mc_boards=args.mc_boards,
        )
        print(
            f"Wrote row slice {out} btn_bucket={args.only_row} stats={stats}",
            flush=True,
        )
        if args.validate_samples > 0:
            print(f"Row validation: checking {args.validate_samples} triples ...", flush=True)
            v_started = time.monotonic()
            checks, ok, werr, umax = validate_row_samples(
                args.only_row,
                samples=args.validate_samples,
                shares=shares,
                pair_eq=pair_eq,
                combo_counts=combo_counts,
            )
            print(
                f"Row validation: weighted_error={werr:.6f} unweighted_max={umax:.6f} "
                f"samples={args.validate_samples} elapsed={time.monotonic() - v_started:.1f}s "
                f"-> {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
            if not ok:
                raise SystemExit(1)
        return

    layout = pool_layout(
        row_workers if row_par else 1,
        native_trio=True,
        openmp_threads=omp_per,
        row_parallel=row_par,
    )
    log_step(f"Full tensor pool layout: {layout} parallel={par_info}")
    ckpt = None if args.no_checkpoint else Path(args.checkpoint_dir)
    t0 = time.monotonic()
    shares, pair_eq, deal_weights, combo_counts = build_trio_equity_tensor(
        workers=row_workers if row_par else 1,
        chunk_rows=args.chunk_rows,
        mc_boards=args.mc_boards,
        progress=True,
        row_parallel=row_par,
        openmp_threads=omp_per,
        checkpoint_dir=ckpt,
        resume=not args.no_resume,
        seed_gate_slices=not args.no_seed_gate_slices,
    )
    wall = time.monotonic() - t0
    out = Path(args.out)
    save_trio_tensor(
        out,
        shares,
        pair_eq,
        deal_weights,
        combo_counts,
        mc_boards=args.mc_boards,
        row_workers=row_workers if row_par else 1,
        row_parallel=row_par,
    )
    valid = int((combo_counts > 0).sum())
    log_step(
        f"Wrote {out} shares={shares.shape} valid_triples={valid}/{combo_counts.size} "
        f"wall_s={wall:.1f} wall_min={wall/60:.1f} boards_per_combo={args.mc_boards} "
        f"row_workers={row_workers}"
    )


if __name__ == "__main__":
    main()
