"""Benchmark: pure-Python vs batched-native 4-max marginal payoff precompute.

The Python ``precompute_payoffs`` calls the native ``mc_nway_shares`` once per (deal, alive-subset)
showdown (~11 crossings/deal), single-threaded. The batched path
(``fp_4max_vec._precompute_payoffs_batched``) collects every distinct showdown for the chunk and
runs ONE OpenMP-parallel native call (``mc_nway_shares_batch``), reusing the same OMPEval evaluator
+ Fisher-Yates sampler + _N_BOARDS, so its output is bit-identical to the Python loop.

Reports single-core deals/sec (Python vs batched threads=1) and projects the full-job wall-clock
(25M deals x 7 depths on N ProcessPool workers, 1 core/worker). An all-core thread sweep is opt-in
(``--threads-scan``); it oversubscribes all cores, so do NOT run it alongside other native jobs.

    python bench/showdown_batch.py [--py-deals 200] [--bat-deals 4000] [--depth 8.0] [--cores 62]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import aof_cfr.fp_4max_vec as fp
from aof_cfr.fp_4max_vec import sample_deals
from aof_cfr.pushfold_payoffs import game_model_default


def _time(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return time.perf_counter() - t0, out


def fmt_hms(secs: float) -> str:
    secs = int(round(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _python_precompute(holes, depth, gm):
    fp._DISABLE_BATCH = True
    fp._SHARE_CACHE.clear()
    try:
        return fp.precompute_payoffs(holes, depth, game_model=gm)
    finally:
        fp._DISABLE_BATCH = False


def _batched_precompute(holes, depth, gm, threads):
    fp._DISABLE_BATCH = False
    os.environ["AOF_MARG4_BATCH_THREADS"] = str(threads)
    fp._SHARE_CACHE.clear()
    return fp.precompute_payoffs(holes, depth, game_model=gm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-deals", type=int, default=200)
    ap.add_argument("--bat-deals", type=int, default=4000)
    ap.add_argument("--depth", type=float, default=8.0)
    ap.add_argument("--cores", type=int, default=62)
    ap.add_argument("--total-deals", type=int, default=25_000_000)
    ap.add_argument("--n-depths", type=int, default=7)
    ap.add_argument("--threads-scan", action="store_true",
                    help="run the all-core OpenMP sweep (spawns cpu-count threads — do NOT run "
                         "alongside other native processes or it oversubscribes all cores)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    from tools.ompeval_cli.native import mc_nway_shares_batch_available  # noqa: E402
    if not mc_nway_shares_batch_available():
        print("mc_nway_shares_batch NOT available — build tools/ompeval_cli (build.ps1). Nothing to bench.")
        return

    gm = game_model_default()
    depth = args.depth
    print(f"game_model={gm}  depth={depth}  _N_BOARDS={fp._N_BOARDS}")

    # --- single-core Python baseline ---
    _, py_holes = sample_deals(args.py_deals, 4242)
    py_t, py_out = _time(_python_precompute, py_holes, depth, gm)
    py_dps = args.py_deals / py_t
    print(f"\nPython (single-core, {args.py_deals} deals): {py_t:.2f}s  -> {py_dps:.1f} deals/s "
          f"({py_t / args.py_deals * 1000:.2f} ms/deal)")

    # --- statistical sanity on the shared py batch (different RNG stream -> per-showdown MC noise,
    #     NOT bit-identical; the load-bearing check is the marginal-level test, see
    #     tests/test_showdown_batch.py). Mean payoff over deals must agree to ~MC stderr. ---
    bat_ref = _batched_precompute(py_holes, depth, gm, 1)
    mean_abs = float(np.mean(np.abs(py_out.mean(axis=1) - bat_ref.mean(axis=1))))
    print(f"mean |E_py - E_batched| over {args.py_deals} deals: {mean_abs:.4f} bb "
          f"(per-showdown MC noise; marginal validation in tests/test_showdown_batch.py)")

    # --- single-core batched (threads=1, process-pool-safe) ---
    _, bat_holes = sample_deals(args.bat_deals, 4242)
    bat_t1, _ = _time(_batched_precompute, bat_holes, depth, gm, 1)
    bat_dps1 = args.bat_deals / bat_t1
    print(f"batched (threads=1, {args.bat_deals} deals): {bat_t1:.3f}s -> {bat_dps1:.0f} deals/s "
          f"({bat_t1 / args.bat_deals * 1000:.4f} ms/deal)")
    print(f"\nSINGLE-CORE SPEEDUP (batched / python): {bat_dps1 / py_dps:.1f}x")

    if args.threads_scan:
        print("\nOpenMP thread scaling (same batch, one process):")
        cpu = os.cpu_count() or 4
        base = None
        for th in sorted({1, 2, 4, 8, cpu}):
            t, _ = _time(_batched_precompute, bat_holes, depth, gm, th)
            dps = args.bat_deals / t
            base = base or dps
            print(f"  threads={th:>3}: {t:.3f}s -> {dps:>10.0f} deals/s ({dps / base:.2f}x vs 1)")
    else:
        print("\n(OpenMP thread-scaling sweep skipped; pass --threads-scan to run it)")

    total = args.total_deals * args.n_depths
    print(f"\nFULL-JOB PROJECTION ({args.total_deals:,} deals x {args.n_depths} depths = {total:,} "
          f"deals, {args.cores} workers, 1 core/worker, threads=1 per worker):")
    py_wall = total / (py_dps * args.cores)
    bat_wall = total / (bat_dps1 * args.cores)
    print(f"  python baseline : {fmt_hms(py_wall)}  ({py_wall / 3600:.2f} h)")
    print(f"  batched kernel  : {fmt_hms(bat_wall)}  ({bat_wall / 3600:.2f} h)")
    print(f"  wall-clock speedup: {py_wall / bat_wall:.2f}x")


if __name__ == "__main__":
    main()
