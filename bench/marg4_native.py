"""Benchmark: pure-Python vs native (Rust/rayon) 4-max marginal payoff precompute.

Measures single-core deals/sec for ``fp_4max_vec.precompute_payoffs`` (forced Python loop)
vs ``marg4_native.precompute_payoffs_native``, reports the speedup, shows rayon thread
scaling on one batch, and projects the full-job wall-clock (25M deals x 7 depths on N
worker cores) for both backends.

    python bench/marg4_native.py [--py-deals 800] [--nat-deals 40000] [--depth 8.0] [--cores 62]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from aof_cfr.exploit_mc import _terminal_returns
from aof_cfr.fp_4max_vec import _OUTCOMES, sample_deals
from aof_cfr.marg4_native import native_available, precompute_payoffs_native
from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default
from aof_ranges import active_seats


def _python_precompute(holes, depth, gm):
    seats = active_seats(4)
    n = len(holes)
    pay = np.zeros((len(_OUTCOMES), n, 4), dtype=np.float64)
    for oi, o in enumerate(_OUTCOMES):
        folded = set(o["folded"])
        for d in range(n):
            pay[oi, d, :] = _terminal_returns(4, seats, holes[d], folded, depth, game_model=gm)
    return pay


def _time(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return time.perf_counter() - t0, out


def fmt_hms(secs: float) -> str:
    secs = int(round(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-deals", type=int, default=200)
    ap.add_argument("--nat-deals", type=int, default=3000)   # native is ~21ms/deal; keep small
    ap.add_argument("--depth", type=float, default=8.0)
    ap.add_argument("--cores", type=int, default=62)
    ap.add_argument("--total-deals", type=int, default=25_000_000)
    ap.add_argument("--n-depths", type=int, default=7)
    ap.add_argument("--threads-scan", action="store_true",
                    help="run the rayon thread-scaling sweep (spawns cpu-count threads — do NOT run "
                         "alongside other native processes or it oversubscribes all cores)")
    args = ap.parse_args()
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)  # flush progress so a long run isn't a black box

    if not native_available():
        print("native extension NOT available — nothing to benchmark.")
        return

    gm = game_model_default()
    fee = flat_fee_default()
    depth = args.depth
    print(f"backend=native  game_model={gm}  flat_fee={fee}  depth={depth}")

    # --- single-core Python baseline ---
    _, py_holes = sample_deals(args.py_deals, 4242)
    py_t, py_out = _time(_python_precompute, py_holes, depth, gm)
    py_dps = args.py_deals / py_t
    py_ms = py_t / args.py_deals * 1000.0
    print(f"\nPython (single-core, {args.py_deals} deals): {py_t:.2f}s  "
          f"-> {py_dps:.1f} deals/s  ({py_ms:.2f} ms/deal)")

    # --- single-core native ---
    _, nat_holes = sample_deals(args.nat_deals, 4242)
    # warm + correctness check on the shared py batch
    nat_ref = precompute_payoffs_native(py_holes, depth, game_model=gm, fee_bb=fee, threads=1)
    assert np.array_equal(py_out, nat_ref), "BIT-IDENTITY BROKEN — aborting bench"
    print("bit-identity vs Python: OK (np.array_equal)")
    nat_t1, _ = _time(precompute_payoffs_native, nat_holes, depth,
                      game_model=gm, fee_bb=fee, threads=1)
    nat_dps1 = args.nat_deals / nat_t1
    nat_ms1 = nat_t1 / args.nat_deals * 1000.0
    print(f"native (single-core, {args.nat_deals} deals): {nat_t1:.3f}s  "
          f"-> {nat_dps1:.0f} deals/s  ({nat_ms1:.4f} ms/deal)")

    speedup = nat_dps1 / py_dps
    print(f"\nSINGLE-CORE SPEEDUP (native / python): {speedup:.1f}x")

    # --- rayon thread scaling on one big batch (opt-in: oversubscribes all cores) ---
    if not args.threads_scan:
        print("\n(rayon thread-scaling sweep skipped; pass --threads-scan to run it)")
        _project(args, py_dps, nat_dps1)
        return
    print("\nrayon thread scaling (same batch, one process):")
    cpu = os.cpu_count() or 4
    thread_opts = sorted({1, 2, 4, 8, cpu})
    base_dps = None
    for th in thread_opts:
        t, _ = _time(precompute_payoffs_native, nat_holes, depth,
                     game_model=gm, fee_bb=fee, threads=th)
        dps = args.nat_deals / t
        if base_dps is None:
            base_dps = dps
        print(f"  threads={th:>3}: {t:.3f}s  -> {dps:>10.0f} deals/s  ({dps/base_dps:.2f}x vs 1)")
    _project(args, py_dps, nat_dps1)


def _project(args, py_dps: float, nat_dps1: float) -> None:
    """Full-job wall-clock projection (process-pool: one core per worker, native serial per core)."""
    total = args.total_deals * args.n_depths
    print(f"\nFULL-JOB PROJECTION ({args.total_deals:,} deals x {args.n_depths} depths "
          f"= {total:,} deals, {args.cores} worker cores, 1 core/worker):")
    py_wall = total / (py_dps * args.cores)
    nat_wall = total / (nat_dps1 * args.cores)
    print(f"  python baseline : {fmt_hms(py_wall)}  ({py_wall/3600:.1f} h)")
    print(f"  native kernel   : {fmt_hms(nat_wall)}  ({nat_wall/3600:.2f} h)")
    print(f"  wall-clock speedup: {py_wall/nat_wall:.2f}x  (>1 = native faster)")


if __name__ == "__main__":
    main()
