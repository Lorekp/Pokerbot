"""Clean single-threaded wall-clock bench: ms/solve & solves/sec at iters 20/50/100 for both leaves.
Uses a subset count from argv to keep iters=100 tractable. No cProfile overhead."""
from __future__ import annotations
import os, pickle, sys, time
import numpy as np
from scripts.bench_search_run import load_snapshot, build_root, make_solver  # reuse

def run(roots, snap, iters, leaf_select, rollout):
    solver = make_solver(snap, iters, leaf_select, rollout)
    r0, p0 = build_root(roots[0]); solver.solve(r0, p0)  # warm
    t0 = time.perf_counter(); n = 0; nodes = 0.0
    for rec in roots:
        root, pbs = build_root(rec)
        res = solver.solve(root, pbs)
        nodes += res.metadata.get("leaf_eval_count", 0.0) + res.metadata.get("leaf_cache_hits", 0.0)
        n += 1
    dt = time.perf_counter() - t0
    return dt / n * 1000.0, n / dt, nodes / n

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    leaf = sys.argv[2] if len(sys.argv) > 2 else "uniform"
    rollout = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    roots = pickle.load(open("artifacts/search_roots.pkl", "rb"))[:n]
    snap = load_snapshot("checkpoints/serve_blueprint_6max.pkl")
    print(f"clean bench: leaf={leaf} rollout_samples={rollout} n_roots={len(roots)}", flush=True)
    print(f"{'iters':>6} {'ms/solve':>10} {'solves/s':>9} {'nodes/solve':>12}", flush=True)
    for iters in (20, 50, 100):
        ms, sps, nodes = run(roots, snap, iters, leaf, rollout)
        print(f"{iters:>6} {ms:>10.2f} {sps:>9.3f} {nodes:>12.1f}", flush=True)

if __name__ == "__main__":
    main()
