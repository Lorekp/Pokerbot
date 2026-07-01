#!/usr/bin/env python3
"""DOES THE PRODUCTION SEARCH CAP (max_iterations=200) ACTUALLY CONVERGE?

Prod serve uses PluribusReBeLConfig.max_iterations=200 / ~2-2.5s. Our quality tool used
ref_iters=1500. If the subgame CFR needs >>200 iters to converge, LIVE play is UNDER-searched
and weaker than the 1500-iter measurements implied.

METHOD (native exploit path, native=True native_exploit=True dag=True, time_budget_ms=1e12 so
ITERATIONS bind, not time):
  * Solve each of the first ~4-5 captured roots at max_iters in {25,50,100,200,400,800,1500,3000}.
  * Reference = the 3000-iter solve (deepest). Per-root L1 = sum|served_action_probs - ref_probs|.
  * Curve = mean-over-roots L1-to-3000ref at each iter count.
  * (a) by how many iters has the strategy stopped moving? (consecutive-iter mean L1 delta < ~0.02)
  * (b) how far is the prod 200-iter cap from convergence? (its mean L1 to the 3000ref)
  * (c) is prod UNDER-searching?
  * Also TIME the native 1500-iter solve/root -> iters reachable in a 2-2.5s live budget
    (native headroom to raise the cap for free).

The native exploit solve returns BEFORE the starvation blueprint-blend branch, so these are the
PURE CFR-average served probs (no blend contamination) — exactly the raw convergence signal.

Env MUST be set before pokerbot import (encoder bakes flags at import):
  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 \
  POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS=200 python scripts/search_cap_convergence.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from native_exploit_equiv import build_root, load_snapshot, make_opponent_model  # noqa: E402
from search_quality_3s_ab import make_solver, _solve_once  # noqa: E402


def main():
    from pokerbot.search import native_solver as ns
    print("native available:", ns.available(), "subgame:", ns.native_subgame_available(),
          "exploit:", ns.native_subgame_exploit_available(), flush=True)

    bp = os.environ.get("BLUEPRINT_PATH", "checkpoints/serve_blueprint_6max.pkl")
    roots_path = os.environ.get("ROOTS_PATH", "artifacts/search_roots.pkl")
    n_roots = int(os.environ.get("SC_ROOTS", "5"))
    salt = int(os.environ.get("SC_SALT", "12345"))
    prod_cap = int(os.environ.get("SC_PROD_CAP", "200"))
    live_budget_lo = float(os.environ.get("SC_BUDGET_LO_MS", "2000"))
    live_budget_hi = float(os.environ.get("SC_BUDGET_HI_MS", "2500"))

    iter_grid = [25, 50, 100, 200, 400, 800, 1500, 3000]
    ref_iters = iter_grid[-1]

    import pickle
    snap = load_snapshot(bp)
    roots = pickle.load(open(roots_path, "rb"))[:n_roots]
    built = [build_root(r) for r in roots]
    hero = int(roots[0]["hero_seat"])
    model = make_opponent_model(list(range(6)), hero)
    print(f"blueprint={bp} infosets={len(snap):,}  roots={len(built)}  "
          f"iter_grid={iter_grid}  ref={ref_iters}", flush=True)

    # One solver per iter count (native runs all max_iterations in ONE FFI call; time_budget huge so
    # iterations bind). Solve every root at every iter count. Store served action_probs (11-vector).
    # served_probs[k, ri, :] for iter_grid[k], root ri.
    served = np.zeros((len(iter_grid), len(built), 11), dtype=np.float64)
    served_tags = {}

    # Warm once (pays the first frozen-table build) at a small iter count so it doesn't dominate timing.
    warm = make_solver(snap, max_iters=25, time_budget_ms=10**12,
                       native=True, native_exploit=True, dag=True)
    warm.solve(*built[0])

    solve_time_ms = {}  # (iters -> list of per-root ms)
    for k, it in enumerate(iter_grid):
        s = make_solver(snap, max_iters=it, time_budget_ms=10**12,
                        native=True, native_exploit=True, dag=True)
        s.solve(*built[0])  # warm this particular solver's tables
        per_root_ms = []
        for ri, (root, pbs) in enumerate(built):
            t0 = time.perf_counter()
            p, ret_it, tag = _solve_once(s, root, pbs, model, salt)
            ms = (time.perf_counter() - t0) * 1000.0
            per_root_ms.append(ms)
            served[k, ri, :p.shape[0]] = p
            served_tags[tag] = served_tags.get(tag, 0) + 1
        solve_time_ms[it] = per_root_ms
        print(f"  iters={it:>5}: mean {np.mean(per_root_ms):8.1f} ms/root  "
              f"(returned iters={ret_it}, tag={tag})", flush=True)

    print(f"\nserved tags: {served_tags}", flush=True)
    frac_native = served_tags.get("search_native_exploit", 0) / (len(iter_grid) * len(built))
    print(f"native-exploit-served fraction: {frac_native:.3f} "
          f"(must be 1.000 or the convergence signal is off-path)", flush=True)

    # --- Convergence curve: L1 of each iter-count served-probs vs the 3000ref (per root, then mean) ---
    ref = served[-1]  # (n_roots, 11) at ref_iters
    l1_to_ref = np.abs(served - ref[None, :, :]).sum(axis=2)  # (n_grid, n_roots)
    mean_l1 = l1_to_ref.mean(axis=1)  # (n_grid,)
    max_l1 = l1_to_ref.max(axis=1)

    print("\n=== CONVERGENCE CURVE (mean L1 of served action_probs vs the %d-iter reference) ===" % ref_iters)
    print(f"{'iters':>6} {'mean L1->ref':>13} {'max L1->ref':>12}  per-root L1")
    for k, it in enumerate(iter_grid):
        pr = "  ".join(f"{v:.3f}" for v in l1_to_ref[k])
        print(f"{it:>6} {mean_l1[k]:>13.4f} {max_l1[k]:>12.4f}  [{pr}]", flush=True)

    # --- (a) consecutive-iter mean-L1 delta: where does the strategy stop moving? ---
    # Track how the served strategy itself moves between successive iter counts (not just vs ref):
    # delta[k] = mean-over-roots L1( served[k] , served[k-1] ). "stopped moving" when < ~0.02.
    print("\n=== STRATEGY MOTION between successive iter counts (mean L1 vs previous grid point) ===")
    print(f"{'from->to':>14} {'mean dL1':>10} {'stopped(<0.02)':>16}")
    converged_by = None
    for k in range(1, len(iter_grid)):
        step = np.abs(served[k] - served[k - 1]).sum(axis=1)  # per-root
        d = float(step.mean())
        stopped = d < 0.02
        print(f"{iter_grid[k-1]:>6}->{iter_grid[k]:<6} {d:>10.4f} {str(stopped):>16}", flush=True)
        if stopped and converged_by is None:
            converged_by = iter_grid[k]

    # Also: at which grid point does L1-to-ref itself drop below thresholds.
    def first_below(thr):
        for k, it in enumerate(iter_grid):
            if mean_l1[k] < thr:
                return it
        return None

    prod_k = iter_grid.index(prod_cap) if prod_cap in iter_grid else None
    prod_l1 = float(mean_l1[prod_k]) if prod_k is not None else None

    print("\n=== ANSWERS ===")
    print(f"(a) strategy effectively stopped moving (consecutive dL1<0.02) by iters = {converged_by}")
    print(f"    L1-to-ref < 0.05 by iters = {first_below(0.05)};  < 0.02 by iters = {first_below(0.02)}")
    if prod_l1 is not None:
        print(f"(b) prod cap {prod_cap} iters: mean L1-to-{ref_iters}ref = {prod_l1:.4f}  "
              f"(max over roots {float(max_l1[prod_k]):.4f})")
    # under-searching verdict: prod is under-searching if its L1-to-ref is materially > the
    # "stopped moving" floor AND convergence happens well past the cap.
    under = None
    if prod_l1 is not None:
        under = (converged_by is not None and converged_by > prod_cap) or (prod_l1 >= 0.05)
        print(f"(c) prod UNDER-searching? {under}  "
              f"(convergence-by-iters {converged_by} vs cap {prod_cap}; prod L1 {prod_l1:.4f})")

    # --- Timing / native headroom: time a 1500-iter solve/root, derive iters reachable in the budget ---
    ms1500 = np.mean(solve_time_ms[1500])
    per_iter_ms = ms1500 / 1500.0
    reach_lo = live_budget_lo / per_iter_ms
    reach_hi = live_budget_hi / per_iter_ms
    print("\n=== NATIVE TIMING / HEADROOM ===")
    print(f"native 1500-iter solve: mean {ms1500:.1f} ms/root  ->  {per_iter_ms*1000:.3f} us/iter")
    print(f"iters reachable in a {live_budget_lo:.0f}-{live_budget_hi:.0f} ms live budget: "
          f"~{reach_lo:.0f} - {reach_hi:.0f} iters")
    print(f"prod cap = {prod_cap}; convergence-by = {converged_by}; "
          f"native headroom vs cap = {reach_lo/prod_cap:.1f}-{reach_hi/prod_cap:.1f}x "
          f"the prod cap for free")

    # Emit a machine-readable summary for the report.
    import json
    summary = {
        "iter_grid": iter_grid,
        "ref_iters": ref_iters,
        "mean_l1_to_ref": {str(it): float(mean_l1[k]) for k, it in enumerate(iter_grid)},
        "max_l1_to_ref": {str(it): float(max_l1[k]) for k, it in enumerate(iter_grid)},
        "successive_dL1": {f"{iter_grid[k-1]}->{iter_grid[k]}":
                           float(np.abs(served[k] - served[k - 1]).sum(axis=1).mean())
                           for k in range(1, len(iter_grid))},
        "converged_by_iters_consecutive_lt_0p02": converged_by,
        "l1_to_ref_below_0p05_by": first_below(0.05),
        "l1_to_ref_below_0p02_by": first_below(0.02),
        "prod_cap": prod_cap,
        "prod_cap_l1_to_ref": prod_l1,
        "prod_under_searching": under,
        "native_ms_per_root_1500iter": float(ms1500),
        "native_us_per_iter": float(per_iter_ms * 1000),
        "iters_reachable_budget_lo": float(reach_lo),
        "iters_reachable_budget_hi": float(reach_hi),
        "budget_lo_ms": live_budget_lo,
        "budget_hi_ms": live_budget_hi,
        "native_served_fraction": float(frac_native),
        "per_iter_solve_ms_mean": {str(it): float(np.mean(solve_time_ms[it])) for it in iter_grid},
    }
    outp = os.path.join("artifacts", "search_cap_convergence.json")
    with open(outp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {outp}", flush=True)


if __name__ == "__main__":
    main()
