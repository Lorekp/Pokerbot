#!/usr/bin/env python3
"""SPEED x ITERS x CONVERGENCE matrix for picking the live search budget (native exploit path).

The prod cap (max_iterations=200, ~0.5s native) is UNDER-SEARCHED. This maps, across an iteration grid
spanning BELOW / AT / OVER the 2-2.5s prod window, what each budget buys:
  * wall_ms          -- native ms per DECISION (one solve) at that iter count (the cost)
  * l1_to_ref        -- L1 of the SALT-AVERAGED served policy vs a deep salt-avg reference; this is the
                        running-average CFR convergence (the served policy IS the running average). Lower
                        = closer to the converged strategy.
  * argmax_match     -- fraction of roots whose salt-avg served ARGMAX matches the deep ref's argmax
                        (does it pick the RIGHT action). This is the play-relevant signal.
  * top_mass         -- mean salt-avg mass on the top action (decisiveness; low = diffuse/under-converged)

Single-stream MC jitter (~0.5 L1) is irreducible and FLAT across iters (no unique 6-max equilibrium), so
we report the salt-AVERAGED convergence -- which does shrink -- not the single-stream jitter.

  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 \
  POKERBOT_POSTFLOP_NUM_BUCKETS=200 python scripts/search_budget_matrix.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from native_exploit_equiv import build_root, load_snapshot, make_opponent_model  # noqa: E402
from search_quality_3s_ab import _solve_once, make_solver  # noqa: E402


def main():
    bp = os.environ.get("BLUEPRINT_PATH", "checkpoints/serve_blueprint_6max.pkl")
    roots_path = os.environ.get("ROOTS_PATH", "artifacts/search_roots.pkl")
    n_roots = int(os.environ.get("BM_ROOTS", "5"))
    ref_iters = int(os.environ.get("BM_REF_ITERS", "6000"))
    salts = [12345, 23456, 34567, 45678][: int(os.environ.get("BM_SALTS", "4"))]
    grid = [int(x) for x in os.environ.get("BM_GRID", "200,1000,1500,2000,2500,3000,4000").split(",")]

    import pickle
    snap = load_snapshot(bp)
    roots = pickle.load(open(roots_path, "rb"))[:n_roots]
    built = [build_root(r) for r in roots]
    model = make_opponent_model(list(range(6)), int(roots[0]["hero_seat"]))
    print(f"blueprint={bp} infosets={len(snap):,} roots={len(built)} ref_iters={ref_iters} "
          f"salts={len(salts)} grid={grid}", flush=True)

    def served_avg(iters):
        """mean-over-salts served probs, per root -> (n_roots, 11)."""
        s = make_solver(snap, max_iters=iters, time_budget_ms=10**12,
                        native=True, native_exploit=True, dag=True)
        s.solve(*built[0])  # warm
        acc = np.zeros((len(built), 11))
        for si in salts:
            for ri, (root, pbs) in enumerate(built):
                p, _, _ = _solve_once(s, root, pbs, model, si)
                acc[ri, : p.shape[0]] += p
        return acc / len(salts)

    print(f"\n[ref] deep salt-avg reference @ {ref_iters} iters ...", flush=True)
    ref = served_avg(ref_iters)
    ref_argmax = ref.argmax(axis=1)

    rows = []
    for iters in grid:
        # timing: one solve (1 salt), mean over roots
        st = make_solver(snap, max_iters=iters, time_budget_ms=10**12,
                         native=True, native_exploit=True, dag=True)
        st.solve(*built[0])
        tms = []
        for root, pbs in built:
            t0 = time.perf_counter(); _solve_once(st, root, pbs, model, salts[0])
            tms.append((time.perf_counter() - t0) * 1000.0)
        wall_ms = float(np.mean(tms))
        # convergence: salt-avg served vs deep ref
        sa = served_avg(iters)
        l1 = float(np.abs(sa - ref).sum(axis=1).mean())
        amatch = float((sa.argmax(axis=1) == ref_argmax).mean())
        top_mass = float(sa.max(axis=1).mean())
        rows.append(dict(iters=iters, wall_ms=wall_ms, l1_to_ref=l1,
                         argmax_match=amatch, top_mass=top_mass))
        print(f"  iters={iters:>5}  wall={wall_ms:7.0f}ms  L1->ref={l1:.3f}  "
              f"argmax_match={amatch*100:3.0f}%  top_mass={top_mass:.2f}", flush=True)

    print("\n=== SEARCH BUDGET MATRIX (native exploit path, 160k blueprint) ===")
    print(f"{'iters':>6} {'wall ms/dec':>12} {'in 2-2.5s?':>11} {'L1->conv':>9} {'argmax OK':>10} {'top-mass':>9}")
    for r in rows:
        inb = "yes" if r["wall_ms"] <= 2500 else ("edge" if r["wall_ms"] <= 3000 else "OVER")
        print(f"{r['iters']:>6} {r['wall_ms']:>12.0f} {inb:>11} {r['l1_to_ref']:>9.3f} "
              f"{r['argmax_match']*100:>9.0f}% {r['top_mass']:>9.2f}")
    out = dict(blueprint=bp, ref_iters=ref_iters, salts=salts, roots=len(built), grid=grid, rows=rows)
    json.dump(out, open("artifacts/search_budget_matrix.json", "w"), indent=1)
    print("\nwrote artifacts/search_budget_matrix.json")


if __name__ == "__main__":
    main()
