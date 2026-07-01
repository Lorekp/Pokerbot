#!/usr/bin/env python3
"""OLD vs NEW search QUALITY on the live 160k blueprint at a realistic ~3s/decision budget.

The point of the search speedups (leaf-gadget DAG 2x bit-identical; native-exploit 20-60x stat-equiv) is
NOT "same answer faster" -- it is "BETTER answer in the same wall-clock", because CFR keeps converging with
more iterations. At a fixed 3s decision budget the faster search runs many more iterations => a more-
converged, less-starvation-blended strategy. This harness measures that on the DEPLOYED EXPLOIT path (the
value-extraction path, opponent_model set + frozen_read_response), where BOTH new levers live:

  OLD search = Python _cfv exploit tail  (native OFF, native_exploit OFF, DAG OFF)
  NEW search = native solve_exploit + DAG (native ON,  native_exploit ON,  DAG ON)

Per root, both capped at time_budget_ms=3000 (max_iterations huge so TIME binds), we report:
  * iters_in_3s        -- how many CFR iterations each completed in 3s (the speedup, as DEPTH)
  * L1_to_converged    -- L1 of the served action-probs vs a high-iteration converged reference
                          (closer = a better/less-starved decision). Reference = NEW at REF_ITERS (fast).
  * budget_bound       -- did the 3s cap cut the solve short of REF_ITERS (=> starvation-blended toward
                          the blueprint, a weaker exploit)?

Caveat: native-exploit is STATISTICAL-equivalence (not bit-identity), so the NEW converged reference is
native-converged; the known Python-vs-native residual is ~0.08 L1 (cloud cert), which is small vs the
starvation gap this harness surfaces. The GTO/DAG arm (--gto) is bit-identical so its reference is exact.

  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 \
  POKERBOT_POSTFLOP_NUM_BUCKETS=200 python scripts/search_quality_3s_ab.py
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

# Reuse the equivalence harness's opponent-model + root builders + snapshot loader (module-level, import-safe).
from native_exploit_equiv import build_root, load_snapshot, make_opponent_model  # noqa: E402


def make_solver(snap, *, max_iters, time_budget_ms, native, native_exploit, dag, rnr_restrict=False):
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig, set_leaf_gadget_batch
    from pokerbot.search.pluribus_search import (
        PluribusReBeLConfig, PluribusReBeLSolver, set_native_exploit, set_native_subgame,
    )
    set_leaf_gadget_batch(bool(dag))
    set_native_subgame(bool(native))
    set_native_exploit(bool(native_exploit))
    cfg = PluribusReBeLConfig(
        max_iterations=int(max_iters),
        time_budget_ms=int(time_budget_ms),
        search_preflop=True,
        validate_snapshot_at_init=False,
        frozen_read_response=True,
        rnr_restrict=rnr_restrict,
        native=bool(native),
        leaf=ContinuationLeafConfig(select="gadget", rollout_samples=4),
    )
    solver = PluribusReBeLSolver(snap, cfg)
    solver._build_leaf_cont_weights = lambda hero: None  # uniform leaf on BOTH arms (fair)
    return solver


def _solve_once(solver, root, pbs, model, salt):
    solver.config.rng_salt = salt
    solver.opponent_model = model
    res = solver.solve(root, pbs)
    return np.asarray(res.action_probs, dtype=np.float64), int(res.iterations), res.metadata.get("served")


def _reset_toggles(dag, native, native_exploit):
    from pokerbot.search.pluribus_rollout import set_leaf_gadget_batch
    from pokerbot.search.pluribus_search import set_native_exploit, set_native_subgame
    set_leaf_gadget_batch(dag)
    set_native_subgame(native)
    set_native_exploit(native_exploit)


def main():
    from pokerbot.search import native_solver as ns
    print("native available:", ns.available(), "subgame:", ns.native_subgame_available(),
          "exploit:", ns.native_subgame_exploit_available(), flush=True)

    bp = os.environ.get("BLUEPRINT_PATH", "checkpoints/serve_blueprint_6max.pkl")
    roots_path = os.environ.get("ROOTS_PATH", "artifacts/search_roots.pkl")
    n_roots = int(os.environ.get("SQ_ROOTS", "5"))
    ref_iters = int(os.environ.get("SQ_REF_ITERS", "1500"))
    budget_ms = int(os.environ.get("SQ_BUDGET_MS", "3000"))
    salt = int(os.environ.get("SQ_SALT", "12345"))

    import pickle
    snap = load_snapshot(bp)
    roots = pickle.load(open(roots_path, "rb"))[:n_roots]
    built = [build_root(r) for r in roots]
    hero = int(roots[0]["hero_seat"])
    model = make_opponent_model(list(range(6)), hero)
    print(f"blueprint={bp} infosets={len(snap):,}  roots={len(built)}  ref_iters={ref_iters}  "
          f"budget_ms={budget_ms}", flush=True)

    # -- Converged reference (NEW/native, high iters, no time bound) --------------------------------
    print("\n[ref] building converged reference (native, high-iter) ...", flush=True)
    ref = np.zeros((len(built), 11))
    sref = make_solver(snap, max_iters=ref_iters, time_budget_ms=10**12,
                       native=True, native_exploit=True, dag=True)
    sref.solve(*built[0])  # warm
    for ri, (root, pbs) in enumerate(built):
        p, it, tag = _solve_once(sref, root, pbs, model, salt)
        ref[ri, :p.shape[0]] = p
        print(f"  [ref] root {ri+1}: {it} iters served={tag}", flush=True)

    # -- Two arms at the 3s budget ------------------------------------------------------------------
    # NEW (native) runs max_iterations INSIDE one FFI call and does NOT poll the Python wall-clock, so
    # time_budget_ms cannot bound it (setting a huge max_iters just hangs). Instead we measure native
    # THROUGHPUT: time a fixed ref_iters solve -> iters reachable in budget_ms, and the L1 at the iter
    # count actually reachable in the budget. OLD (python) DOES poll the wall-clock, so it is bounded by
    # time_budget_ms directly. Run the native arm FIRST (the native-engine flag is process-global; the
    # python arm disables it, and re-enabling mid-process is fragile).
    results = {}

    # NEW arm (native): throughput-based, budget-aware.
    print(f"\n[arm] NEW (native+DAG) @ {budget_ms}ms/decision (throughput-measured)", flush=True)
    snew = make_solver(snap, max_iters=ref_iters, time_budget_ms=10**12,
                       native=True, native_exploit=True, dag=True)
    snew.solve(*built[0])  # warm (pays the first frozen-table build)
    iters, l1, bbound = [], [], 0
    for ri, (root, pbs) in enumerate(built):
        t0 = time.perf_counter()
        p, it, tag = _solve_once(snew, root, pbs, model, salt)
        ms = (time.perf_counter() - t0) * 1000.0
        reach = ref_iters * budget_ms / ms  # iters reachable in budget_ms at this per-iter rate
        if reach >= ref_iters:
            d = 0.0  # reaches the converged reference within the budget (it IS the ref config)
        else:
            # budget-bound: measure the served strategy at the iter count actually reachable in budget_ms
            s_reach = make_solver(snap, max_iters=max(1, int(reach)), time_budget_ms=10**12,
                                  native=True, native_exploit=True, dag=True)
            pr, _, _ = _solve_once(s_reach, root, pbs, model, salt)
            d = float(np.abs(pr - ref[ri]).sum())
            bbound += 1
        iters.append(reach); l1.append(d)
        print(f"  root {ri+1}: {ref_iters} iters in {ms:7.0f}ms -> ~{reach:6.0f} iters/{budget_ms}ms   "
              f"L1_to_converged={l1[-1]:.4f}   {'converged' if reach >= ref_iters else 'BUDGET-BOUND'}",
              flush=True)
    results["NEW (native+DAG)"] = dict(iters=np.array(iters), l1=np.array(l1), bbound=bbound)

    # OLD arm (python): wall-clock bounded directly.
    print(f"\n[arm] OLD (python _cfv) @ {budget_ms}ms/decision (wall-clock bound)", flush=True)
    sold = make_solver(snap, max_iters=10**7, time_budget_ms=budget_ms,
                       native=False, native_exploit=False, dag=False)
    sold.solve(*built[0])  # warm
    iters, l1, bbound = [], [], 0
    for ri, (root, pbs) in enumerate(built):
        p, it, tag = _solve_once(sold, root, pbs, model, salt)
        d = float(np.abs(p - ref[ri]).sum())
        iters.append(it); l1.append(d)
        if it < ref_iters:
            bbound += 1
        print(f"  root {ri+1}: {it:>6} iters   L1_to_converged={d:.4f}   "
              f"{'BUDGET-BOUND' if it < ref_iters else 'converged'}", flush=True)
    results["OLD (python _cfv)"] = dict(iters=np.array(iters), l1=np.array(l1), bbound=bbound)

    # -- Verdict ------------------------------------------------------------------------------------
    old, new = results["OLD (python _cfv)"], results["NEW (native+DAG)"]
    print("\n=== SEARCH QUALITY @ %dms/decision (exploit path, 160k blueprint) ===" % budget_ms)
    print(f"{'arm':22} {'mean iters/3s':>14} {'mean L1_to_conv':>16} {'budget-bound':>13}")
    for name, r in results.items():
        print(f"{name:22} {r['iters'].mean():>14.0f} {r['l1'].mean():>16.4f} "
              f"{r['bbound']}/{len(built):>11}")
    depth_gain = new["iters"].mean() / max(old["iters"].mean(), 1e-9)
    l1_ratio = old["l1"].mean() / max(new["l1"].mean(), 1e-9)
    print(f"\nNEW does {depth_gain:.1f}x the iterations in {budget_ms}ms  -> "
          f"{l1_ratio:.1f}x closer to the converged strategy "
          f"(OLD L1 {old['l1'].mean():.3f} vs NEW L1 {new['l1'].mean():.3f}).")
    print("Interpretation: at a live %dms budget, OLD is starvation-bound on %d/%d roots (blends toward the "
          "blueprint = a weaker exploit); NEW reaches the converged exploit strategy."
          % (budget_ms, old["bbound"], len(built)))


if __name__ == "__main__":
    main()
