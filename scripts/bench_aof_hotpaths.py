"""Benchmark + identity harness for the AoF eval/solve hot paths.

Three micro-benchmarks, each printing wall time and a bit-identity signature so a refactor can be
verified to preserve the CRN-paired streams (the frozen baseline artifact was measured on exactly
these deals/uniforms):

  * ``arena``      — one small paired-arena cell through ``run_cell``; prints per-seed bb/100 for
                     every arm (the identity anchor — must be bit-identical pre/post any change).
  * ``precompute`` — ``precompute_payoffs`` for a fresh deal batch (the 7-depth batch-regen step);
                     showdown shares are depth-invariant + cached, so only the first depth is cold.
  * ``routed``     — ``routed_arrays`` refresh calls (3-max + 4-max), the per-refresh router cost.

Usage:  python scripts/bench_aof_hotpaths.py [all|arena|precompute|routed] [profile]
``profile`` adds a cProfile dump (cumulative + tottime) for the arena cell.

Optimizations validated with this harness (all bit-identical on the arena + drift signatures):
  1. ``fp_br._legal_marginals_3max(..., keys=)`` — compute only the agent's needed marginals
     (``marginals_for`` consumes 1-3 of 6); removes five unused 169^3 cubes per refresh.
  2. ``trio_eval_mc._mc_nway_shares_and_pair_eq_native`` — same RNG board sequence + native eval7,
     per-board Python overhead stripped (no np.asarray / seat_share_from_ranks temporaries).
  3. ``aof_cfr.marg3_native`` (Round 2) — a fused, rayon-parallel Rust kernel materializes the
     contiguous (w, prod=w*pc) cubes for each needed marginal bit-identically; the order-sensitive
     np.sum reductions stay in NumPy. ~2.6-3x on the arena/drift cells and the routed refresh, with
     an automatic pure-NumPy fallback when the extension is absent.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time


def bench_arena_cell(profile: bool = False):
    from aof_cfr.evalsuite.paired_arena import ArenaConfig, run_cell
    cfg = ArenaConfig(players=3, stack_bb=8.0, field="all_leaky", hands=150,
                      seeds=(1, 2), warmup_hands=200, refresh_every=50)
    pr = cProfile.Profile() if profile else None
    if pr:
        pr.enable()
    t0 = time.perf_counter()
    res = run_cell(cfg)
    dt = time.perf_counter() - t0
    if pr:
        pr.disable()
        for sort in ("cumulative", "tottime"):
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats(sort).print_stats(25)
            print(s.getvalue())
    sig = {a: list(r.per_seed_bb100) for a, r in res["arms"].items()}
    print(f"[arena] run_cell {dt:.2f}s")
    print(f"[arena] sig(adj)={sig}")
    return dt, sig


def bench_drift_cell(profile: bool = False):
    """Drift-mode arena cell — stacks evolve per hand, triggering many multi-depth refreshes (the
    cube-math hot path under the depth-grid). Same identity anchor: per-seed bb/100 must be
    bit-identical pre/post any cube-kernel change."""
    from aof_cfr.evalsuite.paired_arena import ArenaConfig, run_cell
    cfg = ArenaConfig(players=3, stack_bb=8.0, field="all_leaky", hands=150, seeds=(1, 2),
                      warmup_hands=200, refresh_every=50, depth_mode="drift")
    pr = cProfile.Profile() if profile else None
    if pr:
        pr.enable()
    t0 = time.perf_counter()
    res = run_cell(cfg)
    dt = time.perf_counter() - t0
    if pr:
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(20)
        print(s.getvalue())
    sig = {a: list(r.per_seed_bb100) for a, r in res["arms"].items()}
    print(f"[drift] run_cell {dt:.2f}s")
    print(f"[drift] sig(adj)={sig}")
    return dt, sig


def bench_precompute(n_deals: int = 2000):
    from aof_cfr.fp_4max_vec import precompute_payoffs, sample_deals
    _buckets, holes = sample_deals(n_deals, seed=999)
    t0 = time.perf_counter()
    pay = precompute_payoffs(holes, 8.0)
    dt = time.perf_counter() - t0
    print(f"[precompute] precompute_payoffs({n_deals}) {dt:.2f}s  sum={pay.sum():.6f}")
    return dt, float(pay.sum())


def bench_routed(n_iter: int = 20):
    import random

    import aof_cfr.query as q
    q.load_cfr_base()
    from aof_cfr.evalsuite.paired_arena import _mem_db
    from aof_cfr.evalsuite.policy import routed_arrays
    from aof_cfr.infoset import aggregate_infoset_key
    from aof_cfr.live_exploit import record_observed_decision
    from aof_cfr.profiler import _base_arrays_seathist, _query_base_push

    rng = random.Random(7)
    results = []
    for n, agent, opp in ((3, "BTN", ("SB", "BB")), (4, "UTG", ("BTN", "SB", "BB"))):
        db = _mem_db()
        for (seat, hist) in _base_arrays_seathist(n, 8.0, _query_base_push):
            aggregate_infoset_key(seat, hist, 8.0)
            for _ in range(20):
                record_observed_decision(db, "bot:0:overfolder", seat, hist, 8.0,
                                         pushed=(rng.random() < 0.3))
        opp_map = {s: "bot:0:overfolder" for s in opp}
        t0 = time.perf_counter()
        for _ in range(n_iter):
            r = routed_arrays(db, n, agent, 8.0, opp_map, min_conf=0.25, margin=0.30)
        dt = time.perf_counter() - t0
        sig = float(sum(float(v[0].sum()) for v in r.values()))
        print(f"[routed] {n_iter}x {n}-max {dt:.2f}s ({dt / n_iter * 1000:.1f}ms/call) sig={sig:.6f}")
        results.append((dt, sig))
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    profile = "profile" in sys.argv
    if mode in ("all", "arena"):
        bench_arena_cell(profile=profile)
    if mode in ("all", "drift"):
        bench_drift_cell(profile=profile)
    if mode in ("all", "precompute"):
        bench_precompute()
    if mode in ("all", "routed"):
        bench_routed()
