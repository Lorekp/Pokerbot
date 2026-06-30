"""Benchmark + cProfile the Pluribus depth-limited search on captured subgame roots.

Re-solves a FIXED set of captured postflop (and optional preflop) roots at several max_iterations and
reports ms/solve, solves/sec, and nodes-per-solve metadata. Optional cProfile pass.

Usage (env MUST be set before pokerbot import — see the launcher):
    python scripts/bench_search_run.py <roots.pkl> [mode]
      mode = bench   -> iteration-scaling table (default)
             profile -> cProfile a batch and print top functions
             native  -> python-vs-native A/B at fixed iters
             one     -> a single solve (for py-spy flamegraph; loops to fill wall time)
"""
from __future__ import annotations

import cProfile
import io
import os
import pickle
import pstats
import sys
import time

import numpy as np

from pokerbot.search.pluribus_search import (
    PluribusReBeLConfig,
    PluribusReBeLSolver,
    set_native_subgame,
)
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig


# Minimal SubgameRoot shim (the real pokerbot.search.state_bridge import pulls torch via pbs_tracker;
# the solver only reads root.env / root.hero_seat / root.pbs_vector / root.legal_mask).
class SubgameRoot:  # noqa: D401
    __slots__ = ("env", "hero_seat", "pbs_vector", "legal_mask", "opponent_profiles")

    def __init__(self, env, hero_seat, pbs_vector, legal_mask, opponent_profiles=None):
        self.env = env
        self.hero_seat = hero_seat
        self.pbs_vector = pbs_vector
        self.legal_mask = legal_mask
        self.opponent_profiles = opponent_profiles


def load_snapshot(path: str) -> dict:
    obj = pickle.load(open(path, "rb"))
    if isinstance(obj, dict) and "strat_sum" in obj:
        return {bytes(k): np.asarray(v) for k, v in obj["strat_sum"].items()}
    return obj


class _PBS:
    """Minimal PBSTracker-like shim: solve() only needs sample_opponent_holes(rng)."""

    def __init__(self, env):
        # Build a real PBSTracker if possible (uniform), else a uniform sampler over unseen cards.
        self.env = env

    def sample_opponent_holes(self, rng):
        # Sample 2 distinct unseen cards per non-hero opponent seat that is in the hand.
        env = self.env
        holes = np.asarray(env.holes)
        board = np.asarray(env.board)
        seen = set(int(c) for c in holes.reshape(-1) if int(c) >= 0)
        seen |= set(int(c) for c in board.reshape(-1) if int(c) >= 0)
        avail = [c for c in range(52) if c not in seen]
        rng.shuffle(avail)
        out = {}
        ptr = 0
        from pokerbot.engine.np_engine import SEAT_FOLDED
        status = np.asarray(env.seat_status)
        for s in range(holes.shape[0]):
            if int(holes[s, 0]) >= 0:
                continue  # known hole (hero)
            if int(status[s]) == int(SEAT_FOLDED):
                continue
            out[s] = (int(avail[ptr]), int(avail[ptr + 1]))
            ptr += 2
        return out


def build_root(rec) -> tuple[SubgameRoot, object]:
    env = rec["env"]
    hero = int(rec["hero_seat"])
    pbs_vec = rec.get("pbs_vector")
    root = SubgameRoot(env=env, hero_seat=hero,
                       pbs_vector=pbs_vec if pbs_vec is not None else np.zeros(1),
                       legal_mask=np.ones(1, dtype=bool))
    pbs = _PBS(env)
    return root, pbs


def make_solver(snap, iters, leaf_select="gadget", rollout_samples=4) -> PluribusReBeLSolver:
    cfg = PluribusReBeLConfig(
        max_iterations=iters,
        time_budget_ms=10**9,
        search_preflop=True,
        validate_snapshot_at_init=False,
        leaf=ContinuationLeafConfig(select=leaf_select, rollout_samples=rollout_samples),
    )
    return PluribusReBeLSolver(snap, cfg)


def bench(roots, snap, iters_list, leaf_select):
    print(f"\n=== BENCH (leaf_select={leaf_select}, n_roots={len(roots)}) ===")
    print(f"{'iters':>6} {'ms/solve':>10} {'solves/s':>10} {'nodes/solve':>12} {'leaf_evals':>11} {'cache_hits':>11}")
    results = {}
    for iters in iters_list:
        solver = make_solver(snap, iters, leaf_select)
        # warm one solve (JIT-ish caches, import side effects)
        r0, p0 = build_root(roots[0])
        solver.solve(r0, p0)
        t0 = time.perf_counter()
        leaf_total = 0.0
        cache_total = 0.0
        n = 0
        for rec in roots:
            root, pbs = build_root(rec)
            res = solver.solve(root, pbs)
            leaf_total += res.metadata.get("leaf_eval_count", 0.0)
            cache_total += res.metadata.get("leaf_cache_hits", 0.0)
            n += 1
        dt = time.perf_counter() - t0
        ms = dt / n * 1000.0
        sps = n / dt
        nodes = (leaf_total + cache_total) / n
        print(f"{iters:>6} {ms:>10.2f} {sps:>10.2f} {nodes:>12.1f} {leaf_total/n:>11.1f} {cache_total/n:>11.1f}")
        results[iters] = (ms, sps)
    return results


def profile(roots, snap, iters, leaf_select, repeats):
    print(f"\n=== cPROFILE (iters={iters}, leaf_select={leaf_select}, repeats={repeats}) ===")
    solver = make_solver(snap, iters, leaf_select)
    built = [build_root(rec) for rec in roots]
    # warm
    solver.solve(*built[0])

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(repeats):
        for root, pbs in built:
            solver.solve(root, pbs)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(30)
    print("---- TOP BY TOTTIME ----")
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("cumulative")
    ps2.print_stats(30)
    print("---- TOP BY CUMTIME ----")
    print(s2.getvalue())

    # Dump a .prof for flameprof/snakeviz fallback
    out = f"artifacts/search_iters{iters}_{leaf_select}.prof"
    pr.dump_stats(out)
    print(f"wrote cProfile stats -> {out}")


def native_ab(roots, snap, iters, leaf_select):
    print(f"\n=== NATIVE A/B (iters={iters}, leaf_select={leaf_select}) ===")
    built = [build_root(rec) for rec in roots]

    def run(native_on):
        set_native_subgame(native_on)
        solver = make_solver(snap, iters, leaf_select)
        if native_on:
            solver.config.native = True
        solver.solve(*built[0])
        t0 = time.perf_counter()
        served = {}
        for root, pbs in built:
            res = solver.solve(root, pbs)
            served[res.metadata.get("served")] = served.get(res.metadata.get("served"), 0) + 1
        dt = time.perf_counter() - t0
        return dt / len(built) * 1000.0, served

    py_ms, py_served = run(False)
    print(f"python:  {py_ms:8.2f} ms/solve   served={py_served}")
    try:
        nat_ms, nat_served = run(True)
        print(f"native:  {nat_ms:8.2f} ms/solve   served={nat_served}")
        if nat_served.get("search_native", 0) > 0:
            print(f"speedup: {py_ms / nat_ms:.2f}x")
        else:
            print("native path did NOT serve search_native (fell back to python) — check ext build/gates")
    except Exception as e:  # noqa: BLE001
        print(f"native run failed: {e!r}")
    set_native_subgame(False)


def one(roots, snap, iters, leaf_select, seconds):
    """Loop solves for ~`seconds` so py-spy/flamegraph has a stable sampling target."""
    solver = make_solver(snap, iters, leaf_select)
    built = [build_root(rec) for rec in roots]
    print(f"flamegraph target: looping {len(built)} roots @ iters={iters} for ~{seconds}s", flush=True)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        for root, pbs in built:
            solver.solve(root, pbs)
            n += 1
            if time.perf_counter() - t0 >= seconds:
                break
    print(f"did {n} solves in {time.perf_counter()-t0:.1f}s", flush=True)


def main():
    roots_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/search_roots.pkl"
    mode = sys.argv[2] if len(sys.argv) > 2 else "bench"
    leaf_select = os.environ.get("BENCH_LEAF_SELECT", "gadget")
    roots = pickle.load(open(roots_path, "rb"))
    print(f"loaded {len(roots)} roots from {roots_path}; leaf_select={leaf_select}", flush=True)
    snap = load_snapshot("checkpoints/serve_blueprint_6max.pkl")

    if mode == "bench":
        bench(roots, snap, [20, 50, 100], leaf_select)
    elif mode == "profile":
        iters = int(os.environ.get("BENCH_ITERS", "50"))
        reps = int(os.environ.get("BENCH_REPEATS", "2"))
        profile(roots, snap, iters, leaf_select, reps)
    elif mode == "native":
        native_ab(roots, snap, int(os.environ.get("BENCH_ITERS", "50")), leaf_select)
    elif mode == "one":
        one(roots, snap, int(os.environ.get("BENCH_ITERS", "50")), leaf_select,
            float(os.environ.get("BENCH_SECONDS", "30")))
    else:
        print(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
