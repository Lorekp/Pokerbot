"""Search-path baseline bench: one real postflop state, 200 iters, no opponent model.

Pure measurement. No source files are modified; we wrap instance methods on a
local solver with thin timing/counting callables (runtime substitution only,
restored on exit). The production path is unchanged on disk.

Metrics (per the task):
  - wall per iter                       (ms/iter)
  - leaf-value ns per call              (leaf_value)
  - subgame-traversal ns per iter       (CFR _cfv)
  - infoset-key lookup ns per call      (_infoset_key)
  - allocation per iter                 (KiB/iter, tracemalloc peak delta / iters)

Configuration: 10 repeats, 200 iters each, p50 + p95 reported per metric.

Usage:
  python scripts/bench_search_baseline.py \
      --iters 200 --repeats 10 \
      --out artifacts/search_bench_baseline.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

# Force the interpreted search path. The task is "no opponent model", which the
# config below already ensures; we additionally disable the optional native
# subgame / CFV paths so the timing isolates the Python traversal + leaf eval
# + infoset-key lookup stack that this bench is profiling.
os.environ.setdefault("POKERBOT_NATIVE_CFV", "0")
os.environ.setdefault("POKERBOT_NATIVE_SUBGAME", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokerbot.engine.actions import ActionType
from pokerbot.engine.cards import Card
from pokerbot.engine.game import GameState, play_hand
from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS
from pokerbot.search import native_solver as ns
from pokerbot.search import pluribus_search as ps
from pokerbot.search.pbs_tracker import PBSTracker
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
from pokerbot.search.state_bridge import SubgameRoot

# --- helpers --------------------------------------------------------------


_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _card(idx: int) -> Card:
    i = int(idx)
    return Card.from_str(_RANKS[i // 4] + _SUITS[i % 4])


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def _summarize(xs: list[float]) -> dict[str, Any]:
    return {
        "n": len(xs),
        "p50": _pct(xs, 50),
        "p95": _pct(xs, 95),
        "mean": float(np.mean(xs)) if xs else None,
        "min": float(np.min(xs)) if xs else None,
        "max": float(np.max(xs)) if xs else None,
        "stdev": float(np.std(xs)) if len(xs) > 1 else 0.0,
    }


def _build_fixed_state(seed: int):
    """Build a deterministic 6-max postflop state at hero seat.

    Plays a single hand to the flop with a deterministic policy (seeded), then
    captures the first flop decision at hero's seat. The SAME state is reused
    for every repeat so the per-iter numbers reflect pure path costs, not deal
    variance.
    """
    rng = random.Random(seed)
    st0 = GameState.new_hand(button_seat=3, rng=random.Random(rng.randint(0, 10**9)))

    captured: dict[str, Any] = {}

    def pol(state):
        legal = state.legal_actions()
        s = state.action_seat
        street = int(state.street.value if hasattr(state.street, "value") else state.street)
        if (
            s is not None
            and s >= 0
            and legal
            and street >= 2
            and "env" not in captured
        ):
            from pokerbot.search.state_bridge import game_state_to_env
            captured["env"] = game_state_to_env(state, int(s))
            captured["hero"] = int(s)
            captured["street"] = street
        if not legal:
            return None
        passive = [a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)]
        if passive and rng.random() < 0.5:
            return rng.choice(passive)
        return rng.choice(legal)

    play_hand({s: pol for s in range(6)}, state=st0,
              rng=random.Random(rng.randint(0, 10**9)))

    if "env" not in captured:
        raise RuntimeError("fixed postflop state not captured (policy never reached street>=2)")

    return captured["env"], captured["hero"]


def _make_solver(seed: int, iters: int):
    """Build a PluribusReBeLSolver pinned to the interpreted path.

    No opponent model (task spec). Configured for 200 iters, no DCFR (so the
    path matches the legacy default used by tests; enabling DCFR changes
    per-iter shape).
    """
    cfg = ps.PluribusReBeLConfig(
        time_budget_ms=10**9,   # wall-clock never bounds -> exactly `iters` iters
        leaf=ContinuationLeafConfig(select="uniform"),
        max_iterations=int(iters),
        max_depth=2,
        per_decision_seed=True,
        force_budget_bound=int(iters),  # measurement-only: run exactly `iters` iters
        dcfr=False,
        linear_averaging=True,
    )
    solver = ps.PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(seed))
    solver.opponent_model = None
    # Belt-and-braces: make sure native subgame/CFV are off for this process.
    ps.set_native_subgame(False)
    ps.set_native_cfv(False)
    try:
        ns.disable()
    except Exception:
        pass
    return solver


def _build_root_and_pbs(env, hero: int):
    board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
    h0, h1 = int(env.holes[hero][0]), int(env.holes[hero][1])
    pbs = PBSTracker.init_uniform(hero, (_card(h0), _card(h1)), board)
    root = SubgameRoot(
        env=env,
        hero_seat=hero,
        pbs_vector=pbs.get_public_belief_state(),
        legal_mask=None,
        opponent_profiles=None,
    )
    return root, pbs


# --- instrumentation ------------------------------------------------------


class _Tape:
    """Per-repeat cumulative timers + call counters.

    All times are summed nanoseconds; rates (ns/call, ns/iter) are derived at
    the end from the totals + counts. We deliberately don't time the wrapper
    itself (perf_counter_ns is ~30-40ns/call) — only the *inside* of the wrapped
    call is summed, so the tape captures the work, not the measurement.
    """

    def __init__(self) -> None:
        self.leaf_ns = 0
        self.leaf_calls = 0
        self.cfv_ns = 0
        self.cfv_calls = 0
        self.key_ns = 0
        self.key_calls = 0


def _wrap(solver, tape: _Tape):
    """Bind instance-method wrappers that time the body of each hot method.

    The original methods are saved on the instance under private aliases and
    restored at the end of the run. We do NOT modify the class or any source
    file.
    """
    orig_leaf = solver._leaf_value
    orig_cfv = solver._cfv
    orig_key = solver._infoset_key

    def timed_leaf(env, hero):
        t0 = time.perf_counter_ns()
        try:
            return orig_leaf(env, hero)
        finally:
            tape.leaf_ns += time.perf_counter_ns() - t0
            tape.leaf_calls += 1

    def timed_cfv(env, hero, depth, t=0):
        t0 = time.perf_counter_ns()
        try:
            return orig_cfv(env, hero, depth, t)
        finally:
            tape.cfv_ns += time.perf_counter_ns() - t0
            tape.cfv_calls += 1

    def timed_key(env, seat):
        t0 = time.perf_counter_ns()
        try:
            return orig_key(env, seat)
        finally:
            tape.key_ns += time.perf_counter_ns() - t0
            tape.key_calls += 1

    solver._orig_leaf_value = orig_leaf
    solver._orig_cfv = orig_cfv
    solver._orig_infoset_key = orig_key
    solver._leaf_value = timed_leaf
    solver._cfv = timed_cfv
    solver._infoset_key = timed_key
    return timed_leaf, timed_cfv, timed_key


def _unwrap(solver):
    """Restore original (untimed) bound methods on the solver instance."""
    if hasattr(solver, "_orig_leaf_value"):
        solver._leaf_value = solver._orig_leaf_value
    if hasattr(solver, "_orig_cfv"):
        solver._cfv = solver._orig_cfv
    if hasattr(solver, "_orig_infoset_key"):
        solver._infoset_key = solver._orig_infoset_key
    for attr in ("_orig_leaf_value", "_orig_cfv", "_orig_infoset_key"):
        if hasattr(solver, attr):
            delattr(solver, attr)


# --- per-repeat driver ----------------------------------------------------


def _run_one_repeat(solver, env, hero, iters: int) -> dict[str, Any]:
    """Run `iters` iterations of CFR on a fixed state, return per-iter metrics."""
    tape = _Tape()
    _wrap(solver, tape)
    try:
        # Force a clean alloc baseline per repeat.
        gc.collect()
        tracemalloc.start()
        alloc_start = tracemalloc.get_traced_memory()[0]

        # Fresh root + PBSTracker per repeat (mirrors live serve: every
        # decision is a fresh solve with no carry-over state).
        root, pbs = _build_root_and_pbs(env, hero)

        wall_t0 = time.perf_counter()
        result = solver.solve(root, pbs, single_table=True)
        wall = time.perf_counter() - wall_t0

        alloc_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    finally:
        _unwrap(solver)

    iterations = int(result.iterations) if result and result.iterations else 0
    if iterations == 0:
        # Fall back to configured target.
        iterations = int(iters)

    wall_per_iter_ms = (wall / iterations) * 1000.0
    leaf_ns_per_call = (tape.leaf_ns / tape.leaf_calls) if tape.leaf_calls else 0.0
    cfv_ns_per_iter = (tape.cfv_ns / iterations) if iterations else 0.0
    key_ns_per_call = (tape.key_ns / tape.key_calls) if tape.key_calls else 0.0
    alloc_kib_per_iter = ((alloc_peak - alloc_start) / 1024.0 / iterations) if iterations else 0.0

    return {
        "wall_per_iter_ms": wall_per_iter_ms,
        "leaf_ns_per_call": leaf_ns_per_call,
        "cfv_ns_per_iter": cfv_ns_per_iter,
        "key_ns_per_call": key_ns_per_call,
        "alloc_kib_per_iter": alloc_kib_per_iter,
        # diagnostic
        "iters_done": iterations,
        "leaf_calls": tape.leaf_calls,
        "cfv_calls": tape.cfv_calls,
        "key_calls": tape.key_calls,
        "wall_total_s": wall,
        "alloc_peak_kib": (alloc_peak - alloc_start) / 1024.0,
    }


# --- main -----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "search_bench_baseline.json")
    args = ap.parse_args()

    # 1) Build the fixed postflop state (used for ALL repeats -> isolates path
    #    cost from deal variance).
    print(f"[bench] building fixed postflop state (seed={args.seed}) ...", flush=True)
    env, hero = _build_fixed_state(args.seed)
    board_cards = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
    hero_hole = (_card(int(env.holes[hero][0])), _card(int(env.holes[hero][1])))
    print(
        f"[bench]   hero_seat={hero} street={int(env.street)} "
        f"board=[{','.join(str(c) for c in board_cards)}] "
        f"hero_hole=[{hero_hole[0]},{hero_hole[1]}]",
        flush=True,
    )

    # 2) Warm-up: one full solve to JIT/touch caches, so the first measured
    #    repeat isn't paying one-time import / adapter-init costs.
    print("[bench] warm-up solve (uncounted) ...", flush=True)
    warm_solver = _make_solver(args.seed, args.iters)
    root_w, pbs_w = _build_root_and_pbs(env, hero)
    _ = warm_solver.solve(root_w, pbs_w, single_table=True)
    del warm_solver

    # 3) The 10 measured repeats. Each repeat uses a FRESH solver so per-solve
    #    caches (regrets/strategy_sum/legal_cache/leaf_cache) start cold, which
    #    is the realistic per-decision cost the live serve pays (every
    #    decision is a fresh solve).
    print(f"[bench] running {args.repeats} repeats x {args.iters} iters ...", flush=True)
    per_repeat: list[dict[str, Any]] = []
    for r in range(args.repeats):
        solver = _make_solver(args.seed + 1 + r, args.iters)
        m = _run_one_repeat(solver, env, hero, args.iters)
        per_repeat.append(m)
        del solver
        print(
            f"  repeat {r+1:2d}/{args.repeats}: "
            f"wall/iter={m['wall_per_iter_ms']:.3f}ms  "
            f"leaf/call={m['leaf_ns_per_call']:.0f}ns  "
            f"cfv/iter={m['cfv_ns_per_iter']/1e3:.1f}us  "
            f"key/call={m['key_ns_per_call']:.0f}ns  "
            f"alloc/iter={m['alloc_kib_per_iter']:.1f}KiB  "
            f"iters={m['iters_done']}",
            flush=True,
        )

    # 4) Aggregate p50 + p95 per metric.
    keys_metric = [
        "wall_per_iter_ms",
        "leaf_ns_per_call",
        "cfv_ns_per_iter",
        "key_ns_per_call",
        "alloc_kib_per_iter",
    ]
    summary: dict[str, Any] = {}
    for k in keys_metric:
        xs = [m[k] for m in per_repeat]
        s = _summarize(xs)
        summary[k] = s

    out = {
        "profile": "search_path_baseline",
        "config": {
            "iters": args.iters,
            "repeats": args.repeats,
            "seed": args.seed,
            "opponent_model": None,
            "force_budget_bound": True,
            "path": "interpreted (native_cfv=0, native_subgame=0)",
            "solver_config": {
                "max_iterations": args.iters,
                "max_depth": 2,
                "dcfr": False,
                "linear_averaging": True,
                "per_decision_seed": True,
            },
        },
        "fixed_state": {
            "hero_seat": hero,
            "street": int(env.street),
            "board": [str(c) for c in board_cards],
            "hero_hole": [str(hero_hole[0]), str(hero_hole[1])],
            "postflop_buckets": int(_POSTFLOP_NUM_BUCKETS),
        },
        "summary": summary,
        "per_repeat": per_repeat,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[bench] wrote {args.out}", flush=True)
    print("[bench] p50 / p95:", flush=True)
    for k in keys_metric:
        s = summary[k]
        print(f"  {k:24s}  p50={s['p50']:.4f}   p95={s['p95']:.4f}   "
              f"mean={s['mean']:.4f}   n={s['n']}", flush=True)


if __name__ == "__main__":
    main()
