#!/usr/bin/env python3
"""Profile TurboReBeL decision latency with traversal vs leaf breakdown."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pokerbot.engine.game import GameState
from pokerbot.search.leaf_evaluator import HistoryValueEvaluator, MockHistoryValueEvaluator
from pokerbot.search.pbs_tracker import PBSTracker
from pokerbot.search.state_bridge import build_subgame_root
from pokerbot.search.turbo_rebel_search import TurboReBeLConfig, TurboReBeLSolver


def _sample_roots(n: int, seed: int) -> list[tuple[GameState, int]]:
    rng = np.random.default_rng(seed)
    roots: list[tuple[GameState, int]] = []
    for i in range(n):
        state = GameState.new_hand(button_seat=int(rng.integers(0, 6)))
        state.deal_hole_cards()
        hero = i % 6
        for _ in range(int(rng.integers(0, 4))):
            if state.is_terminal() or not state.actors_remaining():
                break
            legal = state.legal_actions()
            if not legal:
                break
            state.apply_action(legal[int(rng.integers(0, len(legal)))])
            state.advance_to_next_actor()
        roots.append((state, hero))
    return roots


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4 TurboReBeL latency profiler")
    p.add_argument("--value-checkpoint", default="checkpoints/history_value_net.pkl")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-budget-ms", type=int, default=2000)
    p.add_argument("--p95-budget-ms", type=float, default=2000.0)
    p.add_argument("--use-mock-leaf", action="store_true")
    p.add_argument("--device", default="cpu", help="Reserved for future GPU leaf path")
    p.add_argument("--output", default="profile_phase4_latency.json")
    p.add_argument("--assert-budget", action="store_true")
    args = p.parse_args()

    ckpt = Path(args.value_checkpoint)
    if args.use_mock_leaf or not ckpt.exists():
        leaf = MockHistoryValueEvaluator()
        using_mock = True
    else:
        leaf = HistoryValueEvaluator(ckpt)
        using_mock = not leaf.has_trained_checkpoint

    solver = TurboReBeLSolver(
        leaf,
        TurboReBeLConfig(time_budget_ms=args.time_budget_ms, max_depth=4),
    )

    latencies: list[float] = []
    traversal_fracs: list[float] = []
    native_recommended = 0
    for state, hero in _sample_roots(args.samples, args.seed):
        hole = state.players[hero].hole
        pbs = PBSTracker.init_uniform(hero, hole, state.board)
        root = build_subgame_root(state, pbs, hero)
        t0 = time.perf_counter()
        result = solver.solve(root, pbs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed)
        meta = result.metadata
        traversal_fracs.append(float(meta.get("traversal_fraction", 0.0)))
        if meta.get("native_cfr_recommended"):
            native_recommended += 1

    arr = np.array(latencies, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    tf = np.array(traversal_fracs, dtype=np.float64) if traversal_fracs else np.array([0.0])
    mean_traversal_frac = float(tf.mean())

    report = {
        "samples": args.samples,
        "time_budget_ms": args.time_budget_ms,
        "device": args.device,
        "p50_ms": p50,
        "p95_ms": p95,
        "max_ms": float(arr.max()),
        "mean_ms": float(arr.mean()),
        "mean_traversal_fraction": mean_traversal_frac,
        "native_cfr_recommended_count": native_recommended,
        "native_cfr_decision": (
            "defer_native_bindings"
            if mean_traversal_frac <= 0.4
            else "spec_rust_pyo3_cfv_traversal"
        ),
        "p95_budget_ms": args.p95_budget_ms,
        "budget_assert_passed": p95 <= args.p95_budget_ms,
        "value_checkpoint": str(ckpt),
        "use_mock_leaf": using_mock,
    }

    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")

    if args.assert_budget and p95 > args.p95_budget_ms:
        raise SystemExit(f"p95 latency {p95:.1f}ms exceeds budget {args.p95_budget_ms}ms")


if __name__ == "__main__":
    main()
