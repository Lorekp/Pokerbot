#!/usr/bin/env python3
"""Diagnose postflop-solver convergence vs iteration cap on a single spot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from pokerbot.solver.postflop_solver_client import (
    build_solve_request,
    run_solver,
)
from pokerbot.solver.spot_enumerator import enumerate_spots, load_solver_config

# Old unconverged snapshot (iteration-150 run) for comparison — spot 0 board Td9d6h SB hero
LEGACY_150_DIST = {
    "check": 0.9980919408098544,
    "bet_0.50": 0.0019080591901456281,
    "all_in": 0.0,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/solver_labels_smoke.yaml"))
    p.add_argument(
        "--iteration-caps",
        type=int,
        nargs="+",
        default=[150, 500, 1000, 2000, 5000, 10000],
    )
    p.add_argument("--target-pct", type=float, default=0.5)
    p.add_argument("--output", type=Path, default=Path("logs/solver_convergence_diagnosis.json"))
    args = p.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = load_solver_config(args.config)
    spot = next(iter(enumerate_spots(config)))

    bet_tree = raw.get("bet_tree", {})
    solver_cfg = raw.get("solver", {})
    fractions = [float(x) for x in bet_tree.get("pot_fractions", [0.5])]

    rows: list[dict] = []
    final_dist: dict[str, float] | None = None

    print(f"Spot: {spot.spot_id} board={''.join(c.to_str() for c in spot.board_cards)} hero_seat={spot.hero_seat}")
    print(f"Ranges: OOP={spot.oop_range!r} IP={spot.ip_range!r}")
    print(f"Bet tree: {fractions} raise={bet_tree.get('raise_multiplier', 2.5)}")
    print(f"Target exploitability: {args.target_pct}% of pot\n")

    for cap in args.iteration_caps:
        req = build_solve_request(
            spot,
            bet_fractions=fractions,
            raise_multiplier=float(bet_tree.get("raise_multiplier", 2.5)),
            max_iterations=cap,
            target_exploitability_pct=args.target_pct,
            chip_unit_bb=float(solver_cfg.get("chip_unit_bb", 100)),
        )
        t0 = time.perf_counter()
        result = run_solver(req, timeout_sec=float(solver_cfg.get("timeout_sec", 3600)))
        elapsed = time.perf_counter() - t0
        row = {
            "max_iterations": cap,
            "elapsed_sec": round(elapsed, 2),
            "iterations_used": result.raw.get("iterations_used") if result.raw else result.iterations,
            "exploitability_chips": result.exploitability,
            "exploitability_pct_of_pot": result.raw.get("exploitability_pct_of_pot") if result.raw else None,
            "target_exploitability_chips": result.target_exploitability,
            "converged": result.converged,
            "error": result.error,
        }
        rows.append(row)
        exploit_pct = row["exploitability_pct_of_pot"]
        exploit_str = f"{exploit_pct:.3f}" if exploit_pct is not None else "n/a"
        print(
            f"cap={cap:5d}  used={row['iterations_used']:5}  "
            f"exploit={exploit_str}% pot  "
            f"converged={result.converged}  time={elapsed:.1f}s"
        )
        if result.converged and result.action_distribution:
            final_dist = result.action_distribution

    report = {
        "spot_id": spot.spot_id,
        "board": [c.to_str() for c in spot.board_cards],
        "oop_range": spot.oop_range,
        "ip_range": spot.ip_range,
        "bet_tree": bet_tree,
        "target_exploitability_pct": args.target_pct,
        "iteration_sweep": rows,
        "legacy_150_distribution": LEGACY_150_DIST,
        "converged_distribution": final_dist,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
