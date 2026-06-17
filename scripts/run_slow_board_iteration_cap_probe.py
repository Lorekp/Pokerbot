#!/usr/bin/env python3
"""Probe slow-board iteration cap: exploitability at cap vs full convergence (KL-gated)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from pokerbot.solver.postflop_solver_client import (
    DEFAULT_CLI_PATH,
    build_solve_request,
    run_solver,
)
from pokerbot.solver.spot_enumerator import enumerate_spots, load_solver_config, spot_key


def main() -> None:
    p = argparse.ArgumentParser(description="Slow-board iteration cap probe")
    p.add_argument("--config", type=Path, default=Path("config/solver_labels_prod600.yaml"))
    p.add_argument("--timing-manifest", type=Path, default=Path("data/solver/manifest_prod600.json"))
    p.add_argument("--slow-median-sec", type=float, default=120.0)
    p.add_argument("--iteration-cap", type=int, default=800)
    p.add_argument("--target-pct", type=float, default=1.0)
    p.add_argument("--max-spots", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("logs/slow_board_iteration_cap_probe.json"))
    args = p.parse_args()

    timing = json.loads(args.timing_manifest.read_text(encoding="utf-8")).get("board_timing", {})
    slow_boards = {
        b
        for b, s in timing.items()
        if isinstance(s, dict) and float(s.get("median_sec", 0)) >= args.slow_median_sec
    }

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = load_solver_config(args.config)
    solver_cfg = raw.get("solver", {})

    rows: list[dict[str, Any]] = []
    for spot in enumerate_spots(config):
        board = "".join(c.to_str() for c in spot.board_cards)
        if board not in slow_boards or spot.spot_id != "sb_vs_bb_flop":
            continue
        sk = spot_key(spot)
        if any(r["spot_key"] == sk for r in rows):
            continue
        req_full = build_solve_request(
            spot,
            bet_fractions=[float(x) for x in raw["bet_tree"]["pot_fractions"]],
            raise_multiplier=float(raw["bet_tree"]["raise_multiplier"]),
            max_iterations=int(solver_cfg.get("max_iterations", 10000)),
            target_exploitability_pct=args.target_pct,
            chip_unit_bb=float(solver_cfg.get("chip_unit_bb", 100)),
        )
        req_cap = dict(req_full)
        req_cap["max_iterations"] = args.iteration_cap

        for label, req in (("full", req_full), (f"cap_{args.iteration_cap}", req_cap)):
            t0 = time.perf_counter()
            res = run_solver(req, cli_path=DEFAULT_CLI_PATH, timeout_sec=3600)
            rows.append(
                {
                    "spot_key": sk,
                    "mode": label,
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                    "iterations_used": res.iterations_used,
                    "exploitability_pct_of_pot": res.exploitability_pct_of_pot,
                    "converged": res.converged,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
        if len({r["spot_key"] for r in rows}) >= args.max_spots:
            break

    report = {
        "slow_boards_in_probe": sorted({r["spot_key"].split("|")[1] for r in rows}),
        "iteration_cap": args.iteration_cap,
        "target_exploitability_pct": args.target_pct,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
