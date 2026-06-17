"""Subprocess client for postflop_solver_cli (b-inary/postflop-solver)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pokerbot.solver.label_diff import canonicalize_action_distribution
from pokerbot.solver.spot_enumerator import SolverSpot


DEFAULT_CLI_PATH = (
    Path(__file__).resolve().parents[2]
    / "external"
    / "postflop_solver_cli"
    / "target"
    / "release"
    / ("postflop_solver_cli.exe" if sys.platform == "win32" else "postflop_solver_cli")
)

DEFAULT_CPP_CLI_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "texas_solver_cli"
    / "build"
    / ("texas_solver_cli.exe" if sys.platform == "win32" else "texas_solver_cli")
)


def resolve_cli_path(cli_path: Path | str | None = None) -> Path:
    """Resolve solver CLI: explicit arg, POSTFLOP_SOLVER_CLI env, or Rust default."""
    if cli_path is not None:
        return Path(cli_path)
    env = __import__("os").environ.get("POSTFLOP_SOLVER_CLI")
    if env:
        return Path(env)
    return DEFAULT_CLI_PATH


@dataclass(frozen=True, slots=True)
class SolverSolveResult:
    exploitability: float
    target_exploitability: float
    exploitability_pct_of_pot: float
    converged: bool
    iterations: int
    iterations_used: int
    available_actions: list[str]
    action_distribution: dict[str, float]
    solver_version: str
    error: str | None = None
    raw: dict[str, Any] | None = None


def bet_sizes_string(pot_fractions: list[float], include_all_in: bool = True) -> str:
    parts = [f"{int(f * 100)}%" if f < 1 else f"{int(f * 100)}%" for f in pot_fractions]
    # postflop-solver expects "33%,50%,..." format
    parts = [f"{round(f * 100)}%" for f in pot_fractions]
    if include_all_in:
        parts.append("a")
    return ",".join(parts)


def build_solve_request(
    spot: SolverSpot,
    *,
    bet_fractions: list[float],
    raise_multiplier: float,
    max_iterations: int,
    target_exploitability_pct: float,
    chip_unit_bb: float,
) -> dict[str, Any]:
    board_strs = [c.to_str() for c in spot.board_cards]
    return {
        "oop_range": spot.oop_range,
        "ip_range": spot.ip_range,
        "board": board_strs,
        "turn": spot.turn_card.to_str() if spot.turn_card else None,
        "river": spot.river_card.to_str() if spot.river_card else None,
        "starting_pot_bb": spot.starting_pot_bb,
        "effective_stack_bb": spot.effective_stack_bb,
        "chip_unit_bb": chip_unit_bb,
        "bet_sizes": bet_sizes_string(bet_fractions),
        "raise_sizes": f"{raise_multiplier}x",
        "max_iterations": max_iterations,
        "target_exploitability_pct": target_exploitability_pct,
        "action_path": spot.action_path,
    }



def _canonical_action_distribution(data: dict[str, Any]) -> dict[str, float]:
    dist = data.get("action_distribution", {})
    if not isinstance(dist, dict):
        return {}
    return canonicalize_action_distribution({str(k): float(v) for k, v in dist.items()})


def solver_subprocess_env(solver_threads_per_worker: int | None = None) -> dict[str, str]:
    """Build subprocess env for postflop_solver_cli; honor existing RAYON_NUM_THREADS."""
    env = os.environ.copy()
    if "RAYON_NUM_THREADS" not in env and solver_threads_per_worker is not None:
        env["RAYON_NUM_THREADS"] = str(solver_threads_per_worker)
    return env


def parse_solve_response(data: dict[str, Any]) -> SolverSolveResult:
    return SolverSolveResult(
        exploitability=float(data.get("exploitability", 0.0)),
        target_exploitability=float(data.get("target_exploitability", 0.0)),
        exploitability_pct_of_pot=float(data.get("exploitability_pct_of_pot", 0.0)),
        converged=bool(data.get("converged", False)),
        iterations=int(data.get("iterations", 0)),
        iterations_used=int(data.get("iterations_used", data.get("iterations", 0))),
        available_actions=list(data.get("available_actions", [])),
        action_distribution=_canonical_action_distribution(data),
        solver_version=str(data.get("solver_version", "unknown")),
        error=data.get("error"),
        raw=data,
    )


def run_solver(
    request: dict[str, Any],
    *,
    cli_path: Path | None = None,
    timeout_sec: float = 120.0,
    solver_threads_per_worker: int | None = None,
) -> SolverSolveResult:
    exe = resolve_cli_path(cli_path)
    if not exe.is_file():
        raise FileNotFoundError(
            f"postflop_solver_cli not found at {exe}. "
            "Build with: cd external/postflop_solver_cli && cargo build --release"
        )
    proc = subprocess.run(
        [str(exe)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
        env=solver_subprocess_env(solver_threads_per_worker),
    )
    stdout = proc.stdout.strip()
    if not stdout:
        return SolverSolveResult(
            exploitability=0.0,
            target_exploitability=0.0,
            exploitability_pct_of_pot=0.0,
            converged=False,
            iterations=0,
            iterations_used=0,
            available_actions=[],
            action_distribution={},
            solver_version="postflop-solver@main",
            error=f"cli exit {proc.returncode}: {proc.stderr[:500]}",
        )
    # Progress lines may precede JSON on older binaries; take last JSON object line.
    json_line = stdout.splitlines()[-1] if stdout else ""
    try:
        data = json.loads(json_line)
    except json.JSONDecodeError as exc:
        return SolverSolveResult(
            exploitability=0.0,
            target_exploitability=0.0,
            exploitability_pct_of_pot=0.0,
            converged=False,
            iterations=0,
            iterations_used=0,
            available_actions=[],
            action_distribution={},
            solver_version="postflop-solver@main",
            error=f"invalid json stdout: {exc}; stderr={proc.stderr[:200]}",
        )
    return parse_solve_response(data)


def solve_spot(
    spot: SolverSpot,
    config: dict[str, Any],
    *,
    cli_path: Path | None = None,
    worker: Any | None = None,
    request_id: str | None = None,
    solver_threads_per_worker: int | None = None,
    max_iterations_override: int | None = None,
    target_exploitability_pct_override: float | None = None,
) -> SolverSolveResult:
    bet_tree = config.get("bet_tree", {})
    solver_cfg = config.get("solver", {})
    req = build_solve_request(
        spot,
        bet_fractions=[float(x) for x in bet_tree.get("pot_fractions", [0.33, 0.5, 0.75, 1.0])],
        raise_multiplier=float(bet_tree.get("raise_multiplier", 2.5)),
        max_iterations=int(max_iterations_override or solver_cfg.get("max_iterations", 10000)),
        target_exploitability_pct=float(
            target_exploitability_pct_override or solver_cfg.get("target_exploitability_pct", 0.5)
        ),
        chip_unit_bb=float(solver_cfg.get("chip_unit_bb", 100)),
    )
    timeout_sec = float(solver_cfg.get("timeout_sec", 3600))
    if worker is not None:
        return worker.solve(req, request_id=request_id or "0", timeout_sec=timeout_sec)
    return run_solver(
        req,
        cli_path=cli_path,
        timeout_sec=timeout_sec,
        solver_threads_per_worker=solver_threads_per_worker,
    )
