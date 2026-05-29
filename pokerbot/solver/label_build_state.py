"""Shared spot outcome processing for build_solver_labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pokerbot.encoding.postflop_solver_encoder import encode_solver_spot_row, record_to_parquet_dict
from pokerbot.encoding.weva_encoder import WEVAMapper
from pokerbot.solver.postflop_solver_client import SolverSolveResult
from pokerbot.solver.spot_enumerator import SolverSpot


@dataclass(slots=True)
class LabelBuildState:
    included_rows: list[dict[str, Any]]
    excluded_by_reason: dict[str, int]
    skipped_solver: dict[str, int]
    not_converged_details: list[dict[str, Any]]
    per_spot_results: list[dict[str, Any]]
    first_solver_error: str | None
    convergence_stats: list[float]
    sample_labels: list[dict[str, Any]]
    solve_times_sec: list[float]


def apply_solve_result(
    *,
    spot: SolverSpot,
    spot_key: str,
    result: SolverSolveResult,
    elapsed: float,
    state: LabelBuildState,
    mapper: WEVAMapper,
    bet_sizes_dsl: str,
    raise_sizes_dsl: str,
    target_pct: float,
    include_unconverged: bool,
    prof_measure_encode: Any | None = None,
) -> bool:
    """Process one solver result. Returns True if a row was included."""
    spot_result = {
        "spot": spot_key,
        "elapsed_sec": round(elapsed, 2),
        "iterations_used": result.iterations_used,
        "exploitability_pct_of_pot": result.exploitability_pct_of_pot,
        "target_exploitability_pct": target_pct,
        "converged": result.converged,
        "oop_range": spot.oop_range,
        "ip_range": spot.ip_range,
        "bet_sizes": bet_sizes_dsl,
    }
    state.per_spot_results.append(spot_result)
    state.solve_times_sec.append(elapsed)

    if result.error:
        state.skipped_solver["solver_error"] = state.skipped_solver.get("solver_error", 0) + 1
        if state.first_solver_error is None:
            state.first_solver_error = result.error[:200]
        spot_result["flagged_failure"] = True
        spot_result["failure_reason"] = "solver_error"
        spot_result["error"] = result.error[:200]
        return False

    if not result.converged and not include_unconverged:
        state.skipped_solver["not_converged"] = state.skipped_solver.get("not_converged", 0) + 1
        flagged = {
            **spot_result,
            "flagged_failure": True,
            "failure_reason": "not_converged_within_backstop",
            "action_distribution_preview": result.action_distribution,
        }
        state.not_converged_details.append(flagged)
        print(
            f"  SKIP not converged: {result.exploitability_pct_of_pot:.3f}% pot "
            f"(target {target_pct}%) after {result.iterations_used} iters, {elapsed:.1f}s",
            flush=True,
        )
        return False

    try:
        if prof_measure_encode is not None:
            with prof_measure_encode:
                record = encode_solver_spot_row(
                    spot,
                    result,
                    mapper=mapper,
                    bet_sizes=bet_sizes_dsl,
                    raise_sizes=raise_sizes_dsl,
                    target_exploitability_pct=target_pct,
                )
                row = record_to_parquet_dict(record)
        else:
            record = encode_solver_spot_row(
                spot,
                result,
                mapper=mapper,
                bet_sizes=bet_sizes_dsl,
                raise_sizes=raise_sizes_dsl,
                target_exploitability_pct=target_pct,
            )
            row = record_to_parquet_dict(record)
    except ValueError as exc:
        state.skipped_solver["encode_rejected"] = state.skipped_solver.get("encode_rejected", 0) + 1
        print(f"  SKIP encode: {exc}", flush=True)
        return False

    state.included_rows.append(row)
    state.convergence_stats.append(record.solver_exploitability_pct)
    if len(state.sample_labels) < 5:
        state.sample_labels.append(
            {
                "spot_id": spot.spot_id,
                "spot_class": spot.spot_class.value,
                "board": [c.to_str() for c in spot.board_cards],
                "hero_hole": [c.to_str() for c in spot.hero_hole],
                "pot_bb": spot.starting_pot_bb,
                "converged": result.converged,
                "iterations_used": result.iterations_used,
                "exploitability_pct_of_pot": result.exploitability_pct_of_pot,
                "action_distribution": result.action_distribution,
            }
        )
    print(
        f"  OK converged={result.converged} exploit={result.exploitability_pct_of_pot:.3f}% "
        f"iters={result.iterations_used} time={elapsed:.1f}s",
        flush=True,
    )
    return True


def apply_solve_error(
    *,
    spot_key: str,
    error_kind: str,
    error_msg: str,
    state: LabelBuildState,
    timeout_sec: float | None = None,
) -> None:
    if error_kind == "TimeoutExpired":
        state.skipped_solver["timeout"] = state.skipped_solver.get("timeout", 0) + 1
        state.per_spot_results.append(
            {
                "spot": spot_key,
                "status": "timeout",
                "flagged_failure": True,
                "failure_reason": "timeout",
                "timeout_sec": timeout_sec,
            }
        )
        print(f"  FLAG timeout after {timeout_sec}s backstop", flush=True)
        return
    state.skipped_solver[error_kind] = state.skipped_solver.get(error_kind, 0) + 1
    state.per_spot_results.append({"spot": spot_key, "status": error_kind, "error": error_msg[:200]})
