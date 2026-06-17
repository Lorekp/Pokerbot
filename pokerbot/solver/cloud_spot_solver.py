"""In-process single-spot solver for cloud_label_worker."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from pokerbot.encoding.weva_encoder import WEVAMapper
from pokerbot.solver.export import write_shard
from pokerbot.solver.label_build_state import LabelBuildState, apply_solve_result
from pokerbot.solver.label_completion import spot_key_hash
from pokerbot.solver.postflop_solver_client import (
    bet_sizes_string,
    resolve_cli_path,
    solve_spot,
)
from pokerbot.solver.solver_worker import PersistentSolverWorker, WorkerError
from pokerbot.solver.spot_enumerator import (
    SolverSpot,
    enumerate_spots,
    load_solver_config,
    spot_key as format_spot_key,
)
from pokerbot.solver.spot_validity import is_two_player_valid, spot_node_from_state


def board_timing_from_manifest(manifest: dict[str, Any]) -> dict[str, float]:
    """Extract per-board median seconds from work manifest or nested timing source."""
    timing = manifest.get("board_timing")
    if not isinstance(timing, dict):
        return {}
    out: dict[str, float] = {}
    for board, stats in timing.items():
        if isinstance(stats, dict) and "median_sec" in stats:
            out[str(board)] = float(stats["median_sec"])
    return out


def estimate_spot_seconds(spot_key: str, board_timing: dict[str, float]) -> float:
    from pokerbot.solver.label_completion import parse_spot_key

    parsed = parse_spot_key(spot_key)
    if parsed is None:
        return 0.0
    return board_timing.get(parsed["board"], 0.0)


def sort_spot_keys_slow_first(spot_keys: list[str], board_timing: dict[str, float]) -> list[str]:
    return sorted(spot_keys, key=lambda k: estimate_spot_seconds(k, board_timing), reverse=True)


def slow_board_threshold_sec(board_timing: dict[str, float]) -> float:
    """Median of per-board median solve times — boards above are 'slow' bucket."""
    if not board_timing:
        return 120.0
    values = sorted(board_timing.values())
    return values[len(values) // 2]


def order_pending_for_worker(
    pending_keys: list[str],
    worker_id: str,
    *,
    board_timing: dict[str, float] | None = None,
    slow_threshold_sec: float | None = None,
) -> list[str]:
    """Per-worker manifest walk order: slow-bucket LPT first, shuffled within buckets."""
    import hashlib
    import random

    if not pending_keys:
        return []
    threshold = slow_threshold_sec
    if threshold is None and board_timing:
        threshold = slow_board_threshold_sec(board_timing)
    elif threshold is None:
        threshold = 120.0

    seed = int(hashlib.sha256(worker_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)

    if board_timing:
        slow_bucket: list[str] = []
        fast_bucket: list[str] = []
        for key in pending_keys:
            if estimate_spot_seconds(key, board_timing) >= threshold:
                slow_bucket.append(key)
            else:
                fast_bucket.append(key)
        slow_bucket = sort_spot_keys_slow_first(slow_bucket, board_timing)
        fast_bucket = sort_spot_keys_slow_first(fast_bucket, board_timing)
        rng.shuffle(slow_bucket)
        rng.shuffle(fast_bucket)
        return slow_bucket + fast_bucket

    ordered = list(pending_keys)
    rng.shuffle(ordered)
    return ordered


class CloudSpotSolver:
    """Long-lived solver resources: WEVA mapper, optional persistent Rust worker, spot lookup."""

    def __init__(
        self,
        *,
        config_path: Path,
        weva_index: Path,
        shard_dir: Path,
        cli_path: Path | None = None,
        solver_backend: str = "worker",
        solver_threads_per_worker: int = 1,
        spot_keys: set[str] | None = None,
        timeout_sec: float = 3600.0,
        board_timing: dict[str, float] | None = None,
    ) -> None:
        self._config_path = config_path
        self._shard_dir = shard_dir
        self._solver_backend = solver_backend
        self._solver_threads_per_worker = solver_threads_per_worker
        self._timeout_sec = timeout_sec
        self._raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self._config = load_solver_config(config_path)
        self._mapper = WEVAMapper.from_directory(weva_index)
        bet_tree = self._raw_cfg.get("bet_tree", {})
        fractions = [float(x) for x in bet_tree.get("pot_fractions", [0.5])]
        self._bet_sizes_dsl = bet_sizes_string(
            fractions, bool(bet_tree.get("include_all_in", True))
        )
        self._raise_sizes_dsl = f"{float(bet_tree.get('raise_multiplier', 2.5))}x"
        solver_cfg = self._raw_cfg.get("solver", {})
        self._target_pct = float(solver_cfg.get("target_exploitability_pct", 0.5))
        self._max_iterations = int(solver_cfg.get("max_iterations", 10000))
        self._slow_board_max_iterations = solver_cfg.get("slow_board_max_iterations")
        self._slow_board_median_sec = float(solver_cfg.get("slow_board_median_sec", 120.0))
        self._board_timing: dict[str, float] = {}
        self._include_unconverged = bool(solver_cfg.get("include_unconverged", False))
        self._cli_path = resolve_cli_path(cli_path)
        self._spot_lookup = self._build_spot_lookup(spot_keys)
        self._board_timing = board_timing or {}
        self._worker: PersistentSolverWorker | None = None
        if solver_backend == "worker":
            self._worker = PersistentSolverWorker(
                cli_path=self._cli_path,
                solver_threads_per_worker=solver_threads_per_worker,
            )

    def _build_spot_lookup(self, spot_keys: set[str] | None) -> dict[str, SolverSpot]:
        lookup: dict[str, SolverSpot] = {}
        for spot in enumerate_spots(self._config):
            key = format_spot_key(spot)
            if spot_keys is not None and key not in spot_keys:
                continue
            validity = is_two_player_valid(
                spot_node_from_state(spot.state, spot.hero_seat, bunching=spot.bunching)
            )
            if not validity.valid:
                continue
            lookup[key] = spot
        return lookup

    def _recreate_worker(self) -> None:
        if self._worker is not None:
            self._worker.close()
        self._worker = PersistentSolverWorker(
            cli_path=self._cli_path,
            solver_threads_per_worker=self._solver_threads_per_worker,
        )

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def __enter__(self) -> CloudSpotSolver:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def solve(self, spot_key: str, *, request_id: str | None = None) -> dict[str, Any]:
        """Solve one spot and write shard. Returns metadata for S3 coordination."""
        spot = self._spot_lookup.get(spot_key)
        if spot is None:
            return {
                "ok": False,
                "spot_key": spot_key,
                "error": "spot_not_in_lookup",
            }

        state = LabelBuildState(
            included_rows=[],
            excluded_by_reason={},
            skipped_solver={},
            not_converged_details=[],
            per_spot_results=[],
            first_solver_error=None,
            convergence_stats=[],
            sample_labels=[],
            solve_times_sec=[],
        )

        t0 = time.perf_counter()
        max_iters = self._max_iterations
        if self._slow_board_max_iterations is not None:
            board = "".join(c.to_str() for c in spot.board_cards)
            median = self._board_timing.get(board, 0.0)
            if median >= self._slow_board_median_sec:
                max_iters = min(max_iters, int(self._slow_board_max_iterations))
        try:
            if self._solver_backend == "worker":
                assert self._worker is not None
                result = solve_spot(
                    spot,
                    self._raw_cfg,
                    worker=self._worker,
                    request_id=request_id or spot_key,
                    max_iterations_override=max_iters,
                )
            else:
                result = solve_spot(
                    spot,
                    self._raw_cfg,
                    cli_path=self._cli_path,
                    request_id=request_id or spot_key,
                    solver_threads_per_worker=self._solver_threads_per_worker,
                    max_iterations_override=max_iters,
                )
        except WorkerError as exc:
            elapsed = time.perf_counter() - t0
            if exc.code == "timeout" and self._solver_backend == "worker":
                self._recreate_worker()
            return {
                "ok": False,
                "spot_key": spot_key,
                "elapsed_sec": elapsed,
                "error": exc.code,
                "message": str(exc)[:500],
            }
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return {
                "ok": False,
                "spot_key": spot_key,
                "elapsed_sec": elapsed,
                "error": type(exc).__name__,
                "message": str(exc)[:500],
            }

        elapsed = time.perf_counter() - t0
        included = apply_solve_result(
            spot=spot,
            spot_key=spot_key,
            result=result,
            elapsed=elapsed,
            state=state,
            mapper=self._mapper,
            bet_sizes_dsl=self._bet_sizes_dsl,
            raise_sizes_dsl=self._raise_sizes_dsl,
            target_pct=self._target_pct,
            include_unconverged=self._include_unconverged,
        )

        if not included:
            per_spot = state.per_spot_results[-1] if state.per_spot_results else {}
            return {
                "ok": False,
                "spot_key": spot_key,
                "elapsed_sec": elapsed,
                "converged": per_spot.get("converged", False),
                "exploitability_pct": per_spot.get("exploitability_pct_of_pot"),
                "iterations_used": per_spot.get("iterations_used"),
                "error": per_spot.get("failure_reason") or per_spot.get("error") or "not_included",
            }

        row = state.included_rows[-1]
        shard_path = write_shard(
            row,
            self._shard_dir,
            spot_key,
            spot_key_hash_fn=spot_key_hash,
        )
        per_spot = state.per_spot_results[-1]
        return {
            "ok": True,
            "spot_key": spot_key,
            "elapsed_sec": elapsed,
            "exploitability_pct": per_spot.get("exploitability_pct_of_pot"),
            "iterations_used": per_spot.get("iterations_used"),
            "converged": per_spot.get("converged", True),
            "near_timeout": elapsed >= 0.95 * self._timeout_sec,
            "shard_path": str(shard_path),
        }
