#!/usr/bin/env python3
"""Build 2-player-valid postflop solver labels → data/solver/train.parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from aof_rl.profiling.timer import TimingAccumulator

from pokerbot.encoding.postflop_solver_encoder import LABEL_SOURCE_POSTFLOP_SOLVER
from pokerbot.encoding.weva_encoder import WEVAMapper
from pokerbot.solver.export import merge_shards, write_shard, write_solver_parquet
from pokerbot.solver.label_build_state import LabelBuildState, apply_solve_error, apply_solve_result
from pokerbot.solver.label_completion import (
    CompletionLedger,
    analyze_shard_completion,
    bootstrap_and_validate,
    has_shard,
    spot_key_hash,
    unique_manifest_keys,
    write_completion_report,
)
from pokerbot.solver.postflop_solver_client import (
    DEFAULT_CLI_PATH,
    bet_sizes_string,
    solve_spot,
)
from pokerbot.solver.spot_enumerator import SolverSpot, enumerate_spots, load_solver_config, spot_key as format_spot_key
from pokerbot.solver.spot_validity import is_two_player_valid, spot_node_from_state

DISCLAIMER = (
    "2-player-valid GTO labels only; postflop HU scope in v1; "
    "not valid supervision for multiway 6-max nodes."
)


def _persist_completed_spot(
    *,
    spot_key_str: str,
    row: dict[str, Any],
    result_exploit: float | None,
    result_iters: int | None,
    elapsed: float,
    ledger: CompletionLedger,
    shard_dir: Path,
    resume: bool,
) -> None:
    write_shard(row, shard_dir, spot_key_str, spot_key_hash_fn=spot_key_hash)
    if resume:
        ledger.append(
            spot_key_str,
            exploitability_pct=result_exploit,
            iterations_used=result_iters,
            elapsed_sec=elapsed,
            source="solve",
        )


def _config_hash(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]


def _solver_git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "ls-remote", "https://github.com/b-inary/postflop-solver", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return out.stdout.split()[0][:12] if out.stdout.strip() else "unknown"
    except Exception:
        return "postflop-solver@main"


def _solver_config_snapshot(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    bet_tree = raw_cfg.get("bet_tree", {})
    solver_cfg = raw_cfg.get("solver", {})
    fractions = [float(x) for x in bet_tree.get("pot_fractions", [0.5])]
    raise_mult = float(bet_tree.get("raise_multiplier", 2.5))
    return {
        "ranges": raw_cfg.get("ranges", {}),
        "bet_tree": bet_tree,
        "bet_sizes_dsl": bet_sizes_string(fractions, bool(bet_tree.get("include_all_in", True))),
        "raise_sizes_dsl": f"{raise_mult}x",
        "max_iterations": int(solver_cfg.get("max_iterations", 10000)),
        "target_exploitability_pct": float(solver_cfg.get("target_exploitability_pct", 0.5)),
        "timeout_sec": float(solver_cfg.get("timeout_sec", 3600)),
        "include_unconverged": bool(solver_cfg.get("include_unconverged", False)),
        "label_validity_note": (
            "Each label is GTO for the configured OOP/IP ranges and bet tree only — "
            "not unconditional full-range HU GTO."
        ),
    }


def build_labels(
    config_path: Path,
    output: Path,
    manifest_path: Path,
    report_path: Path,
    weva_index: Path,
    cli_path: Path | None,
    dry_run: bool = False,
    max_spots: int | None = None,
    allow_plumbing: bool = False,
    prof: TimingAccumulator | None = None,
    cprofile_spot: int | None = None,
    solver_backend: str = "subprocess",
    num_workers: int = 1,
    max_in_flight: int | None = None,
    solver_threads_per_worker: int | None = None,
    disable_worker_cache: bool = False,
    worker_cache_size: int | None = None,
    preserve_output_order: bool = True,
    resume: bool = False,
    completion_ledger: Path | None = None,
    bootstrap_log: Path | None = None,
    bootstrap_expected_count: int | None = None,
    shard_dir: Path | None = None,
    spot_keys_filter: set[str] | None = None,
) -> dict[str, Any]:
    raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = load_solver_config(config_path)
    mapper = WEVAMapper.from_directory(weva_index)
    solver_snapshot = _solver_config_snapshot(raw_cfg)
    solver_cfg = raw_cfg.get("solver", {})
    include_unconverged = bool(solver_cfg.get("include_unconverged", False))
    gto_acceptable = not include_unconverged

    if include_unconverged and not allow_plumbing:
        raise SystemExit(
            "ERROR: include_unconverged=true requires --allow-plumbing. "
            "Unconverged snapshots are NOT GTO and must not land in train.parquet."
        )
    if include_unconverged and output.name == "train.parquet":
        raise SystemExit(
            "ERROR: refuse to write unconverged/plumbing labels to train.parquet. "
            "Use --output data/solver/train_plumbing_NON_GTO.parquet"
        )

    bet_tree = raw_cfg.get("bet_tree", {})
    fractions = [float(x) for x in bet_tree.get("pot_fractions", [0.5])]
    bet_sizes_dsl = bet_sizes_string(fractions, bool(bet_tree.get("include_all_in", True)))
    raise_sizes_dsl = f"{float(bet_tree.get('raise_multiplier', 2.5))}x"
    target_pct = float(solver_cfg.get("target_exploitability_pct", 0.5))

    if shard_dir is None:
        shard_dir = output.parent / f"{output.stem}_shards"
    if completion_ledger is None:
        completion_ledger = output.parent / f"{output.stem}_completed.jsonl"

    ledger = CompletionLedger.load(completion_ledger) if resume else CompletionLedger(path=completion_ledger)
    bootstrap_result = None
    if bootstrap_log is not None:
        bootstrap_result = bootstrap_and_validate(
            bootstrap_log,
            ledger_path=completion_ledger,
            shard_dir=shard_dir,
            parquet_path=output if output.is_file() else None,
            expected_count=bootstrap_expected_count,
            write_ledger=False,
        )
        if bootstrap_expected_count is not None and not bootstrap_result.ok:
            raise SystemExit(
                f"ERROR: bootstrap validation failed — expected {bootstrap_expected_count}, "
                f"got {bootstrap_result.valid_count} valid keys. "
                f"invalid={len(bootstrap_result.invalid_keys)} "
                f"unpaired={len(bootstrap_result.unpaired_solving)} "
                f"duplicates={len(bootstrap_result.duplicate_keys)}"
            )
        if resume:
            ledger = CompletionLedger.load(completion_ledger)

    skipped_resume = 0

    included_rows: list[dict[str, Any]] = []
    excluded_by_reason: dict[str, int] = {}
    skipped_solver: dict[str, int] = {}
    not_converged_details: list[dict[str, Any]] = []
    per_spot_results: list[dict[str, Any]] = []
    first_solver_error: str | None = None
    convergence_stats: list[float] = []
    sample_labels: list[dict[str, Any]] = []
    total_candidates = 0
    solve_times_sec: list[float] = []
    pool_stats: dict[str, Any] | None = None

    import time

    state = LabelBuildState(
        included_rows=included_rows,
        excluded_by_reason=excluded_by_reason,
        skipped_solver=skipped_solver,
        not_converged_details=not_converged_details,
        per_spot_results=per_spot_results,
        first_solver_error=first_solver_error,
        convergence_stats=convergence_stats,
        sample_labels=sample_labels,
        solve_times_sec=solve_times_sec,
    )

    # Pre-enumerate valid candidates with stable indices.
    candidates: list[tuple[int, int, str, SolverSpot]] = []
    for spot in enumerate_spots(config):
        if max_spots is not None and total_candidates >= max_spots:
            break
        total_candidates += 1
        spot_idx = total_candidates - 1
        key = format_spot_key(spot)
        if spot_keys_filter is not None and key not in spot_keys_filter:
            continue
        if resume and has_shard(shard_dir, key):
            skipped_resume += 1
            continue
        if prof is not None:
            with prof.measure("spot.validity_check"):
                validity = is_two_player_valid(
                    spot_node_from_state(spot.state, spot.hero_seat, bunching=spot.bunching)
                )
        else:
            validity = is_two_player_valid(
                spot_node_from_state(spot.state, spot.hero_seat, bunching=spot.bunching)
            )
        if not validity.valid:
            excluded_by_reason[validity.reason] = excluded_by_reason.get(validity.reason, 0) + 1
            continue
        if dry_run:
            included_rows.append({"spot_id": spot.spot_id, "dry_run": True})
            continue
        candidates.append((spot_idx, total_candidates, key, spot))

    timeout_sec = float(solver_cfg.get("timeout_sec", 3600))
    use_pool = solver_backend == "worker" and num_workers > 1 and candidates

    if prof is not None:
        prof.start_wall()

    if use_pool:
        from pokerbot.solver.solver_worker_pool import SolverWorkerPool, build_job_from_spot

        jobs = [
            build_job_from_spot(spot_idx, cand_num, spot_key, spot, raw_cfg)
            for spot_idx, cand_num, spot_key, spot in candidates
        ]
        with SolverWorkerPool(
            num_workers=num_workers,
            cli_path=cli_path,
            max_in_flight=max_in_flight,
            solver_threads_per_worker=solver_threads_per_worker,
            disable_worker_cache=disable_worker_cache,
            worker_cache_size=worker_cache_size,
            timeout_sec=timeout_sec,
        ) as pool:
            if prof is not None:
                with prof.measure("spot.solve_subprocess"):
                    outcomes = pool.solve_jobs(jobs)
            else:
                outcomes = pool.solve_jobs(jobs)
            pool_stats = {
                "num_workers": pool.stats.num_workers,
                "max_in_flight": pool.stats.max_in_flight,
                "solver_threads_per_worker": pool.stats.solver_threads_per_worker,
                "jobs_submitted": pool.stats.jobs_submitted,
                "jobs_completed": pool.stats.jobs_completed,
                "queue_wait_sec": round(pool.stats.queue_wait_sec, 4),
            }
        ordered = sorted(candidates, key=lambda c: c[0])
        for spot_idx, cand_num, spot_key, spot in ordered:
            print(f"[{cand_num}] solving {spot_key}", flush=True)
            outcome = outcomes.get(spot_idx)
            if outcome is None:
                apply_solve_error(
                    spot_key=spot_key,
                    error_kind="MissingOutcome",
                    error_msg="no pool outcome",
                    state=state,
                )
                continue
            if outcome.error is not None:
                apply_solve_error(
                    spot_key=spot_key,
                    error_kind=outcome.error_kind or type(outcome.error).__name__,
                    error_msg=str(outcome.error),
                    state=state,
                )
                continue
            if outcome.result is None:
                apply_solve_error(
                    spot_key=spot_key,
                    error_kind="MissingResult",
                    error_msg="empty result",
                    state=state,
                )
                continue
            encode_ctx = prof.measure("spot.encode_row") if prof is not None else None
            included = apply_solve_result(
                spot=spot,
                spot_key=spot_key,
                result=outcome.result,
                elapsed=outcome.elapsed_sec,
                state=state,
                mapper=mapper,
                bet_sizes_dsl=bet_sizes_dsl,
                raise_sizes_dsl=raise_sizes_dsl,
                target_pct=target_pct,
                include_unconverged=include_unconverged,
                prof_measure_encode=encode_ctx,
            )
            if included:
                row = state.included_rows[-1]
                _persist_completed_spot(
                    spot_key_str=spot_key,
                    row=row,
                    result_exploit=outcome.result.exploitability_pct_of_pot,
                    result_iters=outcome.result.iterations_used,
                    elapsed=outcome.elapsed_sec,
                    ledger=ledger,
                    shard_dir=shard_dir,
                    resume=resume,
                )
            if included and prof is not None:
                prof.inc_counter("spots_solved", 1)
    else:
        worker = None
        if solver_backend == "worker":
            from pokerbot.solver.solver_worker import PersistentSolverWorker

            worker = PersistentSolverWorker(
                cli_path=cli_path,
                solver_threads_per_worker=solver_threads_per_worker,
                disable_worker_cache=disable_worker_cache,
                worker_cache_size=worker_cache_size,
            )
        elif solver_backend != "subprocess":
            raise ValueError(f"unknown solver_backend: {solver_backend}")

        try:
            for spot_idx, cand_num, spot_key, spot in candidates:
                if prof is not None:
                    prof.begin_spot(spot_idx, spot_key, spot.spot_id)

                do_cprof = cprofile_spot is not None and spot_idx == cprofile_spot
                if do_cprof:
                    from aof_rl.profiling.league_profiler import start_cprofile

                    start_cprofile()
                try:
                    print(f"[{cand_num}] solving {spot_key}", flush=True)
                    t0 = time.perf_counter()
                    try:
                        if prof is not None:
                            with prof.measure("spot.solve_subprocess"):
                                result = solve_spot(
                                    spot,
                                    raw_cfg,
                                    cli_path=cli_path,
                                    worker=worker,
                                    request_id=str(spot_idx),
                                    solver_threads_per_worker=solver_threads_per_worker,
                                )
                        else:
                            result = solve_spot(
                                spot,
                                raw_cfg,
                                cli_path=cli_path,
                                worker=worker,
                                request_id=str(spot_idx),
                                solver_threads_per_worker=solver_threads_per_worker,
                            )
                    except subprocess.TimeoutExpired:
                        apply_solve_error(
                            spot_key=spot_key,
                            error_kind="TimeoutExpired",
                            error_msg="timeout",
                            state=state,
                            timeout_sec=timeout_sec,
                        )
                        continue
                    except Exception as exc:
                        apply_solve_error(
                            spot_key=spot_key,
                            error_kind=type(exc).__name__,
                            error_msg=str(exc),
                            state=state,
                        )
                        continue
                    elapsed = time.perf_counter() - t0
                    encode_ctx = prof.measure("spot.encode_row") if prof is not None else None
                    included = apply_solve_result(
                        spot=spot,
                        spot_key=spot_key,
                        result=result,
                        elapsed=elapsed,
                        state=state,
                        mapper=mapper,
                        bet_sizes_dsl=bet_sizes_dsl,
                        raise_sizes_dsl=raise_sizes_dsl,
                        target_pct=target_pct,
                        include_unconverged=include_unconverged,
                        prof_measure_encode=encode_ctx,
                    )
                    if included:
                        row = state.included_rows[-1]
                        _persist_completed_spot(
                            spot_key_str=spot_key,
                            row=row,
                            result_exploit=result.exploitability_pct_of_pot,
                            result_iters=result.iterations_used,
                            elapsed=elapsed,
                            ledger=ledger,
                            shard_dir=shard_dir,
                            resume=resume,
                        )
                    if included and prof is not None:
                        prof.inc_counter("spots_solved", 1)
                finally:
                    if do_cprof:
                        from aof_rl.profiling.league_profiler import stop_cprofile

                        stop_cprofile()
                    if prof is not None:
                        prof.end_spot()
        finally:
            if worker is not None:
                worker.close()

    first_solver_error = state.first_solver_error

    # Stable ordering for manifest regardless of parallel completion order.
    per_spot_results.sort(key=lambda r: str(r.get("spot", "")))

    converged_count = sum(1 for r in included_rows if r.get("solver_converged"))
    flagged_failures = [
        r for r in per_spot_results if r.get("flagged_failure") or r.get("status") in ("timeout",)
    ]
    if not flagged_failures:
        flagged_failures = [
            r for r in per_spot_results
            if r.get("converged") is False or r.get("failure_reason")
        ]

    # Board-level timing for slow-converger diagnosis
    board_times: dict[str, list[float]] = {}
    for r in per_spot_results:
        if "elapsed_sec" not in r:
            continue
        spot_str = str(r.get("spot", ""))
        board_key = spot_str.split("|")[1] if "|" in spot_str else "unknown"
        board_times.setdefault(board_key, []).append(float(r["elapsed_sec"]))
    board_timing = {
        board: {
            "count": len(times),
            "median_sec": sorted(times)[len(times) // 2],
            "max_sec": max(times),
        }
        for board, times in sorted(board_times.items())
    }

    range_scopes: dict[str, int] = {}
    for r in included_rows:
        key = (
            f"spot_class={r.get('spot_class')}|oop={r.get('oop_range')}|"
            f"ip={r.get('ip_range')}|bet={r.get('bet_sizes')}"
        )
        range_scopes[key] = range_scopes.get(key, 0) + 1

    completed_without_shard = ledger.keys_without_shards(shard_dir) if resume else []

    scope_keys: list[str] = []
    if spot_keys_filter is not None:
        scope_keys = sorted(spot_keys_filter)
    else:
        for spot in enumerate_spots(config):
            key = format_spot_key(spot)
            validity = is_two_player_valid(
                spot_node_from_state(spot.state, spot.hero_seat, bunching=spot.bunching)
            )
            if validity.valid:
                scope_keys.append(key)
        if max_spots is not None:
            scope_keys = scope_keys[:max_spots]

    scope_unique = unique_manifest_keys(scope_keys)
    shard_completion = analyze_shard_completion(
        manifest_keys=scope_unique,
        shard_dir=shard_dir,
        ledger_path=completion_ledger if resume else None,
    )
    shard_complete = shard_completion.shard_backed >= shard_completion.expected_count

    flagged_count = len(flagged_failures)
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
        "gto_acceptable": gto_acceptable,
        "not_for_sft": not gto_acceptable,
        "config_path": str(config_path),
        "config_hash": _config_hash(config_path),
        "seed": config.seed,
        "enumeration_mode": config.enumeration_mode,
        "disable_bunching": config.disable_bunching,
        "label_source": LABEL_SOURCE_POSTFLOP_SOLVER,
        "solver_engine": "postflop-solver",
        "solver_git_rev": _solver_git_rev(),
        "cli_path": str(cli_path or DEFAULT_CLI_PATH),
        "build_method": f"rust_{solver_backend} (bincode disabled; rayon; compressed memory)",
        "solver_orchestration": {
            "backend": solver_backend,
            "num_workers": num_workers if solver_backend == "worker" else 1,
            "max_in_flight": max_in_flight,
            "solver_threads_per_worker": solver_threads_per_worker,
            "disable_worker_cache": disable_worker_cache,
            "worker_cache_size": worker_cache_size,
            "preserve_output_order": preserve_output_order,
            "pool_stats": pool_stats,
        },
        "scope_v1": "postflop_hu_only — preflop HU deferred to hu_preflop_dcfr",
        "solver_config": solver_snapshot,
        "enumeration": {
            "total_candidates": total_candidates,
            "included": len(included_rows),
            "excluded": sum(excluded_by_reason.values()),
            "excluded_by_reason": excluded_by_reason,
            "skipped_solver": skipped_solver,
            "skipped_resume": skipped_resume,
            "first_solver_error": first_solver_error,
            "not_converged_details": not_converged_details,
            "flagged_failures": flagged_failures,
            "flagged_failure_count": flagged_count,
            "label_set_note": (
                f"{len(included_rows)} converged labels written; "
                f"{total_candidates} candidates attempted; "
                f"{skipped_resume} skipped (resume ledger); "
                f"{flagged_count} flagged failure(s) excluded from Parquet."
            ),
        },
        "resume": {
            "enabled": resume,
            "completion_ledger": str(completion_ledger),
            "shard_dir": str(shard_dir),
            "ledger_completed": len(ledger.completed),
            "completed_without_shard": len(completed_without_shard),
            "shard_authority": True,
            "shard_completion": shard_completion.to_dict(),
            "shard_complete": shard_complete,
            "scope_unique_keys": len(scope_unique),
            "scope_manifest_entries": len(scope_keys),
            "bootstrap_log": str(bootstrap_log) if bootstrap_log else None,
            "bootstrap_valid_count": (
                bootstrap_result.valid_count if bootstrap_result is not None else None
            ),
            "bootstrap_ok": bootstrap_result.ok if bootstrap_result is not None else None,
        },
        "range_scope_distribution": range_scopes,
        "board_timing": board_timing,
        "convergence": {
            "count": len(convergence_stats),
            "converged_count": converged_count,
            "attempted": total_candidates,
            "converged_of_attempted": f"{len(included_rows)}/{total_candidates}",
            "pct_converged_of_written": (
                100.0 * converged_count / len(included_rows) if included_rows else 0.0
            ),
            "median_exploitability_pct": (
                sorted(convergence_stats)[len(convergence_stats) // 2] if convergence_stats else None
            ),
            "max_exploitability_pct": max(convergence_stats) if convergence_stats else None,
            "median_solve_time_sec": (
                sorted(solve_times_sec)[len(solve_times_sec) // 2] if solve_times_sec else None
            ),
            "max_solve_time_sec": max(solve_times_sec) if solve_times_sec else None,
        },
        "per_spot_results": per_spot_results,
        "sample_labels": sample_labels,
    }

    wrote_parquet = False
    shard_row_count = 0
    if not dry_run and gto_acceptable:
        has_shards = shard_dir.is_dir() and any(
            p.suffix == ".parquet" and not p.name.endswith(".tmp")
            for p in shard_dir.iterdir()
        )
        if has_shards:
            shard_row_count = merge_shards(shard_dir, output, spot_key_hash_fn=spot_key_hash)
            wrote_parquet = shard_row_count > 0
        elif included_rows:
            if prof is not None:
                with prof.measure("export.parquet_write"):
                    write_solver_parquet(included_rows, output)
            else:
                write_solver_parquet(included_rows, output)
            wrote_parquet = True
            shard_row_count = len(included_rows)
    elif not dry_run and included_rows and allow_plumbing:
        if prof is not None:
            with prof.measure("export.parquet_write"):
                write_solver_parquet(included_rows, output)
        else:
            write_solver_parquet(included_rows, output)
        wrote_parquet = True
        shard_row_count = len(included_rows)

    if prof is not None:
        prof.stop_wall()

    manifest["output"] = {
        "path": str(output),
        "written": wrote_parquet,
        "row_count": shard_row_count if wrote_parquet else len(included_rows),
        "shard_row_count": shard_row_count,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    _write_report(report_path, manifest, excluded_by_reason, skipped_solver, sample_labels)

    if not dry_run and gto_acceptable and len(included_rows) == 0:
        print(
            "\n*** FAILURE: zero converged GTO labels — no Parquet written for SFT. ***",
            file=sys.stderr,
        )
        print(
            f"Skipped: {json.dumps(skipped_solver)}",
            file=sys.stderr,
        )
        if not_converged_details:
            print(
                f"Example not-converged: {json.dumps(not_converged_details[0], indent=2)}",
                file=sys.stderr,
            )

    if resume and scope_keys and not shard_complete:
        manifest["shard_incomplete"] = True
        print(
            f"\n*** INCOMPLETE: {shard_completion.shard_backed}/{shard_completion.expected_count} "
            f"shard-backed spots (todo={shard_completion.todo_count}). ***",
            file=sys.stderr,
        )

    return manifest


def _write_report(
    report_path: Path,
    manifest: dict[str, Any],
    excluded_by_reason: dict[str, int],
    skipped_solver: dict[str, int],
    sample_labels: list[dict[str, Any]],
) -> None:
    conv = manifest["convergence"]
    lines = [
        "# Solver label pipeline run report",
        "",
        f"Generated: {manifest['generated_at']}",
        "",
        f"**gto_acceptable:** {manifest['gto_acceptable']} | **not_for_sft:** {manifest['not_for_sft']}",
        "",
        "## Scope",
        "",
        DISCLAIMER,
        "",
        manifest["solver_config"]["label_validity_note"],
        "",
        f"- Ranges: `{manifest['solver_config']['ranges']}`",
        f"- Bet tree: `{manifest['solver_config']['bet_sizes_dsl']}` / `{manifest['solver_config']['raise_sizes_dsl']}`",
        f"- Target: {manifest['solver_config']['target_exploitability_pct']}% of pot",
        f"- Max iterations: {manifest['solver_config']['max_iterations']}",
        "",
        "## Enumeration",
        "",
        f"- Total candidates: {manifest['enumeration']['total_candidates']}",
        f"- Included (written): {manifest['enumeration']['included']}",
        f"- Excluded (scope guard): {manifest['enumeration']['excluded']}",
        "",
        "### Skipped at solver",
        "",
    ]
    for reason, count in sorted(skipped_solver.items()):
        lines.append(f"- `{reason}`: {count}")
    lines.extend(
        [
            "",
            "## Convergence",
            "",
            f"- Converged rows: {conv.get('converged_count', 0)} / {conv.get('attempted', conv.get('count', 0))} attempted",
            f"- Median exploitability % of pot: {conv.get('median_exploitability_pct')}",
            f"- Max exploitability % of pot: {conv.get('max_exploitability_pct')}",
            f"- Median solve time (s): {conv.get('median_solve_time_sec')}",
            f"- Max solve time (s): {conv.get('max_solve_time_sec')}",
            "",
        ]
    )
    flagged = manifest["enumeration"].get("flagged_failures", [])
    if flagged:
        lines.extend(["## Flagged failures (excluded from Parquet)", ""])
        for row in flagged:
            lines.append(
                f"- {row.get('spot')}: {row.get('failure_reason') or row.get('status')} "
                f"exploit={row.get('exploitability_pct_of_pot')}% iters={row.get('iterations_used')}"
            )
        lines.append("")
    board_timing = manifest.get("board_timing", {})
    if board_timing:
        lines.extend(["## Board timing", ""])
        for board, stats in board_timing.items():
            lines.append(
                f"- {board}: median={stats['median_sec']:.1f}s max={stats['max_sec']:.1f}s n={stats['count']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Per-spot results",
            "",
        ]
    )
    for row in manifest.get("per_spot_results", []):
        lines.append(
            f"- {row.get('spot')}: converged={row.get('converged')} "
            f"exploit={row.get('exploitability_pct_of_pot')}% iters={row.get('iterations_used')} "
            f"time={row.get('elapsed_sec')}s"
        )
    lines.extend(["", "## Sample labels", ""])
    for i, sample in enumerate(sample_labels, 1):
        lines.append(f"### Sample {i}: {sample['spot_id']}")
        lines.append(f"- Board: {sample['board']}")
        lines.append(f"- Converged: {sample['converged']} @ {sample.get('iterations_used')} iters")
        lines.append(f"- Exploitability: {sample.get('exploitability_pct_of_pot')}% of pot")
        lines.append("- action_distribution:")
        for k, v in sorted(sample["action_distribution"].items(), key=lambda x: -x[1]):
            lines.append(f"  - {k}: {v:.4f}")
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Build postflop-solver training labels")
    p.add_argument("--config", type=Path, default=Path("config/solver_labels.yaml"))
    p.add_argument("--output", type=Path, default=Path("data/solver/train.parquet"))
    p.add_argument("--manifest", type=Path, default=Path("data/solver/manifest.json"))
    p.add_argument("--report", type=Path, default=Path("logs/solver_label_run.md"))
    p.add_argument("--weva-index", type=Path, default=Path("data/weva"))
    p.add_argument("--cli-path", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="Enumerate only, no solver calls")
    p.add_argument("--max-spots", type=int, default=None, help="Cap solver invocations (dev/CI)")
    p.add_argument(
        "--allow-plumbing",
        action="store_true",
        help="Allow include_unconverged=true (NON-GTO snapshots; not for SFT)",
    )
    p.add_argument(
        "--solver-backend",
        choices=["subprocess", "worker"],
        default="subprocess",
        help="Solver orchestration backend (subprocess=one-shot per spot, worker=persistent JSONL)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Worker processes when --solver-backend=worker (use >1 for parallel solves)",
    )
    p.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Max concurrent in-flight solves (default: 2 * num-workers)",
    )
    p.add_argument(
        "--solver-threads-per-worker",
        type=int,
        default=None,
        help="RAYON_NUM_THREADS per worker process (default: cpu_count // num_workers)",
    )
    p.add_argument(
        "--preserve-output-order",
        action="store_true",
        default=True,
        help="Emit rows in enumeration order (always enforced)",
    )
    p.add_argument(
        "--disable-worker-cache",
        action="store_true",
        help="Disable Rust worker-local parse caches",
    )
    p.add_argument(
        "--worker-cache-size",
        type=int,
        default=None,
        help="Max entries per worker cache type",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip spots already in completion ledger; write shard Parquet per spot",
    )
    p.add_argument(
        "--completion-ledger",
        type=Path,
        default=None,
        help="Append-only JSONL ledger of completed spot keys (default: {output_stem}_completed.jsonl)",
    )
    p.add_argument(
        "--bootstrap-log",
        type=Path,
        default=None,
        help="Chain log to bootstrap completed keys before run (validate before trusting)",
    )
    p.add_argument(
        "--bootstrap-expected-count",
        type=int,
        default=None,
        help="Require exact bootstrap key count (e.g. 1331 for prod6000 cutover)",
    )
    p.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Directory for per-spot Parquet shards (default: {output_stem}_shards)",
    )
    p.add_argument(
        "--work-manifest",
        type=Path,
        default=None,
        help="JSON manifest with spot_keys list — only solve keys in manifest",
    )
    p.add_argument(
        "--manifest-slice",
        type=str,
        default=None,
        help="Optional slice start:end into work-manifest spot_keys (for parallel workers)",
    )
    args = p.parse_args()

    spot_keys_filter: set[str] | None = None
    if args.work_manifest is not None:
        wm = json.loads(args.work_manifest.read_text(encoding="utf-8"))
        keys = list(wm.get("spot_keys", []))
        if args.manifest_slice:
            start_s, end_s = args.manifest_slice.split(":", 1)
            start_i = int(start_s) if start_s else 0
            end_i = int(end_s) if end_s else len(keys)
            keys = keys[start_i:end_i]
        spot_keys_filter = set(keys)

    manifest = build_labels(
        args.config,
        args.output,
        args.manifest,
        args.report,
        args.weva_index,
        args.cli_path,
        dry_run=args.dry_run,
        max_spots=args.max_spots,
        allow_plumbing=args.allow_plumbing,
        solver_backend=args.solver_backend,
        num_workers=args.num_workers,
        max_in_flight=args.max_in_flight,
        solver_threads_per_worker=args.solver_threads_per_worker,
        disable_worker_cache=args.disable_worker_cache,
        worker_cache_size=args.worker_cache_size,
        preserve_output_order=args.preserve_output_order,
        resume=args.resume,
        completion_ledger=args.completion_ledger,
        bootstrap_log=args.bootstrap_log,
        bootstrap_expected_count=args.bootstrap_expected_count,
        shard_dir=args.shard_dir,
        spot_keys_filter=spot_keys_filter,
    )
    print(json.dumps(manifest["enumeration"], indent=2))
    if not args.dry_run and manifest["gto_acceptable"] and manifest["enumeration"]["included"] == 0:
        sys.exit(1)
    if (
        not args.dry_run
        and args.resume
        and manifest.get("resume", {}).get("enabled")
        and not manifest.get("resume", {}).get("shard_complete", True)
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
