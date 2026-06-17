#!/usr/bin/env python3
"""Re-analyze gate data with pooled UCB policy (validation-only replay or from-json)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.cpu_pool import kill_stale_tensor_builds, native_openmp_threads, pin_compute_threads
from aof_cfr.run_logger import fmt_elapsed, init_run_log, log_step
from aof_cfr.trio_pair_eq_table import DEFAULT_PAIR_EQ_TABLE, require_pair_eq_table
from aof_cfr.trio_payoff_tensor import N_BUCKETS, build_one_bucket_row, save_trio_row_slice
from aof_cfr.trio_row_validate import (
    DEFAULT_GATE_MARGIN,
    MAX_WEIGHTED_EQUITY_ERROR,
    TripleCheck,
    compact_checks,
    pooled_gate_result,
    triple_check_from_dict,
    validate_row_samples,
)
from aof_equity import warmup
from phase_a_gate_derisk import (
    DEFAULT_SAMPLE_COUNTS,
    DEFAULT_SEEDS,
    DEFAULT_SPOT_ROWS,
    OUT,
    _gate_matrix,
    _log_pooled_gate,
    _row_slice_path,
)


def _load_row_slice(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    data = np.load(path, allow_pickle=True)
    boards = int(data["mc_boards"]) if "mc_boards" in data else 0
    return (
        int(data["bucket_row"]),
        np.asarray(data["shares"], dtype=np.float64),
        np.asarray(data["pair_eq"], dtype=np.float64),
        np.asarray(data["combo_counts"], dtype=np.int32),
        boards,
    )


def _load_or_build_row(
    row: int,
    boards: int,
    *,
    validation_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = _row_slice_path(row, boards)
    if cache.is_file():
        _, shares, pair_eq, combo_counts, cached_bpc = _load_row_slice(cache)
        if cached_bpc == boards or cached_bpc == 0:
            log_step(f"Loaded cached row slice {cache}")
            return shares, pair_eq, combo_counts

    if not validation_only:
        raise SystemExit(f"Missing row slice {cache} and --validation-only not set")

    log_step(f"Building row {row} ({LABELS_169[row]}) at bpc={boards} for validation replay ...")
    t0 = time.monotonic()
    shares, pair_eq, combo_counts, _stats = build_one_bucket_row(row=row, mc_boards=boards, progress=True)
    save_trio_row_slice(cache, row, shares, pair_eq, combo_counts, mc_boards=boards)
    log_step(f"Row {row} built in {fmt_elapsed(t0)}, cached {cache}")
    return shares, pair_eq, combo_counts


def _collect_checks(
    rows: tuple[int, ...],
    boards: int,
    seeds: tuple[int, ...],
    sample_counts: tuple[int, ...],
    *,
    validation_only: bool,
) -> tuple[list[dict], list[TripleCheck], dict[int, dict]]:
    matrix: list[dict] = []
    all_checks: list[TripleCheck] = []
    row_builds: dict[int, dict] = {}

    for row in rows:
        require_pair_eq_table()
        shares, pair_eq, combo_counts = _load_or_build_row(
            row, boards, validation_only=validation_only
        )
        row_matrix, row_checks = _gate_matrix(
            row,
            boards_per_combo=boards,
            shares=shares,
            pair_eq=pair_eq,
            combo_counts=combo_counts,
            seeds=seeds,
            sample_counts=sample_counts,
        )
        matrix.extend(row_matrix)
        all_checks.extend(row_checks)
        row_builds[row] = {"slice_cache": str(_row_slice_path(row, boards))}

    return matrix, all_checks, row_builds


def _boards_target(mu_192: float, *, target: float = 0.0040, bpc_ref: int = 192) -> float:
    if mu_192 <= target:
        return float(bpc_ref)
    return float(bpc_ref * (mu_192 / target) ** 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_SPOT_ROWS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--samples", type=int, nargs="+", default=list(DEFAULT_SAMPLE_COUNTS))
    ap.add_argument("--boards", type=int, default=192)
    ap.add_argument(
        "--validation-only",
        action="store_true",
        help="Rebuild row slices at locked bpc if cache missing (no boards sweep)",
    )
    ap.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Pool triple_checks from existing gate JSON (skip replay)",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--kill-stale", action="store_true")
    args = ap.parse_args()

    log_path = init_run_log("phase_a_gate_reanalyze")
    if args.kill_stale:
        kill_stale_tensor_builds(include_pair_eq=False)

    if not DEFAULT_PAIR_EQ_TABLE.is_file():
        raise SystemExit(f"pair-eq table missing: {DEFAULT_PAIR_EQ_TABLE}")

    pin_compute_threads(native_openmp_threads())
    warmup()

    rows = tuple(args.rows)
    seeds = tuple(args.seeds)
    sample_counts = tuple(args.samples)
    boards = int(args.boards)

    if args.from_json:
        src = args.from_json
        if not src.is_file():
            raise SystemExit(f"Missing {src}")
        data = json.loads(src.read_text(encoding="utf-8"))
        raw = data.get("triple_checks") or []
        if not raw:
            raise SystemExit(f"{src} has no triple_checks — run with --validation-only")
        all_checks = [triple_check_from_dict(d) for d in raw]
        matrix = data.get("gate_matrix") or []
        row_builds = data.get("row_builds") or {}
        log_step(f"Loaded {len(all_checks)} triple checks from {src}")
    else:
        log_step(
            f"Validation replay rows={rows} boards={boards} seeds={seeds} "
            f"samples={sample_counts} validation_only={args.validation_only}"
        )
        matrix, all_checks, row_builds = _collect_checks(
            rows,
            boards,
            seeds,
            sample_counts,
            validation_only=args.validation_only or True,
        )

    pooled = pooled_gate_result(all_checks)
    _log_pooled_gate(pooled)

    margin_cutoff = MAX_WEIGHTED_EQUITY_ERROR - DEFAULT_GATE_MARGIN
    decision = "ship"
    if not pooled.pass_threshold:
        decision = "tighten_samples_or_bump_boards"
    elif not pooled.pass_margin:
        decision = "ship_at_threshold"

    bpc_target = _boards_target(pooled.weighted_mean) if not pooled.pass_threshold else boards

    report = {
        "decision": decision,
        "boards_per_combo": boards,
        "pooled_weighted_mean": pooled.weighted_mean,
        "pooled_stderr": pooled.stderr,
        "pooled_upper_bound_95": pooled.upper_bound_95,
        "n_triples_deduped": pooled.n_triples_deduped,
        "pass_threshold_ucb_005": pooled.pass_threshold,
        "pass_margin_mu_0045": pooled.pass_margin,
        "margin_cutoff": margin_cutoff,
        "worst_spot_diagnostic": pooled.worst_spot_label,
        "computed_bpc_target_if_bump": bpc_target,
    }
    log_step(f"Decision: {decision} — {json.dumps(report, indent=None)}")

    prior: dict = {}
    if args.out.is_file():
        try:
            prior = json.loads(args.out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}

    out = {
        **prior,
        "locked_boards_per_combo": boards if pooled.pass_threshold else prior.get("locked_boards_per_combo"),
        "gate_threshold": MAX_WEIGHTED_EQUITY_ERROR,
        "margin": DEFAULT_GATE_MARGIN,
        "gate_policy": "pooled_ucb95",
        "rows_checked": list(rows),
        "seeds": list(seeds),
        "sample_counts": list(sample_counts),
        "boards_candidates": [boards],
        "gate_matrix": matrix,
        "row_builds": row_builds,
        "pooled_gate": pooled.to_dict(),
        "triple_checks": compact_checks(all_checks),
        "reanalyze_report": report,
        "ready_for_full_build": pooled.pass_threshold and boards == (prior.get("locked_boards_per_combo") or boards),
        "log_file": str(log_path),
        "validation_mode": "share_only_native_oracle",
    }
    if pooled.pass_threshold:
        out["locked_boards_per_combo"] = boards
        out["ready_for_full_build"] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"pooled_gate": pooled.to_dict(), "reanalyze_report": report}, indent=2))
    log_step(f"Wrote {args.out}")

    if not pooled.pass_threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
