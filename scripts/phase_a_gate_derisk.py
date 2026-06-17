#!/usr/bin/env python3
"""Phase A: de-risk weighted gate before full tensor build.

Re-runs row validation with more samples and multiple seeds, spot-checks rows
across bucket space, and locks boards-per-combo when pooled UCB clears threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.buckets import LABELS_169
from aof_cfr.cpu_pool import kill_stale_tensor_builds, native_openmp_threads, pin_compute_threads
from aof_cfr.run_logger import fmt_elapsed, init_run_log, log_step
from aof_cfr.trio_pair_eq_table import DEFAULT_PAIR_EQ_TABLE, hu_pair_eq_values, require_pair_eq_table
from aof_cfr.trio_payoff_tensor import (
    DEFAULT_BOARDS_PER_COMBO,
    N_BUCKETS,
    build_one_bucket_row,
    save_trio_row_slice,
)
from aof_cfr.trio_row_validate import (
    DEFAULT_GATE_MARGIN,
    MAX_WEIGHTED_EQUITY_ERROR,
    TripleCheck,
    compact_checks,
    pooled_gate_result,
    validate_row_samples,
)
from aof_equity import warmup
from tools.ompeval_cli import native as omp_native

OUT = Path("data/aof/cfr_base/phase_a_gate_derisk.json")
PARTIAL = Path("data/aof/cfr_base/phase_a_gate_derisk.partial.json")
GATE_ROW_SLICE_DIR = Path("data/aof/cfr_base/gate_row_slices")

# Prior run at 128 bpc (row 0) before validation speed fix — gate threshold straddle confirmed.
KNOWN_128_ROW0 = [
    {"seed": 0, "samples": 50, "weighted_error": 0.004696, "pass": True, "margin_pass": False},
    {"seed": 0, "samples": 75, "weighted_error": 0.004875, "pass": True, "margin_pass": False},
    {"seed": 0, "samples": 100, "weighted_error": 0.005196, "pass": False, "margin_pass": False},
    {"seed": 1, "samples": 50, "weighted_error": 0.005971, "pass": False, "margin_pass": False},
]

# Row 0 = 22 (low pair), 34 = 72o (trash), 83 = T9s (mid), 167 = AKs (broadway)
DEFAULT_SPOT_ROWS = (0, 34, 83, 167)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_SAMPLE_COUNTS = (50, 75, 100)
DEFAULT_BOARDS_CANDIDATES = (128, 160, 192)
MARGIN = DEFAULT_GATE_MARGIN


def _write_partial(payload: dict) -> None:
    PARTIAL.parent.mkdir(parents=True, exist_ok=True)
    PARTIAL.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _row_slice_path(row: int, boards: int) -> Path:
    return GATE_ROW_SLICE_DIR / f"row{row}_bpc{boards}.npz"


def _cache_row_slice(row: int, boards: int, shares, pair_eq, combo_counts) -> Path:
    path = _row_slice_path(row, boards)
    save_trio_row_slice(path, row, shares, pair_eq, combo_counts, mc_boards=boards)
    return path


def _ensure_rows(rows: tuple[int, ...]) -> None:
    log_step("Loading complete 169×169 pair-eq table (read-only) ...")
    t0 = time.monotonic()
    require_pair_eq_table()
    log_step(f"Pair-eq table ready in {fmt_elapsed(t0)}")


def _pair_eq_parity(table, row: int, n: int = 8) -> dict:
    """Parity: expanded HU lookup vs hu_pair_eq_values."""
    idx = [0, 1, 12, 22, 48, 96, 120, 168][:n]
    samples = []
    max_err = 0.0
    for j in idx:
        for k in idx:
            from aof_cfr.trio_pair_eq_table import expand_pair_eq_row_slice

            tab = expand_pair_eq_row_slice(row, table)[j, k]
            want = hu_pair_eq_values(row, j, k, table)
            err = float(np.max(np.abs(tab - want)))
            max_err = max(max_err, err)
            samples.append({"j": j, "k": k, "max_pair_eq_err": err})
    return {"max_err": max_err, "samples": samples}


def _gate_matrix(
    row: int,
    *,
    boards_per_combo: int,
    shares,
    pair_eq,
    combo_counts,
    seeds: tuple[int, ...],
    sample_counts: tuple[int, ...],
) -> tuple[list[dict], list[TripleCheck]]:
    results: list[dict] = []
    all_checks: list[TripleCheck] = []
    total = len(seeds) * len(sample_counts)
    done = 0
    for seed in seeds:
        for samples in sample_counts:
            done += 1
            t0 = time.monotonic()
            log_step(
                f"Gate check {done}/{total}: row={row} ({LABELS_169[row]}) "
                f"seed={seed} n={samples} bpc={boards_per_combo} ..."
            )
            checks, ok, werr, umax = validate_row_samples(
                row,
                samples=samples,
                seed=seed,
                mc_boards=boards_per_combo,
                shares=shares,
                pair_eq=pair_eq,
                combo_counts=combo_counts,
                share_only=True,
                parallel=max(1, native_openmp_threads() // 2),
            )
            all_checks.extend(checks)
            worst = max(checks, key=lambda c: c.max_error)
            entry = {
                "row": row,
                "label": LABELS_169[row],
                "seed": seed,
                "samples": samples,
                "boards_per_combo": boards_per_combo,
                "pass": ok,
                "weighted_error": werr,
                "unweighted_max": umax,
                "worst_spot": worst.label,
                "worst_spot_error": worst.max_error,
                "worst_spot_deal_weight": worst.deal_weight,
                "margin_pass": werr <= (MAX_WEIGHTED_EQUITY_ERROR - MARGIN),
                "elapsed_s": round(time.monotonic() - t0, 1),
            }
            results.append(entry)
            log_step(
                f"  -> werr={werr:.6f} umax={umax:.6f} worst={worst.label} "
                f"({'PASS' if ok else 'FAIL-threshold' if werr > MAX_WEIGHTED_EQUITY_ERROR else 'PASS'}) "
                f"margin_diag={'yes' if entry['margin_pass'] else 'no'} "
                f"in {entry['elapsed_s']}s"
            )
    return results, all_checks


def _all_pass(matrix: list[dict]) -> bool:
    return all(r["pass"] for r in matrix)


def _log_pooled_gate(pooled) -> None:
    log_step(
        f"Pooled gate: n={pooled.n_triples_deduped} (raw={pooled.n_triples}) "
        f"mu={pooled.weighted_mean:.6f} stderr={pooled.stderr:.6f} "
        f"UCB95={pooled.upper_bound_95:.6f} worst={pooled.worst_spot_label} "
        f"threshold={'PASS' if pooled.pass_threshold else 'FAIL'} "
        f"margin={'PASS' if pooled.pass_margin else 'no'}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_SPOT_ROWS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--samples", type=int, nargs="+", default=list(DEFAULT_SAMPLE_COUNTS))
    ap.add_argument("--boards", type=int, nargs="+", default=list(DEFAULT_BOARDS_CANDIDATES))
    ap.add_argument("--skip-build", action="store_true", help="Only validate cached row slices")
    ap.add_argument(
        "--kill-stale",
        action="store_true",
        help="Kill other tensor/gate build processes before starting (not pair-eq builds)",
    )
    ap.add_argument(
        "--skip-boards",
        type=int,
        nargs="*",
        default=[],
        help="Skip board budgets already measured (e.g. 128)",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    log_path = init_run_log("phase_a_gate_derisk")

    if not DEFAULT_PAIR_EQ_TABLE.is_file():
        raise SystemExit(f"pair-eq table missing: {DEFAULT_PAIR_EQ_TABLE}")

    if args.kill_stale:
        n = kill_stale_tensor_builds(include_pair_eq=False)
        log_step(f"kill_stale_tensor_builds removed {n} conflicting process(es)")
    threads = native_openmp_threads()
    pin_compute_threads(threads)
    warmup()

    rows = tuple(args.rows)
    seeds = tuple(args.seeds)
    sample_counts = tuple(args.samples)
    boards_candidates = tuple(args.boards)
    skip = set(args.skip_boards)
    boards_candidates = tuple(b for b in boards_candidates if b not in skip)
    if 128 in skip:
        log_step(f"Skipping bpc=128 (prior straddle confirmed: {KNOWN_128_ROW0})")

    log_step(
        f"Gate de-risk start: rows={rows} seeds={seeds} samples={sample_counts} "
        f"boards={boards_candidates} threads={threads} log={log_path}"
    )
    _ensure_rows(rows)
    table = require_pair_eq_table()
    parity = {int(r): _pair_eq_parity(table, r) for r in rows}
    log_step(
        "Pair-eq parity max errs: "
        + ", ".join(f"row{k}={v['max_err']:.2e}" for k, v in parity.items())
    )

    locked_boards: int | None = None
    row_builds: dict[int, dict] = {}
    all_matrix: list[dict] = []
    all_checks: list[TripleCheck] = []
    pooled = pooled_gate_result([])

    for boards in boards_candidates:
        log_step(f"=== boards_per_combo={boards} ===")
        matrix: list[dict] = []
        board_checks: list[TripleCheck] = []
        for row in rows:
            if not args.skip_build:
                omp_native.reset_trio_row_profile()
                t0 = time.monotonic()
                log_step(f"Building tensor row {row} ({LABELS_169[row]}) bpc={boards} ...")
                shares, pair_eq, combo_counts, stats = build_one_bucket_row(
                    row=row,
                    mc_boards=boards,
                    progress=True,
                )
                wall = time.monotonic() - t0
                prof = omp_native.get_trio_row_profile()
                slice_path = _cache_row_slice(row, boards, shares, pair_eq, combo_counts)
                row_builds[row] = {
                    "elapsed_s": round(wall, 2),
                    "triples_per_sec": round(N_BUCKETS * N_BUCKETS / wall, 2),
                    "mc_hidden_pair_exact_ns": prof["mc_hidden_pair_exact_ns"],
                    "stats": stats,
                    "slice_cache": str(slice_path),
                }
                log_step(
                    f"Row {row} build done in {fmt_elapsed(t0)} "
                    f"({row_builds[row]['triples_per_sec']} triples/s, "
                    f"hidden_pair_exact_ns={prof['mc_hidden_pair_exact_ns']}, "
                    f"cached={slice_path})"
                )
            else:
                raise SystemExit("--skip-build not implemented (no row cache)")

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
            board_checks.extend(row_checks)
            pooled = pooled_gate_result(board_checks)
            _write_partial(
                {
                    "phase": "gate_checks",
                    "boards_per_combo": boards,
                    "row": row,
                    "gate_matrix": matrix,
                    "row_builds": row_builds,
                    "pooled_gate": pooled.to_dict(),
                    "triple_checks": compact_checks(board_checks),
                }
            )

        pooled = pooled_gate_result(board_checks)
        _log_pooled_gate(pooled)
        all_matrix = matrix
        all_checks = board_checks

        if pooled.pass_threshold:
            locked_boards = boards
            log_step(f"Locked boards_per_combo={boards} (pooled UCB95 <= {MAX_WEIGHTED_EQUITY_ERROR})")
            break
        if _all_pass(matrix):
            locked_boards = boards
            log_step(
                f"Locked boards_per_combo={boards} (per-run pass at threshold; pooled UCB did not clear)"
            )
            break
        log_step(f"boards_per_combo={boards}: pooled UCB failed — trying next")

    row0_werr = [r["weighted_error"] for r in all_matrix if r["row"] == 0]
    spot_summary = {}
    for row in rows:
        if row == 0:
            continue
        row_runs = [r for r in all_matrix if r["row"] == row]
        if row_runs:
            worst = max(row_runs, key=lambda r: r["weighted_error"])
            spot_summary[str(row)] = {
                "label": LABELS_169[row],
                "max_weighted_error": worst["weighted_error"],
                "worst_spot": worst["worst_spot"],
                "all_pass": _all_pass(row_runs),
            }

    out = {
        "locked_boards_per_combo": locked_boards,
        "gate_threshold": MAX_WEIGHTED_EQUITY_ERROR,
        "margin": MARGIN,
        "gate_policy": "pooled_ucb95",
        "rows_checked": list(rows),
        "seeds": list(seeds),
        "sample_counts": list(sample_counts),
        "boards_candidates": list(boards_candidates),
        "row0_weighted_errors": row0_werr,
        "row0_straddles_threshold": any(
            abs(r["weighted_error"] - MAX_WEIGHTED_EQUITY_ERROR) < 1e-4 for r in all_matrix if r["row"] == 0
        ),
        "spot_check": spot_summary,
        "row_builds": row_builds,
        "pair_eq_parity": {str(k): v["max_err"] for k, v in parity.items()},
        "gate_matrix": all_matrix,
        "pooled_gate": pooled.to_dict(),
        "triple_checks": compact_checks(all_checks),
        "ready_for_full_build": locked_boards is not None and pooled.pass_threshold,
        "log_file": str(log_path),
        "validation_mode": "share_only_native_oracle",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    log_step(f"Wrote {args.out}")

    if locked_boards is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
