#!/usr/bin/env python3
"""Instrumented row-0 build: path split, threshold projection, gate, CPU + MC microbench."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.cpu_pool import kill_stale_tensor_builds, native_openmp_threads, pin_compute_threads
from aof_cfr.trio_payoff_tensor import (
    DEFAULT_TRIPLE_SAMPLE_BUDGET,
    EXACT_TRIPLE_COMBO_MAX,
    MC_SHARE_STDERR_GATE,
    MIN_BOARDS_PER_COMBO,
    N_BUCKETS,
    _all_combo_triples,
    _native_build_row_columns,
    tensor_build_backend_name,
    trio_deal_weights,
)
from aof_cfr.trio_row_validate import (
    MAX_WEIGHTED_EQUITY_ERROR,
    TripleCheck,
    validate_row_samples,
    validation_gate_passes,
)
from aof_cfr.trio_eval_mc import eval7_backend_name
from aof_equity import warmup
from aof_equity_grids import _card_str_to_id
from scripts.sweep_exact_enum_fraction import _analytic_row
from tools.ompeval_cli import native as omp_native

N = N_BUCKETS
DEFAULT_OUT = Path("data/aof/cfr_base/row0_cost_profile.json")


def _flip_triples(row: int, threshold: int, combo_counts: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Triples that switch exact→MC when lowering threshold from 36 to T."""
    out: list[tuple[int, int]] = []
    for j in range(N):
        for k in range(N):
            if combo_counts is not None:
                count = int(combo_counts[j, k])
            else:
                count = len(_all_combo_triples(row, j, k))
            if threshold < count <= EXACT_TRIPLE_COMBO_MAX:
                out.append((j, k))
    return out


def _pack_native(row: int, j: int, k: int, *, mc_boards: int, exact_combo_max: int) -> np.ndarray:
    nat = omp_native.avg_metrics_for_triple(
        row,
        j,
        k,
        mc_boards=mc_boards,
        exact_combo_max=exact_combo_max,
        min_boards_per_combo=MIN_BOARDS_PER_COMBO,
        stderr_gate=MC_SHARE_STDERR_GATE,
    )
    return np.array(nat[:6], dtype=np.float64)


def _pack_oracle(row: int, j: int, k: int) -> np.ndarray:
    from aof_cfr.trio_payoff_tensor import exact_shares_for_triple_oracle

    s0, s1, s2, bb, bs, sb, _ = exact_shares_for_triple_oracle(row, j, k)
    return np.array([s0, s1, s2, bb, bs, sb], dtype=np.float64)


def _validate_flip_set(
    row: int,
    threshold: int,
    *,
    mc_boards: int,
    combo_counts: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    flip = _flip_triples(row, threshold, combo_counts)
    deal_w = trio_deal_weights()
    checks: list[TripleCheck] = []
    for j, k in flip:
        got = _pack_native(row, j, k, mc_boards=mc_boards, exact_combo_max=threshold)
        oracle = _pack_oracle(row, j, k)
        err = float(np.max(np.abs(got - oracle)))
        cnt = int(combo_counts[j, k]) if combo_counts is not None else len(_all_combo_triples(row, j, k))
        checks.append(
            TripleCheck(
                row=row,
                col_j=j,
                col_k=k,
                label=f"{LABELS_169[row]} / {LABELS_169[j]} / {LABELS_169[k]}",
                max_error=err,
                mc=got,
                oracle=oracle,
                combo_count=cnt,
                deal_weight=float(deal_w[row, j, k]),
            )
        )
    ok, werr, umax = validation_gate_passes(checks)
    return {
        "threshold": threshold,
        "flip_count": len(flip),
        "weighted_error": werr,
        "unweighted_max_error": umax,
        "gate_passes": ok,
    }


def _profile_dict_to_report(prof: dict, wall_s: float) -> dict:
    exact_n = int(prof["exact_triple_count"])
    mc_n = int(prof["mc_triple_count"])
    exact_ns = float(prof["exact_triple_ns"])
    mc_ns = float(prof["mc_triple_ns"])
    hidden_ns = float(prof["mc_hidden_pair_exact_ns"])
    total_ns = exact_ns + mc_ns
    exact_s = exact_ns / 1e9
    mc_s = mc_ns / 1e9
    hidden_s = hidden_ns / 1e9
    return {
        "wall_s": round(wall_s, 2),
        "exact_triple_count": exact_n,
        "mc_triple_count": mc_n,
        "exact_path_s": round(exact_s, 3),
        "mc_path_s": round(mc_s, 3),
        "mc_hidden_pair_exact_s": round(hidden_s, 3),
        "mean_exact_ms_per_triple": round(1000.0 * exact_s / exact_n, 3) if exact_n else 0.0,
        "mean_mc_ms_per_triple": round(1000.0 * mc_s / mc_n, 3) if mc_n else 0.0,
        "exact_path_pct_of_row_time": round(100.0 * exact_s / wall_s, 2) if wall_s > 0 else 0.0,
        "mc_path_pct_of_row_time": round(100.0 * mc_s / wall_s, 2) if wall_s > 0 else 0.0,
        "exact_triple_count_share_pct": round(100.0 * exact_n / (exact_n + mc_n), 2)
        if (exact_n + mc_n)
        else 0.0,
        "mc_board_evals": int(prof["mc_board_evals"]),
        "profiled_ns_total": total_ns,
    }


def _cpu_sampler(stop: threading.Event, interval_s: float, samples: list[float]) -> None:
    ps = (
        "$c = Get-Counter '\\Processor(_Total)\\% Processor Time' -ErrorAction SilentlyContinue; "
        "if ($c) { [math]::Round($c.CounterSamples[0].CookedValue, 2) } else { 0 }"
    )
    while not stop.wait(interval_s):
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=30,
            )
            val = float((r.stdout.strip() or "0").replace(",", "."))
            samples.append(val)
            print(f"  cpu_sample: {val:.1f}%", flush=True)
        except Exception as exc:
            print(f"  cpu_sample error: {exc}", flush=True)


def _run_instrumented_build(
    row: int,
    *,
    mc_boards: int,
    exact_combo_max: int,
    threads: int,
    cpu_sample_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    shares = np.zeros((N, N, 3), dtype=np.float64)
    pair_eq = np.zeros((N, N, 3), dtype=np.float64)
    combo_counts = np.zeros((N, N), dtype=np.int32)

    omp_native.reset_trio_row_profile()
    cpu_samples: list[float] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=_cpu_sampler,
        args=(stop, 5.0, cpu_samples),
        daemon=True,
    )
    sampler.start()

    t0 = time.monotonic()
    _native_build_row_columns(
        row,
        shares,
        pair_eq,
        combo_counts,
        mc_boards=mc_boards,
        n=N,
        omp_threads=threads,
        progress=True,
        progress_label=f"profile row {row} T={exact_combo_max}",
        started=t0,
        exact_combo_max=exact_combo_max,
    )
    wall_s = time.monotonic() - t0
    time.sleep(min(cpu_sample_sec, 5.0))
    stop.set()
    sampler.join(timeout=10.0)

    prof = omp_native.get_trio_row_profile()
    path_report = _profile_dict_to_report(prof, wall_s)
    logical = os.cpu_count() or 4
    cpu_report = {
        "threads": threads,
        "logical_cpus": logical,
        "expected_saturation_pct": round(100.0 * threads / logical, 1),
        "samples": cpu_samples,
        "mean_cpu_pct": round(float(np.mean(cpu_samples)), 2) if cpu_samples else None,
        "min_cpu_pct": round(float(np.min(cpu_samples)), 2) if cpu_samples else None,
        "max_cpu_pct": round(float(np.max(cpu_samples)), 2) if cpu_samples else None,
    }
    if path_report["exact_path_pct_of_row_time"] > path_report["exact_triple_count_share_pct"] * 1.5:
        print(
            "  NOTE: exact path time share >> count share — exact triples dominate wall time",
            flush=True,
        )
    return shares, pair_eq, combo_counts, path_report, cpu_report


def _threshold_table(
    row: int,
    *,
    mc_boards: int,
    mean_exact_s: float,
    mean_mc_s: float,
    hidden_pair_s_per_mc_triple: float,
) -> list[dict]:
    thresholds = [0, 8, 16, 36]
    rows = []
    for th in thresholds:
        analytic = _analytic_row(row, th, mc_boards)
        n_exact = int(analytic["exact_by_threshold"])
        n_mc = N * N - n_exact
        projected_s = n_exact * mean_exact_s + n_mc * mean_mc_s
        optimistic_s = projected_s - n_mc * hidden_pair_s_per_mc_triple
        exact_path_pct = round(100.0 * (n_exact * mean_exact_s) / projected_s, 2) if projected_s > 0 else 0.0
        rows.append(
            {
                "threshold": th,
                "exact_enum_pct_of_triples": round(100.0 * analytic["frac_exact_by_threshold"], 2),
                "exact_path_pct_of_row_time": exact_path_pct,
                "exact_triple_count": n_exact,
                "mc_triple_count": n_mc,
                "projected_row0_wall_s": round(projected_s, 1),
                "projected_row0_wall_min": round(projected_s / 60.0, 2),
                "optimistic_if_no_hidden_pair_eq_s": round(max(optimistic_s, 0.0), 1),
            }
        )
    return rows


def _mc_microbench(*, mc_boards: int) -> dict:
    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    i, j, k = idx["AKs"], idx["AQs"], idx["KQo"]
    combo = _all_combo_triples(i, j, k)[0]
    cards = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    n_combos = len(_all_combo_triples(i, j, k))
    boards_per_combo = max(MIN_BOARDS_PER_COMBO, mc_boards // n_combos)

    mod = omp_native.load_native_module()
    assert mod is not None

    # Standalone eval7 loop (one MC board pattern, 3 eval7 calls per board)
    c = cards.tolist()
    board = [10, 11, 12, 13, 14]
    n_iters = 200_000
    t0 = time.monotonic()
    for _ in range(n_iters):
        mod.eval7(c[0], c[1], board[0], board[1], board[2], board[3], board[4])
        mod.eval7(c[2], c[3], board[0], board[1], board[2], board[3], board[4])
        mod.eval7(c[4], c[5], board[0], board[1], board[2], board[3], board[4])
    standalone_elapsed = time.monotonic() - t0
    standalone_eval7_s = 3 * n_iters / standalone_elapsed

    # mc_nway_shares only
    reps = 500
    t0 = time.monotonic()
    total_boards = 0
    for r in range(reps):
        mod.mc_nway_shares(cards, 3, boards_per_combo, 12345 + r)
        total_boards += boards_per_combo
    mc_elapsed = time.monotonic() - t0
    mc_eval7_s = (3 * total_boards) / mc_elapsed

    # full eval_combo_adaptive
    t0 = time.monotonic()
    total_boards_adaptive = 0
    for r in range(reps):
        _, boards_used, _ = mod.eval_combo_adaptive(
            cards, 999 + r, boards_per_combo, MIN_BOARDS_PER_COMBO, MC_SHARE_STDERR_GATE
        )
        total_boards_adaptive += int(boards_used)
    adaptive_elapsed = time.monotonic() - t0
    adaptive_eval7_s = (3 * total_boards_adaptive) / adaptive_elapsed if total_boards_adaptive else 0.0

    return {
        "triple": "AKs/AQs/KQo",
        "n_combos": n_combos,
        "boards_per_combo": boards_per_combo,
        "standalone_eval7_per_s": round(standalone_eval7_s, 0),
        "mc_nway_shares_eval7_per_s": round(mc_eval7_s, 0),
        "eval_combo_adaptive_eval7_per_s": round(adaptive_eval7_s, 0),
        "reference_upper_bound_eval7_per_s": "100M+ (HU batch runbook, not apples-to-apples)",
    }


def _fleet_recommendation(projected_min_at_best_threshold: float) -> str:
    if projected_min_at_best_threshold < 30:
        return (
            "Full 169-row tensor locally (~tens of minutes with row-parallel); "
            "EC2 spot fleet unnecessary."
        )
    if projected_min_at_best_threshold < 120:
        return "Local row-parallel or optional cloud; fleet not urgent."
    return (
        "Row-0 stays in tens-of-minutes-to-hours band after threshold tuning; "
        "EC2 spot fleet justified — size with per_row_s × 169 / parallelism."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Profile row-0 trio build cost (measurement only)")
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--mc-boards", type=int, default=DEFAULT_TRIPLE_SAMPLE_BUDGET)
    p.add_argument("--exact-combo-max", type=int, default=EXACT_TRIPLE_COMBO_MAX)
    p.add_argument("--cpu-sample-sec", type=float, default=30.0)
    p.add_argument("--skip-build", action="store_true", help="Skip instrumented build (use prior artifacts)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    if not omp_native.native_trio_available():
        raise SystemExit("native trio backend not available; build tools/ompeval_cli first")

    kill_stale_tensor_builds()
    warmup()
    threads = native_openmp_threads()
    pin_compute_threads(threads)

    print(
        f"profile_trio_row0: row={args.row} mc_boards={args.mc_boards} "
        f"T={args.exact_combo_max} threads={threads} "
        f"backend={tensor_build_backend_name()} eval7={eval7_backend_name()}",
        flush=True,
    )

    microbench = _mc_microbench(mc_boards=args.mc_boards)
    print(f"microbench: {json.dumps(microbench)}", flush=True)

    if args.skip_build:
        path_report = {}
        cpu_report = {}
        shares = pair_eq = combo_counts = None
        wall_for_projection = 76.4 * 60.0
        mean_exact_s = 0.0
        mean_mc_s = wall_for_projection / (N * N)
        hidden_per_mc = 0.0
    else:
        shares, pair_eq, combo_counts, path_report, cpu_report = _run_instrumented_build(
            args.row,
            mc_boards=args.mc_boards,
            exact_combo_max=args.exact_combo_max,
            threads=threads,
            cpu_sample_sec=args.cpu_sample_sec,
        )
        print(f"path_split: {json.dumps(path_report)}", flush=True)
        print(f"cpu: {json.dumps(cpu_report)}", flush=True)
        exact_n = path_report["exact_triple_count"]
        mc_n = path_report["mc_triple_count"]
        mean_exact_s = path_report["exact_path_s"] / exact_n if exact_n else 0.0
        mean_mc_s = path_report["mc_path_s"] / mc_n if mc_n else 0.0
        hidden_per_mc = path_report["mc_hidden_pair_exact_s"] / mc_n if mc_n else 0.0
        wall_for_projection = path_report["wall_s"]

    threshold_rows = _threshold_table(
        args.row,
        mc_boards=args.mc_boards,
        mean_exact_s=mean_exact_s,
        mean_mc_s=mean_mc_s,
        hidden_pair_s_per_mc_triple=hidden_per_mc,
    )

    gate_results = []
    for th in [0, 8, 16, 36]:
        print(f"flip gate threshold={th} ...", flush=True)
        gate = _validate_flip_set(
            args.row,
            th,
            mc_boards=args.mc_boards,
            combo_counts=combo_counts,
        )
        gate_results.append(gate)
        print(
            f"  T={th}: flip={gate['flip_count']} werr={gate['weighted_error']:.5f} "
            f"pass={gate['gate_passes']}",
            flush=True,
        )
        for row in threshold_rows:
            if row["threshold"] == th:
                row["weighted_gate_passes"] = gate["gate_passes"]
                row["flip_weighted_error"] = round(gate["weighted_error"], 6)

    sample_gate = None
    if shares is not None and pair_eq is not None and combo_counts is not None:
        print("validate_row_samples T=36 ...", flush=True)
        _, ok, werr, umax = validate_row_samples(
            args.row,
            samples=15,
            seed=0,
            mc_boards=args.mc_boards,
            shares=shares,
            pair_eq=pair_eq,
            combo_counts=combo_counts,
        )
        sample_gate = {"passes": ok, "weighted_error": werr, "unweighted_max": umax}
        print(f"  sample gate: pass={ok} werr={werr:.5f} umax={umax:.5f}", flush=True)

    best = min(threshold_rows, key=lambda r: r["projected_row0_wall_s"])
    fleet = _fleet_recommendation(best["projected_row0_wall_min"])

    report = {
        "row": args.row,
        "mc_boards": args.mc_boards,
        "exact_combo_max_measured": args.exact_combo_max,
        "backend": tensor_build_backend_name(),
        "eval7_backend": eval7_backend_name(),
        "threads": threads,
        "path_split": path_report,
        "cpu_saturation": cpu_report,
        "microbench": microbench,
        "threshold_decision_table": threshold_rows,
        "flip_gate_by_threshold": gate_results,
        "validate_row_samples_T36": sample_gate,
        "fleet_recommendation": fleet,
        "notes": [
            "projected_row0_wall_s = n_exact(T)*mean_exact + n_mc(T)*mean_mc from one instrumented build",
            "optimistic_if_no_hidden_pair_eq_s subtracts hidden exact pair-eq cost from MC triples",
            f"MAX_WEIGHTED_EQUITY_ERROR={MAX_WEIGHTED_EQUITY_ERROR}",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)

    print("\nDecision table:")
    print("| Threshold | exact-enum % | projected row-0 | gate passes? |")
    print("|-----------|--------------|-----------------|--------------|")
    for row in threshold_rows:
        gate_ok = row.get("weighted_gate_passes", "?")
        print(
            f"| {row['threshold']} | {row['exact_enum_pct_of_triples']}% | "
            f"{row['projected_row0_wall_min']:.1f} min | {gate_ok} |"
        )
    print(f"\nFleet: {fleet}")


if __name__ == "__main__":
    main()
