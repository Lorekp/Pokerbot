#!/usr/bin/env python3
"""Validate full trio_eq_tensor.npz: structural checks + pooled UCB95 gate on all 169 rows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.cpu_pool import assert_no_oversubscription, native_openmp_threads, pin_compute_threads
from aof_cfr.run_logger import fmt_elapsed, init_run_log, log_step
from aof_cfr.trio_pair_eq_table import hu_pair_eq_values, pair_eq_table_file_hash, require_pair_eq_table
from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, N_BUCKETS
from aof_cfr.trio_row_validate import (
    MAX_WEIGHTED_EQUITY_ERROR,
    batch_check_triple_tensor_shares,
    pooled_gate_result,
    row_slice_sanity,
    sample_triple_indices,
)
from tools.ompeval_cli import native as omp_native

DEFAULT_OUT = Path("data/aof/cfr_base/phase_a_full_tensor_validate.json")


def structural_checks(data: np.lib.npyio.NpzFile) -> dict:
    shares = np.asarray(data["shares"], dtype=np.float64)
    pair_eq = np.asarray(data["pair_eq"], dtype=np.float64)
    combo_counts = np.asarray(data["combo_counts"], dtype=np.int32)
    expected = (N_BUCKETS, N_BUCKETS, N_BUCKETS, 3)
    ok_shape = shares.shape == expected and pair_eq.shape == expected
    labels = data["labels"] if "labels" in data else None
    unique_labels = int(np.unique(labels).size) if labels is not None else 0
    sanity = row_slice_sanity(shares, pair_eq, combo_counts)
    boards = int(data["boards_per_combo"]) if "boards_per_combo" in data else int(data.get("mc_boards", 0))
    manifest = {
        "build_mode": str(data["build_mode"]) if "build_mode" in data else "unknown",
        "boards_per_combo": boards,
        "row_workers": int(data["row_workers"]) if "row_workers" in data else 1,
        "row_parallel": bool(data["row_parallel"]) if "row_parallel" in data else False,
        "eval_backend": str(data["eval_backend"]) if "eval_backend" in data else "unknown",
        "pair_eq_table": str(data["pair_eq_table"]) if "pair_eq_table" in data else "unknown",
        "pair_eq_table_hash": str(data["pair_eq_table_hash"]) if "pair_eq_table_hash" in data else "",
        "seed_scheme": str(data["seed_scheme"]) if "seed_scheme" in data else "unknown",
        "row_seeds_count": int(len(data["row_seeds"])) if "row_seeds" in data else 0,
    }
    current_hash = pair_eq_table_file_hash()
    hash_match = bool(manifest["pair_eq_table_hash"]) and manifest["pair_eq_table_hash"] == current_hash
    return {
        "shares_shape": list(shares.shape),
        "pair_eq_shape": list(pair_eq.shape),
        "shape_ok": ok_shape,
        "unique_bucket_labels": unique_labels,
        "labels_ok": unique_labels == N_BUCKETS,
        "sanity": sanity,
        "manifest": manifest,
        "pair_eq_table_hash_match": hash_match,
        "current_pair_eq_table_hash": current_hash,
    }


def pair_eq_parity_samples(
    pair_eq: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict:
    """Vectorized HU-table expand vs tensor pair_eq cells."""
    hu = require_pair_eq_table()
    rng = np.random.default_rng(seed + 999)
    rows = rng.integers(0, N_BUCKETS, size=samples, dtype=np.int32)
    js = rng.integers(0, N_BUCKETS, size=samples, dtype=np.int32)
    ks = rng.integers(0, N_BUCKETS, size=samples, dtype=np.int32)
    got = np.asarray(pair_eq[rows, js, ks, :], dtype=np.float64)
    want = np.column_stack(
        [
            hu[rows, ks],
            hu[rows, js],
            hu[js, ks],
        ]
    ).astype(np.float64)
    errs = np.max(np.abs(got - want), axis=1)
    worst_i = int(np.argmax(errs))
    return {
        "samples": samples,
        "max_err": float(errs[worst_i]),
        "worst_spot": (
            f"{LABELS_169[int(rows[worst_i])]} / {LABELS_169[int(js[worst_i])]} / "
            f"{LABELS_169[int(ks[worst_i])]}"
        ),
        "ok": float(errs.max()) <= 1e-6,
    }


def run_pooled_gate(
    shares: np.ndarray,
    pair_eq: np.ndarray,
    combo_counts: np.ndarray,
    *,
    samples_per_row: int,
    seed: int,
    omp_threads: int,
) -> tuple[list, dict]:
    rows, js, ks = sample_triple_indices(samples_per_row=samples_per_row, seed=seed)
    omp_native.reset_trio_row_profile()
    checks = batch_check_triple_tensor_shares(
        shares,
        pair_eq,
        combo_counts,
        rows,
        js,
        ks,
        omp_threads=omp_threads,
    )
    pooled = pooled_gate_result(checks)
    return checks, pooled.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensor", type=Path, default=DEFAULT_TENSOR)
    ap.add_argument("--samples-per-row", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pair-eq-samples", type=int, default=24)
    ap.add_argument(
        "--omp-threads",
        type=int,
        default=0,
        help="OpenMP threads for batch oracle (0=physical-core default)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    omp_threads = args.omp_threads if args.omp_threads > 0 else native_openmp_threads()
    assert_no_oversubscription(1, omp_threads, label="tensor validate")
    pin_compute_threads(omp_threads)

    log_path = init_run_log("phase_a_full_tensor_validate")
    path = Path(args.tensor)
    if not path.is_file():
        raise SystemExit(f"Missing tensor: {path}")

    log_step(f"Loading {path} ...")
    data = np.load(path, allow_pickle=True)
    shares = np.asarray(data["shares"], dtype=np.float64)
    pair_eq = np.asarray(data["pair_eq"], dtype=np.float64)
    combo_counts = np.asarray(data["combo_counts"], dtype=np.int32)

    struct = structural_checks(data)
    log_step(f"Structural: shape_ok={struct['shape_ok']} labels_ok={struct['labels_ok']}")

    n_samples = args.samples_per_row * N_BUCKETS
    log_step(
        f"Pooled gate (batch native): {args.samples_per_row} samples/row x {N_BUCKETS} = "
        f"{n_samples} triples omp={omp_threads} seed={args.seed}"
    )
    t0 = time.monotonic()
    checks, pooled = run_pooled_gate(
        shares,
        pair_eq,
        combo_counts,
        samples_per_row=args.samples_per_row,
        seed=args.seed,
        omp_threads=omp_threads,
    )
    gate_elapsed = time.monotonic() - t0
    log_step(
        f"Pooled gate done in {fmt_elapsed(t0)} ({gate_elapsed:.1f}s, "
        f"{n_samples / max(gate_elapsed, 1e-9):.1f} triples/s): "
        f"mu={pooled['weighted_mean']:.6f} stderr={pooled['stderr']:.6f} "
        f"n_eff={pooled['effective_n']:.1f} UCB95={pooled['upper_bound_95']:.6f} "
        f"pass={pooled['pass_threshold']}"
    )

    pe_parity = pair_eq_parity_samples(pair_eq, samples=args.pair_eq_samples, seed=args.seed)
    prof = omp_native.get_trio_row_profile()
    hidden = float(prof.get("mc_hidden_pair_exact_ns", 0.0))

    all_ok = (
        struct["shape_ok"]
        and struct["labels_ok"]
        and pooled["pass_threshold"]
        and pe_parity["ok"]
        and hidden == 0.0
    )

    report = {
        "tensor": str(path),
        "samples_per_row": args.samples_per_row,
        "total_triples_sampled": len(checks),
        "gate_elapsed_s": gate_elapsed,
        "gate_triples_per_s": n_samples / max(gate_elapsed, 1e-9),
        "omp_threads": omp_threads,
        "validation_backend": "batch_exact_shares_native",
        "structural": struct,
        "pooled_gate": pooled,
        "gate_threshold": MAX_WEIGHTED_EQUITY_ERROR,
        "pair_eq_parity": pe_parity,
        "hidden_exact_pair_eq_ns": hidden,
        "trusted_for_solve": all_ok,
        "log_file": str(log_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_step(f"Wrote {args.out} trusted={all_ok}")

    print(json.dumps({"pooled_gate": pooled, "trusted_for_solve": all_ok}, indent=2))
    if not all_ok:
        if not pooled["pass_threshold"]:
            log_step(
                f"STOP: UCB95={pooled['upper_bound_95']:.6f} > {MAX_WEIGHTED_EQUITY_ERROR} — "
                "do not solve; consider boards bump from 1/sqrt(n) fit"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
