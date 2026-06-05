#!/usr/bin/env python3
"""Validate trio equity tensor accuracy vs exact enumeration oracle."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.trio_pair_eq_table import expand_pair_eq_row_slice, hu_pair_eq_values, require_pair_eq_table
from aof_cfr.trio_row_validate import (
    MAX_WEIGHTED_EQUITY_ERROR,
    check_triple_tensor,
    validation_gate_passes,
)
from tools.ompeval_cli import native as omp_native


def main() -> None:
    p = argparse.ArgumentParser(description="Spot-check trio tensor vs exact oracle")
    p.add_argument("--tensor", default="data/aof/cfr_base/trio_eq_tensor.npz")
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pair-eq-samples", type=int, default=12)
    args = p.parse_args()

    path = Path(args.tensor)
    if not path.is_file():
        raise SystemExit(f"Missing {path}")

    data = np.load(path, allow_pickle=True)
    shares = np.asarray(data["shares"], dtype=np.float64)
    pair_eq = np.asarray(data["pair_eq"], dtype=np.float64)
    combo_counts = np.asarray(data["combo_counts"], dtype=np.int32)
    n = shares.shape[0]

    print(f"Tensor shape shares={shares.shape} pair_eq={pair_eq.shape}", flush=True)
    if shares.shape != (N_BUCKETS, N_BUCKETS, N_BUCKETS, 3):
        print(f"FAIL: shares shape {shares.shape} != ({N_BUCKETS},{N_BUCKETS},{N_BUCKETS},3)", file=sys.stderr)
        sys.exit(1)
    if pair_eq.shape != shares.shape:
        print(f"FAIL: pair_eq shape {pair_eq.shape} != shares shape", file=sys.stderr)
        sys.exit(1)

    unique_rows = n
    if unique_rows != N_BUCKETS:
        print(f"FAIL: expected {N_BUCKETS} bucket rows, got {unique_rows}", file=sys.stderr)
        sys.exit(1)

    build_mode = str(data["build_mode"]) if "build_mode" in data else "unknown"
    boards = int(data["boards_per_combo"]) if "boards_per_combo" in data else int(data.get("mc_boards", 0))
    row_workers = int(data["row_workers"]) if "row_workers" in data else 1
    row_parallel = bool(data["row_parallel"]) if "row_parallel" in data else False
    backend = str(data["eval_backend"]) if "eval_backend" in data else "unknown"
    pair_eq_table = str(data["pair_eq_table"]) if "pair_eq_table" in data else "unknown"

    print(
        f"Manifest: build_mode={build_mode} boards_per_combo={boards} "
        f"row_workers={row_workers} row_parallel={row_parallel} "
        f"backend={backend} pair_eq_table={pair_eq_table}",
        flush=True,
    )

    valid = combo_counts > 0
    share_sums = shares[valid].sum(axis=-1)
    max_share_dev = float(np.max(np.abs(share_sums - 1.0))) if valid.any() else 0.0
    print(f"Sanity: valid_triples={int(valid.sum())} max_share_sum_dev={max_share_dev:.2e}", flush=True)

    rng = random.Random(args.seed)
    checks = []
    for _ in range(args.samples):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        k = rng.randint(0, n - 1)
        checks.append(check_triple_tensor(i, j, k, shares, pair_eq, combo_counts))

    ok, werr, umax = validation_gate_passes(checks)
    worst = max(checks, key=lambda c: c.max_error)
    print(
        f"Share gate: checked={args.samples} weighted_error={werr:.6f} "
        f"unweighted_max={umax:.6f} worst={worst.label} err={worst.max_error:.6f}",
        flush=True,
    )

    hu = require_pair_eq_table()
    pe_rng = random.Random(args.seed + 999)
    pe_max = 0.0
    for _ in range(args.pair_eq_samples):
        i = pe_rng.randint(0, n - 1)
        j = pe_rng.randint(0, n - 1)
        k = pe_rng.randint(0, n - 1)
        got = pair_eq[i, j, k]
        want = expand_pair_eq_row_slice(i, hu)[j, k]
        err = float(np.max(np.abs(want - got)))
        pe_max = max(pe_max, err)
    print(f"Pair-eq parity vs HU table expand: max_err={pe_max:.2e} over {args.pair_eq_samples} samples", flush=True)

    prof = omp_native.get_trio_row_profile()
    hidden = prof.get("mc_hidden_pair_exact_ns", 0.0)
    print(f"Hidden exact pair-eq counter: {hidden}", flush=True)

    if not ok:
        print(
            f"FAIL: weighted error {werr:.4f} > {MAX_WEIGHTED_EQUITY_ERROR}",
            file=sys.stderr,
        )
        sys.exit(1)
    if pe_max > 1e-6:
        print(f"FAIL: pair-eq parity max_err {pe_max:.2e} > 1e-6", file=sys.stderr)
        sys.exit(1)
    if hidden > 0:
        print(f"WARN: mc_hidden_pair_exact_ns={hidden} (expected 0)", file=sys.stderr)
    print("PASS")


if __name__ == "__main__":
    main()
