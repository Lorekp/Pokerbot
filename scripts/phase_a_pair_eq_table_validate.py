#!/usr/bin/env python3
"""Phase A: exact pair-eq table + per-combo share MC — row-0 gate + bench."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.cpu_pool import kill_stale_tensor_builds, native_openmp_threads, pin_compute_threads
from aof_cfr.run_logger import init_run_log, log_step
from aof_cfr.trio_pair_eq_table import DEFAULT_PAIR_EQ_TABLE, require_pair_eq_table
from aof_cfr.trio_payoff_tensor import (
    DEFAULT_BOARDS_PER_COMBO,
    N_BUCKETS,
    build_one_bucket_row,
    exact_shares_for_triple_oracle,
)
from aof_cfr.trio_row_validate import validate_row_samples
from aof_equity import warmup
from tools.ompeval_cli import native as omp_native

OUT = Path("data/aof/cfr_base/row0_pair_eq_table_bench.json")


def _parity_sample_triples(hu: np.ndarray, row: int = 0, n: int = 8) -> list[dict]:
    """Compare expanded HU pair-eq vs hu_pair_eq_values on sample triples."""
    from aof_cfr.trio_pair_eq_table import expand_pair_eq_row_slice, hu_pair_eq_values

    idx = [0, 1, 12, 22, 48, 96, 120, 168]
    idx = idx[:n]
    out: list[dict] = []
    for j in idx:
        for k in idx:
            tab = expand_pair_eq_row_slice(row, hu)[j, k]
            want = hu_pair_eq_values(row, j, k, hu)
            err = float(np.max(np.abs(tab - want)))
            out.append({"j": j, "k": k, "max_pair_eq_err": err})
    return out


def main() -> None:
    if not DEFAULT_PAIR_EQ_TABLE.is_file():
        raise SystemExit(f"pair-eq table missing: {DEFAULT_PAIR_EQ_TABLE} — run build_trio_pair_eq_table.py")

    kill_stale_tensor_builds(include_pair_eq=False)
    log_path = init_run_log("phase_a_pair_eq_table_validate")
    threads = native_openmp_threads()
    pin_compute_threads(threads)
    warmup()
    log_step(f"Row-0 bench+gate start threads={threads} log={log_path}")
    table = require_pair_eq_table()
    parity = _parity_sample_triples(table, row=0)

    omp_native.reset_trio_row_profile()
    t0 = time.monotonic()
    shares, pair_eq, combo_counts, _stats = build_one_bucket_row(
        row=0,
        mc_boards=DEFAULT_BOARDS_PER_COMBO,
        progress=True,
    )
    wall = time.monotonic() - t0
    prof = omp_native.get_trio_row_profile()

    checks, ok, werr, umax = validate_row_samples(
        0,
        samples=15,
        seed=0,
        mc_boards=DEFAULT_BOARDS_PER_COMBO,
        shares=shares,
        pair_eq=pair_eq,
        combo_counts=combo_counts,
    )
    worst = max(checks, key=lambda c: c.max_error)

    out = {
        "row": 0,
        "boards_per_combo": DEFAULT_BOARDS_PER_COMBO,
        "pair_eq_table": str(DEFAULT_PAIR_EQ_TABLE),
        "pair_eq_table_shape": list(table.shape),
        "pair_eq_table_bytes_mb": round(table.nbytes / 1e6, 2),
        "pair_eq_parity_samples": parity,
        "pair_eq_parity_max_err": max(p["max_pair_eq_err"] for p in parity),
        "mc_hidden_pair_exact_ns": prof["mc_hidden_pair_exact_ns"],
        "threads": threads,
        "logical_cpus": os.cpu_count() or 4,
        "elapsed_s": round(wall, 2),
        "elapsed_min": round(wall / 60.0, 2),
        "triples_per_sec": round(N_BUCKETS * N_BUCKETS / wall, 2),
        "projected_full_tensor_h": round(wall * N_BUCKETS / 3600.0, 2),
        "validate_row_samples_pass": ok,
        "weighted_error": werr,
        "unweighted_max_error": umax,
        "worst_spot": worst.label,
        "worst_spot_error": worst.max_error,
        "worst_spot_deal_weight": worst.deal_weight,
        "fix": "exact_pair_eq_table + per_combo_share_mc",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
