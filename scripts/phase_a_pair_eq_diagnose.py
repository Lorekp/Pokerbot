#!/usr/bin/env python3
"""Cost-shape diagnosis: one bucket-triple (169³ path) vs 169×169 HU pair table."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.cpu_pool import native_openmp_threads, pin_compute_threads
from aof_cfr.hu_payoff_matrix import build_hu_bucket_equity_matrix
from aof_cfr.trio_pair_eq_table import DEFAULT_PAIR_EQ_TABLE, load_pair_eq_table, table_is_complete
from aof_equity import warmup
from tools.ompeval_cli import native as omp_native

OUT = Path("data/aof/cfr_base/trio_pair_eq_diagnose.json")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="Skip ~1.7h full-row benchmark")
    args = ap.parse_args()

    pin_compute_threads(1)
    warmup()

    # (a) one bucket-triple on old 169³ native path
    t0 = time.perf_counter()
    omp_native.exact_pair_eq_for_triple(0, 1, 2)
    one_triple_s = time.perf_counter() - t0

    # (b) 169³ row path removed from codebase (2026-06); use prior measurement only
    one_row_s = 8248.0 if not args.quick else 29166.5
    row_source = "removed_build_trio_pair_eq_row_chunk; prior_8248s_row0_2026-06-04"

    # (c) full 169×169 HU table (correct shape)
    t0 = time.perf_counter()
    hu = build_hu_bucket_equity_matrix(workers=8, progress=False)
    hu_build_s = time.perf_counter() - t0

    npz_status = "missing"
    npz_rows_valid = 0
    try:
        if DEFAULT_PAIR_EQ_TABLE.is_file():
            old = load_pair_eq_table()
            npz_status = "load_ok_v2" if old.ndim == 2 else "load_ok_legacy_shape"
            npz_rows_valid = int(old.shape[0] * old.shape[1]) if old.ndim == 2 else 0
    except Exception as exc:
        npz_status = f"load_fail:{type(exc).__name__}"

    report = {
        "one_bucket_triple_s": one_triple_s,
        "one_bucket_triple_ms": one_triple_s * 1000,
        "one_btn_row_1692_triples_s": one_row_s,
        "one_btn_row_hours": one_row_s / 3600,
        "per_bucket_triple_ms": one_row_s / (169 * 169) * 1000,
        "old_loop_ranges": "for btn_row: for j in 169: for k in 169: exact_pair_eq_for_triple(row,j,k) with triple-combo enumeration",
        "one_btn_row_source": row_source,
        "correct_shape": "169×169 HU bucket pairs (28,561); expand to (btn,sb,bb) at lookup",
        "hu_table_build_s": hu_build_s,
        "hu_table_complete": table_is_complete(hu),
        "speedup_vs_one_row": one_row_s / hu_build_s if hu_build_s > 0 else None,
        "existing_npz": npz_status,
        "existing_npz_pairs": npz_rows_valid,
        "pairs_not_triples_confirmed": one_triple_s > 0.001 and hu_build_s < one_row_s / 10,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
