#!/usr/bin/env python3
"""Build exact 169×169 pair-equity table once (single writer, eager pass)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.cpu_pool import default_pool_workers, kill_stale_tensor_builds, pin_compute_threads
from aof_cfr.hu_payoff_matrix import DEFAULT_TABLE as HU_SHOWDOWN_TABLE
from aof_cfr.hu_payoff_matrix import build_hu_bucket_equity_matrix
from aof_cfr.run_logger import init_run_log, log_step
from aof_cfr.trio_pair_eq_table import (
    DEFAULT_PAIR_EQ_TABLE,
    N_BUCKETS,
    build_pair_eq_table,
    hu_showdown_file_hash,
    load_pair_eq_table,
    parity_vs_hu_oracle,
    pair_eq_table_file_hash,
    require_pair_eq_table,
    save_pair_eq_table,
    table_is_complete,
)
from aof_equity import warmup

OUT_JSON = Path("data/aof/cfr_base/trio_pair_eq_build_report.json")
TEMP_DIR = Path("data/aof/cfr_base/pair_eq_build_temps")


def _worker_build_rows(row_indices: list[int], hu_source: str, temp_path: str) -> str:
    pin_compute_threads(1)
    from aof_cfr.hu_payoff_matrix import _load_eq_dict, _collect_key_counts, _fill_missing_keys
    from aof_cfr.hu_payoff_matrix import DEFAULT_KEY_CACHE

    eq_dict = _load_eq_dict(hu_source)
    _fill_missing_keys(eq_dict, sorted(_collect_key_counts().keys()), workers=1, progress=False)
    n = N_BUCKETS
    chunk = np.zeros((len(row_indices), n), dtype=np.float64)
    for ri, i in enumerate(row_indices):
        chunk[ri, i] = 0.5
        for j in range(n):
            if i == j:
                continue
            total = 0.0
            count = 0
            from aof_cfr.buckets import COMBOS_169
            from aof_equity_showdown import canonical_showdown_key, cards_to_ids

            for c1, c2 in COMBOS_169[i]:
                for c3, c4 in COMBOS_169[j]:
                    if len({c1, c2, c3, c4}) < 4:
                        continue
                    h0, h1, v0, v1 = cards_to_ids((c1, c2), (c3, c4))
                    k = canonical_showdown_key(h0, h1, v0, v1)
                    total += eq_dict[k]
                    count += 1
            chunk[ri, j] = total / count if count else 0.5
    np.savez_compressed(temp_path, rows=np.array(row_indices, dtype=np.int32), chunk=chunk)
    return temp_path


def build_parallel_merge(
    *,
    workers: int,
    hu_source: Path,
    out: Path,
    progress: bool = True,
) -> np.ndarray:
    """Workers write row-chunk temps; one process merges (single writer to final npz)."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(range(N_BUCKETS))
    chunks = [rows[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    temps: list[Path] = []
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for wi, chunk_rows in enumerate(chunks):
            temp = TEMP_DIR / f"pair_eq_rows_{wi}.npz"
            temps.append(temp)
            futs[pool.submit(_worker_build_rows, chunk_rows, str(hu_source), str(temp))] = chunk_rows
        for fut in as_completed(futs):
            if progress:
                log_step(f"Worker finished rows {futs[fut][0]}..{futs[fut][-1]}")
    hu = np.full((N_BUCKETS, N_BUCKETS), 0.5, dtype=np.float64)
    np.fill_diagonal(hu, 0.5)
    for temp in temps:
        data = np.load(temp)
        row_idx = data["rows"].tolist()
        chunk = np.asarray(data["chunk"], dtype=np.float64)
        for ri, i in enumerate(row_idx):
            hu[i, :] = chunk[ri, :]
    elapsed = time.monotonic() - t0
    save_pair_eq_table(
        hu,
        out,
        build_elapsed_s=elapsed,
        hu_source=hu_source,
        writer=f"parallel_{workers}_merge_single",
    )
    for temp in temps:
        temp.unlink(missing_ok=True)
    return hu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_PAIR_EQ_TABLE)
    ap.add_argument("--hu-source", type=Path, default=HU_SHOWDOWN_TABLE)
    ap.add_argument("--workers", type=int, default=0, help="0=fast single-pass (default)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--kill-stale", action="store_true")
    args = ap.parse_args()

    log_path = init_run_log("build_trio_pair_eq_table")
    if args.kill_stale:
        kill_stale_tensor_builds(include_pair_eq=True)

    pin_compute_threads(1)
    warmup()
    log_step(f"Pair-eq table build log={log_path} out={args.out}")

    if args.verify_only:
        hu = require_pair_eq_table(args.out)
        report = {
            "verify_only": True,
            "shape": list(hu.shape),
            "complete": table_is_complete(hu),
            "file_hash": pair_eq_table_file_hash(args.out),
            "parity": parity_vs_hu_oracle(hu),
        }
        OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    t0 = time.monotonic()
    workers = args.workers if args.workers > 0 else 0
    if workers > 1:
        hu = build_parallel_merge(workers=workers, hu_source=args.hu_source, out=args.out)
        writer = f"parallel_{workers}_merge_single"
    else:
        hu = build_pair_eq_table(path=args.out, hu_source=args.hu_source, workers=8, progress=True)
        writer = "single_process"
    wall = time.monotonic() - t0

    parity = parity_vs_hu_oracle(hu, samples=32)
    report = {
        "table": str(args.out),
        "shape": list(hu.shape),
        "n_pairs": N_BUCKETS * N_BUCKETS,
        "version": 2,
        "writer": writer,
        "wall_s": wall,
        "complete": table_is_complete(hu),
        "hu_source": str(args.hu_source),
        "hu_source_hash": hu_showdown_file_hash(args.hu_source),
        "file_hash": pair_eq_table_file_hash(args.out),
        "parity": parity,
        "labels_count": len(LABELS_169),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_step(f"Done wall={wall:.1f}s complete={report['complete']} parity_max={parity['max_err']:.2e}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
