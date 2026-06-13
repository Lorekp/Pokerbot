"""Precomputed 3-way bucket showdown shares for 3-max push/fold solvers."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from aof_cfr.buckets import COMBOS_169, LABELS_169
from aof_cfr.build_progress import log_build_progress
from aof_cfr.cpu_pool import (
    assert_no_oversubscription,
    default_pool_workers,
    describe_parallel_layout,
    native_openmp_threads,
    native_row_pool_workers,
    omp_threads_per_row_worker,
    pin_compute_threads,
    pool_layout,
    tensor_build_workers,
)
from aof_cfr.trio_eval_mc import eval7_backend_name, mc_nway_shares, mc_nway_shares_adaptive
from aof_cfr.trio_pair_eq_table import (
    hu_pair_eq_values,
    pair_eq_row_slice,
    pair_eq_table_file_hash,
    require_pair_eq_table,
)
from aof_equity_grids import _card_str_to_id, _exact_six_card_combo_metrics, _three_pair_equities_exact

_MC_CACHE: dict[tuple[int, ...], np.ndarray] = {}

DEFAULT_TENSOR = Path("data/aof/cfr_base/trio_eq_tensor.npz")
DEFAULT_TENSOR_CHECKPOINT_DIR = Path("data/aof/cfr_base/tensor_row_checkpoints")
GATE_ROW_SLICE_DIR = Path("data/aof/cfr_base/gate_row_slices")
N_BUCKETS = 169
DEFAULT_MC_BOARDS = 768
# Per-combo adaptive MC cap for 3-way shares (pair-eq comes from precomputed exact table).
DEFAULT_BOARDS_PER_COMBO = 192
DEFAULT_TRIPLE_SAMPLE_BUDGET = DEFAULT_BOARDS_PER_COMBO
EXACT_TRIPLE_COMBO_MAX = 36
MIN_BOARDS_PER_COMBO = 16
MC_SHARE_STDERR_GATE = 0.001
BUILD_MODE = "combo_exact_adaptive_mc"


def _use_native_trio() -> bool:
    try:
        from tools.ompeval_cli.native import native_trio_available

        return native_trio_available()
    except Exception:
        return False


def tensor_build_backend_name() -> str:
    if _use_native_trio():
        return "trio_native_openmp"
    return eval7_backend_name()


def require_native_trio_build(*, context: str = "tensor build") -> None:
    """Refuse slow Python/Numba per-triple loops for production builds."""
    if _use_native_trio():
        return
    raise RuntimeError(
        f"{context} requires native trio kernel (hu_table_native). "
        "Rebuild tools/ompeval_cli — do not use Python/Numba fallback for full tensors."
    )


def _native_column_batch_size(n: int, omp_threads: int) -> int:
    """Columns per OpenMP chunk: large enough for triple-parallel saturation, ~8 progress lines."""
    min_cols = max(omp_threads // 2, 16)
    target_batches = 8
    return max(min_cols, math.ceil(n / target_batches))


def _native_build_row_columns(
    row: int,
    shares: np.ndarray,
    pair_eq: np.ndarray,
    combo_counts: np.ndarray,
    *,
    mc_boards: int,
    n: int,
    omp_threads: int,
    progress: bool,
    progress_label: str | None = None,
    started: float | None = None,
    exact_combo_max: int = EXACT_TRIPLE_COMBO_MAX,
    pair_eq_lookup: np.ndarray | None = None,
) -> None:
    """Fill one bucket row via native OpenMP in column batches (parallel + progress logs)."""
    from tools.ompeval_cli import native as omp_native

    boards_per_combo = mc_boards if mc_boards > 0 else DEFAULT_BOARDS_PER_COMBO
    if pair_eq_lookup is None:
        pair_eq_lookup = pair_eq_row_slice(row, require_pair_eq_table())
    t0 = started if started is not None else time.monotonic()
    label = progress_label if progress_label is not None else f"row {row}"
    batch_cols = _native_column_batch_size(n, omp_threads)
    triples_total = n * n
    if progress:
        backend = tensor_build_backend_name()
        print(
            f"  {label}: building {triples_total} triples "
            f"(boards_per_combo={boards_per_combo}, omp={omp_threads}, "
            f"batch_cols={batch_cols}, backend={backend}, pair_eq=table) ...",
            flush=True,
        )
    last_log: list[float] = [0.0]
    triples_done = 0
    for j0 in range(0, n, batch_cols):
        j1 = min(j0 + batch_cols, n)
        chunk_sh, chunk_pe, chunk_cnt = omp_native.build_trio_row_chunk(
            row,
            j0,
            j1,
            mc_boards=boards_per_combo,
            exact_combo_max=exact_combo_max,
            min_boards_per_combo=MIN_BOARDS_PER_COMBO,
            stderr_gate=MC_SHARE_STDERR_GATE,
            n_buckets=n,
            threads=omp_threads,
            pair_eq_lookup=pair_eq_lookup,
        )
        width = j1 - j0
        shares[j0:j1] = np.asarray(chunk_sh, dtype=np.float64)
        pair_eq[j0:j1] = np.asarray(chunk_pe, dtype=np.float64)
        combo_counts[j0:j1] = np.asarray(chunk_cnt, dtype=np.int32)
        triples_done += width * n
        if progress:
            log_build_progress(
                triples_done,
                triples_total,
                t0,
                label=label,
                unit="triples",
                min_interval_s=30.0,
                _last_log=last_log,
            )
    if progress:
        elapsed = time.monotonic() - t0
        rate = triples_total / elapsed if elapsed > 0 else 0.0
        print(
            f"  {label}: done {triples_total} triples in {elapsed:.1f}s ({rate:.1f} triples/s)",
            flush=True,
        )


def _pair_eq_all(h0: int, h1: int, h2: int, h3: int, h4: int, h5: int) -> tuple[float, float, float]:
    bb, bs, sb = _three_pair_equities_exact(h0, h1, h2, h3, h4, h5)
    return float(bb), float(bs), float(sb)

# INVARIANT (correctness trap): tensor cells are computed to best accuracy per (i,j,k).
# Deal-frequency weighting is used ONLY in validation gates — never here, never in FP/BR.


def trio_deal_weights(n: int = N_BUCKETS) -> np.ndarray:
    """Uniform ordered bucket triples — matches HU ``solver_hu_deal_weights`` pattern."""
    w = np.ones((n, n, n), dtype=np.float64)
    w /= w.sum()
    return w


def _combo_seed(i: int, j: int, k: int) -> int:
    h = hashlib.sha256(f"{i}:{j}:{k}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def _all_combo_triples(
    i: int, j: int, k: int
) -> list[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]]:
    out: list[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]] = []
    for c0, c1 in COMBOS_169[i]:
        for c2, c3 in COMBOS_169[j]:
            for c4, c5 in COMBOS_169[k]:
                if len({c0, c1, c2, c3, c4, c5}) == 6:
                    out.append(((c0, c1), (c2, c3), (c4, c5)))
    return out


def _avg_shares_for_triple(
    i: int,
    j: int,
    k: int,
    *,
    mc_boards: int = DEFAULT_MC_BOARDS,
    triple_budget: int | None = None,
    pair_eq_table: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, float, int]:
    """Combo-averaged shares + exact table pair-eq for bucket triple (i,j,k).

    Low-combo collision triples use exact board enumeration per combo.
    High-combo triples use adaptive MC for shares only; pair-eq from precomputed table.
    """
    boards_per_combo = int(triple_budget if triple_budget is not None else mc_boards)
    if boards_per_combo <= 0:
        boards_per_combo = DEFAULT_BOARDS_PER_COMBO

    if _use_native_trio():
        from tools.ompeval_cli import native as omp_native

        if pair_eq_table is None:
            pair_eq_table = require_pair_eq_table()
        lookup = pair_eq_row_slice(i, pair_eq_table)
        return omp_native.avg_metrics_for_triple(
            i,
            j,
            k,
            mc_boards=boards_per_combo,
            exact_combo_max=EXACT_TRIPLE_COMBO_MAX,
            min_boards_per_combo=MIN_BOARDS_PER_COMBO,
            stderr_gate=MC_SHARE_STDERR_GATE,
            pair_eq_lookup=lookup,
        )

    combos = _all_combo_triples(i, j, k)
    count = len(combos)
    if count == 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.5, 0.5, 0.5, 0)

    if pair_eq_table is None:
        pair_eq_table = require_pair_eq_table()
    pe = hu_pair_eq_values(i, j, k, pair_eq_table)

    if count <= EXACT_TRIPLE_COMBO_MAX:
        s0, s1, s2, _, _, _, _ = exact_shares_for_triple_oracle(i, j, k)
        return float(s0), float(s1), float(s2), float(pe[0]), float(pe[1]), float(pe[2]), count

    shares_sum = np.zeros(3, dtype=np.float64)
    sub_seed = _combo_seed(i, j, k)
    for (c0, c1), (c2, c3), (c4, c5) in combos:
        h0 = _card_str_to_id(c0)
        h1 = _card_str_to_id(c1)
        h2 = _card_str_to_id(c2)
        h3 = _card_str_to_id(c3)
        h4 = _card_str_to_id(c4)
        h5 = _card_str_to_id(c5)
        flat = np.array([h0, h1, h2, h3, h4, h5], dtype=np.int32)
        cache_key = (h0, h1, h2, h3, h4, h5, boards_per_combo, 1)
        cached = _MC_CACHE.get(cache_key)
        if cached is None:
            sh, _, _, _ = mc_nway_shares_adaptive(
                flat,
                3,
                sub_seed,
                stderr_gate=MC_SHARE_STDERR_GATE,
                max_boards=boards_per_combo,
                min_boards=MIN_BOARDS_PER_COMBO,
            )
            cached = sh.copy()
            _MC_CACHE[cache_key] = cached
        sub_seed += 1
        shares_sum += cached
    inv = 1.0 / count
    avg = shares_sum * inv
    return (
        float(avg[0]),
        float(avg[1]),
        float(avg[2]),
        float(pe[0]),
        float(pe[1]),
        float(pe[2]),
        count,
    )


def _avg_shares_for_triple_legacy_per_combo(
    i: int,
    j: int,
    k: int,
    *,
    mc_boards: int = DEFAULT_MC_BOARDS,
) -> tuple[float, float, float, float, float, float, int]:
    """Legacy per-combo fixed boards — Stage 1 diagnosis only."""
    shares_sum = np.zeros(3, dtype=np.float64)
    btn_bb_sum = 0.0
    btn_sb_sum = 0.0
    sb_bb_sum = 0.0
    count = 0
    sub_seed = _combo_seed(i, j, k)
    for (c0, c1), (c2, c3), (c4, c5) in _all_combo_triples(i, j, k):
        h0 = _card_str_to_id(c0)
        h1 = _card_str_to_id(c1)
        h2 = _card_str_to_id(c2)
        h3 = _card_str_to_id(c3)
        h4 = _card_str_to_id(c4)
        h5 = _card_str_to_id(c5)
        flat = np.array([h0, h1, h2, h3, h4, h5], dtype=np.int32)
        sh = mc_nway_shares(flat, 3, mc_boards, sub_seed)
        sub_seed += 1
        shares_sum += sh
        bb, bs, sb = _pair_eq_all(h0, h1, h2, h3, h4, h5)
        btn_bb_sum += bb
        btn_sb_sum += bs
        sb_bb_sum += sb
        count += 1
    if count == 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.5, 0.5, 0.5, 0)
    inv = 1.0 / count
    avg = shares_sum * inv
    return (
        float(avg[0]),
        float(avg[1]),
        float(avg[2]),
        btn_bb_sum * inv,
        btn_sb_sum * inv,
        sb_bb_sum * inv,
        count,
    )


def exact_shares_for_triple_oracle(
    i: int, j: int, k: int
) -> tuple[float, float, float, float, float, float, int]:
    """Validation oracle: exact 3-way boards + exact 2-way combo equities."""
    shares_sum = np.zeros(3, dtype=np.float64)
    btn_bb_sum = 0.0
    btn_sb_sum = 0.0
    sb_bb_sum = 0.0
    count = 0
    for (c0, c1), (c2, c3), (c4, c5) in _all_combo_triples(i, j, k):
        h0 = _card_str_to_id(c0)
        h1 = _card_str_to_id(c1)
        h2 = _card_str_to_id(c2)
        h3 = _card_str_to_id(c3)
        h4 = _card_str_to_id(c4)
        h5 = _card_str_to_id(c5)
        s0, s1, s2, bb, bs, sb = _exact_six_card_combo_metrics(h0, h1, h2, h3, h4, h5)
        shares_sum += np.array([s0, s1, s2], dtype=np.float64)
        btn_bb_sum += bb
        btn_sb_sum += bs
        sb_bb_sum += sb
        count += 1
    if count == 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.5, 0.5, 0.5, 0)
    inv = 1.0 / count
    avg = shares_sum * inv
    return (
        float(avg[0]),
        float(avg[1]),
        float(avg[2]),
        btn_bb_sum * inv,
        btn_sb_sum * inv,
        sb_bb_sum * inv,
        count,
    )


def exact_shares_oracle_for_triple(
    row: int, j: int, k: int
) -> tuple[float, float, float, int]:
    """Exact combo-averaged 3-way shares only (fast native path when available)."""
    if _use_native_trio():
        from tools.ompeval_cli import native as omp_native

        lookup = pair_eq_row_slice(row, require_pair_eq_table())
        s0, s1, s2, _, _, _, cnt = omp_native.avg_metrics_for_triple(
            row,
            j,
            k,
            mc_boards=DEFAULT_BOARDS_PER_COMBO,
            exact_combo_max=999999,
            min_boards_per_combo=MIN_BOARDS_PER_COMBO,
            stderr_gate=MC_SHARE_STDERR_GATE,
            pair_eq_lookup=lookup,
        )
        return float(s0), float(s1), float(s2), int(cnt)
    s0, s1, s2, _, _, _, cnt = exact_shares_for_triple_oracle(row, j, k)
    return float(s0), float(s1), float(s2), int(cnt)


def _warm_kernels() -> None:
    if _use_native_trio():
        return
    flat = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    mc_nway_shares(flat, 3, 32, 1)
    _three_pair_equities_exact(0, 1, 2, 3, 4, 5)
    _exact_six_card_combo_metrics(0, 1, 2, 3, 4, 5)


def _tensor_worker_init() -> None:
    """One-time setup per pool worker: single-thread BLAS/Numba + warm kernels."""
    pin_compute_threads(1)
    _warm_kernels()


def _build_chunk(args: tuple[int, int, int, int, int, int]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    i_start, i_end, n, mc_boards, _worker_id, omp_threads = args
    width = i_end - i_start
    chunk_shares = np.zeros((width, n, n, 3), dtype=np.float64)
    chunk_pair = np.zeros((width, n, n, 3), dtype=np.float64)
    chunk_counts = np.zeros((width, n, n), dtype=np.int32)

    if _use_native_trio():
        pin_compute_threads(omp_threads)
        _warm_kernels()
        chunk_started = time.monotonic()
        for ii, i in enumerate(range(i_start, i_end)):
            _native_build_row_columns(
                i,
                chunk_shares[ii],
                chunk_pair[ii],
                chunk_counts[ii],
                mc_boards=mc_boards,
                n=n,
                omp_threads=omp_threads,
                progress=True,
                progress_label=f"bucket row {i} cols",
                started=chunk_started,
            )
        return i_start, chunk_shares, chunk_pair, chunk_counts

    for ii, i in enumerate(range(i_start, i_end)):
        for j in range(n):
            for k in range(n):
                s0, s1, s2, bb, bs, sb, cnt = _avg_shares_for_triple(i, j, k, mc_boards=mc_boards)
                chunk_shares[ii, j, k, 0] = s0
                chunk_shares[ii, j, k, 1] = s1
                chunk_shares[ii, j, k, 2] = s2
                chunk_pair[ii, j, k, 0] = bb
                chunk_pair[ii, j, k, 1] = bs
                chunk_pair[ii, j, k, 2] = sb
                chunk_counts[ii, j, k] = cnt
    return i_start, chunk_shares, chunk_pair, chunk_counts


def _build_row_j_chunk(
    args: tuple[int, int, int, int, int, int],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Process j in [j_start, j_end) for a fixed bucket row (pool worker)."""
    row, j_start, j_end, mc_boards, n, _worker_id = args
    width = j_end - j_start
    chunk_sh = np.zeros((width, n, 3), dtype=np.float64)
    chunk_pe = np.zeros((width, n, 3), dtype=np.float64)
    chunk_cnt = np.zeros((width, n), dtype=np.int32)
    for jj, j in enumerate(range(j_start, j_end)):
        for k in range(n):
            s0, s1, s2, bb, bs, sb, cnt = _avg_shares_for_triple(
                row, j, k, mc_boards=mc_boards
            )
            chunk_sh[jj, k, 0] = s0
            chunk_sh[jj, k, 1] = s1
            chunk_sh[jj, k, 2] = s2
            chunk_pe[jj, k, 0] = bb
            chunk_pe[jj, k, 1] = bs
            chunk_pe[jj, k, 2] = sb
            chunk_cnt[jj, k] = cnt
    return j_start, chunk_sh, chunk_pe, chunk_cnt


def _benchmark_j_chunk(args: tuple[int, int, int, int, int, int]) -> tuple[int, int, float, int]:
    """Timing-only path for benchmark (discards computed values)."""
    row, j_start, j_end, mc_boards, n, _worker_id = args
    started = time.monotonic()
    done = 0
    for j in range(j_start, j_end):
        for k in range(n):
            _avg_shares_for_triple(row, j, k, mc_boards=mc_boards)
            done += 1
    return j_start, j_end, time.monotonic() - started, done


def build_one_bucket_row(
    *,
    row: int = 0,
    mc_boards: int = DEFAULT_MC_BOARDS,
    n: int = N_BUCKETS,
    workers: int | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Build one bucket row (169×169 slice) with optional column-parallel pool."""
    require_pair_eq_table()
    native = _use_native_trio()
    if workers is None:
        workers = tensor_build_workers(native_trio=native)
    omp_threads = native_openmp_threads() if native else 1
    layout = pool_layout(workers, native_trio=native, openmp_threads=omp_threads)
    backend = tensor_build_backend_name()
    if progress:
        print(
            f"Building bucket row {row} mc_boards={mc_boards} backend={backend} "
            f"workers={layout['workers']} x {layout['threads_per_worker']} thread ...",
            flush=True,
        )
    pin_compute_threads(omp_threads if native else 1)
    _warm_kernels()
    shares = np.zeros((n, n, 3), dtype=np.float64)
    pair_eq = np.zeros((n, n, 3), dtype=np.float64)
    combo_counts = np.zeros((n, n), dtype=np.int32)
    started = time.monotonic()
    triples = n * n

    if native:
        _native_build_row_columns(
            row,
            shares,
            pair_eq,
            combo_counts,
            mc_boards=mc_boards,
            n=n,
            omp_threads=omp_threads,
            progress=progress,
            started=started,
        )
    elif workers <= 1:
        last_log: list[float] = [0.0]
        for j in range(n):
            for k in range(n):
                s0, s1, s2, bb, bs, sb, cnt = _avg_shares_for_triple(
                    row, j, k, mc_boards=mc_boards
                )
                shares[j, k, 0] = s0
                shares[j, k, 1] = s1
                shares[j, k, 2] = s2
                pair_eq[j, k, 0] = bb
                pair_eq[j, k, 1] = bs
                pair_eq[j, k, 2] = sb
                combo_counts[j, k] = cnt
            if progress:
                log_build_progress(
                    j + 1,
                    n,
                    started,
                    label=f"row {row} cols",
                    unit="cols",
                    min_interval_s=30.0,
                    _last_log=last_log,
                )
    else:
        chunk_cols = max(1, math.ceil(n / workers))
        tasks = [
            (row, j0, min(j0 + chunk_cols, n), mc_boards, n, wi)
            for wi, j0 in enumerate(range(0, n, chunk_cols))
            if j0 < n
        ]
        done = 0
        last_log: list[float] = [0.0]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_tensor_worker_init,
        ) as pool:
            futures = [pool.submit(_build_row_j_chunk, t) for t in tasks]
            for fut in as_completed(futures):
                j_start, chunk_sh, chunk_pe, chunk_cnt = fut.result()
                end = j_start + chunk_sh.shape[0]
                shares[j_start:end] = chunk_sh
                pair_eq[j_start:end] = chunk_pe
                combo_counts[j_start:end] = chunk_cnt
                done += int(chunk_cnt.sum()) if chunk_cnt.size else chunk_sh.shape[0] * n
                if progress:
                    log_build_progress(
                        min(done, triples),
                        triples,
                        started,
                        label=f"row {row} triples",
                        unit="triples",
                        min_interval_s=30.0,
                        _last_log=last_log,
                    )

    elapsed = time.monotonic() - started
    valid = int((combo_counts > 0).sum())
    stats = {
        "row": float(row),
        "mc_boards": float(mc_boards),
        "workers": float(workers),
        "elapsed_s": elapsed,
        "valid_triples": float(valid),
        "total_triples": float(triples),
        "projected_full_s": elapsed * math.ceil(n / max(1, workers)),
        "eval_backend": backend,
    }
    if progress:
        print(f"Done row build: {stats}", flush=True)
    return shares, pair_eq, combo_counts, stats


def save_trio_row_slice(
    path: Path,
    row: int,
    shares: np.ndarray,
    pair_eq: np.ndarray,
    combo_counts: np.ndarray,
    *,
    mc_boards: int = DEFAULT_MC_BOARDS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shares=shares,
        pair_eq=pair_eq,
        combo_counts=combo_counts,
        bucket_row=np.int32(row),
        labels=np.array(LABELS_169, dtype=object),
        build_mode=np.array(BUILD_MODE),
        mc_boards=np.int32(mc_boards),
        eval_backend=np.array(tensor_build_backend_name()),
    )


def tensor_row_checkpoint_path(
    row: int,
    mc_boards: int,
    checkpoint_dir: Path = DEFAULT_TENSOR_CHECKPOINT_DIR,
) -> Path:
    return checkpoint_dir / f"row{row:03d}_bpc{mc_boards}.npz"


def row_timing_manifest_path(
    mc_boards: int,
    checkpoint_dir: Path = DEFAULT_TENSOR_CHECKPOINT_DIR,
) -> Path:
    return checkpoint_dir / f"row_timing_bpc{mc_boards}.json"


def _process_rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def load_trio_row_slice(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    data = np.load(path, allow_pickle=True)
    row = int(np.asarray(data["bucket_row"]).item())
    boards = int(np.asarray(data["mc_boards"]).item())
    shares = np.asarray(data["shares"], dtype=np.float64)
    pair_eq = np.asarray(data["pair_eq"], dtype=np.float64)
    combo_counts = np.asarray(data["combo_counts"], dtype=np.int32)
    return row, shares, pair_eq, combo_counts, boards


def list_tensor_row_checkpoints(
    checkpoint_dir: Path,
    mc_boards: int,
) -> list[int]:
    if not checkpoint_dir.is_dir():
        return []
    rows: list[int] = []
    for path in sorted(checkpoint_dir.glob(f"row*_bpc{mc_boards}.npz")):
        try:
            row, shares, _, _, boards = load_trio_row_slice(path)
        except Exception:
            continue
        if boards != mc_boards:
            continue
        if shares.shape != (N_BUCKETS, N_BUCKETS, 3):
            continue
        if 0 <= row < N_BUCKETS:
            rows.append(row)
    return sorted(set(rows))


def load_tensor_row_checkpoint(
    path: Path,
    *,
    mc_boards: int,
    expected_row: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row, shares, pair_eq, combo_counts, boards = load_trio_row_slice(path)
    if boards != mc_boards:
        raise ValueError(f"{path}: mc_boards={boards} != {mc_boards}")
    if expected_row is not None and row != expected_row:
        raise ValueError(f"{path}: bucket_row={row} != expected {expected_row}")
    if shares.shape != (N_BUCKETS, N_BUCKETS, 3):
        raise ValueError(f"{path}: bad shares shape {shares.shape}")
    return shares, pair_eq, combo_counts


def save_tensor_row_checkpoint(
    checkpoint_dir: Path,
    row: int,
    shares: np.ndarray,
    pair_eq: np.ndarray,
    combo_counts: np.ndarray,
    *,
    mc_boards: int,
) -> Path:
    path = tensor_row_checkpoint_path(row, mc_boards, checkpoint_dir)
    save_trio_row_slice(path, row, shares, pair_eq, combo_counts, mc_boards=mc_boards)
    return path


def seed_tensor_checkpoints_from_gate_slices(
    checkpoint_dir: Path,
    mc_boards: int,
) -> list[int]:
    """Copy spot-check row slices into checkpoint dir when missing."""
    if not GATE_ROW_SLICE_DIR.is_dir():
        return []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seeded: list[int] = []
    for src in sorted(GATE_ROW_SLICE_DIR.glob(f"row*_bpc{mc_boards}.npz")):
        try:
            row, _, _, _, boards = load_trio_row_slice(src)
        except Exception:
            continue
        if boards != mc_boards:
            continue
        dst = tensor_row_checkpoint_path(row, mc_boards, checkpoint_dir)
        if dst.is_file():
            continue
        shutil.copy2(src, dst)
        seeded.append(row)
    return sorted(seeded)


def append_row_timing_entry(
    checkpoint_dir: Path,
    mc_boards: int,
    entry: dict,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = row_timing_manifest_path(mc_boards, checkpoint_dir)
    rows: list[dict] = []
    if path.is_file():
        rows = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("row") != entry.get("row")]
    rows.append(entry)
    rows.sort(key=lambda r: int(r["row"]))
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def load_completed_tensor_rows(
    checkpoint_dir: Path,
    mc_boards: int,
    shares: np.ndarray,
    pair_eq: np.ndarray,
    combo_counts: np.ndarray,
) -> set[int]:
    completed: set[int] = set()
    for row in list_tensor_row_checkpoints(checkpoint_dir, mc_boards):
        path = tensor_row_checkpoint_path(row, mc_boards, checkpoint_dir)
        sh, pe, cnt = load_tensor_row_checkpoint(path, mc_boards=mc_boards, expected_row=row)
        shares[row] = sh
        pair_eq[row] = pe
        combo_counts[row] = cnt
        completed.add(row)
    return completed


def benchmark_one_bucket_row(
    *,
    row: int = 0,
    mc_boards: int = DEFAULT_MC_BOARDS,
    n: int = N_BUCKETS,
    workers: int | None = None,
    progress_every: int = 500,
) -> dict[str, float]:
    """Benchmark one bucket row; project full-tensor wall time."""
    native = _use_native_trio()
    if workers is None:
        workers = tensor_build_workers(native_trio=native)
    omp_threads = native_openmp_threads() if native else 1
    layout = pool_layout(workers, native_trio=native, openmp_threads=omp_threads)
    backend = tensor_build_backend_name()
    triples = n * n
    print(
        f"Benchmark row {row} mc_boards={mc_boards} backend={backend} "
        f"workers={layout['workers']} x {layout['threads_per_worker']} thread ...",
        flush=True,
    )
    pin_compute_threads(omp_threads if native else 1)
    _warm_kernels()
    started = time.monotonic()

    if native:
        shares = np.zeros((n, n, 3), dtype=np.float64)
        pair_eq = np.zeros((n, n, 3), dtype=np.float64)
        combo_counts = np.zeros((n, n), dtype=np.int32)
        _native_build_row_columns(
            row,
            shares,
            pair_eq,
            combo_counts,
            mc_boards=mc_boards,
            n=n,
            omp_threads=omp_threads,
            progress=True,
            started=started,
        )
    elif workers <= 1:
        done = 0
        last_log: list[float] = [0.0]
        for j in range(n):
            for k in range(n):
                _avg_shares_for_triple(row, j, k, mc_boards=mc_boards)
                done += 1
            if progress_every > 0:
                log_build_progress(
                    done,
                    triples,
                    started,
                    label=f"row {row} triples",
                    unit="triples",
                    min_interval_s=30.0,
                    _last_log=last_log,
                )
    else:
        chunk_cols = max(1, math.ceil(n / workers))
        tasks = [
            (row, j0, min(j0 + chunk_cols, n), mc_boards, n, wi)
            for wi, j0 in enumerate(range(0, n, chunk_cols))
            if j0 < n
        ]
        done = 0
        last_log = [0.0]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_tensor_worker_init,
        ) as pool:
            futures = [pool.submit(_benchmark_j_chunk, t) for t in tasks]
            for fut in as_completed(futures):
                j_start, j_end, chunk_s, chunk_done = fut.result()
                done += chunk_done
                log_build_progress(
                    done,
                    triples,
                    started,
                    label=f"row {row} triples",
                    unit="triples",
                    min_interval_s=15.0,
                    _last_log=last_log,
                )

    elapsed = time.monotonic() - started
    projected_single = elapsed * n
    projected_pool = elapsed * math.ceil(n / max(1, workers))
    stats = {
        "row": float(row),
        "mc_boards": float(mc_boards),
        "workers": float(workers),
        "elapsed_s": elapsed,
        "triples_per_row": float(triples),
        "projected_full_s": projected_pool,
        "projected_single_thread_s": projected_single,
        "eval_backend": backend,
    }
    print(f"Done row benchmark: {stats}", flush=True)
    return stats


def build_trio_equity_tensor(
    *,
    workers: int | None = None,
    chunk_rows: int = 1,
    mc_boards: int = DEFAULT_MC_BOARDS,
    progress: bool = True,
    row_parallel: bool = False,
    openmp_threads: int | None = None,
    checkpoint_dir: Path | None = DEFAULT_TENSOR_CHECKPOINT_DIR,
    resume: bool = True,
    seed_gate_slices: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build combo-exact-averaged 3-way shares (MC boards per combo)."""
    require_native_trio_build(context="Full trio equity tensor")
    require_pair_eq_table()
    native = True
    row_par = bool(row_parallel)
    if workers is None:
        workers = tensor_build_workers(native_trio=True, row_parallel=row_par)
    if row_par:
        chunk_rows = 1
        omp_threads = (
            int(openmp_threads)
            if openmp_threads is not None
            else omp_threads_per_row_worker(workers)
        )
        assert_no_oversubscription(workers, omp_threads, label="row-parallel tensor")
        pin_compute_threads(omp_threads)
    else:
        workers = 1
        omp_threads = (
            int(openmp_threads) if openmp_threads is not None else native_openmp_threads()
        )
        assert_no_oversubscription(1, omp_threads, label="serial tensor")
        pin_compute_threads(omp_threads)
    _warm_kernels()
    layout = pool_layout(
        workers,
        native_trio=True,
        openmp_threads=omp_threads,
        row_parallel=row_par,
    )
    par_info = describe_parallel_layout(workers, omp_threads)
    backend = tensor_build_backend_name()
    n = N_BUCKETS
    shares = np.zeros((n, n, n, 3), dtype=np.float64)
    pair_eq = np.zeros((n, n, n, 3), dtype=np.float64)
    combo_counts = np.zeros((n, n, n), dtype=np.int32)
    deal_weights = trio_deal_weights(n)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    completed: set[int] = set()
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if seed_gate_slices:
            seeded = seed_tensor_checkpoints_from_gate_slices(ckpt_dir, mc_boards)
            if seeded and progress:
                print(
                    f"  seeded {len(seeded)} row checkpoints from gate slices: {seeded}",
                    flush=True,
                )
        if resume:
            completed = load_completed_tensor_rows(ckpt_dir, mc_boards, shares, pair_eq, combo_counts)
            if completed and progress:
                print(
                    f"  resumed {len(completed)} rows from {ckpt_dir} "
                    f"(rows {min(completed)}–{max(completed)})",
                    flush=True,
                )

    tasks = [
        (i, min(i + chunk_rows, n), n, mc_boards, wi, omp_threads)
        for wi, i in enumerate(range(0, n, chunk_rows))
        if not resume or not all(r in completed for r in range(i, min(i + chunk_rows, n)))
    ]
    started = time.monotonic()
    last_log: list[float] = [0.0]
    if progress:
        print(
            f"Building 3-way bucket tensor ({n}^3 triples, "
            f"{len(tasks)} row chunks pending, {layout['workers']} workers x "
            f"{layout['threads_per_worker']} OpenMP, "
            f"total_threads={par_info['total_compute_threads']}/{par_info['logical_cpus']} "
            f"oversubscribed={par_info['oversubscribed']}, "
            f"row_parallel={row_par}, mc_boards={mc_boards}, backend={backend}, "
            f"checkpoint={ckpt_dir}) ...",
            flush=True,
        )

    if not row_par:
        for i in range(n):
            if i in completed:
                continue
            row_started = time.monotonic()
            rss_before = _process_rss_mb()
            i_start, chunk_sh, chunk_pe, chunk_cnt = _build_chunk(
                (i, i + 1, n, mc_boards, 0, omp_threads)
            )
            shares[i_start : i_start + 1] = chunk_sh
            pair_eq[i_start : i_start + 1] = chunk_pe
            combo_counts[i_start : i_start + 1] = chunk_cnt
            row_elapsed = time.monotonic() - row_started
            rss_after = _process_rss_mb()
            if ckpt_dir is not None:
                save_tensor_row_checkpoint(
                    ckpt_dir,
                    i,
                    chunk_sh[0],
                    chunk_pe[0],
                    chunk_cnt[0],
                    mc_boards=mc_boards,
                )
                append_row_timing_entry(
                    ckpt_dir,
                    mc_boards,
                    {
                        "row": i,
                        "label": LABELS_169[i],
                        "elapsed_s": round(row_elapsed, 2),
                        "rss_mb_before": rss_before,
                        "rss_mb_after": rss_after,
                        "triples_per_s": round((n * n) / row_elapsed, 2) if row_elapsed > 0 else 0.0,
                    },
                )
            if progress:
                log_build_progress(
                    i + 1,
                    n,
                    started,
                    label="bucket rows",
                    unit="rows",
                    min_interval_s=30.0,
                    _last_log=last_log,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_tensor_worker_init,
        ) as pool:
            futures = [pool.submit(_build_chunk, t) for t in tasks]
            for fut in as_completed(futures):
                i_start, chunk_sh, chunk_pe, chunk_cnt = fut.result()
                end = i_start + chunk_sh.shape[0]
                shares[i_start:end] = chunk_sh
                pair_eq[i_start:end] = chunk_pe
                combo_counts[i_start:end] = chunk_cnt
                if ckpt_dir is not None:
                    for ii, row in enumerate(range(i_start, end)):
                        save_tensor_row_checkpoint(
                            ckpt_dir,
                            row,
                            chunk_sh[ii],
                            chunk_pe[ii],
                            chunk_cnt[ii],
                            mc_boards=mc_boards,
                        )
                if progress:
                    log_build_progress(
                        end,
                        n,
                        started,
                        label="bucket rows",
                        unit="rows",
                        min_interval_s=30.0,
                        _last_log=last_log,
                    )
    if progress:
        elapsed = time.monotonic() - started
        print(f"  full tensor build elapsed={elapsed / 3600:.2f}h", flush=True)
    return shares, pair_eq, deal_weights, combo_counts


def row_combo_seed_scheme(n: int = N_BUCKETS) -> np.ndarray:
    """Per-row anchor seeds (combo_seed(row,0,0)) for manifest reproducibility."""
    return np.array([_combo_seed(r, 0, 0) for r in range(n)], dtype=np.uint32)


def save_trio_tensor(
    path: Path,
    shares: np.ndarray,
    pair_eq: np.ndarray,
    deal_weights: np.ndarray,
    combo_counts: np.ndarray,
    *,
    mc_boards: int = DEFAULT_MC_BOARDS,
    row_workers: int = 1,
    row_parallel: bool = False,
) -> None:
    from aof_cfr.trio_pair_eq_table import DEFAULT_PAIR_EQ_TABLE

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shares=shares,
        pair_eq=pair_eq,
        deal_weights=deal_weights,
        combo_counts=combo_counts,
        labels=np.array(LABELS_169, dtype=object),
        build_mode=np.array(BUILD_MODE),
        mc_boards=np.int32(mc_boards),
        boards_per_combo=np.int32(mc_boards),
        row_workers=np.int32(row_workers),
        row_parallel=np.array(row_parallel),
        pair_eq_table=np.array(str(DEFAULT_PAIR_EQ_TABLE)),
        pair_eq_table_hash=np.array(pair_eq_table_file_hash(DEFAULT_PAIR_EQ_TABLE)),
        seed_scheme=np.array("combo_sha256_v1"),
        row_seeds=row_combo_seed_scheme(),
        eval_backend=np.array(tensor_build_backend_name()),
    )


def load_trio_tensor(path: Path | str, *, game_model: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(shares, pair_eq, deal_weights). Deal weights per the game model: v2 derives the TRUE
    chance distribution from the tensor's stored ``combo_counts`` (combo frequencies + card
    removal over ordered bucket triples — the shipped tensor's ``deal_weights`` field is
    exactly uniform, a v1 artifact); v1 legacy returns the stored uniform weights."""
    from aof_cfr.pushfold_payoffs import game_model_default
    data = np.load(path, allow_pickle=True)
    shares = np.asarray(data["shares"], dtype=np.float64)
    deal_weights = np.asarray(data["deal_weights"], dtype=np.float64)
    gm = game_model_default() if game_model is None else int(game_model)
    if gm >= 2 and "combo_counts" in data:
        cc = np.asarray(data["combo_counts"], dtype=np.float64)
        if cc.sum() > 0:
            deal_weights = cc / cc.sum()
    if "pair_eq" in data:
        pair_eq = np.asarray(data["pair_eq"], dtype=np.float64)
    else:
        pair_eq = np.full_like(shares, 0.5)
    return shares, pair_eq, deal_weights
