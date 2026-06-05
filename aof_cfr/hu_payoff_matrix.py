"""Exact HU bucket-pair equity matrix (NPZ table average over suit-compatible combos)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from aof_cfr.buckets import COMBOS_169, LABELS_169
from aof_equity_score2 import score2_two_combos_exact
from aof_equity_showdown import canonical_showdown_key, cards_to_ids, load_showdown_table

DEFAULT_TABLE = Path("data/aof/hu_showdown_equity.npz")
DEFAULT_KEY_CACHE = Path("data/aof/cfr_base/hu_eq_key_cache.npz")
DENOM2 = 3_424_608


def solver_hu_deal_weights(n: int = 169) -> np.ndarray:
    """Uniform ordered bucket pairs — matches ``solver_hu_dcfr`` deal_w."""
    return np.ones((n, n), dtype=np.float64) / (n * n)


def build_hu_bucket_equity_matrix_solver(
    *,
    equity_samples: int = 500,
    seed: int = 42,
    bucket_stride: int = 1,
) -> np.ndarray:
    """Oracle-family eq matrix via in-repo DCFR builder (MC bucket vs bucket)."""
    from pokerbot.solver.hu_preflop_dcfr import _build_169_buckets, _build_eq_matrix

    labels, combos = _build_169_buckets()
    if bucket_stride > 1:
        idx = list(range(0, len(labels), bucket_stride))
        labels = [labels[i] for i in idx]
        combos = [combos[i] for i in idx]
    eq = _build_eq_matrix(labels, combos, samples=equity_samples, seed=seed)
    if bucket_stride == 1:
        return eq
    from pokerbot.solver.hu_preflop_dcfr import expand_to_169

    full_labels, _ = _build_169_buckets()
    n = len(full_labels)
    coarse_map = {lab: i for i, lab in enumerate(labels)}
    out = np.zeros((n, n), dtype=np.float64)
    for i, li in enumerate(full_labels):
        out[i, i] = 0.5
        for j, lj in enumerate(full_labels):
            if i == j:
                continue
            ci = coarse_map.get(li, 0)
            cj = coarse_map.get(lj, 0)
            out[i, j] = float(eq[ci, cj])
    return out


def _load_eq_dict(table_path: Path | str) -> dict[int, float]:
    load_showdown_table(table_path)
    from aof_equity_showdown import _table_eq, _table_keys

    assert _table_keys is not None and _table_eq is not None
    return {int(k): float(v) for k, v in zip(_table_keys, _table_eq)}


def _load_key_cache(path: Path) -> dict[int, float]:
    if not path.is_file():
        return {}
    data = np.load(path, allow_pickle=False)
    keys = np.asarray(data["keys"], dtype=np.uint64)
    eq = np.asarray(data["eq"], dtype=np.float64)
    return {int(k): float(v) for k, v in zip(keys, eq)}


def save_key_cache(path: Path, extra: dict[int, float]) -> None:
    if not extra:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.array(sorted(extra.keys()), dtype=np.uint64)
    eq = np.array([extra[int(k)] for k in keys], dtype=np.float64)
    np.savez_compressed(path, keys=keys, eq=eq)


def _fill_missing_keys(
    eq_dict: dict[int, float],
    missing: list[int],
    *,
    workers: int = 8,
    batch_size: int = 256,
    cache_path: Path = DEFAULT_KEY_CACHE,
    progress: bool = True,
) -> None:
    """Exact-enumerate missing canonical keys in parallel batches."""
    if not missing:
        return
    cached = _load_key_cache(cache_path)
    still_missing = [k for k in missing if k not in eq_dict and k not in cached]
    eq_dict.update(cached)
    if progress:
        print(
            f"NPZ eq: {len(missing)} unique keys, "
            f"{len(missing) - len(still_missing)} cached, "
            f"{len(still_missing)} to enumerate ({workers} workers)",
            flush=True,
        )
    if not still_missing:
        return
    new_entries: dict[int, float] = {}
    done = 0
    total = len(still_missing)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, total, batch_size):
            batch = still_missing[start : start + batch_size]
            futures = [pool.submit(_exact_for_key, k) for k in batch]
            for fut in as_completed(futures):
                key, val = fut.result()
                eq_dict[key] = val
                new_entries[key] = val
                done += 1
            if progress:
                print(f"  enumerated {done}/{total} missing keys", flush=True)
            save_key_cache(cache_path, {**cached, **new_entries})
            cached.update(new_entries)


def _exact_for_key(key: int) -> tuple[int, float]:
    from bench._keys import representative_from_key

    h0, h1, v0, v1 = representative_from_key(key)
    return key, score2_two_combos_exact(h0, h1, v0, v1) / DENOM2

def _collect_key_counts() -> dict[int, int]:
    counts: dict[int, int] = {}
    for i in range(len(LABELS_169)):
        for j in range(len(LABELS_169)):
            if i == j:
                continue
            for c1, c2 in COMBOS_169[i]:
                for c3, c4 in COMBOS_169[j]:
                    if len({c1, c2, c3, c4}) < 4:
                        continue
                    h0, h1, v0, v1 = cards_to_ids((c1, c2), (c3, c4))
                    k = canonical_showdown_key(h0, h1, v0, v1)
                    counts[k] = counts.get(k, 0) + 1
    return counts


def build_hu_bucket_equity_matrix(
    *,
    table_path: Path | str = DEFAULT_TABLE,
    workers: int = 8,
    cache_path: Path = DEFAULT_KEY_CACHE,
    progress: bool = True,
) -> np.ndarray:
    """
    Average combo-pair equity from NPZ. Missing canonical keys are filled once
    via exact enumeration (parallel), then bucket averages are computed.
    """
    if progress:
        print("Collecting bucket combo keys ...", flush=True)
    eq_dict = _load_eq_dict(table_path)
    key_counts = _collect_key_counts()
    missing = sorted(key_counts.keys())
    npz_missing = [k for k in missing if k not in eq_dict]
    _fill_missing_keys(
        eq_dict,
        npz_missing,
        workers=workers,
        cache_path=cache_path,
        progress=progress,
    )

    if progress:
        print("Averaging into 169x169 bucket matrix ...", flush=True)
    n = len(LABELS_169)
    eq = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        eq[i, i] = 0.5
        for j in range(n):
            if i == j:
                continue
            total = 0.0
            count = 0
            for c1, c2 in COMBOS_169[i]:
                for c3, c4 in COMBOS_169[j]:
                    if len({c1, c2, c3, c4}) < 4:
                        continue
                    h0, h1, v0, v1 = cards_to_ids((c1, c2), (c3, c4))
                    k = canonical_showdown_key(h0, h1, v0, v1)
                    total += eq_dict[k]
                    count += 1
            eq[i, j] = total / count if count else 0.5
        if progress and (i + 1) % 20 == 0:
            print(f"  bucket rows {i + 1}/{n}", flush=True)
    return eq


def save_equity_matrix(path: Path, eq: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, eq=eq, labels=np.array(LABELS_169, dtype=object))


def load_equity_matrix(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["eq"], dtype=np.float64)
