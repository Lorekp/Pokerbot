"""Import shim for the hu_table_native pybind11 extension."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from .paths import build_dirs

_CLI_DIR = Path(__file__).resolve().parent
_BUILD_DIRS = build_dirs()

SENTINEL_KEY_AA_AKO = 12948042205429808
SENTINEL_SCORE2_AA_AKO = 2806331
# Bootstrap: python scripts/bootstrap_sentinel_score2.py @ fb3650f75c03

_native_verified: bool | None = None
_native_trio_verified: bool | None = None
_native_module: ModuleType | None = None


def _ensure_build_path() -> None:
    for d in _BUILD_DIRS:
        s = str(d)
        if d.is_dir() and s not in sys.path:
            sys.path.insert(0, s)


def load_native_module() -> ModuleType | None:
    global _native_module
    if _native_module is not None:
        return _native_module
    _ensure_build_path()
    try:
        _native_module = importlib.import_module("hu_table_native")
    except ImportError:
        _native_module = None
    return _native_module


def _verify_native_oracle() -> bool:
    mod = load_native_module()
    if mod is None:
        return False
    try:
        if not bool(mod.verify_sentinel()):
            return False
        got = int(mod.enumerate_score2_one(int(SENTINEL_KEY_AA_AKO)))
        return got == int(SENTINEL_SCORE2_AA_AKO)
    except Exception:
        return False


def _verify_trio_oracle() -> bool:
    mod = load_native_module()
    if mod is None or not hasattr(mod, "verify_trio_sentinel"):
        return False
    try:
        return bool(mod.verify_trio_sentinel())
    except Exception:
        return False


def native_backend_available() -> bool:
    global _native_verified
    if _native_verified is None:
        _native_verified = _verify_native_oracle()
    return _native_verified


def native_trio_available() -> bool:
    global _native_trio_verified
    if not native_backend_available():
        return False
    if _native_trio_verified is None:
        _native_trio_verified = _verify_trio_oracle()
    return _native_trio_verified


def reset_native_backend_cache() -> None:
    global _native_verified, _native_trio_verified, _native_module
    _native_verified = None
    _native_trio_verified = None
    _native_module = None


def build_scores(keys, threads: int = 0):
    """Batch score2 for canonical keys via native build_scores (GIL released)."""
    mod = load_native_module()
    if mod is None:
        raise RuntimeError("hu_table_native extension not available")
    return mod.build_scores(keys, threads)


def exact_six_card_combo_metrics(cards) -> tuple[float, float, float, float, float, float]:
    mod = load_native_module()
    if mod is None:
        raise RuntimeError("hu_table_native extension not available")
    out = mod.exact_six_card_combo_metrics(cards)
    return tuple(float(x) for x in out)


def mc_nway_shares_native(cards, n_players: int, n_boards: int, seed: int):
    mod = load_native_module()
    if mod is None:
        raise RuntimeError("hu_table_native extension not available")
    return mod.mc_nway_shares(cards, n_players, n_boards, int(seed) & 0xFFFFFFFF)


def mc_nway_shares_batch_native(holes_flat, sizes, seeds, n_boards: int, threads: int = 1):
    """Batched OpenMP n-way MC shares; returns (n_show, max_players) float64 (zero-padded rows).

    ``threads=1`` is process-pool-safe (no core oversubscription when run under a
    ProcessPoolExecutor); pass ``threads=0`` (OpenMP default = all cores) only in a single-process
    context. Statistically identical to per-showdown ``mc_nway_shares`` (same evaluator + sampler)."""
    mod = load_native_module()
    if mod is None or not hasattr(mod, "mc_nway_shares_batch"):
        raise RuntimeError("hu_table_native.mc_nway_shares_batch not available")
    return mod.mc_nway_shares_batch(holes_flat, sizes, seeds, int(n_boards), int(threads))


def mc_nway_shares_batch_available() -> bool:
    mod = load_native_module()
    return mod is not None and hasattr(mod, "mc_nway_shares_batch")


def eval_combo_adaptive_native(
    cards,
    seed: int,
    max_boards: int,
    *,
    min_boards: int = 16,
    stderr_gate: float = 0.001,
):
    mod = load_native_module()
    if mod is None:
        raise RuntimeError("hu_table_native extension not available")
    metrics, boards_used, mode = mod.eval_combo_adaptive(
        cards, int(seed) & 0xFFFFFFFF, int(max_boards), int(min_boards), float(stderr_gate)
    )
    return metrics, int(boards_used), int(mode)


def avg_metrics_for_triple(
    row: int,
    j: int,
    k: int,
    *,
    mc_boards: int,
    exact_combo_max: int = 36,
    min_boards_per_combo: int = 16,
    stderr_gate: float = 0.001,
    pair_eq_lookup=None,
):
    mod = load_native_module()
    if mod is None:
        raise RuntimeError("hu_table_native extension not available")
    return mod.avg_metrics_for_triple(
        int(row),
        int(j),
        int(k),
        int(mc_boards),
        int(exact_combo_max),
        int(min_boards_per_combo),
        float(stderr_gate),
        pair_eq_lookup,
    )


def build_trio_row_chunk(
    row: int,
    j_start: int,
    j_end: int,
    *,
    mc_boards: int,
    exact_combo_max: int = 36,
    min_boards_per_combo: int = 16,
    stderr_gate: float = 0.001,
    n_buckets: int = 169,
    threads: int = 0,
    pair_eq_lookup=None,
):
    mod = load_native_module()
    if mod is None:
        raise RuntimeError("hu_table_native extension not available")
    return mod.build_trio_row_chunk(
        int(row),
        int(j_start),
        int(j_end),
        int(mc_boards),
        int(exact_combo_max),
        int(min_boards_per_combo),
        float(stderr_gate),
        int(n_buckets),
        int(threads),
        pair_eq_lookup,
    )


def exact_pair_eq_for_triple(row: int, j: int, k: int):
    mod = load_native_module()
    if mod is None:
        raise RuntimeError("hu_table_native extension not available")
    return mod.exact_pair_eq_for_triple(int(row), int(j), int(k))


def batch_exact_shares_for_triples(
    rows,
    js,
    ks,
    *,
    threads: int = 0,
):
    """Batch exact share oracle; returns (shares float32 [N,3], combo_counts int32 [N])."""
    mod = load_native_module()
    if mod is None or not hasattr(mod, "batch_exact_shares_for_triples"):
        raise RuntimeError("hu_table_native.batch_exact_shares_for_triples not available")
    return mod.batch_exact_shares_for_triples(rows, js, ks, int(threads))


def reset_trio_row_profile() -> None:
    mod = load_native_module()
    if mod is None:
        raise RuntimeError("hu_table_native extension not available")
    mod.reset_trio_row_profile()


def get_trio_row_profile() -> dict[str, float | int]:
    mod = load_native_module()
    if mod is None:
        raise RuntimeError("hu_table_native extension not available")
    return dict(mod.get_trio_row_profile())
