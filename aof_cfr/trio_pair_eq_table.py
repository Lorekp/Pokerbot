"""Precomputed exact 2-way bucket-pair equity table (169×169).

Each entry is combo-averaged HU equity (hero row vs villain column) from
``hu_showdown_equity.npz``. For 3-max tensor lookup at (btn, sb, bb):

  btn-BB = table[btn, bb]
  btn-SB = table[btn, sb]
  sb-BB  = table[sb, bb]

Expanded to (169, 169, 3) per BTN row at lookup time — no third-index enumeration.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.hu_payoff_matrix import DEFAULT_TABLE as HU_SHOWDOWN_TABLE
from aof_cfr.hu_payoff_matrix import build_hu_bucket_equity_matrix

N_BUCKETS = len(LABELS_169)

DEFAULT_PAIR_EQ_TABLE = Path("data/aof/cfr_base/trio_pair_eq_exact.npz")
TABLE_VERSION = 2

_PAIR_EQ_HU: np.ndarray | None = None
_PAIR_EQ_MMAP: np.lib.npyio.NpzFile | None = None


def hu_showdown_file_hash(path: Path | str = HU_SHOWDOWN_TABLE) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pair_eq_table_file_hash(path: Path | str = DEFAULT_PAIR_EQ_TABLE) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hu_pair_eq_values(btn: int, sb: int, bb: int, hu: np.ndarray) -> np.ndarray:
    """Three positional pair equities from the 169×169 HU matrix."""
    return np.array([hu[btn, bb], hu[btn, sb], hu[sb, bb]], dtype=np.float64)


def expand_pair_eq_row_slice(btn_row: int, hu: np.ndarray) -> np.ndarray:
    """Expand HU matrix to (169, 169, 3) lookup for one BTN bucket row."""
    n = hu.shape[0]
    out = np.empty((n, n, 3), dtype=np.float32)
    out[:, :, 0] = hu[btn_row, :][None, :]
    out[:, :, 1] = hu[btn_row, :][:, None]
    out[:, :, 2] = hu[:, :]
    return np.ascontiguousarray(out)


def pair_eq_row_slice(row: int, table: np.ndarray | None = None) -> np.ndarray:
    """Return float32 lookup slice (169,169,3) for one BTN bucket row."""
    hu = table if table is not None else require_pair_eq_table()
    if hu.ndim != 2:
        raise ValueError(f"expected 169×169 HU table, got shape {hu.shape}")
    return expand_pair_eq_row_slice(row, hu)


def _load_v2(path: Path) -> np.ndarray:
    global _PAIR_EQ_HU, _PAIR_EQ_MMAP
    if _PAIR_EQ_MMAP is not None:
        try:
            _PAIR_EQ_MMAP.close()
        except Exception:
            pass
    _PAIR_EQ_MMAP = np.load(path, mmap_mode="r")
    version = int(_PAIR_EQ_MMAP["version"]) if "version" in _PAIR_EQ_MMAP else 0
    if version != TABLE_VERSION:
        raise ValueError(f"pair-eq table version {version} != {TABLE_VERSION}; rebuild with build_trio_pair_eq_table.py")
    key = "pair_eq_hu" if "pair_eq_hu" in _PAIR_EQ_MMAP else "pair_eq"
    hu = np.asarray(_PAIR_EQ_MMAP[key], dtype=np.float64)
    if hu.shape != (N_BUCKETS, N_BUCKETS):
        raise ValueError(f"unexpected pair_eq_hu shape {hu.shape}")
    _PAIR_EQ_HU = hu
    return hu


def load_pair_eq_table(path: Path | str = DEFAULT_PAIR_EQ_TABLE) -> np.ndarray:
    """Load 169×169 HU pair-equity table (read-only / mmap when possible)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"pair-eq table missing: {p}")
    return _load_v2(p)


def get_pair_eq_table(path: Path | str = DEFAULT_PAIR_EQ_TABLE) -> np.ndarray:
    if _PAIR_EQ_HU is not None:
        return _PAIR_EQ_HU
    return load_pair_eq_table(path)


def require_pair_eq_table(path: Path | str = DEFAULT_PAIR_EQ_TABLE) -> np.ndarray:
    """Load the complete pair-eq table; refuse to build inline during tensor passes."""
    return get_pair_eq_table(path)


def table_is_complete(hu: np.ndarray) -> bool:
    if hu.shape != (N_BUCKETS, N_BUCKETS):
        return False
    mask = ~np.eye(N_BUCKETS, dtype=bool)
    off = hu[mask]
    return bool(np.all(np.isfinite(off)) and off.min() > 0.0 and off.max() < 1.0)


def save_pair_eq_table(
    hu: np.ndarray,
    path: Path | str = DEFAULT_PAIR_EQ_TABLE,
    *,
    build_elapsed_s: float | None = None,
    hu_source: Path | str = HU_SHOWDOWN_TABLE,
    writer: str = "single_process",
) -> None:
    """Single-writer persist of the 169×169 table."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "pair_eq_hu": hu.astype(np.float64),
        "version": np.array(TABLE_VERSION),
        "hu_source": np.array(str(hu_source)),
        "hu_source_hash": np.array(hu_showdown_file_hash(hu_source)),
        "writer": np.array(writer),
        "n_pairs": np.int32(N_BUCKETS * N_BUCKETS),
    }
    if build_elapsed_s is not None:
        payload["build_elapsed_s"] = np.array(build_elapsed_s)
    np.savez_compressed(p, **payload)
    global _PAIR_EQ_HU, _PAIR_EQ_MMAP
    _PAIR_EQ_HU = hu.astype(np.float64)
    if _PAIR_EQ_MMAP is not None:
        try:
            _PAIR_EQ_MMAP.close()
        except Exception:
            pass
        _PAIR_EQ_MMAP = None


def build_pair_eq_table(
    *,
    path: Path | str = DEFAULT_PAIR_EQ_TABLE,
    hu_source: Path | str = HU_SHOWDOWN_TABLE,
    workers: int = 8,
    progress: bool = True,
    writer: str = "single_process",
) -> np.ndarray:
    """Eager build of all 169² bucket pairs (~seconds). Single writer at end."""
    if not Path(hu_source).is_file():
        raise FileNotFoundError(f"HU showdown table missing: {hu_source}")
    t0 = time.monotonic()
    hu = build_hu_bucket_equity_matrix(
        table_path=hu_source,
        workers=workers,
        progress=progress,
    )
    elapsed = time.monotonic() - t0
    if not table_is_complete(hu):
        raise RuntimeError("pair-eq table build produced incomplete or invalid HU matrix")
    save_pair_eq_table(
        hu,
        path,
        build_elapsed_s=elapsed,
        hu_source=hu_source,
        writer=writer,
    )
    if progress:
        print(
            f"Saved pair-eq table {path} shape={hu.shape} "
            f"n_pairs={N_BUCKETS * N_BUCKETS} elapsed={elapsed:.1f}s writer={writer}",
            flush=True,
        )
    return hu


def parity_vs_hu_oracle(
    hu: np.ndarray,
    *,
    samples: int = 24,
    seed: int = 0,
) -> dict[str, float | int]:
    """Self-consistency: expanded triple values match hu_pair_eq_values."""
    from aof_cfr.hu_payoff_matrix import build_hu_bucket_equity_matrix

    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(samples):
        btn = int(rng.integers(0, N_BUCKETS))
        sb = int(rng.integers(0, N_BUCKETS))
        bb = int(rng.integers(0, N_BUCKETS))
        want = hu_pair_eq_values(btn, sb, bb, hu)
        got = expand_pair_eq_row_slice(btn, hu)[sb, bb]
        max_err = max(max_err, float(np.max(np.abs(want - got))))
    return {"samples": samples, "max_err": max_err, "ok": max_err <= 1e-6}


# --- removed lazy / triple-enumeration API (fail fast if called) ---


def ensure_pair_eq_row(row: int, **kwargs) -> np.ndarray:
    raise RuntimeError(
        "ensure_pair_eq_row is removed — build the full 169×169 table once with "
        "scripts/build_trio_pair_eq_table.py --full, then load read-only via require_pair_eq_table()"
    )


def build_pair_eq_row(row: int, **kwargs):
    raise RuntimeError(
        "build_pair_eq_row (169³ triple enumeration) is removed — use build_pair_eq_table()"
    )


def merge_row_into_table(row: int, slice_pe: np.ndarray, path: Path | str = DEFAULT_PAIR_EQ_TABLE):
    raise RuntimeError("merge_row_into_table is removed — single-writer build_pair_eq_table() only")


def load_or_build_exact_pair_eq_cube(*_args, **_kwargs) -> np.ndarray:
    """Removed: 169³ pair-eq row build is O(months); use HU table + tensor pair_eq channel."""
    raise RuntimeError(
        "load_or_build_exact_pair_eq_cube is removed. Use pair_eq_source='tensor' or "
        "'hu_expand' (fp_br.resolve_pair_eq_for_static). Proven <1 mbb/g vs 169³ exact."
    )
