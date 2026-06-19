"""Native (Rust) binding for the blueprint infoset FINGERPRINT — bit-identical to
``pokerbot.engine.jax_weva_fast.fingerprint_fast(state, seat)`` in betting-abstraction mode.

This is the hot path of blueprint+search: the fingerprint is a tabular-CFR DICT KEY
(``make_blueprint_policy`` -> ``snapshot.get(fp_bytes)``), so the native output MUST be byte-for-byte
identical to the Python digest or coverage collapses to 0% (serve-skew). Only the FINAL 16 digest bytes
must match; the native core reproduces the integer features (lossless preflop class, WEVA table lookup,
bounded betting-state key / lossless history key, phevaluator made-hand bucket) and the blake2b
personalization/field-order EXACTLY, so equality is exact by construction (no float enters the digest).

OPT-IN and gated, exactly like ``AOF_MARG4_NATIVE`` / the aivat_native pattern:
  * ``available()``   -> the .pyd is built and importable.
  * ``is_enabled()``  -> available() AND ``POKERBOT_FINGERPRINT_NATIVE`` is truthy (default OFF).
Default serve behavior is byte-for-byte the Python path until parity is trusted and the flag is set.

If the extension is not built (or disabled), ``fingerprint_fast`` here transparently delegates to the
Python oracle, so callers never special-case availability.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_CRATE_DIR = Path(__file__).resolve().parents[2] / "tools" / "fingerprint_native"
_native: ModuleType | None = None
_native_tried = False

_ENABLE_ENV = "POKERBOT_FINGERPRINT_NATIVE"


# ---------------------------------------------------------------------------
# Native loading: locate target/release/fingerprint_native.dll, alias to .pyd, import.
# ---------------------------------------------------------------------------
def _candidate_pyd_dirs():
    yield _CRATE_DIR / "target" / "release"
    yield _CRATE_DIR / "target" / "debug"
    yield _CRATE_DIR


def _ensure_pyd() -> Path | None:
    for d in _candidate_pyd_dirs():
        if not d.is_dir():
            continue
        pyd = d / "fingerprint_native.pyd"
        if pyd.exists():
            return pyd
        so = d / "fingerprint_native.so"
        if so.exists():
            return so
        dll = d / "fingerprint_native.dll"
        if dll.exists():
            try:
                shutil.copy2(dll, pyd)
                return pyd
            except OSError:
                return dll
    return None


def load_native() -> ModuleType | None:
    global _native, _native_tried
    if _native is not None or _native_tried:
        return _native
    _native_tried = True
    pyd = _ensure_pyd()
    if pyd is None:
        return None
    try:
        spec = importlib.util.spec_from_file_location("fingerprint_native", str(pyd))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _native = mod
    except Exception:
        _native = None
    return _native


def available() -> bool:
    """The compiled extension is built and importable."""
    return load_native() is not None


def _flag_truthy(val: str | None) -> bool:
    return (val or "").lower() not in ("", "0", "false", "no")


def is_enabled() -> bool:
    """Native path is BOTH built and opted-in via ``POKERBOT_FINGERPRINT_NATIVE`` (default OFF).

    HARD OVERRIDE: if depth-aware bucketing is on (``POKERBOT_DEPTH_BUCKETS`` truthy) the native path is
    DISABLED regardless of opt-in. The Rust core has no depth byte (it reproduces the legacy non-depth
    preflop digest), so serving a depth blueprint through it would silently drop the depth key -> ~0%
    coverage -> uniform fallback. Forcing the Python oracle keeps train/serve byte-identity. Remove this
    guard only once the Rust core learns the depth byte and re-passes golden parity."""
    if _flag_truthy(os.environ.get("POKERBOT_DEPTH_BUCKETS")):
        return False
    return available() and _flag_truthy(os.environ.get(_ENABLE_ENV))


def reset_native_cache() -> None:
    global _native, _native_tried
    _native = None
    _native_tried = False


# ---------------------------------------------------------------------------
# Bridge: extract the EnvState fields the fingerprint needs + dispatch to native.
# ---------------------------------------------------------------------------
def _as_i32_contig(a: Any) -> np.ndarray:
    """Fast path: pass already-int32 C-contiguous numpy arrays straight through (no copy).

    EnvState fields off ``np_engine.to_numpy_state`` are int32 + C-contiguous (except int8
    ``seat_status``), so the common case is zero-copy. The Rust side reads via ``item_count`` so 2-D
    contiguous arrays (``holes``) need no reshape. Falls back to a copy only when dtype/contiguity differ.
    """
    arr = a if isinstance(a, np.ndarray) else np.asarray(a)
    if arr.dtype == np.int32 and arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr, dtype=np.int32)


# The (52,52,6) weva table is constant across calls; convert to a flat int32 buffer once and key the
# cache by object id (the default table is an lru_cached singleton, so this is a single entry).
_TABLE_CACHE: dict[int, np.ndarray] = {}


def _table_i32(table: np.ndarray) -> np.ndarray:
    key = id(table)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    flat = np.ascontiguousarray(np.asarray(table).reshape(-1), dtype=np.int32)
    if len(_TABLE_CACHE) > 8:
        _TABLE_CACHE.clear()
    _TABLE_CACHE[key] = flat
    return flat


def fingerprint_native(state: Any, seat: int, *, weva_table: np.ndarray | None = None) -> bytes:
    """Native fingerprint bytes for (state, seat). Requires the extension AND betting-abstraction mode.

    Raises ``RuntimeError`` if the extension is unavailable. Callers should use ``fingerprint_fast``
    (below), which gates on ``is_enabled()`` and falls back to the Python oracle.
    """
    mod = load_native()
    if mod is None:
        raise RuntimeError("fingerprint_native extension not built")

    from pokerbot.engine import jax_weva_encode as enc
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table
    from pokerbot.encoding import weva_encoder

    table = weva_table if weva_table is not None else default_weva_bucket_table()
    table_i32 = _table_i32(table)

    return bytes(
        mod.fingerprint(
            _as_i32_contig(state.holes),
            _as_i32_contig(state.board),
            _as_i32_contig(state.seat_status),
            _as_i32_contig(state.stacks),
            _as_i32_contig(state.committed_street),
            int(state.street),
            int(state.pot_dead),
            int(state.current_bet),
            int(state.button),
            int(state.last_aggressor),
            _as_i32_contig(state.hist_seat),
            _as_i32_contig(state.hist_action),
            int(state.hist_len),
            int(seat),
            table_i32,
            bool(weva_encoder._PREFLOP_LOSSLESS),
            bool(weva_encoder._PREFLOP_LOSSLESS_BETTING),
            bool(enc._POSTFLOP_BUCKETS_ENABLED),
            int(enc._POSTFLOP_NUM_BUCKETS),
        )
    )


def fingerprint_batch(
    states: list[Any], seats, *, weva_table: np.ndarray | None = None
) -> list[bytes]:
    """Native batched fingerprint: one FFI crossing for ``len(states)`` (state, seat) pairs.

    This is where the native compute speed (~8x the Python per-fp cost) actually shows up end-to-end:
    the per-call Python/FFI marshaling is amortized across the whole batch. Use it on the search/LBR
    leaf sweep. Bit-identical to calling ``fingerprint_native`` per state.
    """
    mod = load_native()
    if mod is None:
        raise RuntimeError("fingerprint_native extension not built")
    b = len(states)
    if b == 0:
        return []
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table
    from pokerbot.engine import jax_weva_encode as enc
    from pokerbot.encoding import weva_encoder

    table = weva_table if weva_table is not None else default_weva_bucket_table()
    table_i32 = _table_i32(table)

    def stack(field: str) -> np.ndarray:
        return np.ascontiguousarray(
            np.stack([np.asarray(getattr(s, field)) for s in states], axis=0), dtype=np.int32
        )

    hist_cap = int(np.asarray(states[0].hist_seat).shape[-1])
    seats_arr = np.ascontiguousarray(np.asarray(seats).reshape(-1), dtype=np.int32)

    raw = bytes(
        mod.fingerprint_batch(
            b,
            stack("holes"),
            stack("board"),
            stack("seat_status"),
            stack("stacks"),
            stack("committed_street"),
            stack("street"),
            stack("pot_dead"),
            stack("current_bet"),
            stack("button"),
            stack("last_aggressor"),
            stack("hist_seat"),
            stack("hist_action"),
            stack("hist_len"),
            hist_cap,
            seats_arr,
            table_i32,
            bool(weva_encoder._PREFLOP_LOSSLESS),
            bool(weva_encoder._PREFLOP_LOSSLESS_BETTING),
            bool(enc._POSTFLOP_BUCKETS_ENABLED),
            int(enc._POSTFLOP_NUM_BUCKETS),
        )
    )
    return [raw[i * 16 : i * 16 + 16] for i in range(b)]


def fingerprint_fast(state: Any, seat: int, *, weva_table: np.ndarray | None = None) -> bytes:
    """Drop-in for ``jax_weva_fast.fingerprint_fast``: native when enabled+available, else Python oracle.

    Gated by ``is_enabled()`` (built AND ``POKERBOT_FINGERPRINT_NATIVE`` truthy). The native path only
    covers betting-abstraction mode; if that mode is OFF the oracle's own fast path already routes to the
    full encode, so we just defer to the Python oracle there too.
    """
    from pokerbot.engine import jax_weva_encode as enc
    from pokerbot.engine.jax_weva_fast import fingerprint_fast as _py_fp

    if is_enabled() and enc._BETTING_ABSTRACTION:
        return fingerprint_native(state, seat, weva_table=weva_table)
    return _py_fp(state, seat, weva_table=weva_table)


# ---------------------------------------------------------------------------
# Standalone native hooks (testing / verification).
# ---------------------------------------------------------------------------
def evaluate_cards(cards) -> int:
    mod = load_native()
    if mod is None:
        raise RuntimeError("fingerprint_native extension not built")
    return int(mod.evaluate_cards([int(c) for c in cards]))


def postflop_bucket(hole, board, num_buckets: int) -> int:
    mod = load_native()
    if mod is None:
        raise RuntimeError("fingerprint_native extension not built")
    return int(mod.postflop_bucket([int(c) for c in hole], [int(c) for c in board], int(num_buckets)))


def blake2b_digest(digest_size: int, person: bytes, data: bytes) -> bytes:
    mod = load_native()
    if mod is None:
        raise RuntimeError("fingerprint_native extension not built")
    return bytes(mod.blake2b_digest(int(digest_size), bytes(person), bytes(data)))
