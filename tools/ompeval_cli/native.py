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


def native_backend_available() -> bool:
    global _native_verified
    if _native_verified is None:
        _native_verified = _verify_native_oracle()
    return _native_verified


def reset_native_backend_cache() -> None:
    global _native_verified, _native_module
    _native_verified = None
    _native_module = None


def build_scores(keys, threads: int = 0):
    """Batch score2 for canonical keys via native build_scores (GIL released)."""
    mod = load_native_module()
    if mod is None:
        raise RuntimeError("hu_table_native extension not available")
    return mod.build_scores(keys, threads)
