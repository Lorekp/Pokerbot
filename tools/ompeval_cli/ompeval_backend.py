"""OMPEval CLI helpers and native backend detection (no per-key subprocess builds)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from . import native as _native

_CLI_NAME = "enumerate_hu_equity.exe" if sys.platform == "win32" else "enumerate_hu_equity"
_CLI_DIR = Path(__file__).resolve().parent
_BUILD_DIRS = (
    _CLI_DIR / "build" / "Release",
    _CLI_DIR / "build",
    _CLI_DIR,
)


def find_cli_binary() -> Path | None:
    """Locate the legacy single-matchup CLI (benchmark / eval7 only)."""
    for d in _BUILD_DIRS:
        p = d / _CLI_NAME
        if p.is_file():
            return p
    which = shutil.which(_CLI_NAME)
    if which:
        return Path(which)
    return None


def eval7_ompeval(card_ids: list[int]) -> int:
    """Single 7-card hand category via CLI (benchmark / validation)."""
    exe = find_cli_binary()
    if exe is None:
        raise FileNotFoundError("OMPEval CLI not built; run tools/ompeval_cli/build.ps1")
    proc = subprocess.run(
        [str(exe), "--eval7", *[str(c) for c in card_ids]],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def backend_name() -> str:
    if native_backend_available():
        return "hu_table_native"
    return "numba_exact"


def native_backend_available() -> bool:
    """True when the in-process pybind oracle passes the sentinel self-test."""
    return _native.native_backend_available()


def _verify_native_oracle() -> bool:
    return _native._verify_native_oracle()
