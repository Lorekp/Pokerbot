"""Cross-platform paths for OMPEval native build artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

_CLI_DIR = Path(__file__).resolve().parent


def cli_dir() -> Path:
    return _CLI_DIR


def build_dirs() -> tuple[Path, ...]:
    """Directories that may contain hu_table_native extension modules."""
    if sys.platform == "win32":
        return (_CLI_DIR / "build" / "Release", _CLI_DIR / "build")
    return (_CLI_DIR / "build", _CLI_DIR / "build" / "Release")


def hu_table_build_binary() -> Path | None:
    """Return hu_table_build executable if built, else None."""
    if sys.platform == "win32":
        candidates = (
            _CLI_DIR / "build" / "Release" / "hu_table_build.exe",
            _CLI_DIR / "build" / "hu_table_build.exe",
        )
    else:
        candidates = (
            _CLI_DIR / "build" / "hu_table_build",
            _CLI_DIR / "build" / "Release" / "hu_table_build",
        )
    for path in candidates:
        if path.is_file():
            return path
    return None
