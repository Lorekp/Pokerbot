"""Toolchain metadata for HU showdown artifact builds (Phase 5 CI diagnostics)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OMPEVAL_VENDOR = REPO_ROOT / "tools" / "ompeval_cli" / "vendor" / "OMPEval"


def _run(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def collect_toolchain_meta(*, portable_build: bool = True) -> dict[str, object]:
    """Capture build environment fields stored in artifact schema sidecar."""
    meta: dict[str, object] = {
        "portable_build": portable_build,
        "openmp_enabled": True,
    }

    cmake_version = _run(["cmake", "--version"])
    if cmake_version:
        meta["cmake_version"] = cmake_version.splitlines()[0]

    if shutil.which("g++"):
        cxx = _run(["g++", "--version"])
        if cxx:
            meta["cxx_compiler"] = cxx.splitlines()[0]
    elif shutil.which("cl"):
        cxx = _run(["cl"])
        if cxx:
            meta["cxx_compiler"] = cxx.splitlines()[0]

    if OMPEVAL_VENDOR.is_dir():
        sha = _run(["git", "-C", str(OMPEVAL_VENDOR), "rev-parse", "HEAD"])
        if sha:
            meta["ompeval_submodule_sha"] = sha

    return meta


def manifest_path() -> Path:
    return REPO_ROOT / "data" / "aof" / "hu_showdown_equity.manifest.json"


def load_manifest() -> dict[str, object]:
    return json.loads(manifest_path().read_text(encoding="utf-8"))
