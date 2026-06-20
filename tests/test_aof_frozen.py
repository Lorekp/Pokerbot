"""AoF FREEZE GUARD — the AoF engine is ON ICE (2026-06-20).

The AoF engine's STRATEGY is correct, but the game is ~unbeatable at micro rake and AoF is a
tiny EV slice off the critical path (see memory ``aof-bot-correct-but-game-unbeatable`` and the
repo-root ``AOF_FROZEN.md``). The ULTRACODE program therefore treats every AoF *source* path as
READ-ONLY / FROZEN: it may be read for reuse patterns and its data artifacts reused read-only, but
its logic must never be edited, retrained, or refactored.

This guard makes the freeze enforceable. It hashes every tracked AoF SOURCE file (``.py`` / ``.rs`` /
``.pyx`` under the firewall paths) and compares against a committed manifest
(``tests/aof_freeze_manifest.json``). Any modification, addition, or removal of an AoF source file
fails the guard with a precise diff. Data artifacts (``.npz`` / ``.json`` under ``data/aof``) are
intentionally NOT covered — the firewall is about LOGIC, and those artifacts carry pre-existing
training changes; they are reused read-only.

Regenerating the manifest (only with an explicit, reviewed reason — e.g. an intentional one-time
freeze annotation): ``python tests/test_aof_frozen.py`` rewrites ``tests/aof_freeze_manifest.json``
from the current working tree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

# Repo root = parent of tests/.
_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = Path(__file__).resolve().with_name("aof_freeze_manifest.json")

# The AoF firewall surface (repo-root paths — note ``pokerbot/aof_cfr`` does NOT exist; the package
# is the repo-root ``aof_cfr/``). Glob patterns are relative to the repo root and never descend into
# ``.claude/`` worktrees or ``external/`` because they are anchored at these specific top-level names.
_SOURCE_EXTS = (".py", ".rs", ".pyx")
_FROZEN_DIR_GLOBS = (
    "aof_cfr",
    "aof_engine",
    "aof_rl",
    "aof_solver",
    "aof_validation",
    "scripts/aof",
)
# Top-level single-file AoF modules (py-modules declared in pyproject.toml).
_FROZEN_FILE_GLOBS = (
    "aof.py",
    "aof_*.py",
)


def _iter_frozen_sources() -> list[Path]:
    """All tracked AoF SOURCE files (absolute paths), sorted, deduped. Excludes __pycache__."""
    seen: set[Path] = set()
    for d in _FROZEN_DIR_GLOBS:
        for ext in _SOURCE_EXTS:
            for p in _ROOT.glob(f"{d}/**/*{ext}"):
                if "__pycache__" in p.parts:
                    continue
                seen.add(p.resolve())
    for pat in _FROZEN_FILE_GLOBS:
        for p in _ROOT.glob(pat):
            if p.suffix in _SOURCE_EXTS and p.is_file():
                seen.add(p.resolve())
    return sorted(seen)


def _rel(p: Path) -> str:
    return p.resolve().relative_to(_ROOT).as_posix()


def _hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_manifest() -> dict[str, str]:
    """{repo-relative-posix-path: sha256-hex} over the current AoF source tree."""
    return {_rel(p): _hash(p) for p in _iter_frozen_sources()}


def diff_manifests(baseline: dict[str, str], current: dict[str, str]) -> list[str]:
    """PURE comparison (no filesystem) — returns human-readable violations. Empty == frozen/unchanged.

    Exposed separately so the self-test can prove the guard catches edits without touching AoF."""
    violations: list[str] = []
    b_keys, c_keys = set(baseline), set(current)
    for path in sorted(c_keys - b_keys):
        violations.append(f"ADDED (new AoF source not in freeze manifest): {path}")
    for path in sorted(b_keys - c_keys):
        violations.append(f"REMOVED (frozen AoF source deleted): {path}")
    for path in sorted(b_keys & c_keys):
        if baseline[path] != current[path]:
            violations.append(f"MODIFIED (frozen AoF source changed): {path}")
    return violations


def test_manifest_exists_and_nonempty() -> None:
    assert _MANIFEST.exists(), (
        f"AoF freeze manifest missing at {_MANIFEST}. Regenerate with "
        f"`python tests/test_aof_frozen.py`."
    )
    manifest = json.loads(_MANIFEST.read_text())
    # Sanity: the AoF surface is large (~183 source files); a near-empty manifest means the globs
    # silently stopped matching (a broken guard reads as 'all frozen').
    assert len(manifest) >= 150, (
        f"AoF freeze manifest only has {len(manifest)} entries — expected ~183. "
        f"The firewall globs may be broken (a broken guard would silently pass)."
    )


def test_aof_sources_unchanged() -> None:
    """The headline guard: zero AoF source files modified/added/removed since the freeze."""
    baseline = json.loads(_MANIFEST.read_text())
    current = build_manifest()
    violations = diff_manifests(baseline, current)
    assert not violations, (
        "AoF is FROZEN (on ice) — the following AoF source changes are forbidden by the firewall:\n  "
        + "\n  ".join(violations)
        + "\n\nIf this change is an intentional, reviewed one-time freeze annotation, regenerate the "
        "manifest with `python tests/test_aof_frozen.py`. Otherwise REVERT the AoF edit — the ULTRACODE "
        "program must not modify AoF logic (see AOF_FROZEN.md)."
    )


def test_guard_detects_simulated_edit() -> None:
    """Demonstrably fails on a simulated AoF edit (acceptance criterion) — exercises the pure diff."""
    baseline = {"aof_cfr/game_hu.py": "aaa", "aof.py": "bbb", "aof_engine/__init__.py": "ccc"}

    # (1) A modified file is caught.
    modified = dict(baseline, **{"aof_cfr/game_hu.py": "DEADBEEF"})
    v = diff_manifests(baseline, modified)
    assert v == ["MODIFIED (frozen AoF source changed): aof_cfr/game_hu.py"], v

    # (2) A newly added AoF source is caught.
    added = dict(baseline, **{"aof_cfr/sneaky_new.py": "xyz"})
    v = diff_manifests(baseline, added)
    assert v == ["ADDED (new AoF source not in freeze manifest): aof_cfr/sneaky_new.py"], v

    # (3) A removed frozen source is caught.
    removed = {k: val for k, val in baseline.items() if k != "aof.py"}
    v = diff_manifests(baseline, removed)
    assert v == ["REMOVED (frozen AoF source deleted): aof.py"], v

    # (4) No change == no violation.
    assert diff_manifests(baseline, dict(baseline)) == []


def test_guard_catches_real_tree_tamper(tmp_path) -> None:
    """End-to-end: take the REAL current manifest, simulate a one-byte tamper on a real frozen file,
    and assert the guard reports exactly that file (proves it works against the live tree, not just
    synthetic dicts)."""
    real = build_manifest()
    assert real, "no AoF sources discovered — firewall globs broken"
    victim = next(iter(real))
    tampered = dict(real)
    tampered[victim] = "0" * 64  # flip the hash as if the file were edited
    v = diff_manifests(real, tampered)
    assert v == [f"MODIFIED (frozen AoF source changed): {victim}"], v


if __name__ == "__main__":
    manifest = build_manifest()
    _MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {_MANIFEST} with {len(manifest)} AoF source hashes.")
