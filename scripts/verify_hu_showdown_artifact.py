#!/usr/bin/env python3
"""Verify HU showdown artifact v2 against pinned manifest (Phase 5 CI gate)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from aof_equity_score2 import equity_float32_from_score2
from aof_equity_showdown import (
    ARTIFACT_SCHEMA_VERSION,
    N_CANONICAL_KEYS,
    key_spec_hash,
    load_showdown_table,
)
from scripts.hu_showdown_toolchain import load_manifest


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_score2(score2: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(score2, dtype=np.uint32).tobytes()).hexdigest()


def verify_artifact(
    npz_path: Path,
    *,
    schema_path: Path | None = None,
    manifest: dict[str, object] | None = None,
    check_npz_hash: bool = True,
    check_score2_hash: bool = True,
) -> None:
    manifest = manifest or load_manifest()
    schema_path = schema_path or npz_path.with_suffix(".schema.json")

    if not npz_path.is_file():
        raise SystemExit(f"NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    keys = np.asarray(data["keys"], dtype=np.uint64)
    if len(keys) <= 1:
        raise SystemExit("keys array too short")
    if not np.all(keys[:-1] <= keys[1:]):
        raise SystemExit("keys not sorted ascending")

    if "score2" not in data:
        raise SystemExit("missing score2 array (artifact v1?)")
    score2 = np.asarray(data["score2"], dtype=np.uint32)
    eq = np.asarray(data["equity"], dtype=np.float32)

    if len(score2) != len(keys) or len(eq) != len(keys):
        raise SystemExit("array length mismatch")

    expected_eq = equity_float32_from_score2(score2)
    if not np.array_equal(eq, expected_eq):
        raise SystemExit("equity != float32(score2/DENOM2)")

    if int(manifest.get("n_keys", 0)) != N_CANONICAL_KEYS:
        raise SystemExit(f"manifest n_keys != {N_CANONICAL_KEYS}")
    if len(keys) != N_CANONICAL_KEYS:
        raise SystemExit(f"built {len(keys)} keys != {N_CANONICAL_KEYS}")

    if manifest.get("key_spec_hash") != key_spec_hash():
        raise SystemExit("key_spec_hash mismatch vs manifest")

    if int(manifest.get("artifact_schema_version", 0)) != ARTIFACT_SCHEMA_VERSION:
        raise SystemExit("artifact_schema_version mismatch vs manifest")

    if schema_path.is_file():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if schema.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise SystemExit("schema artifact_schema_version != 2")
        if schema.get("key_spec_hash") != key_spec_hash():
            raise SystemExit("schema key_spec_hash mismatch")
        if schema.get("n_keys") != N_CANONICAL_KEYS:
            raise SystemExit("schema n_keys mismatch")

    if check_npz_hash:
        got = _sha256_file(npz_path)
        expected = str(manifest.get("npz_sha256", ""))
        if got != expected:
            raise SystemExit(f"NPZ sha256 mismatch: got {got}, expected {expected}")

    if check_score2_hash:
        got = _sha256_score2(score2)
        expected = str(manifest.get("score2_sha256", ""))
        if got != expected:
            raise SystemExit(f"score2 sha256 mismatch: got {got}, expected {expected}")

    load_showdown_table(npz_path)
    print(f"verify OK: {npz_path} ({len(keys)} keys)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "npz",
        type=Path,
        nargs="?",
        default=Path("data/aof/hu_showdown_equity.npz"),
    )
    p.add_argument("--schema", type=Path, default=None)
    p.add_argument("--skip-npz-hash", action="store_true")
    p.add_argument("--skip-score2-hash", action="store_true")
    p.add_argument("--score2-only", action="store_true", help="Only print score2 sha256")
    args = p.parse_args()

    if args.score2_only:
        data = np.load(args.npz, allow_pickle=False)
        score2 = np.asarray(data["score2"], dtype=np.uint32)
        print(_sha256_score2(score2))
        return

    try:
        verify_artifact(
            args.npz,
            schema_path=args.schema,
            check_npz_hash=not args.skip_npz_hash,
            check_score2_hash=not args.skip_score2_hash,
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
