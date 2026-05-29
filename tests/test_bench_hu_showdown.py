"""Tests for HU showdown benchmark harness (Phase 1)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from aof_equity_showdown import key_spec_hash
from bench._keys import fixtures_dir, load_keys, representative_from_key
from bench.datasets.generate import generate

REPO = Path(__file__).resolve().parents[1]
FIXTURES = fixtures_dir()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_fixtures_exist_and_counts() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["key_spec_hash"] == key_spec_hash()
    assert manifest["files"]["full_keys.uint64.npy"] == 97_513
    assert manifest["files"]["smoke_keys.uint64.npy"] == 256
    assert manifest["files"]["stratified_keys.uint64.npy"] == 4096
    assert manifest["files"]["gate_sample_pairs.jsonl"] == 10_000
    assert manifest["files"]["runtime_sanity.jsonl"] == 1000


def test_fixtures_match_manifest() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_sha in manifest["sha256"].items():
        assert _sha256_file(FIXTURES / name) == expected_sha, name


def test_representative_from_key_disjoint() -> None:
    keys = load_keys("smoke")
    for k in keys[:32]:
        h0, h1, v0, v1 = representative_from_key(int(k))
        ids = {h0, h1, v0, v1}
        assert len(ids) == 4


@pytest.mark.slow
def test_meso_smoke_repeatable() -> None:
    """Two meso smoke runs should agree within 5% (CI uses stratified; smoke is fast)."""
    cmd = [
        sys.executable,
        "-m",
        "bench.meso",
        "--fixture",
        "smoke",
        "--repeat",
        "2",
    ]
    out1 = subprocess.check_output(cmd, cwd=REPO, text=True)
    out2 = subprocess.check_output(cmd, cwd=REPO, text=True)
    def _parse_s_key(text: str) -> float:
        for line in text.splitlines():
            if line.startswith("s/key="):
                return float(line.split("=")[1].split()[0])
        raise AssertionError(f"no s/key in output: {text}")

    s1 = _parse_s_key(out1)
    s2 = _parse_s_key(out2)
    rel = abs(s1 - s2) / s1
    assert rel < 0.05, f"meso smoke not repeatable: {s1} vs {s2} ({rel:.1%})"


def test_regenerate_matches_committed_manifest() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        regen = generate(out, seed=manifest["seed"])
        assert regen["sha256"] == manifest["sha256"]
