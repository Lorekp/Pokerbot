"""Phase 3 hu_table_build CLI tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from bench._keys import load_keys
from tools.ompeval_cli.native import build_scores, load_native_module, native_backend_available
from tools.ompeval_cli.paths import hu_table_build_binary

CLI_BIN = hu_table_build_binary()

pytestmark = pytest.mark.skipif(
    not native_backend_available() or CLI_BIN is None,
    reason="hu_table_build CLI or native backend unavailable",
)


@pytest.fixture(scope="module")
def native():
    mod = load_native_module()
    assert mod is not None
    return mod


def test_cli_sentinel_gate_passes_on_smoke(tmp_path: Path) -> None:
    keys = load_keys("smoke")
    keys_path = tmp_path / "smoke.npy"
    np.save(keys_path, keys)
    out_path = tmp_path / "scores.bin"

    proc = subprocess.run(
        [str(CLI_BIN), "--keys", str(keys_path), "--out", str(out_path), "--threads", "1"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.is_file()


@pytest.mark.slow
def test_cli_output_matches_build_scores(tmp_path: Path) -> None:
    keys = load_keys("smoke")
    keys_path = tmp_path / "smoke.npy"
    np.save(keys_path, keys)
    out_path = tmp_path / "scores.bin"

    expected = np.asarray(build_scores(keys, threads=1), dtype=np.uint32)

    proc = subprocess.run(
        [
            str(CLI_BIN),
            "--keys",
            str(keys_path),
            "--out",
            str(out_path),
            "--threads",
            "1",
            "--format",
            "raw",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    raw = out_path.read_bytes()
    got = np.frombuffer(raw, dtype=np.uint32)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)
