"""Artifact v2 (score2 + derived equity) parity and reproducibility gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aof_equity import warmup
from aof_equity_score2 import (
    DENOM2,
    equity_float32_from_score2,
    score2_from_key,
)
from aof_equity_showdown import (
    ARTIFACT_SCHEMA_VERSION,
    EVALUATOR_BACKEND,
    _git_commit,
    clear_showdown_table_cache,
    equity_hu_showdown_exact,
    key_spec_hash,
    load_showdown_table,
    save_showdown_table,
)
from bench._keys import load_keys
from tools.ompeval_cli.native import build_scores, load_native_module, native_backend_available


pytestmark = pytest.mark.skipif(
    not native_backend_available(),
    reason="native build_scores backend not available",
)


@pytest.fixture(scope="module")
def native():
    warmup()
    mod = load_native_module()
    assert mod is not None
    return mod


@pytest.fixture(scope="module")
def smoke_batch(native):
    keys = load_keys("smoke")
    score2 = np.asarray(build_scores(keys, threads=1), dtype=np.uint32)
    equity = np.asarray(equity_float32_from_score2(score2), dtype=np.float32)
    return keys, score2, equity


@pytest.fixture(scope="module")
def stratified_batch(native):
    keys = load_keys("stratified")
    score2 = np.asarray(build_scores(keys, threads=4), dtype=np.uint32)
    equity = np.asarray(equity_float32_from_score2(score2), dtype=np.float32)
    return keys, score2, equity


def test_v2_score2_parity_smoke(smoke_batch) -> None:
    keys, score2, _ = smoke_batch
    mismatches = sum(
        int(score2[i]) != score2_from_key(int(keys[i])) for i in range(len(keys))
    )
    assert mismatches == 0, f"{mismatches}/{len(keys)} score2 mismatches on smoke fixture"


@pytest.mark.slow
def test_v2_score2_parity_stratified(stratified_batch, native) -> None:
    keys, score2, _ = stratified_batch
    mismatches = sum(
        int(score2[i]) != int(native.enumerate_score2_one(int(keys[i])))
        for i in range(len(keys))
    )
    assert mismatches == 0, f"{mismatches}/{len(keys)} native score2 mismatches on stratified"


@pytest.mark.slow
def test_v2_score2_parity_stratified_python(stratified_batch) -> None:
    """Full 4096-key Python oracle cross-check (nightly / path-filtered CI)."""
    keys, score2, _ = stratified_batch
    mismatches = sum(
        int(score2[i]) != score2_from_key(int(keys[i])) for i in range(len(keys))
    )
    assert mismatches == 0, f"{mismatches}/{len(keys)} score2 mismatches on stratified fixture"


def test_v2_legacy_float_bit_identical(stratified_batch) -> None:
    _, score2, equity = stratified_batch
    expected = equity_float32_from_score2(score2)
    assert np.array_equal(equity, expected)
    assert np.max(np.abs(equity.astype(np.float64) - expected.astype(np.float64))) == 0.0


def test_v2_byte_identical_rebuild(smoke_batch, tmp_path: Path) -> None:
    keys, score2, _ = smoke_batch
    fixed_meta = {
        "build_git_commit": "deadbeef0000",
        "backend": EVALUATOR_BACKEND,
        "n_canonical_keys": int(len(keys)),
    }
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    save_showdown_table(path_a, keys, score2=score2, meta=fixed_meta)
    save_showdown_table(path_b, keys, score2=score2, meta=fixed_meta)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_v2_schema_metadata_roundtrip(smoke_batch, tmp_path: Path) -> None:
    keys, score2, _ = smoke_batch
    out = tmp_path / "table.npz"
    meta = {"build_git_commit": _git_commit(), "backend": EVALUATOR_BACKEND}
    save_showdown_table(out, keys, score2=score2, meta=meta)
    schema_path = out.with_suffix(".schema.json")
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    assert payload["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert payload["denom2"] == DENOM2
    assert payload["key_spec_hash"] == key_spec_hash()
    assert payload["n_keys"] == len(keys)
    assert payload["evaluator_backend"] == EVALUATOR_BACKEND
    assert payload["build_git_commit"] == meta["build_git_commit"]


def test_v2_loader_asserts_unsorted(tmp_path: Path) -> None:
    keys = np.array([3, 1, 2], dtype=np.uint64)
    eq = np.array([0.5, 0.4, 0.6], dtype=np.float32)
    np.savez_compressed(tmp_path / "bad.npz", keys=keys, equity=eq)
    clear_showdown_table_cache()
    with pytest.raises(AssertionError, match="not sorted"):
        load_showdown_table(tmp_path / "bad.npz")


def test_v1_backward_compat_load(tmp_path: Path) -> None:
    keys = np.array([1, 2, 3], dtype=np.uint64)
    eq = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    np.savez_compressed(tmp_path / "v1.npz", keys=keys, equity=eq)
    clear_showdown_table_cache()
    load_showdown_table(tmp_path / "v1.npz")

    from aof_equity_grids import _card_str_to_id
    from aof_equity_showdown import canonical_showdown_key, save_showdown_table

    h0 = _card_str_to_id("Ac")
    h1 = _card_str_to_id("Ad")
    v0 = _card_str_to_id("Kh")
    v1 = _card_str_to_id("Kd")
    k = canonical_showdown_key(h0, h1, v0, v1)
    eq_val = np.float32(0.8195)
    save_showdown_table(tmp_path / "mini.npz", np.array([k], dtype=np.uint64), equity=np.array([eq_val]))
    clear_showdown_table_cache()
    got = equity_hu_showdown_exact(
        ("Ac", "Ad"),
        ("Kh", "Kd"),
        table_path=tmp_path / "mini.npz",
        mode="table",
    )
    assert float(got) == pytest.approx(float(eq_val))
