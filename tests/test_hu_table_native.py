"""Phase 2 native hu_table_native oracle: bit-exact score2 parity gates."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from aof_equity import warmup
from aof_equity_grids import _card_str_to_id, _eval7, _equity_two_combos_exact
from aof_equity_score2 import DENOM2, equity_from_score2, score2_from_key, score2_two_combos_exact
from bench._keys import load_keys, representative_from_key
from tools.ompeval_cli.native import (
    SENTINEL_KEY_AA_AKO,
    SENTINEL_SCORE2_AA_AKO,
    build_scores,
    load_native_module,
    native_backend_available,
    reset_native_backend_cache,
)
from tools.ompeval_cli.ompeval_backend import _verify_native_oracle

pytestmark = pytest.mark.skipif(
    not native_backend_available(),
    reason="hu_table_native extension not built or sentinel self-test failed",
)


@pytest.fixture(scope="module")
def native():
    mod = load_native_module()
    assert mod is not None
    return mod


def test_score2_reference_matches_float_oracle() -> None:
    warmup()
    h0 = _card_str_to_id("Ac")
    h1 = _card_str_to_id("Ad")
    v0 = _card_str_to_id("Kh")
    v1 = _card_str_to_id("Kd")
    score2 = score2_two_combos_exact(h0, h1, v0, v1)
    eq_ref = equity_from_score2(score2)
    eq_float = float(_equity_two_combos_exact(h0, h1, v0, v1))
    assert eq_ref == pytest.approx(eq_float, rel=0, abs=1e-12)
    rng = random.Random(0)
    pairs = []
    for _ in range(100):
        cids = rng.sample(range(52), 4)
        pairs.append(tuple(cids))
    for h0, h1, v0, v1 in pairs:
        if len({h0, h1, v0, v1}) < 4:
            continue
        s2 = score2_two_combos_exact(h0, h1, v0, v1)
        eq = equity_from_score2(s2)
        assert eq == pytest.approx(
            float(_equity_two_combos_exact(h0, h1, v0, v1)), rel=0, abs=1e-12
        )


def test_native_import_and_denom2(native) -> None:
    assert native.DENOM2 == DENOM2 == 3_424_608


def test_representative_round_trip_smoke(native) -> None:
    keys = load_keys("smoke")
    for key in keys:
        py = representative_from_key(int(key))
        cpp = native.representative_from_key(int(key))
        assert py == cpp


def test_rank_preservation_all_52(native) -> None:
    for cid in range(52):
        rank = cid // 4
        suit = cid % 4
        part = ((rank & 0xF) << 2) | (suit & 0x3)
        key = int(part)
        h0, h1, v0, v1 = native.representative_from_key(key)
        assert h0 // 4 == rank


def test_gate_a_sentinel_score2(native) -> None:
    warmup()
    h0 = _card_str_to_id("Ac")
    h1 = _card_str_to_id("Ad")
    v0 = _card_str_to_id("Kh")
    v1 = _card_str_to_id("Kd")
    py = score2_two_combos_exact(h0, h1, v0, v1)
    assert py == SENTINEL_SCORE2_AA_AKO
    assert native.enumerate_score2(h0, h1, v0, v1) == py
    assert native.enumerate_score2_one(SENTINEL_KEY_AA_AKO) == py
    assert native.verify_sentinel()


@pytest.mark.slow
def test_smoke_keys_score2_parity(native) -> None:
    warmup()
    keys = load_keys("smoke")
    for key in keys:
        py = score2_from_key(int(key))
        got = int(native.enumerate_score2_one(int(key)))
        assert got == py, f"key={key} py={py} native={got}"


@pytest.mark.slow
def test_stratified_sample_score2_parity(native) -> None:
    warmup()
    keys = load_keys("stratified")
    sample = sorted(int(k) for k in keys)
    rng = random.Random(0)
    picked = rng.sample(sample, 512)
    for key in picked:
        py = score2_from_key(key)
        got = int(native.enumerate_score2_one(key))
        assert got == py, f"key={key} py={py} native={got}"


def test_symmetry_sample(native) -> None:
    warmup()
    path = Path(__file__).resolve().parents[1] / "bench/datasets/fixtures/gate_sample_pairs.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rng = random.Random(0)
    picked = rng.sample(rows, 200)
    for row in picked:
        h0, h1, v0, v1 = row["h0"], row["h1"], row["v0"], row["v1"]
        a = int(native.enumerate_score2(h0, h1, v0, v1))
        b = int(native.enumerate_score2(v0, v1, h0, h1))
        assert a + b == DENOM2


def test_comparator_sentinels(native) -> None:
    warmup()
    # Hero win: AA full house beats AK high card on 2c3c4c5c6c board pattern
    win_board = (0, 1, 2, 3, 4)
    hr = native.eval7(48, 49, *win_board)
    vr = native.eval7(46, 45, *win_board)
    assert native.native_outcome(hr, vr) == 2
    py_hr = _eval7(48, 49, *win_board)
    py_vr = _eval7(46, 45, *win_board)
    assert py_hr < py_vr

    # Tie: same straight on board for both (play the board)
    tie_board = (8, 12, 16, 20, 24)
    hr = native.eval7(0, 4, *tie_board)
    vr = native.eval7(1, 5, *tie_board)
    assert native.native_outcome(hr, vr) == 1
    py_hr = _eval7(0, 4, *tie_board)
    py_vr = _eval7(1, 5, *tie_board)
    assert py_hr == py_vr

    # Hero loss: swap roles on winning board
    win_hr = native.eval7(48, 49, *win_board)
    win_vr = native.eval7(46, 45, *win_board)
    assert native.native_outcome(win_vr, win_hr) == 0
    assert _eval7(46, 45, *win_board) > _eval7(48, 49, *win_board)


def test_native_self_test_refuses_bad_sentinel(monkeypatch) -> None:
    import tools.ompeval_cli.native as native_mod

    reset_native_backend_cache()
    original = native_mod.SENTINEL_SCORE2_AA_AKO
    monkeypatch.setattr(native_mod, "SENTINEL_SCORE2_AA_AKO", 0)
    assert _verify_native_oracle() is False
    monkeypatch.setattr(native_mod, "SENTINEL_SCORE2_AA_AKO", original)
    reset_native_backend_cache()
    assert _verify_native_oracle() is True


def test_backend_native_available() -> None:
    assert native_backend_available()
    assert _verify_native_oracle()


def test_build_scores_smoke_parity(native) -> None:
    keys = load_keys("smoke")
    expected = np.asarray([native.enumerate_score2_one(int(k)) for k in keys], dtype=np.uint32)
    for threads in (1, 2, 4):
        got = np.asarray(build_scores(keys, threads=threads), dtype=np.uint32)
        assert np.array_equal(got, expected), f"thread parity failed at threads={threads}"


@pytest.mark.slow
def test_build_scores_stratified_parity(native) -> None:
    keys = load_keys("stratified")
    expected = np.asarray([native.enumerate_score2_one(int(k)) for k in keys], dtype=np.uint32)
    for threads in (1, 2, os.cpu_count() or 4):
        threads = max(1, min(int(threads), len(keys)))
        got = np.asarray(build_scores(keys, threads=threads), dtype=np.uint32)
        assert np.array_equal(got, expected), f"thread parity failed at threads={threads}"
