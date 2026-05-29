"""Phase 5 CI gates: uint64 lookup boundary, runtime table parity, comparator proxy."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from aof_equity import warmup
from aof_equity_score2 import GATE_B_FLOAT32_ATOL, equity_float32_from_score2, score2_from_key
from aof_equity_showdown import (
    N_CANONICAL_KEYS,
    canonical_showdown_key,
    clear_showdown_table_cache,
    equity_hu_showdown_exact,
    load_showdown_table,
    save_showdown_table,
    _key_search_index,
)
from bench._keys import fixtures_dir, load_keys, representative_from_key
from tools.ompeval_cli.native import (
    SENTINEL_KEY_AA_AKO,
    build_scores,
    load_native_module,
    native_backend_available,
)

# Frozen from bench/datasets/fixtures/full_keys.uint64.npy (Phase 4 boundary audit).
KEY_MAX_BELOW_2_53 = 8725853130260528
KEY_MIN_ABOVE_2_53 = 9007353873563652
# Odd lane above 2^53: float64-coerced searchsorted lands on the previous key.
KEY_ABOVE_2_53_ODD = 9007353873563653
SENTINEL_KEY = SENTINEL_KEY_AA_AKO

RUNTIME_SANITY = fixtures_dir() / "runtime_sanity.jsonl"
PRODUCTION_NPZ = Path("data/aof/hu_showdown_equity.npz")

pytestmark = pytest.mark.skipif(
    not native_backend_available(),
    reason="native backend not available",
)


@pytest.fixture(scope="module")
def native():
    warmup()
    mod = load_native_module()
    assert mod is not None
    return mod


@pytest.fixture(scope="module")
def full_keys():
    return np.load(fixtures_dir() / "full_keys.uint64.npy")


def _cards_from_key(key: int) -> tuple[tuple[str, str], tuple[str, str]]:
    ranks = "23456789TJQKA"
    suits = "cdhs"
    h0, h1, v0, v1 = representative_from_key(key)

    def one(c: int) -> str:
        return f"{ranks[c // 4]}{suits[c % 4]}"

    return (one(h0), one(h1)), (one(v0), one(v1))


def test_canonical_key_count_exact() -> None:
    keys = load_keys("full")
    assert len(keys) == N_CANONICAL_KEYS == 97_513


def test_searchsorted_uint64_cast_required(full_keys) -> None:
    """Without np.uint64 cast, searchsorted can return the wrong slot above 2^53."""
    keys = full_keys
    key = KEY_ABOVE_2_53_ODD
    assert key in keys or np.any(keys == np.uint64(key))

    bad_idx = int(np.searchsorted(keys, key))
    good_idx = _key_search_index(keys, key)

    assert bad_idx != good_idx, "expected float64 coercion bug for odd key above 2^53"
    assert int(keys[bad_idx]) != key
    assert int(keys[good_idx]) == key


def test_boundary_keys_lookup_in_mini_table(tmp_path: Path) -> None:
    warmup()
    raw_probe = [
        KEY_MAX_BELOW_2_53,
        KEY_MIN_ABOVE_2_53,
        KEY_ABOVE_2_53_ODD,
        SENTINEL_KEY,
    ]
    probe_keys = np.array(
        [
            canonical_showdown_key(*representative_from_key(k))
            for k in raw_probe
        ],
        dtype=np.uint64,
    )
    score2 = np.asarray(build_scores(probe_keys, threads=1), dtype=np.uint32)
    out = tmp_path / "boundary.npz"
    save_showdown_table(out, probe_keys, score2=score2)
    clear_showdown_table_cache()

    for i, key in enumerate(probe_keys):
        hero, vill = _cards_from_key(int(key))
        lut = equity_hu_showdown_exact(hero, vill, table_path=out, mode="table")
        expected = float(equity_float32_from_score2(int(score2[i])))
        assert float(lut) == expected
        assert int(score2[i]) == score2_from_key(int(key))


@pytest.mark.skipif(
    not PRODUCTION_NPZ.is_file(),
    reason="production HU showdown table not built",
)
def test_production_table_boundary_lookup() -> None:
    warmup()
    data = np.load(PRODUCTION_NPZ, allow_pickle=False)
    keys = np.asarray(data["keys"], dtype=np.uint64)
    score2 = np.asarray(data["score2"], dtype=np.uint32)
    eq = np.asarray(data["equity"], dtype=np.float32)

    for key in (KEY_MAX_BELOW_2_53, KEY_MIN_ABOVE_2_53, KEY_ABOVE_2_53_ODD, SENTINEL_KEY):
        idx = _key_search_index(keys, key)
        assert int(keys[idx]) == key
        assert int(score2[idx]) == score2_from_key(key)
        assert float(eq[idx]) == float(equity_float32_from_score2(int(score2[idx])))


@pytest.mark.skipif(
    not PRODUCTION_NPZ.is_file(),
    reason="production HU showdown table not built",
)
def test_runtime_table_matches_exact_enum() -> None:
    warmup()
    clear_showdown_table_cache()
    load_showdown_table(PRODUCTION_NPZ)

    rows = []
    with RUNTIME_SANITY.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rng = random.Random(0)
    picked = rng.sample(rows, min(100, len(rows)))

    max_err = 0.0
    for row in picked:
        hero = tuple(row["hero"])
        vill = tuple(row["vill"])
        h0, h1, v0, v1 = row["h0"], row["h1"], row["v0"], row["v1"]
        k = canonical_showdown_key(h0, h1, v0, v1)
        table_eq = equity_hu_showdown_exact(
            hero, vill, table_path=PRODUCTION_NPZ, mode="table"
        )
        expected = float(equity_float32_from_score2(score2_from_key(k)))
        max_err = max(max_err, abs(float(table_eq) - expected))
    assert max_err <= GATE_B_FLOAT32_ATOL


@pytest.mark.skipif(
    not PRODUCTION_NPZ.is_file(),
    reason="production HU showdown table not built",
)
def test_payoff_hu_mode_rollback_exact_enum() -> None:
    """Rollback flag: exact_enum must match table on runtime sanity samples."""
    warmup()
    clear_showdown_table_cache()
    load_showdown_table(PRODUCTION_NPZ)

    rows = []
    with RUNTIME_SANITY.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rng = random.Random(1)
    picked = rng.sample(rows, min(50, len(rows)))

    max_err_table = 0.0
    max_err_enum = 0.0
    for row in picked:
        hero = tuple(row["hero"])
        vill = tuple(row["vill"])
        h0, h1, v0, v1 = row["h0"], row["h1"], row["v0"], row["v1"]
        k = canonical_showdown_key(h0, h1, v0, v1)
        expected = float(equity_float32_from_score2(score2_from_key(k)))
        table_eq = equity_hu_showdown_exact(
            hero, vill, table_path=PRODUCTION_NPZ, mode="table"
        )
        enum_eq = equity_hu_showdown_exact(
            hero, vill, table_path=PRODUCTION_NPZ, mode="exact_enum"
        )
        max_err_table = max(max_err_table, abs(float(table_eq) - expected))
        max_err_enum = max(max_err_enum, abs(float(enum_eq) - expected))
    assert max_err_table <= GATE_B_FLOAT32_ATOL
    assert max_err_enum <= 1e-7


def test_comparator_sentinel_would_fail_if_flipped(native) -> None:
    warmup()

    def flipped_outcome(hr, vr):
        actual = int(native.native_outcome(hr, vr))
        if actual == 2:
            return 0
        if actual == 0:
            return 2
        return actual

    win_board = (0, 1, 2, 3, 4)
    hr = native.eval7(48, 49, *win_board)
    vr = native.eval7(46, 45, *win_board)
    assert int(native.native_outcome(hr, vr)) == 2
    assert flipped_outcome(hr, vr) == 0

    tie_board = (8, 12, 16, 20, 24)
    hr = native.eval7(0, 4, *tie_board)
    vr = native.eval7(1, 5, *tie_board)
    assert int(native.native_outcome(hr, vr)) == 1
    assert flipped_outcome(hr, vr) == 1

    win_hr = native.eval7(48, 49, *win_board)
    win_vr = native.eval7(46, 45, *win_board)
    assert int(native.native_outcome(win_vr, win_hr)) == 0
    assert flipped_outcome(win_vr, win_hr) == 2
