"""Combo-exact HU showdown table: canonical keys + gates A/B."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from aof_equity import warmup
from aof_equity_grids import _card_str_to_id, _equity_two_combos_exact, aa_vs_ako_equity_exact
from aof_equity_score2 import GATE_B_FLOAT32_ATOL, equity_float32_from_score2, score2_from_key
from aof_equity_showdown import (
    build_showdown_equity_table,
    canonical_showdown_key,
    enumerate_all_disjoint_pairs,
    equity_hu_showdown_exact,
    load_showdown_table,
    save_showdown_table,
    _key_search_index,
)

def test_gate_a_aa_vs_ako() -> None:
    warmup()
    ref = aa_vs_ako_equity_exact()
    assert 0.8190 <= ref <= 0.8200
    try:
        from tools.ompeval_cli.ompeval_backend import native_backend_available

        if native_backend_available():
            from aof_equity_score2 import score2_two_combos_exact
            from tools.ompeval_cli.native import SENTINEL_SCORE2_AA_AKO, load_native_module

            h0 = _card_str_to_id("Ac")
            h1 = _card_str_to_id("Ad")
            v0 = _card_str_to_id("Kh")
            v1 = _card_str_to_id("Kd")
            assert score2_two_combos_exact(h0, h1, v0, v1) == SENTINEL_SCORE2_AA_AKO
            mod = load_native_module()
            assert mod is not None
            assert mod.enumerate_score2(h0, h1, v0, v1) == SENTINEL_SCORE2_AA_AKO
    except ImportError:
        pass


def test_gate_b_canonical_round_trip(tmp_path: Path) -> None:
    warmup()
    pairs = enumerate_all_disjoint_pairs()
    rng = random.Random(0)
    sample_pairs = [rng.choice(pairs) for _ in range(500)]
    keys, eq = build_showdown_equity_table(sample_pairs)
    score2 = np.asarray(
        [score2_from_key(int(k)) for k in keys],
        dtype=np.uint32,
    )
    save_showdown_table(tmp_path / "t.npz", keys, score2=score2)
    load_showdown_table(tmp_path / "t.npz")
    loaded = np.load(tmp_path / "t.npz", allow_pickle=False)
    loaded_score2 = np.asarray(loaded["score2"], dtype=np.uint32)
    max_err = 0.0
    for h0, h1, v0, v1 in sample_pairs:
        k = canonical_showdown_key(h0, h1, v0, v1)
        idx = _key_search_index(keys, k)
        assert int(keys[idx]) == k
        expected_s2 = score2_from_key(k)
        assert int(loaded_score2[idx]) == expected_s2
        hero = _ids_to_cards(h0, h1)
        vill = _ids_to_cards(v0, v1)
        lut = equity_hu_showdown_exact(
            hero, vill, table_path=tmp_path / "t.npz", mode="table"
        )
        direct = float(equity_float32_from_score2(expected_s2))
        max_err = max(max_err, abs(float(lut) - direct))
    assert max_err <= GATE_B_FLOAT32_ATOL


def test_census_key_count_sane() -> None:
    keys: set[int] = set()
    for h0, h1, v0, v1 in enumerate_all_disjoint_pairs():
        keys.add(canonical_showdown_key(h0, h1, v0, v1))
    assert len(keys) == 97_513


def _ids_to_cards(c0: int, c1: int) -> tuple[str, str]:
    ranks = "23456789TJQKA"
    suits = "cdhs"

    def one(c: int) -> str:
        return f"{ranks[c // 4]}{suits[c % 4]}"

    return (one(c0), one(c1))


def test_reward_noise_floor_helper() -> None:
    from aof_rl.reward_noise import reward_noise_floor_bb

    floor = reward_noise_floor_bb([1.0, -1.0, 0.5, -0.5], batch_hands=128, noise_k=0.25)
    assert floor >= 0.0


@pytest.mark.skipif(
    not Path("data/aof/hu_showdown_equity.npz").is_file(),
    reason="production HU showdown table not built",
)
def test_production_table_gate_b_subset() -> None:
    warmup()
    path = Path("data/aof/hu_showdown_equity.npz")
    data = np.load(path, allow_pickle=False)
    keys = np.asarray(data["keys"], dtype=np.uint64)
    eq = np.asarray(data["equity"], dtype=np.float32)
    score2 = np.asarray(data["score2"], dtype=np.uint32)
    pairs = enumerate_all_disjoint_pairs()
    rng = random.Random(1)
    for _ in range(200):
        h0, h1, v0, v1 = rng.choice(pairs)
        k = canonical_showdown_key(h0, h1, v0, v1)
        idx = _key_search_index(keys, k)
        assert int(keys[idx]) == k
        expected_s2 = score2_from_key(k)
        assert int(score2[idx]) == expected_s2
        direct = float(equity_float32_from_score2(expected_s2))
        assert abs(float(eq[idx]) - direct) <= GATE_B_FLOAT32_ATOL
