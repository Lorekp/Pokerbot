"""Regression tests for n-way MC sampler (deck + unbiased boards)."""

from __future__ import annotations

import numpy as np
import pytest

from aof_cfr.buckets import LABELS_169
from aof_cfr.terminal_ev import _exact_nway_shares_numba
from aof_cfr.trio_eval_mc import build_live_deck, mc_nway_shares, mc_nway_shares_and_pair_eq
from aof_cfr.trio_payoff_tensor import _all_combo_triples, exact_shares_for_triple_oracle
from aof_equity_grids import _card_str_to_id, _three_pair_equities_exact


def _triple_indices(a: str, b: str, c: str) -> tuple[int, int, int]:
    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    return idx[a], idx[b], idx[c]


def test_live_deck_22_22_33_probe() -> None:
    combos = _all_combo_triples(0, 0, 1)
    flat = np.array([_card_str_to_id(c) for c in sum(combos[0], ())], dtype=np.int32)
    deck, nd = build_live_deck(flat, 3)
    assert nd == 46
    assert len(set(deck[:nd].tolist())) == 46
    assert not set(deck[:nd].tolist()) & set(flat.tolist())


@pytest.mark.parametrize(
    "label,indices",
    [
        ("22/22/33", (0, 0, 1)),
        ("22/42o/97s", (0, 48, 120)),
    ],
)
def test_collision_combo_mc_converges(label: str, indices: tuple[int, int, int]) -> None:
    i, j, k = indices
    combo = _all_combo_triples(i, j, k)[0]
    flat = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    exact = _exact_nway_shares_numba(flat, 3)
    acc = np.zeros(3, dtype=np.float64)
    for seed in range(50):
        acc += mc_nway_shares(flat, 3, 8192, seed)
    mean = acc / 50.0
    assert np.max(np.abs(mean - exact)) < 0.004, (label, mean, exact)


def test_clean_triple_unchanged_within_mc_noise() -> None:
    i, j, k = _triple_indices("AKs", "AQs", "KQo")
    o0, o1, o2, _, _, _, _ = exact_shares_for_triple_oracle(i, j, k)
    exact = np.array([o0, o1, o2], dtype=np.float64)
    combo = _all_combo_triples(i, j, k)[0]
    flat = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    mc = mc_nway_shares(flat, 3, 8192, 0)
    assert np.max(np.abs(mc - exact)) < 0.02


def test_native_mc_shares_match_numba_at_fixed_seed() -> None:
    from tools.ompeval_cli import native as omp_native

    if not omp_native.native_trio_available():
        pytest.skip("native trio not available")
    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    combo = _all_combo_triples(idx["22"], idx["22"], idx["33"])[0]
    flat = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    exact = _exact_nway_shares_numba(flat, 3)
    acc = np.zeros(3, dtype=np.float64)
    for seed in range(10):
        acc += omp_native.mc_nway_shares_native(flat, 3, 4096, seed)
    mean = acc / 10.0
    assert np.max(np.abs(mean - exact)) < 0.004


def test_mc_fused_pair_eq_converges() -> None:
    i, j, k = _triple_indices("AKs", "AQs", "KQo")
    combo = _all_combo_triples(i, j, k)[0]
    flat = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    exact_bb, exact_bs, exact_sb = _three_pair_equities_exact(*flat)
    acc = np.zeros(3, dtype=np.float64)
    for seed in range(20):
        _, pe = mc_nway_shares_and_pair_eq(flat, 3, 4096, seed)
        acc += pe
    mean = acc / 20.0
    assert np.max(np.abs(mean - np.array([exact_bb, exact_bs, exact_sb]))) < 0.005


def test_four_max_collision_deal() -> None:
    cards = ["2c", "2d", "2h", "2s", "3c", "3d", "4c", "5d"]
    flat = np.array([_card_str_to_id(c) for c in cards], dtype=np.int32)
    deck, nd = build_live_deck(flat, 4)
    assert nd == 44
    exact = _exact_nway_shares_numba(flat, 4)
    acc = np.zeros(4, dtype=np.float64)
    for seed in range(30):
        acc += mc_nway_shares(flat, 4, 8192, seed)
    mean = acc / 30.0
    assert np.max(np.abs(mean - exact)) < 0.004
