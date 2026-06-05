"""Native trio exact metrics parity vs numba oracle."""

from __future__ import annotations

import numpy as np
import pytest

from aof_cfr.buckets import LABELS_169
from aof_cfr.trio_payoff_tensor import _all_combo_triples, exact_shares_for_triple_oracle
from aof_equity import warmup
from aof_equity_grids import _card_str_to_id, _exact_six_card_combo_metrics
from tools.ompeval_cli import native as omp_native

pytestmark = pytest.mark.skipif(
    not omp_native.native_trio_available(),
    reason="native trio kernel not available",
)


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


def test_exact_six_card_combo_metrics_matches_numba() -> None:
    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    combo = _all_combo_triples(idx["22"], idx["22"], idx["33"])[0]
    cards = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    py = _exact_six_card_combo_metrics(*cards)
    nat = omp_native.exact_six_card_combo_metrics(cards)
    assert np.max(np.abs(np.array(py) - np.array(nat))) < 1e-9


def test_avg_metrics_for_triple_matches_python_oracle() -> None:
    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    i, j, k = idx["AKs"], idx["AQs"], idx["KQo"]
    py = exact_shares_for_triple_oracle(i, j, k)
    nat = omp_native.avg_metrics_for_triple(i, j, k, mc_boards=2048)
    assert np.max(np.abs(np.array(py[:6]) - np.array(nat[:6]))) < 0.005
    assert py[6] == nat[6]


def test_eval_combo_adaptive_mc_shares_near_exact() -> None:
    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    combo = _all_combo_triples(idx["AKs"], idx["AQs"], idx["KQo"])[0]
    cards = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    exact = omp_native.exact_six_card_combo_metrics(cards)
    metrics, _, _ = omp_native.eval_combo_adaptive_native(
        cards, 12345, 2048, min_boards=16, stderr_gate=0.001
    )
    assert np.max(np.abs(np.array(metrics[:3]) - np.array(exact[:3]))) < 0.005


def test_exact_pair_eq_for_triple_matches_oracle() -> None:
    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    i, j, k = idx["22"], idx["96o"], idx["K8o"]
    py = exact_shares_for_triple_oracle(i, j, k)
    nat = omp_native.exact_pair_eq_for_triple(i, j, k)
    assert np.max(np.abs(np.array(py[3:6]) - np.array(nat[:3]))) < 1e-9
    assert py[6] == nat[3]


def test_eval_combo_adaptive_no_hidden_exact_profile() -> None:
    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    combo = _all_combo_triples(idx["AKs"], idx["AQs"], idx["KQo"])[0]
    cards = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    omp_native.reset_trio_row_profile()
    omp_native.eval_combo_adaptive_native(cards, 99, 512, min_boards=16, stderr_gate=0.001)
    prof = omp_native.get_trio_row_profile()
    assert prof["mc_hidden_pair_exact_ns"] == 0.0


def test_native_profile_counters_reset() -> None:
    omp_native.reset_trio_row_profile()
    prof = omp_native.get_trio_row_profile()
    assert prof["exact_triple_count"] == 0
    assert prof["mc_triple_count"] == 0
    assert prof["mc_board_evals"] == 0


@pytest.mark.parametrize("label,indices", [("22/22/33", (0, 0, 1)), ("AKs/AQs/KQo", (12, 13, 48))])
def test_native_exact_random_triples(label: str, indices: tuple[int, int, int]) -> None:
    i, j, k = indices
    py = exact_shares_for_triple_oracle(i, j, k)
    nat = omp_native.avg_metrics_for_triple(
        i,
        j,
        k,
        mc_boards=2048,
        exact_combo_max=10_000,
    )
    assert np.max(np.abs(np.array(py[:6]) - np.array(nat[:6]))) < 1e-6, label
