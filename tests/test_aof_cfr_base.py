"""Phase A CFR base tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyspiel")

from aof_cfr.buckets import LABELS_169
from aof_cfr.fp_br import nash_gap_3max, trivial_baselines_3max
from aof_cfr.infoset import decode_infoset_key, encode_infoset_key, infoset_key_spec_hash
from aof_cfr.hu_payoff_matrix import (
    build_hu_bucket_equity_matrix,
    load_equity_matrix,
    solver_hu_deal_weights,
)
from aof_cfr.pushfold_payoffs import TERMINAL, audit_terminal_ev, build_static_payoffs
from aof_cfr.query import base_strategy, load_cfr_base

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data/aof/cfr_base"

FAIL_MBBG_CEILING = 50.0
TRIVIAL_RATIO = 0.25


def test_infoset_key_roundtrip() -> None:
    key = encode_infoset_key("SB", "AKs", "A", 8.0)
    assert key == "SB|AKs|A|d8"
    assert decode_infoset_key(key) == ("SB", "AKs", "A", 8)
    # default depth is 8bb
    assert encode_infoset_key("SB", "AKs", "A") == "SB|AKs|A|d8"
    # nearest-bucket: 9bb -> d8, 11bb -> d10
    assert encode_infoset_key("BTN", "AA", "-", 9.0).endswith("|d8")
    assert encode_infoset_key("BTN", "AA", "-", 11.0).endswith("|d10")
    assert len(infoset_key_spec_hash()) == 16


def test_deal_weights_uniform() -> None:
    w = solver_hu_deal_weights(4)
    assert w.shape == (4, 4)
    assert abs(float(w.sum()) - 1.0) < 1e-12


def test_trio_deal_weights_uniform() -> None:
    from aof_cfr.trio_payoff_tensor import trio_deal_weights

    w = trio_deal_weights(4)
    assert w.shape == (4, 4, 4)
    assert abs(float(w.sum()) - 1.0) < 1e-12


def test_terminal_ev_audit() -> None:
    audit_terminal_ev()
    t = TERMINAL
    assert abs(t.sb_win_uncontested_after_btn_fold - 1.0) < 1e-12
    assert abs(t.btn_win_uncontested_after_sb_fold - 1.5) < 1e-12


def test_trio_cfr_smoke() -> None:
    from pokerbot.solver.trio_preflop_dcfr import solve_3max_pushfold, trio_policy_rows

    n = 4
    shares = np.full((n, n, n, 3), 1.0 / 3.0)
    pair_eq = np.full((n, n, n, 3), 0.5)
    deal_w = np.ones((n, n, n)) / n**3
    r = solve_3max_pushfold(
        iterations=100, shares=shares, pair_eq=pair_eq, deal_w=deal_w
    )
    rows = trio_policy_rows(r)
    assert len(rows) == n * 6
    assert r.exploitability_bb >= 0.0


def test_3max_fp_smoke() -> None:
    from pokerbot.solver.trio_preflop_fp import solve_3max_fp, trio_fp_policy_rows

    n = 4
    shares = np.full((n, n, n, 3), 1.0 / 3.0)
    pair_eq = np.full((n, n, n, 3), 0.5)
    deal_w = np.ones((n, n, n)) / n**3
    r = solve_3max_fp(
        max_rounds=20, shares=shares, pair_eq=pair_eq, deal_w=deal_w
    )
    rows = trio_fp_policy_rows(r)
    assert len(rows) == n * 6
    st = build_static_payoffs(shares, pair_eq)
    trivial = trivial_baselines_3max(st, deal_w)
    assert r.exploitability_bb * 1000 < trivial["always_fold"] * TRIVIAL_RATIO


def test_3max_cfr_plus_smoke() -> None:
    from aof_cfr.solve import solve_3max_cfr_plus

    path = BASE_DIR / "trio_eq_tensor.npz"
    if not path.is_file():
        pytest.skip("run scripts/build_3way_eq_tensor_npz.py first")
    r = solve_3max_cfr_plus(iterations=400, tensor_path=str(path))
    assert r.br_method == "exact"
    assert len(r.policy_rows) == 1014


def test_nash_gap_consistency() -> None:
    n = 4
    shares = np.full((n, n, n, 3), 1.0 / 3.0)
    pair_eq = np.full((n, n, n, 3), 0.5)
    deal_w = np.ones((n, n, n)) / n**3
    st = build_static_payoffs(shares, pair_eq)
    btn = np.full(n, 0.5)
    sb = np.full((2, n), 0.5)
    bb = np.full((3, n), 0.5)
    g1, _ = nash_gap_3max(st, deal_w, btn, sb, bb)
    g2, _ = nash_gap_3max(st, deal_w, btn, sb, bb)
    assert g1 == g2


@pytest.mark.slow
def test_hu_bucket_matrix_diagonal() -> None:
    eq = build_hu_bucket_equity_matrix()
    assert eq.shape == (169, 169)
    np.testing.assert_allclose(np.diag(eq), 0.5, atol=1e-6)


@pytest.mark.slow
def test_hu_cfr_plus_smoke() -> None:
    from aof_cfr.solve import solve_hu_cfr_plus

    eq_path = BASE_DIR / "hu_eq_matrix.npz"
    if not eq_path.is_file():
        pytest.skip("run scripts/solve_aof_cfr_base.py to build eq matrix")
    r = solve_hu_cfr_plus(iterations=400, eq_matrix_path=str(eq_path).replace("\\", "/"))
    assert r.exploitability_mbb_g < 50.0
    assert len(r.policy_rows) >= 169 * 2 - 1


@pytest.mark.skipif(not (BASE_DIR / "cfr_base_2p.npz").is_file(), reason="CFR artifacts not built")
def test_base_strategy_lookup_allocation_free() -> None:
    load_cfr_base(BASE_DIR)
    p = base_strategy(2, "SB", "-", "AA")
    assert 0.0 <= p <= 1.0


@pytest.mark.skipif(not (BASE_DIR / "cfr_base_2p.npz").is_file(), reason="CFR artifacts not built")
@pytest.mark.slow
def test_hu_oracle_agreement() -> None:
    from aof_cfr.hu_validate import compare_freqs, dcfr_oracle_freqs, load_policy_freqs

    sb_sol, bb_sol = load_policy_freqs(BASE_DIR / "cfr_base_2p.npz")
    sb_oracle, bb_oracle = dcfr_oracle_freqs(
        BASE_DIR / "hu_eq_matrix.npz", iterations=2500
    )
    hu = compare_freqs(sb_sol, bb_sol, sb_oracle, bb_oracle, tolerance=0.02)
    assert hu["passed"], f"max_diff={hu['max_diff']} — fix payoff/deal model"
