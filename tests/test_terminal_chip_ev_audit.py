"""Regression: 3-max terminal chip-EV matches terminal_ev oracle (fold lines)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aof_cfr.pushfold_payoffs import (
    BLINDS_DEAD,
    POT_HU_STACKS,
    POT_THREE_STACKS,
    SB_POST,
    STACK_BB,
    TERMINAL,
    build_static_payoffs,
)
from aof_cfr.terminal_ev import fold_terminal_returns
from scripts.audit_terminal_chip_ev import run_audit

ROOT = Path(__file__).resolve().parents[1]
TENSOR = ROOT / "data/aof/cfr_base/trio_eq_tensor.npz"


def test_pot_constants_match_hu_solver() -> None:
    assert POT_HU_STACKS == pytest.approx(1.5 + 2.0 * STACK_BB)
    assert POT_THREE_STACKS == pytest.approx(1.5 + 3.0 * STACK_BB)


def test_fold_terminals_match_oracle() -> None:
    # Spot 1: BTN jam, SB fold, BB fold
    r1 = fold_terminal_returns(set(), [0], np.array([8.0, 0.5, 1.0]))
    assert r1[0] == pytest.approx(TERMINAL.btn_win_uncontested_after_sb_fold)
    assert r1[1] == pytest.approx(TERMINAL.sb_fold_after_btn_jam)
    assert r1[2] == pytest.approx(TERMINAL.bb_fold)

    # Spot 2: BTN fold, SB jam, BB fold
    r2 = fold_terminal_returns(set(), [1], np.array([0.0, 8.0, 1.0]))
    assert r2[0] == pytest.approx(TERMINAL.btn_fold)
    assert r2[1] == pytest.approx(TERMINAL.sb_win_uncontested_after_btn_fold)
    assert r2[2] == pytest.approx(TERMINAL.bb_fold)


@pytest.mark.skipif(not TENSOR.is_file(), reason="trio_eq_tensor.npz missing")
def test_audit_script_passes() -> None:
    report = run_audit(tensor_path=TENSOR)
    assert report["audit_pass"]


@pytest.mark.skipif(not TENSOR.is_file(), reason="trio_eq_tensor.npz missing")
def test_static_btn_bb_pot_includes_all_blinds() -> None:
    from aof_cfr.trio_payoff_tensor import load_trio_tensor

    shares, pair_eq, _ = load_trio_tensor(TENSOR)
    st = build_static_payoffs(shares, pair_eq)
    btn, sb, bb = 0, 1, 2
    eq = float(pair_eq[btn, sb, bb, 0])
    expected = eq * POT_HU_STACKS - STACK_BB
    assert float(st.btn_bb_call0[btn, sb, bb]) == pytest.approx(expected)
    assert float(st.sb_bb_call1[btn, sb, bb]) == pytest.approx(
        float(pair_eq[btn, sb, bb, 2]) * POT_HU_STACKS - STACK_BB
    )
