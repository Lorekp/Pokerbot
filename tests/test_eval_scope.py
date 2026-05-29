"""Tests for eval-time spot scope propagation."""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.encoding.eval_scope import (
    GATE_ISOLATED_HARNESS_SCOPE_NOTE,
    classify_phase1_gate_scope,
    encode_state_for_policy,
    infer_policy_spot_class,
    require_encoded_spot_scope,
    require_gate_policy_scope_encoding,
)
from pokerbot.encoding.spot_scope import encode_spot_scope
from pokerbot.engine.cards import Card
from pokerbot.solver.spot_enumerator import apply_flop_board, build_btn_open_bb_call, pick_hero_hand


def _sample_flop_state():
    board = [Card.from_str(c) for c in ("7c", "5d", "2h")]
    state, oop, _, _ = build_btn_open_bb_call(100.0)
    apply_flop_board(state, board)
    hole = pick_hero_hand([["Ah", "Kd"], ["Qs", "Qh"]], board, __import__("random").Random(0))
    state.players[oop].hole = hole
    return state, oop


def test_encode_spot_scope_one_hot_matches_training():
    vanilla = encode_spot_scope("hu_postflop_2p")
    bunch = encode_spot_scope("pseudo_hu_bunching")
    assert vanilla.tolist() == [1.0, 0.0]
    assert bunch.tolist() == [0.0, 1.0]


def test_encode_state_for_policy_vanilla_scope():
    state, hero = _sample_flop_state()
    enc = encode_state_for_policy(state, hero, spot_class="hu_postflop_2p", require_scope=True)
    assert np.allclose(enc["spot_scope"], encode_spot_scope("hu_postflop_2p"))


def test_require_encoded_spot_scope_fails_on_zero():
    with pytest.raises(SystemExit, match="all-zero"):
        require_encoded_spot_scope({"spot_scope": np.zeros(2, dtype=np.float32)})


def test_gate_isolated_classifies_vanilla_only():
    state, hero = _sample_flop_state()
    ok, spot = classify_phase1_gate_scope(state, hero)
    assert ok
    assert spot == "hu_postflop_2p"
    assert "pseudo_hu_bunching" in GATE_ISOLATED_HARNESS_SCOPE_NOTE


def test_require_gate_policy_scope_encoding():
    checks = require_gate_policy_scope_encoding()
    assert checks["vanilla_matches_training"]
    assert checks["bunching_matches_training"]
    assert checks["training_unevaluated_fraction"]["pseudo_hu_bunching"] == pytest.approx(1 / 3)


def test_infer_policy_spot_class_on_flop():
    state, hero = _sample_flop_state()
    assert infer_policy_spot_class(state, hero) == "hu_postflop_2p"
