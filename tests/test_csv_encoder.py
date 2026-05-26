"""Tests for CSV encoder and loader."""

import numpy as np
import pytest

from pokerbot.encoding.csv_encoder import (
    available_moves_to_legal_mask,
    encode_preflop_row,
    encode_postflop_row,
    position_one_hot,
)
from pokerbot.encoding.tensors import SPEC


def test_position_one_hot():
    pos = position_one_hot("BTN")
    assert pos[3] == 1.0
    assert pos.sum() == 1.0


def test_encode_preflop_row():
    row = {
        "prev_line": "UTG/2.0bb/BTN/call",
        "hero_pos": "CO",
        "hero_holding": "AhKd",
        "correct_decision": "raise 6.5",
        "pot_size": 4.0,
        "available_moves": "['fold', 'call', 'raise 6.5']",
    }
    enc = encode_preflop_row(row)
    assert enc["cards"].shape == SPEC.card_shape
    assert enc["actions"].shape == SPEC.action_shape
    assert enc["position"].shape == (6,)
    assert enc["legal_mask"].any()
    assert enc["label"] >= 0


def test_encode_postflop_row():
    row = {
        "preflop_action": "HJ/2.0bb/BB/call",
        "board_flop": "Ks7h2d",
        "board_turn": "Jc",
        "board_river": "7c",
        "postflop_action": "OOP_CHECK/IP_BET_5/OOP_RAISE_14",
        "evaluation_at": "River",
        "hero_position": "IP",
        "holding": "8h8c",
        "correct_decision": "Check",
        "pot_size": 32.0,
        "available_moves": "['Check', 'Bet 24']",
    }
    enc = encode_postflop_row(row)
    assert enc["cards"][2:5].any()
    assert enc["label"] == 1  # check


def test_encode_preflop_bb_sizing():
    row = {
        "prev_line": "",
        "hero_pos": "UTG",
        "hero_holding": "AsKs",
        "correct_decision": "27.0bb",
        "pot_size": 1.5,
        "available_moves": "['fold', '27.0bb']",
    }
    enc = encode_preflop_row(row)
    assert enc["label"] >= 3  # raise bucket


def test_legal_mask_fold_call():
    mask = available_moves_to_legal_mask(["fold", "call"])
    assert mask[0]
    assert mask[2]
    assert not mask[1]
