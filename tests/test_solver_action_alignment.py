"""Tests for postflop solver action key -> policy index mapping."""

import numpy as np
import pytest

from pokerbot.engine.actions import ACTION_VOCAB
from pokerbot.encoding.toolpoker_encoder import (
    action_distribution_to_policy,
    solver_action_keys_to_legal_mask,
)


def test_solver_action_keys_to_legal_mask_maps_bet_050():
    mask = solver_action_keys_to_legal_mask(["check", "bet_0.50", "all_in"])
    assert mask[ACTION_VOCAB.index("bet_0.50")]
    assert not mask[ACTION_VOCAB.index("bet_2.00")]
    assert mask[ACTION_VOCAB.index("check")]
    assert mask[ACTION_VOCAB.index("all_in")]


def test_action_distribution_preserves_bet_050_mass():
    dist = {"check": 0.4, "bet_0.50": 0.35, "all_in": 0.25}
    legal = solver_action_keys_to_legal_mask(list(dist.keys()))
    policy = action_distribution_to_policy(dist, legal)
    assert policy[ACTION_VOCAB.index("bet_0.50")] == pytest.approx(0.35, abs=1e-5)
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)
