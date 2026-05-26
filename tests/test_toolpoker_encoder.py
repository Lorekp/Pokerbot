"""Tests for ToolPoker JSON encoder."""

import copy

import numpy as np
import pytest

from pokerbot.data.sft_dataloader import LABEL_SOURCE_SOLVER
from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.encoding.toolpoker_encoder import (
    action_distribution_to_policy,
    encode_toolpoker_row,
    encode_toolpoker_to_batch_dict,
    parse_toolpoker_solver_json,
)


SAMPLE_JSON = {
    "success": True,
    "data": {
        "stage": "preflop",
        "hero_position": "UTG",
        "hole_cards": ["Ah", "Kd"],
        "action_history": "",
        "pot_bb": 1.5,
        "num_players": 6,
        "available_actions": ["fold", "call", "raise 3.0"],
        "gto_action": "raise 3.0",
        "action_distribution": {
            "fold": 0.05,
            "call": 0.10,
            "bet_0.50": 0.25,
            "bet_0.75": 0.35,
            "bet_1.00": 0.15,
            "bet_1.50": 0.05,
            "bet_2.00": 0.03,
            "all_in": 0.02,
        },
    },
}


def test_parse_toolpoker_solver_json():
    data = parse_toolpoker_solver_json(SAMPLE_JSON)
    assert data["gto_action"] == "raise 3.0"


def test_action_distribution_to_policy():
    legal = np.ones(NUM_ACTIONS, dtype=bool)
    policy = action_distribution_to_policy(SAMPLE_JSON["data"]["action_distribution"], legal)
    assert policy.shape == (NUM_ACTIONS,)
    assert abs(policy.sum() - 1.0) < 1e-5


def test_encode_toolpoker_row(tmp_path):
    from pokerbot.encoding.weva_encoder import WEVABuilder, WEVAMapper

    WEVABuilder(k=5, feature_mode="hole_only", num_rollouts=8, max_roots=16).save_artifacts(tmp_path / "weva")
    mapper = WEVAMapper.from_directory(tmp_path / "weva")
    record = encode_toolpoker_row(SAMPLE_JSON, mapper=mapper)
    assert record.actions.shape == (6, 4, 12)
    assert record.solver_policy.shape == (NUM_ACTIONS,)
    assert 0 <= record.weva_bucket_id < 5
    assert record.solver_label >= 0


def test_export_dict_includes_label_source():
    from pokerbot.encoding.weva_encoder import WEVABuilder, WEVAMapper

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path

        p = Path(td)
        WEVABuilder(k=3, feature_mode="hole_only", num_rollouts=4, max_roots=12).save_artifacts(p / "weva")
        mapper = WEVAMapper.from_directory(p / "weva")
        record = encode_toolpoker_row(SAMPLE_JSON, mapper=mapper)
        batch = encode_toolpoker_to_batch_dict(record)
        assert batch["label_source"] == LABEL_SOURCE_SOLVER


def test_encode_rejects_missing_action_distribution(tmp_path):
    from pokerbot.encoding.weva_encoder import WEVABuilder, WEVAMapper

    WEVABuilder(k=3, feature_mode="hole_only", num_rollouts=4, max_roots=12).save_artifacts(tmp_path / "weva")
    mapper = WEVAMapper.from_directory(tmp_path / "weva")
    bad = copy.deepcopy(SAMPLE_JSON)
    del bad["data"]["action_distribution"]
    with pytest.raises(ValueError, match="action_distribution"):
        encode_toolpoker_row(bad, mapper=mapper)
