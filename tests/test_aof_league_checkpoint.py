"""League checkpoint schema v2 round-trip."""

from __future__ import annotations

from pathlib import Path

import torch

from aof_rl.policy_v3 import AoFPolicyNetV3, EncoderConfig
from aof_rl.registry import SCHEMA_VERSION, load_league_checkpoint, save_league_checkpoint


def test_league_checkpoint_round_trip(tmp_path: Path) -> None:
    enc = EncoderConfig(encoder_type="deep_sets", d_model=48, n_blocks=1, trunk_hidden=48)
    model = AoFPolicyNetV3(enc)
    path = tmp_path / "league_v3.pt"
    save_league_checkpoint(
        path,
        model,
        update=7,
        training={"algo": "ppo_league", "seed": 42},
        encoder_cfg=enc,
        teacher_version=2,
        evaluation={"bb_per_100": 1.5, "paired_eval_hands": 100},
        provenance={"teacher_sources": ["static_profile"]},
    )
    loaded, meta = load_league_checkpoint(path, "cpu")
    assert meta["obs_version"] == 3
    assert meta["teacher_version"] == 2
    assert meta["encoder"]["encoder_type"] == "deep_sets"
    assert meta["encoder"]["d_model"] == 48
    assert meta["update"] == 7
    assert meta["evaluation"]["bb_per_100"] == 1.5
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["schema_version"] == SCHEMA_VERSION
    for k, v in model.state_dict().items():
        assert torch.allclose(v.cpu(), loaded.state_dict()[k].cpu())
