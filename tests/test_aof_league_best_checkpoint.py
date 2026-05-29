"""Best-checkpoint path helpers and save-on-improvement wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from aof_rl.config import TrainConfig, TrainExperimentConfig
from aof_rl.league_trainer import _maybe_save_best_checkpoint
from aof_rl.policy_v3 import AoFPolicyNetV3, EncoderConfig
from aof_rl.registry import (
    best_checkpoint_path,
    load_league_checkpoint,
    resolve_best_checkpoint_path,
)


def test_best_checkpoint_path_derivation() -> None:
    assert best_checkpoint_path("checkpoints/aof_rl/policy.pt") == Path(
        "checkpoints/aof_rl/policy_best.pt"
    )
    assert resolve_best_checkpoint_path("out/policy.pt") == Path("out/policy_best.pt")
    assert resolve_best_checkpoint_path(
        "out/policy.pt", "out/custom_best.pt"
    ) == Path("out/custom_best.pt")


def test_maybe_save_best_checkpoint_writes_on_improvement(tmp_path: Path) -> None:
    enc = EncoderConfig(encoder_type="deep_sets", d_model=32, n_blocks=1, trunk_hidden=32)
    model = AoFPolicyNetV3(enc)
    final_path = tmp_path / "policy.pt"
    cfg = TrainExperimentConfig(
        checkpoint_path=str(final_path),
        train=TrainConfig(save_checkpoint_on_best=True, min_bb_hand_improvement=0.0),
    )
    best_path = resolve_best_checkpoint_path(cfg.checkpoint_path, cfg.checkpoint_best_path)

    with patch("aof_rl.league_trainer.save_league_checkpoint") as save_mock:
        best = _maybe_save_best_checkpoint(
            cfg=cfg,
            model=model,
            encoder_cfg=enc,
            deterministic=False,
            update=0,
            bb_hand=0.1,
            best_bb_hand=None,
            wall_seconds=1.0,
        )
        assert best == 0.1
        assert save_mock.call_count == 1
        assert save_mock.call_args.kwargs.get("atomic") is True

        best = _maybe_save_best_checkpoint(
            cfg=cfg,
            model=model,
            encoder_cfg=enc,
            deterministic=False,
            update=1,
            bb_hand=0.05,
            best_bb_hand=best,
            wall_seconds=2.0,
        )
        assert best == 0.1
        assert save_mock.call_count == 1

        best = _maybe_save_best_checkpoint(
            cfg=cfg,
            model=model,
            encoder_cfg=enc,
            deterministic=False,
            update=2,
            bb_hand=0.25,
            best_bb_hand=best,
            wall_seconds=3.0,
        )
        assert best == 0.25
        assert save_mock.call_count == 2

    assert str(save_mock.call_args_list[0].args[0]) == str(best_path)


def test_atomic_best_checkpoint_round_trip(tmp_path: Path) -> None:
    from aof_rl.registry import save_league_checkpoint

    enc = EncoderConfig(encoder_type="deep_sets", d_model=32, n_blocks=1, trunk_hidden=32)
    model = AoFPolicyNetV3(enc)
    path = tmp_path / "policy_best.pt"
    save_league_checkpoint(
        path,
        model,
        update=3,
        training={"checkpoint_kind": "best", "best_bb_hand": 0.42},
        encoder_cfg=enc,
        atomic=True,
    )
    loaded, meta = load_league_checkpoint(path, "cpu")
    assert meta["update"] == 3
    assert meta["training"]["checkpoint_kind"] == "best"
    for k, v in model.state_dict().items():
        assert torch.allclose(v.cpu(), loaded.state_dict()[k].cpu())
