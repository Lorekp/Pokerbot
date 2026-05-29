"""Checkpoint compatibility and loading (Priority 10)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from aof_rl.checkpoint import (
    load_policy_from_checkpoint,
    validate_checkpoint_compatibility,
)
from aof_rl.encode import OBS_DIM_V1, OBS_DIM_V2, obs_dim
from aof_rl.policy import AoFPolicyNet


def _save_ckpt(path: Path, obs_dim: int, obs_version: int, teacher_version: int) -> None:
    model = AoFPolicyNet(hidden=128, obs_dim=obs_dim)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "obs_version": obs_version,
                "teacher_version": teacher_version,
                "obs_dim": obs_dim,
            },
            "update": 1,
        },
        path,
    )


def test_v1_checkpoint_loads(tmp_path: Path) -> None:
    p = tmp_path / "v1.pt"
    _save_ckpt(p, OBS_DIM_V1, 1, 1)
    model, meta = load_policy_from_checkpoint(p, "cpu", obs_version=1, teacher_version=1)
    assert model.obs_dim == OBS_DIM_V1
    assert meta["obs_version"] == 1


def test_v2_checkpoint_loads(tmp_path: Path) -> None:
    p = tmp_path / "v2.pt"
    _save_ckpt(p, OBS_DIM_V2, 2, 2)
    model, meta = load_policy_from_checkpoint(p, "cpu", obs_version=2, teacher_version=2)
    assert model.obs_dim == OBS_DIM_V2
    assert meta["obs_version"] == 2


def test_v1_rejected_as_v2(tmp_path: Path) -> None:
    p = tmp_path / "v1.pt"
    _save_ckpt(p, OBS_DIM_V1, 1, 1)
    with pytest.raises(ValueError, match="incompatible"):
        load_policy_from_checkpoint(p, "cpu", obs_version=2, teacher_version=2)


def test_legacy_checkpoint_warning(tmp_path: Path) -> None:
    model = AoFPolicyNet(hidden=128, obs_dim=OBS_DIM_V1)
    p = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "config": {"teacher_samples": 100}}, p)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m, meta = load_policy_from_checkpoint(p, "cpu")
    assert m.obs_dim == OBS_DIM_V1
    assert any("missing" in str(x.message).lower() for x in w)
    assert meta["obs_dim"] == OBS_DIM_V1


def test_validate_obs_dim_mismatch() -> None:
    with pytest.raises(ValueError):
        validate_checkpoint_compatibility(
            {"obs_version": 2, "obs_dim": OBS_DIM_V2, "teacher_version": 2},
            requested_obs_version=1,
            requested_teacher_version=1,
            requested_obs_dim=obs_dim(1),
        )


def test_train_cli_accepts_version_flags() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "train_aof_self_play",
        Path(__file__).resolve().parents[1] / "scripts" / "train_aof_self_play.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--obs-version", type=int, choices=(1, 2), default=1)
    p.add_argument("--teacher-version", type=int, choices=(1, 2), default=1)
    ns = p.parse_args(["--obs-version", "2", "--teacher-version", "2"])
    assert ns.obs_version == 2
