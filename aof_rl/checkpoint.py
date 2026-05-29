"""Checkpoint metadata, compatibility checks, and policy loading."""

from __future__ import annotations

import logging
import subprocess
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from aof_equity_config import EquityRuntimeConfig, default_equity_config
from aof_rl.encode import OBS_DIM_V1, obs_dim
from aof_rl.policy import AoFPolicyNet

logger = logging.getLogger(__name__)

LEGACY_OBS_DIM = OBS_DIM_V1


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def equity_config_to_dict(cfg: EquityRuntimeConfig | None = None) -> dict[str, Any]:
    c = cfg or default_equity_config()
    return asdict(c)


def enrich_checkpoint_meta(
    cfg: Any,
    *,
    training_profile: str = "train",
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build config dict for torch.save from AoFSelfPlayConfig-like object."""
    odim = obs_dim(getattr(cfg, "obs_version", 1))
    return {
        "teacher_samples": getattr(cfg, "teacher_samples", 2000),
        "payoff_samples": getattr(cfg, "payoff_samples", 4000),
        "lambda_teacher": getattr(cfg, "lambda_teacher", 0.3),
        "stack_bb": getattr(cfg, "stack_bb", 8.0),
        "obs_version": getattr(cfg, "obs_version", 1),
        "teacher_version": getattr(cfg, "teacher_version", 1),
        "obs_dim": odim,
        "hidden": getattr(cfg, "hidden", 128),
        "table_sizes": (2, 3, 4),
        "equity_config": equity_config_to_dict(),
        "training_seed": getattr(cfg, "seed", 0),
        "training_profile": training_profile,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit if git_commit is not None else _git_commit(),
    }


def validate_checkpoint_compatibility(
    checkpoint_meta: dict[str, Any],
    *,
    requested_obs_version: int,
    requested_teacher_version: int,
    requested_obs_dim: int,
    in_features: int | None = None,
) -> dict[str, Any]:
    """
    Validate checkpoint vs requested versions. Returns resolved meta (may fill legacy defaults).

    Raises ValueError on incompatible obs dimension / version mismatch.
    """
    meta = dict(checkpoint_meta)
    ckpt_obs_version = int(meta.get("obs_version", 1))
    ckpt_teacher_version = int(meta.get("teacher_version", 1))
    ckpt_obs_dim = meta.get("obs_dim")

    legacy = ckpt_obs_dim is None
    if legacy:
        warnings.warn(
            "Checkpoint missing obs_version/obs_dim metadata; assuming obs v1 (117 dims).",
            UserWarning,
            stacklevel=2,
        )
        ckpt_obs_version = 1
        ckpt_obs_dim = LEGACY_OBS_DIM

    ckpt_obs_dim = int(ckpt_obs_dim)
    expected_dim = obs_dim(requested_obs_version)

    if requested_obs_dim != expected_dim:
        raise ValueError(
            f"requested_obs_dim={requested_obs_dim} does not match "
            f"obs_dim({requested_obs_version})={expected_dim}"
        )

    if ckpt_obs_dim != requested_obs_dim:
        raise ValueError(
            f"Checkpoint obs_dim={ckpt_obs_dim} incompatible with "
            f"requested obs_version={requested_obs_version} (dim={requested_obs_dim})"
        )

    if ckpt_obs_version != requested_obs_version and not legacy:
        raise ValueError(
            f"Checkpoint obs_version={ckpt_obs_version} != requested {requested_obs_version}"
        )

    if in_features is not None and in_features != ckpt_obs_dim:
        raise ValueError(
            f"Model input features={in_features} != checkpoint obs_dim={ckpt_obs_dim}"
        )

    if requested_teacher_version >= 2 and requested_obs_version < 2:
        warnings.warn(
            "teacher_version>=2 with obs_version<2: teacher-path/provenance one-hots "
            "are only in obs v2.",
            UserWarning,
            stacklevel=2,
        )

    meta.setdefault("obs_version", ckpt_obs_version)
    meta.setdefault("teacher_version", ckpt_teacher_version)
    meta.setdefault("obs_dim", ckpt_obs_dim)
    return meta


def load_policy_from_checkpoint(
    path: str | Path,
    device: torch.device | str,
    *,
    obs_version: int | None = None,
    teacher_version: int | None = None,
) -> tuple[AoFPolicyNet, dict[str, Any]]:
    """Load policy weights with compatibility validation."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = dict(ckpt.get("config") or {})
    state = ckpt["model"]

    req_obs = int(obs_version if obs_version is not None else config.get("obs_version", 1))
    req_teacher = int(
        teacher_version if teacher_version is not None else config.get("teacher_version", 1)
    )
    req_dim = obs_dim(req_obs)

    in_features = None
    if isinstance(state, dict) and "net.0.weight" in state:
        in_features = int(state["net.0.weight"].shape[1])

    meta = validate_checkpoint_compatibility(
        config,
        requested_obs_version=req_obs,
        requested_teacher_version=req_teacher,
        requested_obs_dim=req_dim,
        in_features=in_features,
    )

    hidden = int(config.get("hidden", 128))
    model = AoFPolicyNet(hidden=hidden, obs_dim=req_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    meta["checkpoint_path"] = str(path)
    meta["update"] = ckpt.get("update")
    return model, meta
