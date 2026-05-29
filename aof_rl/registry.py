"""Checkpoint registry schema v2 for AoF PPO league (obs v3)."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from aof_rl.policy_v3 import (
    AoFPolicyNetV3,
    EncoderConfig,
    encoder_config_from_dict,
    encoder_config_to_dict,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


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


def build_checkpoint_payload(
    model: AoFPolicyNetV3,
    *,
    update: int,
    training: dict[str, Any],
    encoder_cfg: EncoderConfig,
    teacher_version: int = 2,
    evaluation: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model.state_dict(),
        "config": {
            "obs_version": 3,
            "teacher_version": teacher_version,
            "encoder": encoder_config_to_dict(encoder_cfg),
            "training": training,
            "provenance": provenance
            or {
                "teacher_sources": ["static_profile", "nash_mc"],
                "equity_engine": "exact_plus_seeded_mc",
            },
        },
        "evaluation": evaluation or {"bb_per_100": None, "paired_eval_hands": 0},
        "update": update,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }


def best_checkpoint_path(path: str | Path) -> Path:
    """Sibling path for rolling best-by-training-metric checkpoints."""
    p = Path(path)
    return p.with_name(f"{p.stem}_best{p.suffix}")


def resolve_best_checkpoint_path(
    checkpoint_path: str | Path,
    checkpoint_best_path: str | Path | None = None,
) -> Path:
    if checkpoint_best_path is not None:
        return Path(checkpoint_best_path)
    return best_checkpoint_path(checkpoint_path)


def save_league_checkpoint(
    path: str | Path,
    model: AoFPolicyNetV3,
    *,
    update: int,
    training: dict[str, Any],
    encoder_cfg: EncoderConfig,
    teacher_version: int = 2,
    evaluation: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    atomic: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint_payload(
        model,
        update=update,
        training=training,
        encoder_cfg=encoder_cfg,
        teacher_version=teacher_version,
        evaluation=evaluation,
        provenance=provenance,
    )
    if not atomic:
        torch.save(payload, path)
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_league_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> tuple[AoFPolicyNetV3, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    schema = int(ckpt.get("schema_version", 0))
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema_version={SCHEMA_VERSION}, got {schema} in {path}"
        )
    config = dict(ckpt.get("config") or {})
    if int(config.get("obs_version", 0)) != 3:
        raise ValueError(f"League checkpoint must be obs_version=3, got {config}")
    enc = encoder_config_from_dict(config.get("encoder") or {})
    model = AoFPolicyNetV3(enc).to(device)
    model.load_state_dict(ckpt["model"])
    meta = {
        **config,
        "schema_version": schema,
        "update": ckpt.get("update"),
        "evaluation": ckpt.get("evaluation"),
        "timestamp": ckpt.get("timestamp"),
        "git_commit": ckpt.get("git_commit"),
        "checkpoint_path": str(path),
    }
    return model, meta
