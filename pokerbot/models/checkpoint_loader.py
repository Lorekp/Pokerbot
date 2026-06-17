"""Shared WEVA-compatible policy checkpoint loading with legacy detection."""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import torch

from pokerbot.models.policy import PseudoSiamesePolicy, num_weva_buckets_from_checkpoint


def _torch_load_xplat(ckpt_path, device):
    """torch.load with a cross-platform pathlib shim. Checkpoints saved on Windows pickle a
    ``WindowsPath`` (and POSIX a ``PosixPath``); unpickling the other flavor raises
    ``NotImplementedError: cannot instantiate 'WindowsPath'/'PosixPath' on your system``. Temporarily
    alias the foreign class to the native one so the cloud (Linux) can load a Windows-trained .pt.
    Restored in a finally so global pathlib behavior is unchanged outside this load."""
    if sys.platform == "win32":
        foreign, native = "PosixPath", pathlib.WindowsPath
    else:
        foreign, native = "WindowsPath", pathlib.PosixPath
    orig = getattr(pathlib, foreign)
    setattr(pathlib, foreign, native)
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    finally:
        setattr(pathlib, foreign, orig)

LEGACY_CARD_ENCODER_KEYS = ("card_encoder.0.net.0.weight", "card_encoder.1.net.0.weight")
POLICY_ENCODING_V2_CARD_KEY = "card_encoder.0.net.0.weight"

LEGACY_CHECKPOINT_ERROR = """
*** Legacy checkpoint architecture (card_encoder) — not WEVA-compatible ***

This checkpoint predates the WEVA bucket embedding migration. Current PseudoSiamesePolicy
expects weva_embed.weight, not card_encoder.* weights.

Fix: re-run the Phase 1 pipeline to produce WEVA-compatible artifacts:
  python scripts/fast_train.py --device cuda

Or promote a WEVA checkpoint explicitly:
  python scripts/promote_gate_baseline.py --source checkpoints/policy_sft.pt
"""


PRE_ENCODING_V2_ERROR = """
*** Pre-encoding-v2 checkpoint — missing board card branch ***

This checkpoint predates policy encoding v2 (board cards via encode_cards + spot scope).
It cannot load into the current PseudoSiamesePolicy (requires card_encoder weights).

Fix: re-run training with the upgraded encoding:
  python scripts/train_sft.py --mode solver --parquet-glob data/solver/train_prod60.parquet

Phase 2 self-play / PPO checkpoints (policy_selfplay.pt, policy_selfplay_mixed.pt) also
require regeneration after this encoding upgrade.
"""


def is_legacy_card_encoder_checkpoint(state_dict: dict) -> bool:
    """True when state dict uses pre-WEVA card_encoder weights."""
    if not isinstance(state_dict, dict):
        return False
    if "weva_embed.weight" in state_dict:
        return False
    return any(k in state_dict for k in LEGACY_CARD_ENCODER_KEYS)


def is_pre_encoding_v2_checkpoint(state_dict: dict) -> bool:
    """True when checkpoint lacks policy encoding v2 card_encoder branch."""
    if not isinstance(state_dict, dict):
        return False
    if is_legacy_card_encoder_checkpoint(state_dict):
        return False
    return POLICY_ENCODING_V2_CARD_KEY not in state_dict


def load_pseudo_siamese_policy(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    strict: bool = True,
) -> PseudoSiamesePolicy:
    """
    Load a hero policy checkpoint with WEVA bucket inference and legacy detection.

    Raises ValueError with retrain instructions when card_encoder weights are detected.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {ckpt_path}")

    ckpt = _torch_load_xplat(ckpt_path, device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Invalid policy checkpoint format (missing 'model' key): {ckpt_path}")

    state_dict = ckpt["model"]
    if is_legacy_card_encoder_checkpoint(state_dict):
        raise ValueError(LEGACY_CHECKPOINT_ERROR.strip())
    if is_pre_encoding_v2_checkpoint(state_dict):
        raise ValueError(PRE_ENCODING_V2_ERROR.strip())

    num_buckets = num_weva_buckets_from_checkpoint(ckpt)
    model = PseudoSiamesePolicy(num_weva_buckets=num_buckets).to(device)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model
