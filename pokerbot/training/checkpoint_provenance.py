"""Checkpoint artifact provenance helpers (Phase 1 BC vs solver discrimination)."""

from __future__ import annotations

from pathlib import Path

import torch

from pokerbot.training.sft import BC_WARNING, LABEL_SOURCE_BC

_BC_META_KEYS = ("mode", "gto_verified", "label_source")


def load_checkpoint_metadata(path: Path) -> dict:
    """Return provenance fields stored in a policy checkpoint, if any."""
    if not path.exists():
        return {}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return {}
    return {k: ckpt.get(k) for k in _BC_META_KEYS if k in ckpt}


def is_bc_checkpoint(path: Path) -> bool:
    """
    True when checkpoint metadata marks a PokerBench BC artifact.

    Uses Phase 1 provenance fields (not filename alone).
    """
    meta = load_checkpoint_metadata(path)
    if not meta:
        return False
    if meta.get("gto_verified") is False:
        return True
    if meta.get("mode") == "bc":
        return True
    if meta.get("label_source") == LABEL_SOURCE_BC:
        return True
    return False


def warn_if_bc_checkpoint(path: Path, *, context: str = "hero rollout") -> None:
    """Emit the standard BC banner when a resolved checkpoint is not GTO-verified."""
    if is_bc_checkpoint(path):
        print(BC_WARNING)
        print(f"Resolved {context} checkpoint: {path} (BC, not GTO-verified)\n")
