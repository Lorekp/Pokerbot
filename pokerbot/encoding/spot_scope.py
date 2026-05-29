"""Spot validity scope encoding for policy inputs (swap-friendly)."""

from __future__ import annotations

import numpy as np

# Ordered registry — extend when adding scopes; do not one-hot at call sites.
SPOT_CLASS_ORDER: tuple[str, ...] = (
    "hu_postflop_2p",
    "pseudo_hu_bunching",
)

SPOT_CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(SPOT_CLASS_ORDER)}

NUM_SPOT_SCOPE = len(SPOT_CLASS_ORDER)

# Hero seat one-hot (6) + spot scope vector.
POLICY_POSITION_BASE_DIM = 6
POLICY_POSITION_DIM = POLICY_POSITION_BASE_DIM + NUM_SPOT_SCOPE


def encode_spot_scope(
    spot_class: str,
    oop_range: str = "",
    ip_range: str = "",
) -> np.ndarray:
    """
    Encode label validity scope for the policy.

    Returns one-hot over SPOT_CLASS_ORDER today. ``oop_range`` / ``ip_range`` are
    reserved for a future learned embedding over range identifiers.
    """
    _ = oop_range, ip_range  # reserved for range-id embedding swap
    vec = np.zeros(NUM_SPOT_SCOPE, dtype=np.float32)
    idx = SPOT_CLASS_TO_INDEX.get(str(spot_class).strip())
    if idx is not None:
        vec[idx] = 1.0
    return vec


def fuse_position_scope(position: np.ndarray, spot_scope: np.ndarray) -> np.ndarray:
    """Concatenate hero position one-hot with spot scope (policy pos input)."""
    pos = np.asarray(position, dtype=np.float32).reshape(-1)
    if pos.shape[0] != POLICY_POSITION_BASE_DIM:
        raise ValueError(
            f"position must have length {POLICY_POSITION_BASE_DIM}, got {pos.shape[0]}"
        )
    scope = np.asarray(spot_scope, dtype=np.float32).reshape(-1)
    if scope.shape[0] != NUM_SPOT_SCOPE:
        raise ValueError(f"spot_scope must have length {NUM_SPOT_SCOPE}, got {scope.shape[0]}")
    return np.concatenate([pos, scope], axis=0)
