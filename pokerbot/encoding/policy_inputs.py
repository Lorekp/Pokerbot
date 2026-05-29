"""Build policy-network inputs from encoded game state."""

from __future__ import annotations

from typing import Any

import numpy as np

from pokerbot.encoding.spot_scope import (
    POLICY_POSITION_DIM,
    encode_spot_scope,
    fuse_position_scope,
)
from pokerbot.encoding.tensors import SPEC

# Card tensor from encode_cards(): (NUM_CARD_SLOTS=9, 4 suits, 13 ranks).
# Absent slots (preflop board, turn/river not yet dealt) are zero vectors.
POLICY_CARDS_SHAPE = SPEC.card_shape
ABSENT_CARD_ENCODING = "zero_vector"


def policy_position_from_encoded(enc: dict[str, Any]) -> np.ndarray:
    """Fuse 6-dim hero position with spot scope for the policy forward pass."""
    position = np.asarray(enc["position"], dtype=np.float32)
    if position.shape[0] == POLICY_POSITION_DIM:
        return position
    scope = enc.get("spot_scope")
    if scope is None:
        scope = encode_spot_scope("")
    return fuse_position_scope(position, np.asarray(scope, dtype=np.float32))


def policy_inputs_from_encoded(enc: dict[str, Any]) -> dict[str, np.ndarray]:
    """Numpy tensors ready for PseudoSiamesePolicy (before torch batching)."""
    if "cards" not in enc:
        raise KeyError(
            "encode_state output missing 'cards'. Policy encoding v2 requires "
            "board cards via encode_cards(); re-encode labels or upgrade encode_state."
        )
    return {
        "weva_id": int(enc["weva_id"]),
        "cards": np.asarray(enc["cards"], dtype=np.float32),
        "actions": np.asarray(enc["actions"], dtype=np.float32),
        "position": policy_position_from_encoded(enc),
    }
