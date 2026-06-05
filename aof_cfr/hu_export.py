"""Export HU policy rows directly from bucket frequencies."""

from __future__ import annotations

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.infoset import encode_infoset_key


def hu_policy_rows_from_freqs(
    sb_push: np.ndarray,
    bb_call: np.ndarray,
    stack_bb: float = 8.0,
) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    for i, label in enumerate(LABELS_169):
        p_push = float(sb_push[i])
        rows[encode_infoset_key("SB", label, "-", stack_bb)] = np.array(
            [1.0 - p_push, p_push], dtype=np.float64
        )
        p_call = float(bb_call[i])
        rows[encode_infoset_key("BB", label, "A", stack_bb)] = np.array(
            [1.0 - p_call, p_call], dtype=np.float64
        )
    return rows
