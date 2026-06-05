"""Locked infoset key spec for CFR base (Phases B/D/E must reuse verbatim)."""

from __future__ import annotations

import hashlib
import json

from aof_cfr.buckets import LABELS_169, bucket_index

INFOSET_KEY_VERSION = 2

# Canonical effective-stack-depth buckets (bb) baked into the infoset key. The nearest
# bucket is used. This grid is part of the key spec (so Phases B/D/E and the persistent
# opponent DB share it); keep in sync with aof_ranges.STACK_DEPTHS.
STACK_DEPTH_BUCKETS: tuple[int, ...] = (4, 6, 8, 10, 12, 15, 20)


def stack_depth_bucket(stack_bb: float) -> int:
    """Nearest canonical depth bucket (bb) for an effective stack."""
    return min(STACK_DEPTH_BUCKETS, key=lambda d: abs(d - float(stack_bb)))


def encode_infoset_key(
    position: str, hand_bucket: str, action_history: str, stack_bb: float = 8.0
) -> str:
    """Format (v2): ``{position}|{hand_bucket}|{action_history}|d{stack_depth_bucket}``.

    ``stack_bb`` defaults to 8.0 so single-depth (8bb) callers — the OpenSpiel/MCCFR and
    diagnostic paths — keep producing correct ``d8`` keys; multi-depth producers (FP/CFR+
    grid, 4-max, HU export) pass the real effective stack.
    """
    hist = action_history if action_history else "-"
    return f"{position}|{hand_bucket}|{hist}|d{stack_depth_bucket(stack_bb)}"


def aggregate_infoset_key(position: str, action_history: str, stack_bb: float = 8.0) -> str:
    """Hand-agnostic key ``{position}|*|{history}|d{depth}`` for hidden-card opponents.

    Live we rarely see an opponent's folded cards, so we can only observe their push/fold
    frequency marginalized over their hand at a ``(position, history, depth)``. Recording under
    this aggregate key keeps ``n_obs`` equal to the true decision count (no 169x inflation);
    ``profiler.estimate_policy_rows_shift`` reconstructs a per-bucket policy from it.
    """
    hist = action_history if action_history else "-"
    return f"{position}|*|{hist}|d{stack_depth_bucket(stack_bb)}"


def decode_infoset_key(key: str) -> tuple[str, str, str, int]:
    """Return (position, hand_bucket, history, stack_depth_bb)."""
    parts = key.split("|")
    if len(parts) != 4:
        raise ValueError(f"Invalid infoset key (expected v2 4-part): {key!r}")
    pos, bucket, hist, depth = parts
    if not depth.startswith("d"):
        raise ValueError(f"Invalid depth field in infoset key: {key!r}")
    return pos, bucket, ("-" if hist == "-" else hist), int(depth[1:])


def history_after_action(history: str, action_token: str) -> str:
    """Append F or A to public history."""
    if history in ("", "-"):
        return action_token
    return history + action_token


def infoset_key_spec_hash() -> str:
    payload = json.dumps(
        {
            "version": INFOSET_KEY_VERSION,
            "format": "{position}|{hand_bucket}|{action_history}|d{stack_depth_bucket}",
            "history_tokens": ["F", "A"],
            "root_history": "-",
            "n_buckets": len(LABELS_169),
            "bucket_order": "hu_preflop_dcfr._build_169_buckets",
            "stack_depth_buckets": list(STACK_DEPTH_BUCKETS),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def infoset_index_parts(
    num_players: int,
    position: str,
    history: str,
    hand: str,
) -> tuple[str, str, str]:
    """Normalize inputs for lookup."""
    hist = history if history else "-"
    if hand in LABELS_169:
        bucket = hand
    else:
        from aof_cfr.buckets import combo_to_bucket

        if len(hand) == 4:
            bucket = combo_to_bucket(hand[0:2], hand[2:4])
        elif len(hand) >= 2 and hand[1] in "cdhs":
            bucket = combo_to_bucket(hand[0:2], hand[2:4])
        else:
            bucket = hand
    _ = bucket_index(bucket)
    return position, bucket, hist
