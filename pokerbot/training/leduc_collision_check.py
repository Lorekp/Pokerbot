"""Leduc encode collision check — distinct infosets must not share network inputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from pokerbot.engine.leduc_bridge import (
    collect_reachable_infosets,
    py_info_key,
    py_state_from_deal,
)
from pokerbot.engine.leduc_jax import NUM_PLAYERS
from pokerbot.training.leduc_tabular_cfr import all_leduc_deals
from pokerbot.training.sd_cfr import encode_leduc


def _encode_fingerprint(state, seat: int) -> tuple:
    vec = np.asarray(encode_leduc(state.jax, seat), dtype=np.float32)
    return vec.tobytes()


def check_leduc_encode_collisions(
    deals: list[tuple[tuple[int, int], int]] | None = None,
) -> dict[str, Any]:
    """Fail if two distinct Leduc infosets map to the same ``encode_leduc`` vector."""
    if deals is None:
        deals = list(all_leduc_deals())

    all_infosets: set[tuple] = set()
    for cards, pub in deals:
        root = py_state_from_deal(cards, pub)
        all_infosets |= collect_reachable_infosets([root])

    encode_to_infosets: dict[tuple, list[tuple]] = defaultdict(list)
    for ik in all_infosets:
        seat = ik[0]
        # Reconstruct a representative state from info key is hard; walk one deal tree instead.
        _ = seat

    # Walk every deal and record (infoset_key -> encode) at decision nodes.
    key_to_encode: dict[tuple, tuple] = {}
    collisions: list[dict[str, Any]] = []

    for cards, pub in deals:
        from pokerbot.engine.leduc_bridge import apply_action_py, py_fingerprint, py_legal

        stack = [py_state_from_deal(cards, pub)]
        seen: set[tuple] = set()
        while stack:
            state = stack.pop()
            fp = py_fingerprint(state)
            if fp in seen:
                continue
            seen.add(fp)
            if state.done:
                continue
            seat = state.acting
            ik = py_info_key(state, seat)
            enc = _encode_fingerprint(state, seat)
            if ik in key_to_encode and key_to_encode[ik] != enc:
                collisions.append(
                    {
                        "infoset_key": ik,
                        "encode_a": key_to_encode[ik],
                        "encode_b": enc,
                    }
                )
            else:
                key_to_encode[ik] = enc
            mask = py_legal(state)
            for a in range(3):
                if mask[a]:
                    stack.append(apply_action_py(state, a))

    encode_groups: dict[tuple, list[tuple]] = defaultdict(list)
    for ik, enc in key_to_encode.items():
        encode_groups[enc].append(ik)

    collision_groups = {enc: keys for enc, keys in encode_groups.items() if len(keys) > 1}

    return {
        "passed": len(collision_groups) == 0 and len(collisions) == 0,
        "n_infosets": len(key_to_encode),
        "n_unique_encodes": len(encode_groups),
        "collision_groups": len(collision_groups),
        "inconsistent_keys": len(collisions),
        "sample_collision": next(iter(collision_groups.values()), None),
    }


def require_no_leduc_encode_collisions(deals: list | None = None) -> dict[str, Any]:
    result = check_leduc_encode_collisions(deals)
    if not result["passed"]:
        raise AssertionError(
            f"Leduc encode collision: {result['n_infosets']} infosets, "
            f"{result['collision_groups']} collision groups, "
            f"{result['inconsistent_keys']} inconsistent keys"
        )
    return result
