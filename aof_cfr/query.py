"""Allocation-free runtime lookup for CFR base policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aof_cfr.buckets import LABEL_TO_INDEX, LABELS_169
from aof_cfr.infoset import encode_infoset_key, infoset_index_parts

DEFAULT_DIR = Path("data/aof/cfr_base")

_LOADED: dict[int, tuple[np.ndarray, np.ndarray, dict[str, int]]] = {}


def load_cfr_base(base_dir: Path | str = DEFAULT_DIR) -> None:
    """Load all ``cfr_base_{n}p.npz`` artifacts into memory."""
    global _LOADED
    _LOADED.clear()
    base = Path(base_dir)
    for n in (2, 3, 4):
        path = base / f"cfr_base_{n}p.npz"
        if not path.is_file():
            continue
        data = np.load(path, allow_pickle=True)
        keys = [str(k) for k in data["infoset_keys"]]
        probs = np.asarray(data["action_probs"], dtype=np.float32)
        index = {k: i for i, k in enumerate(keys)}
        _LOADED[n] = (probs, np.asarray(keys, dtype=object), index)


def _lookup_row(num_players: int, key: str) -> np.ndarray | None:
    if num_players not in _LOADED:
        return None
    probs, _keys, index = _LOADED[num_players]
    idx = index.get(key)
    if idx is None:
        return None
    return probs[idx]


def base_strategy(
    num_players: int,
    position: str,
    history: str,
    hand: str,
    stack_bb: float = 8.0,
) -> float:
    """
    Return P(ALL_IN) for the CFR base policy at the given effective stack depth.

    ``stack_bb`` is bucketed to the nearest canonical depth in the infoset key.
    Raises KeyError if artifact missing or infoset not in table.
    """
    pos, bucket, hist = infoset_index_parts(num_players, position, history, hand)
    key = encode_infoset_key(pos, bucket, hist, stack_bb)
    row = _lookup_row(num_players, key)
    if row is None:
        raise KeyError(
            f"CFR base infoset not found: n={num_players} key={key!r} "
            f"(load_cfr_base() called={bool(_LOADED)})"
        )
    return float(row[1])
