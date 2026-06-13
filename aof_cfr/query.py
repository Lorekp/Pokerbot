"""Allocation-free runtime lookup for CFR base policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aof_cfr.buckets import LABEL_TO_INDEX, LABELS_169
from aof_cfr.infoset import encode_infoset_key, infoset_index_parts

DEFAULT_DIR = Path("data/aof/cfr_base")

_LOADED: dict[int, tuple[np.ndarray, np.ndarray, dict[str, int]]] = {}


def _verify_base_stamp(path: Path, data) -> None:
    """Reject a base NPZ whose stamped game model / fee / infoset-key-spec disagrees with the runtime
    (mirrors exploit_router._router_batch). A stamped mismatch is a hard error — a stale v1/no-fee
    base blended against v2 marginals silently poisons the router. UNstamped legacy files are
    accepted with a warning (can't verify) until the stamp-migration re-writes them."""
    keys = set(getattr(data, "files", []))
    if not ({"game_model", "flat_fee", "infoset_key_spec_hash"} & keys):
        print(f"aof_cfr.query: {path.name} is unstamped (legacy) — cannot verify game model/fee/key "
              "spec; run scripts/stamp_aof_artifacts.py to stamp it.")
        return
    from aof_cfr.infoset import infoset_key_spec_hash
    from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default
    gm = int(data["game_model"]) if "game_model" in keys else 1
    fee = float(data["flat_fee"]) if "flat_fee" in keys else 0.0
    spec = str(data["infoset_key_spec_hash"]) if "infoset_key_spec_hash" in keys else None
    want_gm, want_fee, want_spec = game_model_default(), flat_fee_default(), infoset_key_spec_hash()
    if gm != want_gm or abs(fee - want_fee) > 1e-9 or (spec is not None and spec != want_spec):
        raise ValueError(
            f"{path.name} stamp mismatch: game_model={gm} fee={fee} spec={spec} vs runtime "
            f"game_model={want_gm} fee={want_fee} spec={want_spec}. The base was solved under a "
            "different model than the live payoffs price — re-solve or re-stamp "
            "(scripts/stamp_aof_artifacts.py), or set AOF_GAME_MODEL/AOF_FLAT_FEE_BB to match.")


def load_cfr_base(base_dir: Path | str = DEFAULT_DIR) -> None:
    """Load all ``cfr_base_{n}p.npz`` artifacts into memory."""
    global _LOADED
    _LOADED.clear()
    base = Path(base_dir)
    missing = []
    for n in (2, 3, 4):
        path = base / f"cfr_base_{n}p.npz"
        if not path.is_file():
            missing.append(n)
            continue
        data = np.load(path, allow_pickle=True)
        _verify_base_stamp(path, data)
        keys = [str(k) for k in data["infoset_keys"]]
        probs = np.asarray(data["action_probs"], dtype=np.float32)
        index = {k: i for i, k in enumerate(keys)}
        _LOADED[n] = (probs, np.asarray(keys, dtype=object), index)
    if missing and _LOADED:                       # partial deploy: some bases loaded, some absent
        print(f"aof_cfr.query: cfr_base for players={missing} not found in {base} "
              f"(loaded {sorted(_LOADED)}); requests at those player counts will raise at "
              "decision time, not here.")


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
