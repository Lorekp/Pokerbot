"""Information-set keys for experimental CFR skeleton."""

from __future__ import annotations

from aof_solver.game import HuState


def infoset_key(player: str, bucket: str, state: HuState) -> str:
    """Private bucket + public action history."""
    hist = "".join(state.actions) if state.actions else "-"
    return f"{player}|{bucket}|{hist}"
