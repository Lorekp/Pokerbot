"""Generate Gambit EFG for HU AoF push/fold (169-bucket deal, OpenSpiel CFR+)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyspiel

from aof_cfr.buckets import LABELS_169
from aof_cfr.infoset import encode_infoset_key
from aof_cfr.terminal_ev import (
    hu_fold_sb_utility,
    hu_push_fold_sb_utility,
    hu_showdown_sb_utility,
)

_STACK_BB = 8.0
_SB_POSTED = 0.5
_BB_POSTED = 1.0
_N = 169
_DEAL_N = _N * _N
_DEAL_P = f"1/{_DEAL_N}"


def _sb_infoset_name(i: int) -> str:
    return encode_infoset_key("SB", LABELS_169[i], "-")


def _bb_infoset_name(j: int) -> str:
    return encode_infoset_key("BB", LABELS_169[j], "A")


def build_hu_efg_string(eq: np.ndarray) -> str:
    """Return Gambit EFG text for uniform 169x169 bucket deal."""
    if eq.shape != (_N, _N):
        raise ValueError(f"expected eq shape ({_N}, {_N}), got {eq.shape}")

    lines: list[str] = [
        'EFG 2 R "AoF HU push/fold 8bb" { "SB" "BB" }',
        "",
        f'c "ROOT" 0 "" {{ {" ".join(f"\"{k}\" {_DEAL_P}" for k in range(_DEAL_N))} }} 0',
    ]

    for i in range(_N):
        for j in range(_N):
            sb_fold_u = hu_fold_sb_utility(sb_posted=_SB_POSTED)
            sb_push_fold_u = hu_push_fold_sb_utility(sb_posted=_SB_POSTED, bb_posted=_BB_POSTED)
            sb_show_u = hu_showdown_sb_utility(
                i, j, eq, stack_bb=_STACK_BB, sb_posted=_SB_POSTED, bb_posted=_BB_POSTED
            )
            bb_show_u = -sb_show_u

            lines.append(
                f' p "" 1 {i + 1} "{_sb_infoset_name(i)}" {{ "F" "A" }} 0'
            )
            lines.append(f'  t "" 2 "" {{ {sb_fold_u:.9f} {-sb_fold_u:.9f} }}')
            lines.append(
                f'  p "" 2 {j + 1} "{_bb_infoset_name(j)}" {{ "F" "A" }} 0'
            )
            lines.append(
                f'   t "" 2 "" {{ {sb_push_fold_u:.9f} {-sb_push_fold_u:.9f} }}'
            )
            lines.append(f'   t "" 2 "" {{ {sb_show_u:.9f} {bb_show_u:.9f} }}')

    return "\n".join(lines) + "\n"


def write_hu_efg(path: Path, eq: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_hu_efg_string(eq), encoding="utf-8")
    return path


def load_hu_efg_game(eq: np.ndarray) -> pyspiel.Game:
    return pyspiel.load_efg_game(build_hu_efg_string(eq))


def infoset_key_from_efg_state(state: pyspiel.State, player: int) -> str:
    raw = state.information_state_string(player)
    return raw.split("-", 3)[3]
