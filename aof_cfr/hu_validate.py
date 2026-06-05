"""HU oracle validation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aof_cfr.buckets import LABELS_169, LABEL_TO_INDEX
from aof_cfr.hu_payoff_matrix import load_equity_matrix
from aof_cfr.infoset import decode_infoset_key

DEFAULT_EQ = Path("data/aof/cfr_base/hu_eq_matrix.npz")
DEFAULT_JSON_ORACLE = Path("data/aof/hu_pushfold_8bb.json")


def load_policy_freqs(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    sb = np.zeros(len(LABELS_169), dtype=np.float64)
    bb = np.zeros(len(LABELS_169), dtype=np.float64)
    for key, probs in zip(data["infoset_keys"], data["action_probs"]):
        pos, bucket, hist, _depth = decode_infoset_key(str(key))
        idx = LABEL_TO_INDEX[bucket]
        if pos == "SB" and hist == "-":
            sb[idx] = float(probs[1])
        elif pos == "BB" and hist == "A":
            bb[idx] = float(probs[1])
    return sb, bb


def dcfr_oracle_freqs(
    eq_path: Path = DEFAULT_EQ,
    *,
    iterations: int = 2500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    from pokerbot.solver.hu_preflop_dcfr import solve_hu_pushfold

    eq = load_equity_matrix(eq_path)
    result = solve_hu_pushfold(iterations=iterations, eq_matrix=eq, seed=seed)
    return result.sb_push_freq, result.bb_call_freq


def compare_freqs(
    sol_sb: np.ndarray,
    sol_bb: np.ndarray,
    oracle_sb: np.ndarray,
    oracle_bb: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, object]:
    sb_diff = float(np.max(np.abs(sol_sb - oracle_sb)))
    bb_diff = float(np.max(np.abs(sol_bb - oracle_bb)))
    max_diff = max(sb_diff, bb_diff)
    return {
        "max_sb_diff": sb_diff,
        "max_bb_diff": bb_diff,
        "max_diff": max_diff,
        "tolerance": tolerance,
        "passed": max_diff <= tolerance,
    }
