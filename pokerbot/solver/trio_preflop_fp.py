"""3-max AoF push/fold fictitious play (averaged best-response)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.fp_br import (
    StabilityTracker,
    best_responses_3max_legal,
    load_3max_payoffs,
    nash_gap_3max_legal,
    trivial_baselines_3max_legal,
)
from aof_cfr.infoset import encode_infoset_key
from aof_cfr.pushfold_payoffs import BB_HIST, SB_HIST, audit_terminal_ev
from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, load_trio_tensor


@dataclass
class TrioFPSolveResult:
    stack_bb: float
    rounds: int
    exploitability_bb: float
    btn_push_freq: np.ndarray
    sb_push_freq: np.ndarray
    bb_push_freq: np.ndarray
    bucket_labels: list[str]
    player_improvements: list[float]
    trivial_baselines_mbb_g: dict[str, float]
    converged: bool
    manifest: dict = field(default_factory=dict)


def _policy_rows(
    btn_push: np.ndarray,
    sb_push: np.ndarray,
    bb_push: np.ndarray,
    labels: list[str],
    stack_bb: float,
) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    for bi, lab in enumerate(labels):
        p = float(btn_push[bi])
        rows[encode_infoset_key("BTN", lab, "-", stack_bb)] = np.array([1.0 - p, p], dtype=np.float64)
        for hi, hist in enumerate(SB_HIST):
            p = float(sb_push[hi, bi])
            rows[encode_infoset_key("SB", lab, hist, stack_bb)] = np.array([1.0 - p, p], dtype=np.float64)
        for hi, hist in enumerate(BB_HIST):
            p = float(bb_push[hi, bi])
            rows[encode_infoset_key("BB", lab, hist, stack_bb)] = np.array([1.0 - p, p], dtype=np.float64)
    return rows


def solve_3max_fp(
    *,
    max_rounds: int = 1000,  # legal pure-BR target needs more rounds than the old
    # clairvoyant fractional target to settle the time-average; 1000 reaches legal gap
    # ~1 mbb/g with policy delta <1e-3 (see scripts/phase_a_fp_adjudicate.py).
    damping: float = 1.0,
    seed: int = 42,
    shares: np.ndarray | None = None,
    pair_eq: np.ndarray | None = None,
    deal_w: np.ndarray | None = None,
    stack_bb: float = 8.0,
    stability: StabilityTracker | None = None,
    pair_eq_source: str = "tensor",
) -> TrioFPSolveResult:
    _ = seed
    audit_terminal_ev()
    if shares is None or pair_eq is None:
        path = Path(DEFAULT_TENSOR)
        if path.is_file():
            shares, pair_eq, deal_w_loaded = load_trio_tensor(path)
            if deal_w is None:
                deal_w = deal_w_loaded
        else:
            raise FileNotFoundError(
                f"Missing {path}; run scripts/build_3way_eq_tensor_npz.py first"
            )
    st, deal_w = load_3max_payoffs(
        shares, pair_eq, deal_w, stack_bb=stack_bb, pair_eq_source=pair_eq_source
    )
    n = int(shares.shape[0])
    labels = list(LABELS_169[:n])

    btn_avg = np.full(n, 0.5, dtype=np.float64)
    sb_avg = np.full((2, n), 0.5, dtype=np.float64)
    bb_avg = np.full((3, n), 0.5, dtype=np.float64)
    baselines = trivial_baselines_3max_legal(st, deal_w)
    # No absolute gap floor: n>=3 general-sum has no low Nash gap to reach, so an
    # always_push*0.25 floor has no multiplayer justification. Convergence is genuine
    # gap+policy plateau (StabilityTracker); shippability is decided by the BR-consistency
    # gate in scripts/phase_a_fp_adjudicate.py, not by an absolute mbb/g number.
    tracker = stability or StabilityTracker(gap_floor_mbb_g=None)
    converged = False
    gap = 0.0
    imps: list[float] = [0.0, 0.0, 0.0]

    for rnd in range(1, max_rounds + 1):
        br_btn, br_sb, br_bb = best_responses_3max_legal(st, deal_w, btn_avg, sb_avg, bb_avg)
        alpha = damping / rnd
        btn_avg = (1.0 - alpha) * btn_avg + alpha * br_btn
        sb_avg = (1.0 - alpha) * sb_avg + alpha * br_sb
        bb_avg = (1.0 - alpha) * bb_avg + alpha * br_bb
        gap, imps = nash_gap_3max_legal(st, deal_w, btn_avg, sb_avg, bb_avg)
        if tracker.observe(gap * 1000.0, btn_avg):
            converged = True
            break

    return TrioFPSolveResult(
        stack_bb=stack_bb,
        rounds=rnd,
        exploitability_bb=gap,
        btn_push_freq=btn_avg,
        sb_push_freq=sb_avg,
        bb_push_freq=bb_avg,
        bucket_labels=labels,
        player_improvements=imps,
        trivial_baselines_mbb_g=baselines,
        converged=converged,
        manifest={
            "stack_bb": stack_bb,
            "rounds": rnd,
            "exploitability_bb": gap,
            "converged": converged,
            "trivial_baselines_mbb_g": baselines,
            "n_buckets": n,
        },
    )


def trio_fp_policy_rows(result: TrioFPSolveResult) -> dict[str, np.ndarray]:
    return _policy_rows(
        result.btn_push_freq,
        result.sb_push_freq,
        result.bb_push_freq,
        result.bucket_labels,
        result.stack_bb,
    )
