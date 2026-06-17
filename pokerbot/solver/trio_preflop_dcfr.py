"""3-max AoF push/fold vectorized tabular CFR+ (1014 bucket infosets)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.fp_br import load_3max_payoffs, nash_gap_3max
from aof_cfr.infoset import encode_infoset_key
from aof_cfr.pushfold_payoffs import BB_HIST, SB_HIST, TERMINAL, audit_terminal_ev, mix, mix_into
from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, load_trio_tensor

FOLD = 0
AGGRESS = 1


def _regret_matching(regrets: np.ndarray) -> np.ndarray:
    pos = np.maximum(regrets, 0.0)
    s = pos.sum(axis=-1, keepdims=True)
    uniform = np.full_like(pos, 0.5)
    return np.where(s > 1e-12, pos / np.maximum(s, 1e-12), uniform)


@dataclass
class TrioSolveResult:
    stack_bb: float
    iterations: int
    exploitability_bb: float
    btn_push_freq: np.ndarray
    sb_push_freq: np.ndarray
    bb_push_freq: np.ndarray
    bucket_labels: list[str]
    manifest: dict = field(default_factory=dict)
    # Final-iteration regret-matching policy (differs from time-average btn_push_freq).
    btn_push_current: np.ndarray | None = None
    sb_push_current: np.ndarray | None = None
    bb_push_current: np.ndarray | None = None
    btn_reg: np.ndarray | None = None
    sb_reg: np.ndarray | None = None
    bb_reg: np.ndarray | None = None


def _policy_rows(
    btn_push: np.ndarray,
    sb_push: np.ndarray,
    bb_push: np.ndarray,
    labels: list[str],
) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    for bi, lab in enumerate(labels):
        p = float(btn_push[bi])
        rows[encode_infoset_key("BTN", lab, "-")] = np.array([1.0 - p, p], dtype=np.float64)
        for hi, hist in enumerate(SB_HIST):
            p = float(sb_push[hi, bi])
            rows[encode_infoset_key("SB", lab, hist)] = np.array([1.0 - p, p], dtype=np.float64)
        for hi, hist in enumerate(BB_HIST):
            p = float(bb_push[hi, bi])
            rows[encode_infoset_key("BB", lab, hist)] = np.array([1.0 - p, p], dtype=np.float64)
    return rows


def solve_3max_pushfold(
    *,
    iterations: int = 3000,
    seed: int = 42,
    shares: np.ndarray | None = None,
    pair_eq: np.ndarray | None = None,
    deal_w: np.ndarray | None = None,
    stack_bb: float = 8.0,
    pair_eq_source: str = "tensor",
    export_regrets: bool = False,
    compute_exploitability: bool = True,
) -> TrioSolveResult:
    _ = seed
    audit_terminal_ev()
    if shares is None or pair_eq is None or deal_w is None:
        path = Path(DEFAULT_TENSOR)
        if path.is_file():
            shares, pair_eq, deal_w_loaded = load_trio_tensor(path)
            if deal_w is None:
                deal_w = deal_w_loaded
        else:
            raise FileNotFoundError(
                f"Missing {path}; run scripts/build_3way_eq_tensor_npz.py first"
            )
    st, deal_w = load_3max_payoffs(shares, pair_eq, deal_w, pair_eq_source=pair_eq_source)
    n = int(shares.shape[0])
    labels = list(LABELS_169[:n])
    t = TERMINAL

    btn_reg = np.zeros((n, 2), dtype=np.float64)
    sb_reg = np.zeros((2, n, 2), dtype=np.float64)
    bb_reg = np.zeros((3, n, 2), dtype=np.float64)
    btn_sum = np.zeros((n, 2), dtype=np.float64)
    sb_sum = np.zeros((2, n, 2), dtype=np.float64)
    bb_sum = np.zeros((3, n, 2), dtype=np.float64)

    u_btn_push0 = np.empty((n, n, n), dtype=np.float64)
    u_sb_fa1 = np.empty((n, n, n), dtype=np.float64)
    u_sb_aa1 = np.empty((n, n, n), dtype=np.float64)
    u_btn_after_sb_f = np.empty((n, n, n), dtype=np.float64)
    u_btn_after_sb_a = np.empty((n, n, n), dtype=np.float64)
    u_bb_fa2 = np.empty((n, n, n), dtype=np.float64)
    u_bb_af2 = np.empty((n, n, n), dtype=np.float64)
    u_bb_aa2 = np.empty((n, n, n), dtype=np.float64)
    node = np.empty((n, n, n), dtype=np.float64)
    work = np.empty((n, n, n), dtype=np.float64)

    for _ in range(iterations):
        btn_s = _regret_matching(btn_reg)
        sb_s = _regret_matching(sb_reg)
        bb_s = _regret_matching(bb_reg)

        psf = sb_s[0, :, AGGRESS]
        psa = sb_s[1, :, AGGRESS]
        pfa = bb_s[0, :, AGGRESS]
        paf = bb_s[1, :, AGGRESS]
        paa = bb_s[2, :, AGGRESS]

        mix_into(u_sb_fa1, pfa[None, None, :], t.sb_win_uncontested_after_btn_fold, st.sb_bb_call1, work=work)
        mix_into(u_btn_after_sb_f, paf[None, None, :], t.btn_win_uncontested_after_sb_fold, st.btn_bb_call0, work=work)
        mix_into(u_btn_after_sb_a, paa[None, None, :], st.btn_sb_call0, st.three0, work=work)
        mix_into(u_btn_push0, psa[None, :, None], u_btn_after_sb_f, u_btn_after_sb_a, work=work)

        np.multiply(btn_s[:, AGGRESS, None, None], u_btn_push0, out=node)
        np.add(node, btn_s[:, FOLD, None, None] * t.btn_fold, out=node)
        np.subtract(t.btn_fold, node, out=work)
        work *= deal_w
        btn_reg[:, FOLD] += work.sum(axis=(1, 2))
        np.subtract(u_btn_push0, node, out=work)
        work *= deal_w
        btn_reg[:, AGGRESS] += work.sum(axis=(1, 2))

        mix_into(node, psf[None, :, None], t.sb_fold_after_btn_fold, u_sb_fa1, work=work)
        np.subtract(t.sb_fold_after_btn_fold, node, out=work)
        work *= deal_w
        work *= btn_s[:, FOLD, None, None]
        sb_reg[0, :, FOLD] += work.sum(axis=(0, 2))
        np.subtract(u_sb_fa1, node, out=work)
        work *= deal_w
        work *= btn_s[:, FOLD, None, None]
        sb_reg[0, :, AGGRESS] += work.sum(axis=(0, 2))

        mix_into(u_sb_aa1, paa[None, None, :], st.btn_sb_call1, st.three1, work=work)
        mix_into(node, psa[None, :, None], t.sb_fold_after_btn_jam, u_sb_aa1, work=work)
        np.subtract(t.sb_fold_after_btn_jam, node, out=work)
        work *= deal_w
        work *= btn_s[:, AGGRESS, None, None]
        sb_reg[1, :, FOLD] += work.sum(axis=(0, 2))
        np.subtract(u_sb_aa1, node, out=work)
        work *= deal_w
        work *= btn_s[:, AGGRESS, None, None]
        sb_reg[1, :, AGGRESS] += work.sum(axis=(0, 2))

        mix_into(u_bb_fa2, pfa[None, None, :], t.bb_fold, st.sb_bb_call2, work=work)
        np.subtract(t.bb_fold, u_bb_fa2, out=work)
        work *= deal_w
        work *= btn_s[:, FOLD, None, None]
        work *= psf[None, :, None]
        bb_reg[0, :, FOLD] += work.sum(axis=(0, 1))
        np.subtract(st.sb_bb_call2, u_bb_fa2, out=work)
        work *= deal_w
        work *= btn_s[:, FOLD, None, None]
        work *= psf[None, :, None]
        bb_reg[0, :, AGGRESS] += work.sum(axis=(0, 1))

        mix_into(u_bb_af2, paf[None, None, :], t.bb_fold, st.btn_bb_call2, work=work)
        np.subtract(t.bb_fold, u_bb_af2, out=work)
        work *= deal_w
        work *= btn_s[:, AGGRESS, None, None]
        work *= 1.0 - psa[None, :, None]
        bb_reg[1, :, FOLD] += work.sum(axis=(0, 1))
        np.subtract(st.btn_bb_call2, u_bb_af2, out=work)
        work *= deal_w
        work *= btn_s[:, AGGRESS, None, None]
        work *= 1.0 - psa[None, :, None]
        bb_reg[1, :, AGGRESS] += work.sum(axis=(0, 1))

        mix_into(u_bb_aa2, paa[None, None, :], t.bb_fold, st.three2, work=work)
        np.subtract(t.bb_fold, u_bb_aa2, out=work)
        work *= deal_w
        work *= btn_s[:, AGGRESS, None, None]
        work *= psa[None, :, None]
        bb_reg[2, :, FOLD] += work.sum(axis=(0, 1))
        np.subtract(st.three2, u_bb_aa2, out=work)
        work *= deal_w
        work *= btn_s[:, AGGRESS, None, None]
        work *= psa[None, :, None]
        bb_reg[2, :, AGGRESS] += work.sum(axis=(0, 1))

        btn_reg = np.maximum(btn_reg, 0.0)
        sb_reg = np.maximum(sb_reg, 0.0)
        bb_reg = np.maximum(bb_reg, 0.0)
        btn_sum += btn_s
        sb_sum += sb_s
        bb_sum += bb_s

    btn_push = (btn_sum / max(iterations, 1))[:, AGGRESS]
    sb_push = (sb_sum / max(iterations, 1))[:, :, AGGRESS]
    bb_push = (bb_sum / max(iterations, 1))[:, :, AGGRESS]
    btn_s_final = _regret_matching(btn_reg)
    sb_s_final = _regret_matching(sb_reg)
    bb_s_final = _regret_matching(bb_reg)
    btn_current = btn_s_final[:, AGGRESS].copy()
    sb_current = sb_s_final[:, :, AGGRESS].copy()
    bb_current = bb_s_final[:, :, AGGRESS].copy()
    if compute_exploitability:
        exploit, _ = nash_gap_3max(st, deal_w, btn_push, sb_push, bb_push)
    else:
        exploit = float("nan")

    return TrioSolveResult(
        stack_bb=stack_bb,
        iterations=iterations,
        exploitability_bb=exploit,
        btn_push_freq=btn_push,
        sb_push_freq=sb_push,
        bb_push_freq=bb_push,
        bucket_labels=labels,
        btn_push_current=btn_current,
        sb_push_current=sb_current,
        bb_push_current=bb_current,
        btn_reg=btn_reg.copy() if export_regrets else None,
        sb_reg=sb_reg.copy() if export_regrets else None,
        bb_reg=bb_reg.copy() if export_regrets else None,
        manifest={
            "stack_bb": stack_bb,
            "iterations": iterations,
            "exploitability_bb": exploit,
            "n_buckets": n,
        },
    )


def trio_policy_rows(result: TrioSolveResult) -> dict[str, np.ndarray]:
    return _policy_rows(
        result.btn_push_freq,
        result.sb_push_freq,
        result.bb_push_freq,
        result.bucket_labels,
    )
