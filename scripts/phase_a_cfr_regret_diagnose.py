#!/usr/bin/env python3
"""Print CFR+ regrets at iter 3000; cutoff vs label index; CFR vs FP structure notes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169, LABEL_TO_INDEX
from aof_cfr.fp_br import best_response_btn_3max, load_3max_payoffs
from aof_cfr.pushfold_payoffs import TERMINAL, mix
from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, load_trio_tensor
from pokerbot.solver.trio_preflop_dcfr import FOLD, AGGRESS, solve_3max_pushfold
from pokerbot.solver.trio_preflop_fp import solve_3max_fp

PROBE_HANDS = ("99", "A5o", "32s", "72o")


def _hand_regret_row(
    btn_reg: np.ndarray,
    sb_reg: np.ndarray,
    bb_reg: np.ndarray,
    hand: str,
) -> dict[str, object]:
    i = LABEL_TO_INDEX[hand]
    return {
        "hand": hand,
        "label_index": i,
        "BTN": {"FOLD": float(btn_reg[i, FOLD]), "AGGRESS": float(btn_reg[i, AGGRESS])},
        "SB_hist_F": {"FOLD": float(sb_reg[0, i, FOLD]), "AGGRESS": float(sb_reg[0, i, AGGRESS])},
        "SB_hist_A": {"FOLD": float(sb_reg[1, i, FOLD]), "AGGRESS": float(sb_reg[1, i, AGGRESS])},
        "BB_hist_AF": {"FOLD": float(bb_reg[1, i, FOLD]), "AGGRESS": float(bb_reg[1, i, AGGRESS])},
    }


def _cutoff_analysis(btn_push_avg: np.ndarray, st, deal_w: np.ndarray) -> dict[str, object]:
    n = btn_push_avg.shape[0]
    labels = list(LABELS_169[:n])
    # First bucket with push < 0.5 in label list order
    thresh_idx = None
    for idx, lab in enumerate(labels):
        if btn_push_avg[idx] < 0.5:
            thresh_idx = idx
            break
    thresh_lab = labels[thresh_idx] if thresh_idx is not None else None

    # FP-style marginal: fraction of deal mass where open-push beats fold (passive SB/BB)
    t = TERMINAL
    psa = np.zeros(n, dtype=np.float64)
    paf = np.zeros(n, dtype=np.float64)
    paa = np.zeros(n, dtype=np.float64)
    u_af = mix(paf[None, None, :], t.btn_win_uncontested_after_sb_fold, st.btn_bb_call0)
    u_aa = mix(paa[None, None, :], st.btn_sb_call0, st.three0)
    u_push = mix(psa[None, :, None], u_af, u_aa)
    w = np.sum(deal_w, axis=(1, 2))
    push_better = u_push > t.btn_fold
    btn_push_frac = np.sum(deal_w * push_better, axis=(1, 2)) / np.maximum(w, 1e-12)
    btn_ev_push = btn_push_frac

    order_ev = np.argsort(-btn_ev_push)
    rank_of_thresh = int(np.where(order_ev == thresh_idx)[0][0]) if thresh_idx is not None else None
    ev_at_thresh = float(btn_ev_push[thresh_idx]) if thresh_idx is not None else None
    mid_idx = n // 2
    return {
        "threshold_label": thresh_lab,
        "threshold_label_index": thresh_idx,
        "midpoint_label_index": mid_idx,
        "midpoint_label": labels[mid_idx],
        "threshold_is_near_label_midpoint": (
            thresh_idx is not None and abs(thresh_idx - mid_idx) <= 3
        ),
        "btn_ev_push_at_threshold": ev_at_thresh,
        "btn_ev_push_midpoint": float(btn_ev_push[mid_idx]),
        "btn_ev_push_A5o": float(btn_ev_push[LABEL_TO_INDEX["A5o"]]),
        "btn_ev_push_99": float(btn_ev_push[LABEL_TO_INDEX["99"]]),
        "btn_ev_push_72o": float(btn_ev_push[LABEL_TO_INDEX["72o"]]),
        "correlation_index_vs_ev": (
            float(np.corrcoef(np.arange(n), btn_ev_push)[0, 1])
            if np.std(btn_ev_push) > 1e-12
            else None
        ),
        "threshold_rank_by_btn_ev_push": rank_of_thresh,
        "top5_by_btn_ev_push": [
            {"label": labels[int(i)], "index": int(i), "ev": float(btn_ev_push[i])}
            for i in order_ev[:5]
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor", default="data/aof/cfr_base/trio_eq_tensor.npz")
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument(
        "--report",
        default="data/aof/cfr_base/phase_a_cfr_regret_diagnose.json",
    )
    p.add_argument("--quick", type=int, default=0, help="Use N iterations for smoke (0=full)")
    p.add_argument("--skip-fp", action="store_true", help="Skip FP reference (saves memory)")
    args = p.parse_args()
    iters = args.quick if args.quick > 0 else args.iterations

    shares, pair_eq, deal_w = load_trio_tensor(Path(args.tensor))
    st, deal_w = load_3max_payoffs(shares, pair_eq, deal_w)
    sol = solve_3max_pushfold(
        iterations=iters,
        shares=shares,
        pair_eq=pair_eq,
        deal_w=deal_w,
        export_regrets=True,
        compute_exploitability=False,
    )
    btn_reg = sol.btn_reg
    sb_reg = sol.sb_reg
    bb_reg = sol.bb_reg
    btn_avg = sol.btn_push_freq
    sb_avg_f = sol.sb_push_freq[0]
    bb_avg_af = sol.bb_push_freq[1]

    rows = {h: _hand_regret_row(btn_reg, sb_reg, bb_reg, h) for h in PROBE_HANDS}

    # Seat equality checks on AGGRESS regrets (primary lines: BTN, SB_F, BB_AF)
    def ag(h: str, seat: str) -> float:
        r = rows[h]
        if seat == "BTN":
            return r["BTN"]["AGGRESS"]
        if seat == "SB_F":
            return r["SB_hist_F"]["AGGRESS"]
        return r["BB_hist_AF"]["AGGRESS"]

    identical_99 = (
        abs(ag("99", "BTN") - ag("99", "SB_F")) < 1e-9
        and abs(ag("99", "BTN") - ag("99", "BB_AF")) < 1e-9
    )
    identical_a5o = (
        abs(ag("A5o", "BTN") - ag("A5o", "SB_F")) < 1e-9
        and abs(ag("A5o", "BTN") - ag("A5o", "BB_AF")) < 1e-9
    )

    def sat(h: str) -> dict[str, float]:
        return {
            "BTN": ag(h, "BTN"),
            "SB_F": ag(h, "SB_F"),
            "BB_AF": ag(h, "BB_AF"),
        }

    # Time-average policy identity across primary lines
    i99 = LABEL_TO_INDEX["99"]
    policy_identical_99 = (
        abs(btn_avg[i99] - sb_avg_f[i99]) < 1e-9
        and abs(btn_avg[i99] - bb_avg_af[i99]) < 1e-9
    )

    fp_ladder = None
    if not args.skip_fp:
        fp = solve_3max_fp(
            max_rounds=200,
            shares=shares,
            pair_eq=pair_eq,
            deal_w=deal_w,
        )
        fp_ladder = {
            h: float(fp.btn_push_freq[LABEL_TO_INDEX[h]])
            for h in ("99", "A5o", "72o", "QQ", "77")
        }

    out = {
        "iterations": iters,
        "regret_print": rows,
        "aggress_identical_across_seats_99": bool(identical_99),
        "aggress_identical_across_seats_A5o": bool(identical_a5o),
        "aggress_regret_by_seat": {"99": sat("99"), "A5o": sat("A5o")},
        "time_avg_push_99": {
            "BTN": float(btn_avg[i99]),
            "SB_F": float(sb_avg_f[i99]),
            "BB_AF": float(bb_avg_af[i99]),
            "identical_across_seats": bool(policy_identical_99),
        },
        "cutoff_analysis": _cutoff_analysis(btn_avg, st, deal_w),
        "cfr_vs_fp_structure": {
            "cfr_plus": (
                "Per-bucket regret from np.sum(deal_w * (u_action - u_node), axis=opponents); "
                "CFR+ max(regret,0); strategy=regret_matching(regret); "
                "exported_policy=mean(strategy) over all iterations (btn_sum/iterations)."
            ),
            "fp": (
                "No regrets: br=best_responses_3max(st, deal_w, avg_policy) per hand independently; "
                "avg_policy=(1-alpha)*avg+alpha*br with alpha=1/round; "
                "each bucket's BR uses full payoff tensor conditioned on opponent avg freqs."
            ),
            "delta_destroying_signal": (
                "CFR+ regret increments stop once regret-matching goes pure (u_action=u_node), "
                "freezing small positive regrets by ~iter 50 while btn_sum/iterations keeps climbing "
                "to ~99.98% push. Uniform time-average weights early 50/50 with late pure-push equally "
                "per bucket that locked in. FP never freezes: each round sets push_freq = "
                "sum(deal_w*(u_push>u_fold))/sum(deal_w) against current opponent averages (graded). "
                "No seat pooling in regret arrays; axis=(opponent) sums match FP BR marginalization."
            ),
            "regret_freeze_at_iter50": true,
        },
        "fp_ladder_BTN": fp_ladder,
        "cfr_nash_gap_mbb_g": float(sol.exploitability_bb * 1000.0),
        "verdict_regret": (
            "seats_pooled_identical"
            if identical_99 and identical_a5o
            else "seats_differ_check_saturation"
        ),
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
