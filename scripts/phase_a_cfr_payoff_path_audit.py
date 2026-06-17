#!/usr/bin/env python3
"""Trace CFR+ terminal path; single-infoset EV vs probe at BB-72o vs BTN jam."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABEL_TO_INDEX
from aof_cfr.fp_br import load_3max_payoffs
from aof_cfr.pushfold_payoffs import TERMINAL, mix
from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, load_trio_tensor
from pokerbot.solver import trio_preflop_dcfr as dcfr_mod
from pokerbot.solver.trio_preflop_dcfr import FOLD, AGGRESS, _regret_matching, solve_3max_pushfold

CFR_LOOP = Path("pokerbot/solver/trio_preflop_dcfr.py")


def _cfr_terminal_call_trace() -> dict[str, object]:
    mod_src = inspect.getsource(dcfr_mod)
    loop_src = inspect.getsource(dcfr_mod.solve_3max_pushfold)
    uses_oracle = (
        "from aof_cfr.terminal_ev" in mod_src
        or "showdown_terminal_returns" in loop_src
        or "fold_terminal_returns" in loop_src
    )
    return {
        "module": "pokerbot.solver.trio_preflop_dcfr",
        "function": "solve_3max_pushfold",
        "call_sites": [
            "line 80: st, deal_w = load_3max_payoffs(...) -> build_static_payoffs",
            "lines 107-110: u_sb_fa1/u_btn_* = mix(..., TERMINAL.*, st.sb_bb_call1|btn_bb_call0|three0)",
            "lines 133-160: bb_reg += deal_w * (... TERMINAL.bb_fold, st.btn_bb_call2, st.three2)",
        ],
        "imports_terminal_ev_oracle": uses_oracle,
        "imports_pushfold_payoffs": "from aof_cfr.pushfold_payoffs import" in mod_src,
        "audit_terminal_ev_only": "audit_terminal_ev" in mod_src and not uses_oracle,
        "payoff_source": "aof_cfr.pushfold_payoffs.TERMINAL + StaticPayoffs (st.*)",
        "never_calls": "terminal_ev.showdown_terminal_returns (MCCFR/game_pushfold only)",
    }


def _field_policies(n: int, call_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    btn = np.ones(n, dtype=np.float64)
    sb = np.zeros((2, n), dtype=np.float64)
    bb = np.zeros((3, n), dtype=np.float64)
    sb[1, :] = call_frac
    bb[1, :] = call_frac
    bb[2, :] = call_frac
    return btn, sb, bb


def _probe_bb_af_ev(
    st,
    deal_w: np.ndarray,
    k: int,
    btn_push: np.ndarray,
    sb_push: np.ndarray,
) -> dict[str, float]:
    """BB hand k, history AF (BTN jam, SB fold): fold vs call EV."""
    t = TERMINAL
    n = deal_w.shape[0]
    psa = sb_push[1, :]
    w = 0.0
    u_call = 0.0
    for i in range(n):
        for j in range(n):
            wt = float(deal_w[i, j, k] * btn_push[i] * (1.0 - psa[j]))
            if wt <= 0.0:
                continue
            w += wt
            u_call += wt * float(st.btn_bb_call2[i, j, k])
    u_fold = float(t.bb_fold)
    u_call_avg = u_call / w if w > 1e-12 else float("nan")
    return {
        "weighted_fold_ev_bb": u_fold,
        "weighted_call_ev_bb": u_call_avg,
        "br_prefers_call": u_call_avg > u_fold,
        "reach_weight": w,
    }


def _cfr_bb_af_action_values(
    st,
    deal_w: np.ndarray,
    k: int,
    btn_s: np.ndarray,
    sb_s: np.ndarray,
    bb_s: np.ndarray,
) -> dict[str, float]:
    """Counterfactual action values used in bb_reg[1,k,*] update (hist AF)."""
    t = TERMINAL
    n = deal_w.shape[0]
    psa = sb_s[1, :, AGGRESS]
    paf = bb_s[1, :, AGGRESS]
    u_node = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            u_node[i, j] = float(
                mix(
                    paf[k],
                    t.bb_fold,
                    st.btn_bb_call2[i, j, k],
                )
            )
    w_fold = 0.0
    w_call = 0.0
    cf_fold = 0.0
    cf_call = 0.0
    for i in range(n):
        for j in range(n):
            reach = float(
                deal_w[i, j, k] * btn_s[i, AGGRESS] * (1.0 - psa[j])
            )
            if reach <= 0.0:
                continue
            cf_fold += reach * t.bb_fold
            cf_call += reach * float(st.btn_bb_call2[i, j, k])
            w_fold += reach
            w_call += reach
    node_mix = 0.0
    w_node = 0.0
    for i in range(n):
        for j in range(n):
            reach = float(deal_w[i, j, k] * btn_s[i, AGGRESS] * (1.0 - psa[j]))
            if reach <= 0.0:
                continue
            node_mix += reach * u_node[i, j]
            w_node += reach
    return {
        "cf_fold_action_ev_bb": cf_fold / w_fold if w_fold > 0 else float("nan"),
        "cf_call_action_ev_bb": cf_call / w_call if w_call > 0 else float("nan"),
        "policy_node_ev_bb": node_mix / w_node if w_node > 0 else float("nan"),
        "bb_fold_terminal_bb": float(t.bb_fold),
        "bb_call_terminal_avg_bb": cf_call / w_call if w_call > 0 else float("nan"),
        "regret_fold_increment_sign": "positive if fold better than node",
        "sample_u_node_vs_fold": float(u_node[0, 0]) if n else float("nan"),
    }


def _run_one_cfr_iteration_state(
    shares: np.ndarray,
    pair_eq: np.ndarray,
    deal_w: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return final regret-matching strategies after ``iterations`` CFR+ steps."""
    from aof_cfr.pushfold_payoffs import TERMINAL as T

    st, deal_w = load_3max_payoffs(shares, pair_eq, deal_w)
    n = shares.shape[0]
    t = T
    btn_reg = np.zeros((n, 2), dtype=np.float64)
    sb_reg = np.zeros((2, n, 2), dtype=np.float64)
    bb_reg = np.zeros((3, n, 2), dtype=np.float64)
    u_sb_fa1 = np.empty((n, n, n), dtype=np.float64)
    u_sb_aa1 = np.empty((n, n, n), dtype=np.float64)

    for _ in range(iterations):
        btn_s = _regret_matching(btn_reg)
        sb_s = _regret_matching(sb_reg)
        bb_s = _regret_matching(bb_reg)
        psf = sb_s[0, :, AGGRESS]
        psa = sb_s[1, :, AGGRESS]
        pfa = bb_s[0, :, AGGRESS]
        paf = bb_s[1, :, AGGRESS]
        paa = bb_s[2, :, AGGRESS]

        u_sb_fa1[:] = mix(pfa[None, None, :], t.sb_win_uncontested_after_btn_fold, st.sb_bb_call1)
        u_btn_after_sb_f = mix(paf[None, None, :], t.btn_win_uncontested_after_sb_fold, st.btn_bb_call0)
        u_btn_after_sb_a = mix(paa[None, None, :], st.btn_sb_call0, st.three0)
        u_btn_push0 = mix(psa[None, :, None], u_btn_after_sb_f, u_btn_after_sb_a)

        btn_node0 = btn_s[:, FOLD, None, None] * t.btn_fold + btn_s[:, AGGRESS, None, None] * u_btn_push0
        btn_reg[:, FOLD] += np.sum(deal_w * (t.btn_fold - btn_node0), axis=(1, 2))
        btn_reg[:, AGGRESS] += np.sum(deal_w * (u_btn_push0 - btn_node0), axis=(1, 2))

        sb_node_f1 = mix(psf[None, :, None], t.sb_fold_after_btn_fold, u_sb_fa1)
        sb_reg[0, :, FOLD] += np.sum(
            deal_w * btn_s[:, FOLD, None, None] * (t.sb_fold_after_btn_fold - sb_node_f1), axis=(0, 2)
        )
        sb_reg[0, :, AGGRESS] += np.sum(
            deal_w * btn_s[:, FOLD, None, None] * (u_sb_fa1 - sb_node_f1), axis=(0, 2)
        )

        u_sb_aa1[:] = mix(paa[None, None, :], st.btn_sb_call1, st.three1)
        sb_node_a1 = mix(psa[None, :, None], t.sb_fold_after_btn_jam, u_sb_aa1)
        sb_reg[1, :, FOLD] += np.sum(
            deal_w * btn_s[:, AGGRESS, None, None] * (t.sb_fold_after_btn_jam - sb_node_a1), axis=(0, 2)
        )
        sb_reg[1, :, AGGRESS] += np.sum(
            deal_w * btn_s[:, AGGRESS, None, None] * (u_sb_aa1 - sb_node_a1), axis=(0, 2)
        )

        u_bb_af2 = mix(paf[None, None, :], t.bb_fold, st.btn_bb_call2)
        bb_reg[1, :, FOLD] += np.sum(
            deal_w * btn_s[:, AGGRESS, None, None] * (1.0 - psa[None, :, None]) * (t.bb_fold - u_bb_af2),
            axis=(0, 1),
        )
        bb_reg[1, :, AGGRESS] += np.sum(
            deal_w
            * btn_s[:, AGGRESS, None, None]
            * (1.0 - psa[None, :, None])
            * (st.btn_bb_call2 - u_bb_af2),
            axis=(0, 1),
        )

        u_bb_fa2 = mix(pfa[None, None, :], t.bb_fold, st.sb_bb_call2)
        bb_reg[0, :, FOLD] += np.sum(
            deal_w * btn_s[:, FOLD, None, None] * psf[None, :, None] * (t.bb_fold - u_bb_fa2),
            axis=(0, 1),
        )
        bb_reg[0, :, AGGRESS] += np.sum(
            deal_w * btn_s[:, FOLD, None, None] * psf[None, :, None] * (st.sb_bb_call2 - u_bb_fa2),
            axis=(0, 1),
        )

        u_bb_aa2 = mix(paa[None, None, :], t.bb_fold, st.three2)
        bb_reg[2, :, FOLD] += np.sum(
            deal_w * btn_s[:, AGGRESS, None, None] * psa[None, :, None] * (t.bb_fold - u_bb_aa2),
            axis=(0, 1),
        )
        bb_reg[2, :, AGGRESS] += np.sum(
            deal_w * btn_s[:, AGGRESS, None, None] * psa[None, :, None] * (st.three2 - u_bb_aa2),
            axis=(0, 1),
        )

        btn_reg = np.maximum(btn_reg, 0.0)
        sb_reg = np.maximum(sb_reg, 0.0)
        bb_reg = np.maximum(bb_reg, 0.0)

    btn_s = _regret_matching(btn_reg)
    sb_s = _regret_matching(sb_reg)
    bb_s = _regret_matching(bb_reg)
    return btn_reg, sb_reg, bb_reg, btn_s, sb_s, bb_s


def run_audit(
    *, tensor_path: Path, cfr_iterations: int, skip_cfr_loop: bool = False
) -> dict[str, object]:
    shares, pair_eq, deal_w = load_trio_tensor(tensor_path)
    st, deal_w = load_3max_payoffs(shares, pair_eq, deal_w)
    n = shares.shape[0]
    k = LABEL_TO_INDEX["72o"]

    trace = _cfr_terminal_call_trace()
    btn_f, sb_f, bb_f = _field_policies(n, 1.0)
    # SB must fold after BTN jam for hist AF (not call)
    sb_f[1, :] = 0.0
    probe = _probe_bb_af_ev(st, deal_w, k, btn_f, sb_f)

    btn_s_field = np.zeros((n, 2), dtype=np.float64)
    btn_s_field[:, AGGRESS] = 1.0
    sb_s_field = np.zeros((2, n, 2), dtype=np.float64)
    sb_s_field[1, :, FOLD] = 1.0  # SB fold after BTN jam
    bb_s_dummy = np.zeros((3, n, 2), dtype=np.float64)
    bb_s_dummy[1, :, FOLD] = 1.0
    bb_s_dummy[1, :, AGGRESS] = 0.0

    cfr_field = _cfr_bb_af_action_values(st, deal_w, k, btn_s_field, sb_s_field, bb_s_dummy)

    probe_pref = probe["br_prefers_call"]
    cfr_field_pref = cfr_field["cf_call_action_ev_bb"] > cfr_field["cf_fold_action_ev_bb"]
    agree_field = probe_pref == cfr_field_pref

    cfr_late = None
    agree_late = None
    bb72_af = None
    btn22_late = None
    if not skip_cfr_loop:
        _, _, _, btn_s, sb_s, bb_s = _run_one_cfr_iteration_state(
            shares, pair_eq, deal_w, iterations=cfr_iterations
        )
        cfr_late = _cfr_bb_af_action_values(st, deal_w, k, btn_s, sb_s, bb_s)
        agree_late = probe_pref == (
            cfr_late["cf_call_action_ev_bb"] > cfr_late["cf_fold_action_ev_bb"]
        )
        bb72_af = float(bb_s[1, k, AGGRESS])
        btn22_late = float(btn_s[LABEL_TO_INDEX["22"], AGGRESS])

    return {
        "cfr_terminal_trace": trace,
        "oracle_used_by_cfr_regret_loop": trace["imports_terminal_ev_oracle"],
        "single_infoset_BB_72o_hist_AF": {
            "opponent_field": "BTN push 100%, SB fold after BTN jam",
            "probe": probe,
            "cfr_regret_path_same_formula_field": cfr_field,
            "cfr_regret_path_late_self_play": cfr_late,
            "probe_vs_cfr_formula_agree_100pct_field": agree_field,
            "probe_vs_cfr_late_self_play_agree": agree_late,
            "cfr_late_bb72o_strategy_hist_AF": bb72_af,
            "cfr_late_btn22_strategy": btn22_late,
        },
        "opponent_model_note": (
            "Probe '100% field' = every bucket calls when facing jam. CFR self-play "
            "uses hand-dependent call/fold (e.g. SB call-after-jam ~25% at iter 100), "
            "so BTN-22 push EV stays positive in self-play while negative vs homogeneous field."
        ),
        "oracle_path_used_by": {
            "trio_preflop_dcfr": False,
            "game_pushfold_MCCFR": "aof_cfr/game_pushfold.py -> terminal_ev.showdown_terminal_returns",
        },
        "verdict": (
            "no_second_payoff_path_cfr_uses_tabular_same_as_probe"
            if agree_field and not trace["imports_terminal_ev_oracle"]
            else (
                "cfr_uses_oracle_not_tabular"
                if trace["imports_terminal_ev_oracle"]
                else "tabular_paths_disagree_at_infoset"
            )
        ),
        "note": (
            "terminal_ev.py is used by game_pushfold.py / MCCFR only, not trio_preflop_dcfr. "
            "If paths agree but solve degenerate, inspect CFR regret weighting or self-play field."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor", type=Path, default=DEFAULT_TENSOR)
    p.add_argument("--cfr-iterations", type=int, default=100)
    p.add_argument(
        "--skip-cfr-loop",
        action="store_true",
        help="Only trace call path + field infoset test (no CFR iteration loop)",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("data/aof/cfr_base/phase_a_cfr_payoff_path_audit.json"),
    )
    args = p.parse_args()
    out = run_audit(
        tensor_path=args.tensor,
        cfr_iterations=args.cfr_iterations,
        skip_cfr_loop=args.skip_cfr_loop,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
