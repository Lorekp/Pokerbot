#!/usr/bin/env python3
"""Phase A 1547 verdict: CFR+ vs FP on trusted trio_eq_tensor.npz."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.export import export_manifest, export_policy_npz
from aof_cfr.fp_br import (
    StabilityTracker,
    best_responses_3max,
    load_3max_payoffs,
    nash_gap_3max,
    trivial_baselines_3max,
)
from aof_cfr.infoset import encode_infoset_key
from aof_cfr.pushfold_payoffs import audit_terminal_ev
from aof_cfr.solve import SolveResult
from aof_cfr.trio_payoff_tensor import load_trio_tensor
from aof_cfr.trio_pair_eq_table import pair_eq_table_file_hash
from pokerbot.solver.trio_preflop_dcfr import solve_3max_pushfold, trio_policy_rows
from pokerbot.solver.trio_preflop_fp import trio_fp_policy_rows

FP_SOLVER = "fictitious_play_3max"
CFR_SOLVER = "cfr_plus_trio_dcfr"
TRIVIAL_RATIO = 0.25
STALE_CFR_MBBG = 1547.1574091347645

CHART_SPOTS = [
    ("BTN", "-", "AA", 0.85, "push"),
    ("BTN", "-", "AKs", 0.70, "push"),
    ("BTN", "-", "72o", 0.15, "fold"),
    ("SB", "F", "AA", 0.80, "push"),
    ("SB", "F", "72o", 0.20, "fold"),
    ("SB", "A", "AA", 0.85, "call"),
    ("SB", "A", "72o", 0.15, "fold"),
    ("BB", "FA", "AA", 0.85, "call"),
    ("BB", "FA", "72o", 0.15, "fold"),
    ("BB", "AF", "AA", 0.85, "call"),
    ("BB", "AF", "72o", 0.15, "fold"),
    ("BB", "AA", "AA", 0.90, "call"),
    ("BB", "AA", "72o", 0.10, "fold"),
]

ORDER_HANDS = ("AA", "KK", "QQ", "AKs", "AJo", "77", "22", "72o", "32o")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_manifest(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    out: dict[str, object] = {"path": str(path), "sha256": _file_sha256(path)}
    for key in (
        "build_mode",
        "boards_per_combo",
        "mc_boards",
        "pair_eq_table",
        "pair_eq_table_hash",
        "seed_scheme",
        "eval_backend",
        "row_parallel",
    ):
        if key in data:
            val = data[key]
            out[key] = val.item() if hasattr(val, "item") else val
    if "row_workers" in data:
        out["row_workers"] = int(data["row_workers"])
    return out


def _p_push(rows: dict[str, np.ndarray], pos: str, hist: str, hand: str) -> float:
    key = encode_infoset_key(pos, hand, hist)
    return float(rows[key][1])


def chart_sanity(rows: dict[str, np.ndarray], *, label: str) -> dict[str, object]:
    checks = []
    ok = True
    for pos, hist, hand, bound, kind in CHART_SPOTS:
        p = _p_push(rows, pos, hist, hand)
        passed = p >= bound if kind in ("push", "call") else p <= bound
        checks.append(
            {"spot": f"{pos}|{hist}|{hand}", "kind": kind, "p_push": p, "bound": bound, "passed": passed}
        )
        ok = ok and passed

    order = []
    order_ok = True
    btn_aa = _p_push(rows, "BTN", "-", "AA")
    btn_77 = _p_push(rows, "BTN", "-", "77")
    btn_72 = _p_push(rows, "BTN", "-", "72o")
    degenerate = (btn_aa - btn_77) < 0.05
    for hand in ORDER_HANDS:
        btn = _p_push(rows, "BTN", "-", hand)
        sb = _p_push(rows, "SB", "F", hand)
        bb = _p_push(rows, "BB", "FA", hand)
        order.append({"hand": hand, "BTN": btn, "SB_F": sb, "BB_FA": bb})
        if hand in ("AA", "AKs", "77", "72o"):
            order_ok = order_ok and btn >= sb >= bb - 0.05

    return {
        "label": label,
        "checks": checks,
        "position_order_sample": order,
        "position_order_ok": order_ok,
        "degenerate_push_all": degenerate,
        "passed": ok and order_ok and not degenerate,
    }


def _push_threshold(rows: dict[str, np.ndarray], pos: str, hist: str) -> str | None:
    for hand in LABELS_169:
        if _p_push(rows, pos, hist, hand) < 0.5:
            return hand
    return None


def range_thresholds(rows: dict[str, np.ndarray]) -> dict[str, str | None]:
    return {
        "BTN_push": _push_threshold(rows, "BTN", "-"),
        "SB_push_after_btn_fold": _push_threshold(rows, "SB", "F"),
        "SB_call_vs_btn_jam": _push_threshold(rows, "SB", "A"),
        "BB_call_vs_sb_fold_btn_fold": _push_threshold(rows, "BB", "FA"),
        "BB_call_vs_btn_jam_sb_fold": _push_threshold(rows, "BB", "AF"),
        "BB_call_vs_btn_sb_jam": _push_threshold(rows, "BB", "AA"),
    }


def verify_payoff_identity(
    st,
    deal_w: np.ndarray,
    btn: np.ndarray,
    sb: np.ndarray,
    bb: np.ndarray,
    solver_gap: float,
) -> dict[str, object]:
    gap, imps = nash_gap_3max(st, deal_w, btn, sb, bb)
    return {
        "solver_gap_bb": float(solver_gap),
        "metric_gap_bb": float(gap),
        "abs_diff_bb": abs(float(solver_gap) - float(gap)),
        "identical": abs(float(solver_gap) - float(gap)) < 1e-12,
        "player_improvements": imps,
    }


def solve_fp_with_trajectory(
    shares: np.ndarray,
    pair_eq: np.ndarray,
    deal_w: np.ndarray,
    *,
    max_rounds: int,
    damping: float,
    pair_eq_source: str = "tensor",
):
    from pokerbot.solver.trio_preflop_fp import TrioFPSolveResult

    st, deal_w = load_3max_payoffs(shares, pair_eq, deal_w, pair_eq_source=pair_eq_source)
    n = int(shares.shape[0])
    baselines_pre = trivial_baselines_3max(st, deal_w)
    gap_floor = baselines_pre["always_push"] * TRIVIAL_RATIO
    tracker = StabilityTracker(gap_floor_mbb_g=gap_floor)
    btn_avg = np.full(n, 0.5, dtype=np.float64)
    sb_avg = np.full((2, n), 0.5, dtype=np.float64)
    bb_avg = np.full((3, n), 0.5, dtype=np.float64)
    trajectory: list[dict[str, float]] = []
    converged = False
    gap = 0.0
    imps: list[float] = [0.0, 0.0, 0.0]
    rnd = 0
    i72 = LABELS_169.index("72o")

    for rnd in range(1, max_rounds + 1):
        br_btn, br_sb, br_bb = best_responses_3max(st, deal_w, btn_avg, sb_avg, bb_avg)
        alpha = damping / rnd
        btn_avg = (1.0 - alpha) * btn_avg + alpha * br_btn
        sb_avg = (1.0 - alpha) * sb_avg + alpha * br_sb
        bb_avg = (1.0 - alpha) * bb_avg + alpha * br_bb
        gap, imps = nash_gap_3max(st, deal_w, btn_avg, sb_avg, bb_avg)
        trajectory.append(
            {
                "round": rnd,
                "gap_mbb_g": gap * 1000.0,
                "btn_imp_mbb_g": imps[0] * 1000.0,
                "sb_imp_mbb_g": imps[1] * 1000.0,
                "bb_imp_mbb_g": imps[2] * 1000.0,
                "btn_aa": float(btn_avg[0]),
                "btn_72o": float(btn_avg[i72]),
            }
        )
        if tracker.observe(gap * 1000.0, btn_avg):
            converged = True
            break

    result = TrioFPSolveResult(
        stack_bb=8.0,
        rounds=rnd,
        exploitability_bb=gap,
        btn_push_freq=btn_avg,
        sb_push_freq=sb_avg,
        bb_push_freq=bb_avg,
        bucket_labels=list(LABELS_169[:n]),
        player_improvements=imps,
        trivial_baselines_mbb_g=trivial_baselines_3max(st, deal_w),
        converged=converged,
    )
    return result, trajectory


def _fp_stability(trajectory: list[dict[str, float]]) -> dict[str, object]:
    if len(trajectory) < 10:
        return {"plateau": False, "reason": "too_few_rounds"}
    gaps = [t["gap_mbb_g"] for t in trajectory]
    last10 = gaps[-10:]
    spread = max(last10) - min(last10)
    return {
        "final_gap_mbb_g": gaps[-1],
        "last10_min": min(last10),
        "last10_max": max(last10),
        "last10_spread": spread,
        "plateau": spread < 5.0,
        "oscillating": spread > 50.0,
    }


def _verdict_label(cfr_mbb: float, fp_mbb: float) -> tuple[str, str]:
    low = 500.0
    if cfr_mbb < low and fp_mbb < low:
        return (
            "tensor_was_binding",
            "CFR+ dropped from ~1547 mbb/g to chart-aligned low hundreds on the trusted tensor; "
            "the old gap was payoff-layer error, not solver class.",
        )
    if cfr_mbb >= low and fp_mbb < low:
        return (
            "solver_class_mattered",
            "CFR+ stayed high while FP converged low on the same tensor; the FP pivot was load-bearing.",
        )
    return ("mixed", f"CFR+={cfr_mbb:.1f} fp={fp_mbb:.1f} mbb/g — inspect trajectories and charts.")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase A 1547 CFR+ vs FP verdict")
    p.add_argument("--tensor", default="data/aof/cfr_base/trio_eq_tensor.npz")
    p.add_argument("--out-dir", default="data/aof/cfr_base")
    p.add_argument(
        "--report",
        default="",
        help="Output JSON path (default: <out-dir>/phase_a_1547_verdict.json)",
    )
    p.add_argument("--cfr-iterations", type=int, default=3000)
    p.add_argument("--fp-rounds", type=int, default=200)
    p.add_argument("--fp-damping", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write-fp", action="store_true", help="Overwrite cfr_base_3p if FP passes gates")
    p.add_argument(
        "--pair-eq-source",
        choices=("tensor", "hu_expand", "exact_native"),
        default="tensor",
        help="pair_eq volume for build_static_payoffs (isolated experiments)",
    )
    p.add_argument(
        "--fix-phase",
        default="chip_flow",
        help="Report tag: chip_flow | pair_eq | baseline",
    )
    p.add_argument("--baseline-report", default="", help="Prior verdict JSON for pair_eq delta")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    tensor_path = Path(args.tensor)
    report_path = Path(args.report) if args.report else out_dir / "phase_a_1547_verdict.json"

    audit_terminal_ev()
    shares, pair_eq, deal_w = load_trio_tensor(tensor_path)
    st, deal_w = load_3max_payoffs(
        shares, pair_eq, deal_w, pair_eq_source=args.pair_eq_source
    )
    trivial = trivial_baselines_3max(st, deal_w)
    tensor_meta = _tensor_manifest(tensor_path)

    t0 = time.monotonic()
    cfr_result = solve_3max_pushfold(
        iterations=args.cfr_iterations,
        seed=args.seed,
        shares=shares,
        pair_eq=pair_eq,
        deal_w=deal_w,
        pair_eq_source=args.pair_eq_source,
    )
    cfr_elapsed = time.monotonic() - t0
    cfr_rows = trio_policy_rows(cfr_result)
    cfr_mbb = cfr_result.exploitability_bb * 1000.0

    t1 = time.monotonic()
    fp_result, fp_traj = solve_fp_with_trajectory(
        shares,
        pair_eq,
        deal_w,
        max_rounds=args.fp_rounds,
        damping=args.fp_damping,
        pair_eq_source=args.pair_eq_source,
    )
    fp_elapsed = time.monotonic() - t1
    fp_rows = trio_fp_policy_rows(fp_result)
    fp_mbb = fp_result.exploitability_bb * 1000.0

    cfr_id = verify_payoff_identity(
        st,
        deal_w,
        cfr_result.btn_push_freq,
        cfr_result.sb_push_freq,
        cfr_result.bb_push_freq,
        cfr_result.exploitability_bb,
    )
    fp_id = verify_payoff_identity(
        st,
        deal_w,
        fp_result.btn_push_freq,
        fp_result.sb_push_freq,
        fp_result.bb_push_freq,
        fp_result.exploitability_bb,
    )

    cfr_chart = chart_sanity(cfr_rows, label="cfr_plus")
    fp_chart = chart_sanity(fp_rows, label="fp")
    fp_stab = _fp_stability(fp_traj)
    fp_gap_ok = fp_mbb < trivial["always_push"] * TRIVIAL_RATIO
    fp_sane = fp_chart["passed"] and fp_gap_ok and fp_stab.get("plateau", False) and fp_id["identical"]
    verdict, verdict_note = _verdict_label(cfr_mbb, fp_mbb)

    baseline = {}
    if args.baseline_report:
        baseline = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))

    report = {
        "fix_phase": args.fix_phase,
        "pair_eq_source": args.pair_eq_source,
        "tensor": tensor_meta,
        "pair_eq_table_hash": pair_eq_table_file_hash(),
        "seed": args.seed,
        "cfr_iterations": args.cfr_iterations,
        "fp_rounds": args.fp_rounds,
        "fp_damping": args.fp_damping,
        "trivial_baselines_mbb_g": trivial,
        "cfr_plus": {
            "solver": CFR_SOLVER,
            "nash_gap_mbb_g": cfr_mbb,
            "elapsed_s": cfr_elapsed,
            "payoff_identity": cfr_id,
            "chart": cfr_chart,
            "thresholds": range_thresholds(cfr_rows),
        },
        "fp": {
            "solver": FP_SOLVER,
            "nash_gap_mbb_g": fp_mbb,
            "rounds": fp_result.rounds,
            "converged": fp_result.converged,
            "elapsed_s": fp_elapsed,
            "payoff_identity": fp_id,
            "chart": fp_chart,
            "thresholds": range_thresholds(fp_rows),
            "stability": fp_stab,
            "gap_trajectory": fp_traj,
        },
        "verdict": verdict,
        "verdict_note": verdict_note,
        "stale_manifest_cfr_mbb_g": STALE_CFR_MBBG,
        "fp_production_ready": fp_sane,
        "overwrite_cfr_base_3p": False,
        "gap_floor_mbb_g": trivial["always_push"] * TRIVIAL_RATIO,
    }
    if baseline:
        report["delta_vs_baseline"] = {
            "cfr_plus_mbb_g": cfr_mbb - float(baseline["cfr_plus"]["nash_gap_mbb_g"]),
            "fp_mbb_g": fp_mbb - float(baseline["fp"]["nash_gap_mbb_g"]),
            "baseline_fix_phase": baseline.get("fix_phase"),
        }

    wrote_3p = False
    if args.write_fp and fp_sane:
        fp_solve = SolveResult(
            num_players=3,
            solver_name=FP_SOLVER,
            iterations=fp_result.rounds,
            seed=args.seed,
            exploitability_bb_per_hand=fp_result.exploitability_bb,
            exploitability_mbb_g=fp_mbb,
            br_method="exact",
            player_improvements=fp_result.player_improvements,
            policy_rows=fp_rows,
            nash_disclaimer=(
                "N=3: Nash gap is exact tabular BR over bucket deal; multiplayer has no unique Nash."
            ),
        )
        export_policy_npz(out_dir / "cfr_base_3p.npz", fp_solve)
        export_manifest(
            out_dir / "cfr_base.manifest.json",
            [fp_solve],
            extra={
                "trio_tensor": tensor_meta,
                "boards_per_combo": int(tensor_meta.get("boards_per_combo", 192)),
                "phase_a_1547_verdict": verdict,
            },
            merge_existing=True,
        )
        wrote_3p = True
        report["overwrite_cfr_base_3p"] = True

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Phase A 1547 verdict ===")
    print(f"Tensor: {tensor_path}")
    print(f"Trivial baselines mbb/g: push={trivial['always_push']:.1f} fold={trivial['always_fold']:.1f}")
    print(f"Stale CFR+ (bad tensor): {STALE_CFR_MBBG:.1f} mbb/g")
    print(f"CFR+ Nash gap: {cfr_mbb:.3f} mbb/g ({cfr_elapsed:.1f}s)")
    print(f"FP Nash gap:   {fp_mbb:.3f} mbb/g rounds={fp_result.rounds} ({fp_elapsed:.1f}s)")
    print(f"Payoff identity — CFR+: {cfr_id['identical']}  FP: {fp_id['identical']}")
    print(f"Chart sanity — CFR+: {cfr_chart['passed']}  FP: {fp_chart['passed']}")
    print(f"FP stability plateau: {fp_stab.get('plateau')} (last10 spread {fp_stab.get('last10_spread')})")
    print(f"VERDICT: {verdict}")
    print(verdict_note)
    print(f"Report: {report_path}")
    if wrote_3p:
        print(f"Overwrote {out_dir / 'cfr_base_3p.npz'}")
    else:
        print("cfr_base_3p.npz NOT overwritten")


if __name__ == "__main__":
    main()
