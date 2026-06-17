#!/usr/bin/env python3
"""Phase A ground truth: current CFR+/FP policy on trusted tensor (no new fixes)."""

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

from aof_cfr.buckets import LABELS_169, LABEL_TO_INDEX
from aof_cfr.fp_br import (
    StabilityTracker,
    best_response_btn_3max,
    load_3max_payoffs,
    nash_gap_3max,
    trivial_baselines_3max,
)
from aof_cfr.infoset import encode_infoset_key
from aof_cfr.pushfold_payoffs import audit_terminal_ev
from aof_cfr.trio_payoff_tensor import load_trio_tensor
from pokerbot.solver.trio_preflop_dcfr import solve_3max_pushfold, trio_policy_rows
from pokerbot.solver.trio_preflop_fp import solve_3max_fp, trio_fp_policy_rows

# Reuse verdict chart helpers
from scripts.phase_a_1547_verdict import (
    TRIVIAL_RATIO,
    chart_sanity,
    range_thresholds,
    solve_fp_with_trajectory,
)

HAND_LADDER = ("AA", "QQ", "99", "77", "ATs", "A5o", "KTo", "KQo", "T9s", "72o", "32o")
SEATS = (
    ("BTN", "-", "btn"),
    ("SB_after_BTN_fold", "F", "sb", 0),
    ("BB_vs_BTN_jam_SB_fold", "AF", "bb", 1),
)
TRUSTED_SHA = "babe82516019cb9fada1f64f61d5d80bad00f98ffbc414dc0c53ebf2d4dca3ed"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _policy_table(
    btn: np.ndarray,
    sb: np.ndarray,
    bb: np.ndarray,
    *,
    label: str,
) -> dict[str, object]:
    rows: dict[str, dict[str, float]] = {}
    for hand in HAND_LADDER:
        hi = LABEL_TO_INDEX[hand]
        rows[hand] = {
            "BTN": float(btn[hi]),
            "SB_after_BTN_fold": float(sb[0, hi]),
            "BB_vs_BTN_jam_SB_fold": float(bb[1, hi]),
        }
    policy_rows: dict[str, np.ndarray] = {}
    n = btn.shape[0]
    for i, lab in enumerate(LABELS_169[:n]):
        policy_rows[encode_infoset_key("BTN", lab, "-")] = np.array(
            [1.0 - btn[i], btn[i]], dtype=np.float64
        )
        for hi, hist in enumerate(("F", "A")):
            policy_rows[encode_infoset_key("SB", lab, hist)] = np.array(
                [1.0 - sb[hi, i], sb[hi, i]], dtype=np.float64
            )
        for hi, hist in enumerate(("FA", "AF", "AA")):
            policy_rows[encode_infoset_key("BB", lab, hist)] = np.array(
                [1.0 - bb[hi, i], bb[hi, i]], dtype=np.float64
            )
    th = range_thresholds(policy_rows)
    return {
        "label": label,
        "hands": rows,
        "push_threshold_50pct": {
            "BTN": th.get("BTN_push"),
            "SB_after_BTN_fold": th.get("SB_push_after_btn_fold"),
            "BB_vs_BTN_jam_SB_fold": th.get("BB_call_vs_btn_jam_sb_fold"),
        },
        "overall_push_pct": {
            "BTN": float(btn.mean()),
            "SB_after_BTN_fold": float(sb[0].mean()),
            "BB_vs_BTN_jam_SB_fold": float(bb[1].mean()),
        },
    }


def _fp_stability(traj: list[dict[str, float]], *, gap_floor: float) -> dict[str, object]:
    if not traj:
        return {"plateau": False, "note": "empty trajectory"}
    gaps = [t["gap_mbb_g"] for t in traj]
    tail = gaps[-10:] if len(gaps) >= 10 else gaps
    spread = max(tail) - min(tail) if tail else 0.0
    below_floor = all(g <= gap_floor for g in tail)
    return {
        "final_gap_mbb_g": gaps[-1],
        "tail_spread_mbb_g": spread,
        "below_gap_floor": below_floor,
        "plateau": spread < 1.2 and below_floor,
        "rounds": len(traj),
    }


def _reconcile_9998(
    time_avg_rows: dict[str, np.ndarray],
    current_btn: np.ndarray,
) -> dict[str, object]:
    chip_path = Path("data/aof/cfr_base/phase_a_1547_verdict_chip_flow.json")
    stale_note = "cfr_base_3p.npz not in repo; chip_flow verdict JSON is prior measurement"
    chip_sample = None
    if chip_path.is_file():
        chip = json.loads(chip_path.read_text(encoding="utf-8"))
        chip_sample = chip.get("cfr_plus", {}).get("chart", {}).get("position_order_sample", [])

    def _p(rows, pos, hist, hand):
        return float(rows[encode_infoset_key(pos, hand, hist)][1])

    ladder_chip = {}
    if chip_sample:
        for row in chip_sample:
            ladder_chip[row["hand"]] = row

    ladder_now_avg = {h: _p(time_avg_rows, "BTN", "-", h) for h in ("AA", "77", "22", "72o")}
    return {
        "stale_99pct_source": {
            "artifact": str(chip_path) if chip_path.is_file() else "missing",
            "config": "phase_a_1547_verdict chip_flow: trusted tensor + tabular payoffs, 3000 CFR+ iters",
            "policy_metric": "time_average (btn_sum/iterations) via trio_policy_rows",
            "sample": ladder_chip,
        },
        "current_run_time_average_BTN": ladder_now_avg,
        "current_run_final_regret_matching_BTN_mean": float(current_btn.mean()),
        "explanation": (
            "Iteration-sweep ~37% used final regret-matching btn_s[:,AGGRESS], not time-average. "
            "Verdict '99.98%' used time-average export; if time-average still ~99% on premiums "
            "while current ~37%, CFR+ oscillates between push/fold late vs early iterations."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor", default="data/aof/cfr_base/trio_eq_tensor.npz")
    p.add_argument("--cfr-iterations", type=int, default=3000)
    p.add_argument("--fp-rounds", type=int, default=200)
    p.add_argument("--report", default="data/aof/cfr_base/phase_a_ground_truth_policy.json")
    args = p.parse_args()

    tensor_path = Path(args.tensor)
    sha = _sha256(tensor_path)
    if sha != TRUSTED_SHA:
        print(f"WARNING: tensor sha {sha} != trusted {TRUSTED_SHA}", file=sys.stderr)

    audit_terminal_ev()
    shares, pair_eq, deal_w = load_trio_tensor(tensor_path)
    st, deal_w = load_3max_payoffs(shares, pair_eq, deal_w, pair_eq_source="tensor")
    trivial = trivial_baselines_3max(st, deal_w)

    t0 = time.monotonic()
    cfr = solve_3max_pushfold(
        iterations=args.cfr_iterations,
        seed=42,
        shares=shares,
        pair_eq=pair_eq,
        deal_w=deal_w,
        pair_eq_source="tensor",
    )
    cfr_elapsed = time.monotonic() - t0
    cfr_rows = trio_policy_rows(cfr)
    cfr_mbb = cfr.exploitability_bb * 1000.0

    t1 = time.monotonic()
    fp, fp_traj = solve_fp_with_trajectory(
        shares, pair_eq, deal_w, max_rounds=args.fp_rounds, damping=1.0, pair_eq_source="tensor"
    )
    fp_elapsed = time.monotonic() - t1
    fp_rows = trio_fp_policy_rows(fp)
    fp_mbb = fp.exploitability_bb * 1000.0
    gap_floor = trivial["always_push"] * 0.25

    # BR sanity (correct slicing)
    sb_p = cfr.sb_push_freq
    bb_p = cfr.bb_push_freq
    br_btn = best_response_btn_3max(st, deal_w, sb_p, bb_p)
    br_note = {
        "btn_22_time_avg_policy_BR": float(br_btn[LABEL_TO_INDEX["22"]]),
        "btn_72o_time_avg_policy_BR": float(br_btn[LABEL_TO_INDEX["72o"]]),
    }

    out = {
        "tensor_path": str(tensor_path),
        "tensor_sha256": sha,
        "trusted_sha256": TRUSTED_SHA,
        "cfr_iterations": args.cfr_iterations,
        "trivial_baselines_mbb_g": trivial,
        "cfr_plus": {
            "nash_gap_mbb_g": cfr_mbb,
            "elapsed_s": cfr_elapsed,
            "time_average_policy": _policy_table(
                cfr.btn_push_freq, cfr.sb_push_freq, cfr.bb_push_freq, label="time_average"
            ),
            "final_regret_matching_policy": _policy_table(
                cfr.btn_push_current if cfr.btn_push_current is not None else cfr.btn_push_freq,
                cfr.sb_push_current if cfr.sb_push_current is not None else cfr.sb_push_freq,
                cfr.bb_push_current if cfr.bb_push_current is not None else cfr.bb_push_freq,
                label="final_regret_matching",
            ),
            "chart_sanity_time_average": chart_sanity(cfr_rows, label="cfr_time_average"),
            "br_vs_time_average": br_note,
        },
        "fp": {
            "nash_gap_mbb_g": fp_mbb,
            "elapsed_s": fp_elapsed,
            "time_average_policy": _policy_table(
                fp.btn_push_freq, fp.sb_push_freq, fp.bb_push_freq, label="fp_time_average"
            ),
            "chart_sanity": chart_sanity(fp_rows, label="fp"),
            "stability": _fp_stability(fp_traj, gap_floor=gap_floor),
            "gap_floor_mbb_g": gap_floor,
        },
        "reconcile_99pct": _reconcile_9998(
            cfr_rows,
            cfr.btn_push_current if cfr.btn_push_current is not None else cfr.btn_push_freq,
        ),
        "chart_plausible_3max_8bb": None,
        "overwrite_cfr_base_3p": False,
    }
    cfr_chart = out["cfr_plus"]["chart_sanity_time_average"]
    deg = cfr_chart.get("degenerate_push_all", True)
    order_ok = cfr_chart.get("position_order_ok", False)
    gap_ok = cfr_mbb < trivial["always_push"] * TRIVIAL_RATIO
    out["chart_plausible_3max_8bb"] = {
        "passed_spot_checks": cfr_chart.get("passed"),
        "position_order_ok": order_ok,
        "not_degenerate_premium_vs_77": not deg,
        "verdict": (
            "plausible_for_3max_8bb"
            if (order_ok and not deg and cfr_chart.get("passed"))
            else "mixed_not_shove_bot"
        ),
        "note": "3-max has no unique Nash; wide BTN and gap above trivial can be normal.",
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
