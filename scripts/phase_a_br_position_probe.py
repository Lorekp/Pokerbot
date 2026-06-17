#!/usr/bin/env python3
"""Direct best-response position probe (no solve) on trusted 3-max payoffs."""

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
from aof_cfr.fp_br import (
    best_response_bb_3max,
    best_response_btn_3max,
    best_response_sb_3max,
    load_3max_payoffs,
)
from aof_cfr.pushfold_payoffs import BB_POST, SB_POST, STACK_BB, TERMINAL, build_static_payoffs
from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, load_trio_tensor
from aof_ranges_chart import CALL_RANGE_STR

PROBE_HANDS_TIGHT = ("A5o", "22", "KTo")
PROBE_HANDS_FIELD = ("72o", "22", "A5o", "KTo")
SEATS = ("BTN", "SB", "BB")
# ~13% chart call vs BTN open at 8bb (target ~top 15%)
TIGHT_CALL_RANGE = CALL_RANGE_STR[8][("BB", "BTN")]


def _call_mask_from_range(range_str: str, labels: list[str]) -> np.ndarray:
    from pokerkit import parse_range

    from aof_cfr.buckets import combo_to_bucket

    allowed: set[str] = set()
    for combo in parse_range(range_str):
        cards = sorted(combo, key=str)
        s0, s1 = str(cards[0]), str(cards[1])
        # pokerkit repr: "NINE OF CLUBS (9c)" -> "9c"
        c0 = s0[s0.index("(") + 1 : s0.index(")")]
        c1 = s1[s1.index("(") + 1 : s1.index(")")]
        allowed.add(combo_to_bucket(c0, c1))
    out = np.zeros(len(labels), dtype=np.float64)
    for i, lab in enumerate(labels):
        if lab in allowed:
            out[i] = 1.0
    return out


def fixed_call_policies(
    n: int,
    call_mask: np.ndarray,
    *,
    open_jam: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Opponents call per mask when facing jam; never open-jam unless open_jam set."""
    btn = np.full(n, open_jam, dtype=np.float64)
    sb = np.zeros((2, n), dtype=np.float64)
    bb = np.zeros((3, n), dtype=np.float64)
    sb[0, :] = open_jam
    sb[1, :] = call_mask
    bb[0, :] = call_mask
    bb[1, :] = call_mask
    bb[2, :] = call_mask
    return btn, sb, bb


def fixed_tight_call_policies(n: int, call_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return fixed_call_policies(n, call_mask)


def uniform_call_policies(n: int, call_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.full(n, float(call_frac), dtype=np.float64)
    return fixed_call_policies(n, mask)


def hero_br_push_freq(
    seat: str,
    hand_idx: int,
    st,
    deal_w: np.ndarray,
    btn: np.ndarray,
    sb: np.ndarray,
    bb: np.ndarray,
) -> float:
    if seat == "BTN":
        br = best_response_btn_3max(st, deal_w, sb, bb)
        return float(br[hand_idx])
    if seat == "SB":
        br = best_response_sb_3max(st, deal_w, btn, bb)
        return float(br[0, hand_idx])
    if seat == "BB":
        br = best_response_bb_3max(st, deal_w, btn, sb)
        # BTN jam + SB fold: primary BB defense spot (hist AF)
        return float(br[1, hand_idx])
    raise ValueError(seat)


def audit_topology(st) -> dict[str, object]:
    """Report which positional factors are encoded in static payoffs."""
    t = TERMINAL
    fold_by_seat = {
        "BTN": t.btn_fold,
        "SB_after_btn_fold": t.sb_fold_after_btn_fold,
        "SB_after_btn_jam": t.sb_fold_after_btn_jam,
        "BB": t.bb_fold,
    }
    uncontested = {
        "BTN_after_SB_fold": t.btn_win_uncontested_after_sb_fold,
        "SB_after_BTN_fold": t.sb_win_uncontested_after_btn_fold,
    }
    # Showdown uses uniform -STACK_BB for all seats in build_static_payoffs
    showdown_commit = {
        "btn_paths": "all use -STACK_BB (8bb)",
        "sb_paths": "all use -STACK_BB (8bb)",
        "bb_paths": "all use -STACK_BB (8bb)",
    }
    live_opponents = {
        "BTN_push": "mix SB response (psa) then BB (paf/paa); 2 live behind",
        "SB_open_after_BTN_fold": "BB only (pfa on sb_bb_call1); 1 live behind",
        "BB_defense_AF": "single caller path btn_bb_call2; BTN jam SB fold",
        "BB_defense_AA": "three-way three2; BTN jam SB call",
    }
    action_order = {
        "histories": "SB_HIST F/A, BB_HIST FA/AF/AA encode BTN→SB→BB sequence",
        "btn_first": True,
    }
    suspects: list[str] = []
    if t.btn_fold == t.sb_fold_after_btn_fold == t.bb_fold:
        suspects.append("fold terminals identical across seats (they are not)")
    if abs(t.btn_fold) < 1e-12 and abs(t.sb_fold_after_btn_fold + SB_POST) < 1e-12:
        suspects.append("BTN fold=0, SB fold=-SB_POST: per-seat fold EV present")
    if abs(t.bb_fold + BB_POST) < 1e-12:
        suspects.append("BB fold=-BB_POST: per-seat fold EV present")
    # Symmetric pot constants across HU channels
    from aof_cfr.pushfold_payoffs import POT_HU_STACKS, POT_THREE_STACKS

    if POT_HU_STACKS == 2.0 * STACK_BB + SB_POST + BB_POST:
        suspects.append("HU pot includes dead blinds (chip-flow fix applied)")
    return {
        "fold_terminals_bb": fold_by_seat,
        "uncontested_win_bb": uncontested,
        "showdown_commit_model": showdown_commit,
        "live_opponents_behind": live_opponents,
        "action_order": action_order,
        "pot_constants": {"POT_HU_STACKS": POT_HU_STACKS, "POT_THREE_STACKS": POT_THREE_STACKS},
        "suspects": suspects,
        "pair_eq_channels": {
            "0": "btn_bb (HU)",
            "1": "btn_sb (HU)",
            "2": "sb_bb (HU)",
            "shares": "three0/1/2 seat-indexed 3-way",
        },
    }


def _br_table(
    st,
    deal_w: np.ndarray,
    n: int,
    hands: tuple[str, ...],
    opp_btn: np.ndarray,
    opp_sb: np.ndarray,
    opp_bb: np.ndarray,
) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {seat: {} for seat in SEATS}
    for hand in hands:
        hi = LABEL_TO_INDEX[hand]
        for seat in SEATS:
            if seat == "BTN":
                opp = (opp_btn, opp_sb, opp_bb)
            elif seat == "SB":
                opp = (np.zeros(n), opp_sb, opp_bb)  # BTN always folds
            else:
                opp = (opp_btn, opp_sb, opp_bb)
            table[seat][hand] = hero_br_push_freq(
                seat, hi, st, deal_w, opp[0], opp[1], opp[2]
            )
    return table


def _spreads(table: dict[str, dict[str, float]], hands: tuple[str, ...]) -> dict[str, object]:
    spreads = {}
    for hand in hands:
        vals = [table[s][hand] for s in SEATS]
        spreads[hand] = {
            "btn": vals[0],
            "sb": vals[1],
            "bb": vals[2],
            "max_minus_min": float(max(vals) - min(vals)),
        }
    return spreads


def run_probe(*, tensor_path: Path, profile: str = "tight") -> dict[str, object]:
    shares, pair_eq, deal_w = load_trio_tensor(tensor_path)
    st, deal_w = load_3max_payoffs(shares, pair_eq, deal_w, pair_eq_source="tensor")
    n = shares.shape[0]
    labels = list(LABELS_169[:n])

    if profile == "tight":
        call_mask = _call_mask_from_range(TIGHT_CALL_RANGE, labels)
        call_pct = float(call_mask.mean())
        opp_btn, opp_sb, opp_bb = fixed_tight_call_policies(n, call_mask)
        hands = PROBE_HANDS_TIGHT
        opponent_profile = "call top chart range when facing jam; no open-jam"
    elif profile in ("field100", "field80"):
        frac = 1.0 if profile == "field100" else 0.8
        opp_btn, opp_sb, opp_bb = uniform_call_policies(n, frac)
        call_pct = frac
        hands = PROBE_HANDS_FIELD
        opponent_profile = f"all buckets call {frac:.0%} when facing jam; no open-jam"
    else:
        raise ValueError(f"unknown profile={profile!r}")

    table = _br_table(st, deal_w, n, hands, opp_btn, opp_sb, opp_bb)
    spreads = _spreads(table, hands)

    max_spread = max(spreads[h]["max_minus_min"] for h in hands)
    position_blind = max_spread < 0.05
    # Open-jam seats only (BB row is call-vs-jam, not comparable to open freq)
    open_btn = [table["BTN"][h] for h in hands]
    open_sb = [table["SB"][h] for h in hands]
    chart_canonical_order = all(b >= s >= 0.0 for b, s in zip(open_btn, open_sb))

    trash_jam_into_callers = False
    if profile in ("field100", "field80"):
        trash_jam_into_callers = table["BTN"].get("72o", 1.0) > 0.5 or table["SB"].get("72o", 1.0) > 0.5

    if profile == "tight":
        verdict = (
            "position_blind_payoff_layer"
            if position_blind
            else "position_aware_payoffs_bug_likely_solver_or_weighting"
        )
    elif trash_jam_into_callers:
        verdict = "contested_showdown_ev_underpunishes_jam_into_callers"
    else:
        verdict = "sound_payoffs_solver_or_fixed_point"

    out: dict[str, object] = {
        "profile": profile,
        "tensor_path": str(tensor_path),
        "call_pct_effective": call_pct,
        "opponent_profile": opponent_profile,
        "probe_hands": list(hands),
        "br_table": table,
        "spreads": spreads,
        "max_spread_across_seats": max_spread,
        "position_blind": position_blind,
        "trash_jam_into_callers": trash_jam_into_callers,
        "verdict": verdict,
        "notes": {
            "SB": "open jam after BTN fold (sb hist F)",
            "BB": "call vs BTN jam + SB fold (bb hist AF)",
        },
    }
    if profile == "tight":
        out["tight_call_range"] = TIGHT_CALL_RANGE
        out["chart_canonical_open_order_btn_ge_sb"] = chart_canonical_order
        out["cfr_degeneracy_contrast"] = (
            "CFR+ solves ~99.98% push identically across infosets while this probe "
            "shows seat-dependent BR under fixed tight callers — mismatch implicates "
            "solver/update (or equilibrium collapse), not static payoff symmetry."
        )
        out["topology_audit"] = audit_topology(st)
    return out


def run_all_in_field_probes(*, tensor_path: Path) -> dict[str, object]:
    f100 = run_probe(tensor_path=tensor_path, profile="field100")
    f80 = run_probe(tensor_path=tensor_path, profile="field80")
    return {
        "field100": f100,
        "field80": f80,
        "verdict": f100["verdict"],
        "trash_jams_at_100pct_field": f100["trash_jam_into_callers"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor", type=Path, default=DEFAULT_TENSOR)
    p.add_argument(
        "--profile",
        choices=("tight", "field100", "field80", "all_in_field"),
        default="tight",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
    )
    args = p.parse_args()
    if args.profile == "all_in_field":
        out = run_all_in_field_probes(tensor_path=args.tensor)
        default_report = Path("data/aof/cfr_base/phase_a_br_all_in_field_probe.json")
    else:
        out = run_probe(tensor_path=args.tensor, profile=args.profile)
        default_report = Path("data/aof/cfr_base/phase_a_br_position_probe.json")
    report = args.report or default_report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
