#!/usr/bin/env python3
"""Hand-audit 3-max terminal chip-EV: pencil vs tabular vs terminal_ev oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import COMBOS_169, LABELS_169, LABEL_TO_INDEX
from aof_cfr.pushfold_payoffs import (
    BB_POST,
    BLINDS_DEAD,
    POT_HU_STACKS,
    POT_THREE_STACKS,
    SB_POST,
    STACK_BB,
    TERMINAL,
    build_static_payoffs,
)
from aof_cfr.terminal_ev import fold_terminal_returns, showdown_terminal_returns
from aof_cfr.trio_payoff_tensor import load_trio_tensor

SEATS = ("BTN", "SB", "BB")


def _first_valid_combo_triple(
    btn: int, sb: int, bb: int
) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    for c0, c1 in COMBOS_169[btn]:
        for c2, c3 in COMBOS_169[sb]:
            for c4, c5 in COMBOS_169[bb]:
                if len({c0, c1, c2, c3, c4, c5}) == 6:
                    return (c0, c1), (c2, c3), (c4, c5)
    raise ValueError(f"no valid combo triple for {btn},{sb},{bb}")


def _card_ids(combo: tuple[str, str]) -> tuple[int, int]:
    from aof_cfr.combo_deal import card_id

    return card_id(combo[0]), card_id(combo[1])


def _oracle_fold(btn_inv: float, sb_inv: float, bb_inv: float, winner: int) -> dict[str, float]:
    invested = np.array([btn_inv, sb_inv, bb_inv], dtype=np.float64)
    rets = fold_terminal_returns(set(), [winner], invested)
    return {SEATS[i]: float(rets[i]) for i in range(3)}


def _oracle_showdown(btn_inv: float, sb_inv: float, bb_inv: float, holes: list[tuple[int, int]]) -> dict[str, float]:
    invested = np.array([btn_inv, sb_inv, bb_inv], dtype=np.float64)
    rets = showdown_terminal_returns(holes, invested)
    return {SEATS[i]: float(rets[i]) for i in range(3)}


def _tabular_spot1() -> dict[str, float]:
    t = TERMINAL
    return {
        "BTN": t.btn_win_uncontested_after_sb_fold,
        "SB": t.sb_fold_after_btn_jam,
        "BB": t.bb_fold,
    }


def _tabular_spot2() -> dict[str, float]:
    t = TERMINAL
    return {
        "BTN": t.btn_fold,
        "SB": t.sb_win_uncontested_after_btn_fold,
        "BB": t.bb_fold,
    }


def _tabular_spot3(btn_bb_eq: float) -> dict[str, float]:
    pot = POT_HU_STACKS
    return {
        "BTN": btn_bb_eq * pot - STACK_BB,
        "SB": -SB_POST,
        "BB": (1.0 - btn_bb_eq) * pot - STACK_BB,
    }


def _static_spot3(shares: np.ndarray, pair_eq: np.ndarray, btn: int, sb: int, bb: int) -> dict[str, float]:
    st = build_static_payoffs(shares, pair_eq)
    pot_btn_bb = POT_HU_STACKS
    return {
        "BTN": float(st.btn_bb_call0[btn, sb, bb]),
        "SB": -SB_POST,
        "BB": float(st.btn_bb_call2[btn, sb, bb]),
        "pot_btn_bb_code": pot_btn_bb,
        "pot_btn_bb_pencil": 2.0 * STACK_BB + BLINDS_DEAD,
        "eq_btn_bb": float(pair_eq[btn, sb, bb, 0]),
    }


def _pencil_spot3(btn_bb_eq: float) -> dict[str, float]:
    pot = POT_HU_STACKS
    return {
        "BTN": btn_bb_eq * pot - STACK_BB,
        "SB": -SB_POST,
        "BB": (1.0 - btn_bb_eq) * pot - STACK_BB,
        "pot": pot,
    }


def _row(seat: str, pencil: float, tabular: float, oracle: float) -> dict[str, object]:
    return {
        "seat": seat,
        "pencil_bb": pencil,
        "tabular_bb": tabular,
        "oracle_bb": oracle,
        "tabular_vs_pencil": tabular - pencil,
        "oracle_vs_pencil": oracle - pencil,
        "tabular_vs_oracle": tabular - oracle,
    }


def run_audit(*, tensor_path: Path) -> dict[str, object]:
    shares, pair_eq, _ = load_trio_tensor(tensor_path)
    btn_i = LABEL_TO_INDEX["AKs"]
    sb_i = LABEL_TO_INDEX["72o"]
    bb_i = LABEL_TO_INDEX["QQ"]
    combo_triple = _first_valid_combo_triple(btn_i, sb_i, bb_i)
    holes = [_card_ids(combo_triple[i]) for i in range(3)]
    eq_btn_bb = float(pair_eq[btn_i, sb_i, bb_i, 0])

    pot_sb_bb_code = POT_HU_STACKS
    pot_three_code = POT_THREE_STACKS
    pot_btn_sb_code = POT_HU_STACKS
    pot_btn_bb_code = POT_HU_STACKS

    spots = []

    # Spot 1: BTN jam, SB fold, BB fold
    p1 = {"BTN": 1.5, "SB": -0.5, "BB": -1.0}
    t1 = _tabular_spot1()
    o1 = _oracle_fold(STACK_BB, SB_POST, BB_POST, winner=0)
    spots.append(
        {
            "id": 1,
            "line": "BTN jam, SB fold, BB fold",
            "rows": [_row(s, p1[s], t1[s], o1[s]) for s in SEATS],
            "pot_notes": "pencil pot=9.5 (8+0.5+1)",
        }
    )

    # Spot 2: BTN fold, SB jam, BB fold (pot=9, SB nets +1.0)
    p2 = {"BTN": 0.0, "SB": 1.0, "BB": -1.0}
    t2 = _tabular_spot2()
    o2 = _oracle_fold(0.0, STACK_BB, BB_POST, winner=1)
    spots.append(
        {
            "id": 2,
            "line": "BTN fold, SB jam, BB fold",
            "rows": [_row(s, p2[s], t2[s], o2[s]) for s in SEATS],
            "pot_notes": "pencil pot=9.5; tabular SB win uses TERMINAL.sb_win_uncontested_after_btn_fold",
        }
    )

    # Spot 3: BTN jam, SB fold, BB call
    p3 = _pencil_spot3(eq_btn_bb)
    t3 = _tabular_spot3(eq_btn_bb)
    o3 = _oracle_showdown(STACK_BB, SB_POST, STACK_BB, holes)
    s3 = _static_spot3(shares, pair_eq, btn_i, sb_i, bb_i)
    spots.append(
        {
            "id": 3,
            "line": "BTN jam, SB fold, BB call (showdown)",
            "sample_buckets": "AKs/72o/QQ",
            "eq_btn_bb": eq_btn_bb,
            "rows": [_row(s, p3[s], t3[s], o3[s]) for s in SEATS],
            "static_payoffs": {k: s3[k] for k in ("BTN", "SB", "BB")},
            "static_vs_tabular_btn": s3["BTN"] - t3["BTN"],
            "static_vs_tabular_bb": s3["BB"] - t3["BB"],
            "pot_btn_bb": {"code": s3["pot_btn_bb_code"], "pencil": s3["pot_btn_bb_pencil"]},
        }
    )

    suspects = [
        {
            "name": "sb_win_uncontested_after_btn_fold",
            "code": TERMINAL.sb_win_uncontested_after_btn_fold,
            "pencil_spot2_sb": 1.0,
            "delta_vs_pencil": TERMINAL.sb_win_uncontested_after_btn_fold - 1.0,
        },
        {
            "name": "pot_sb_bb",
            "code": pot_sb_bb_code,
            "pencil": 2.0 * STACK_BB + BLINDS_DEAD,
            "delta_vs_pencil": pot_sb_bb_code - (2.0 * STACK_BB + BLINDS_DEAD),
        },
        {
            "name": "pot_three",
            "code": pot_three_code,
            "pencil": 3.0 * STACK_BB + BLINDS_DEAD,
            "delta_vs_pencil": pot_three_code - (3.0 * STACK_BB + BLINDS_DEAD),
        },
        {
            "name": "pot_btn_sb",
            "code": pot_btn_sb_code,
            "pencil": 2.0 * STACK_BB + BLINDS_DEAD,
            "delta_vs_pencil": pot_btn_sb_code - (2.0 * STACK_BB + BLINDS_DEAD),
        },
        {
            "name": "pot_btn_bb",
            "code": pot_btn_bb_code,
            "pencil": POT_HU_STACKS,
            "delta_vs_pencil": pot_btn_bb_code - POT_HU_STACKS,
        },
    ]

    import aof as aof_mod

    advisor = {
        "btn_blind_state": aof_mod._blind_state("BTN"),
        "call_pot_terms_btn": aof_mod._call_pot_terms("BTN", STACK_BB),
    }

    max_err_fold = 0.0
    for spot in spots[:2]:
        for row in spot["rows"]:
            max_err_fold = max(
                max_err_fold,
                abs(row["tabular_vs_pencil"]),
                abs(row["oracle_vs_pencil"]),
                abs(row["tabular_vs_oracle"]),
            )

    spot3 = spots[2]
    max_err_spot3_tab = max(abs(r["tabular_vs_pencil"]) for r in spot3["rows"])

    return {
        "blind_structure": {
            "stack_bb": STACK_BB,
            "sb_post": SB_POST,
            "bb_post": BB_POST,
            "btn_post": 0.0,
            "act_order": list(SEATS),
        },
        "spots": spots,
        "suspects": suspects,
        "advisor_crosscheck": advisor,
        "max_abs_fold_terminal_error_bb": max_err_fold,
        "max_abs_spot3_tabular_vs_pencil_bb": max_err_spot3_tab,
        "spot3_oracle_note": "oracle uses exact combo showdown; tabular uses bucket pair_eq",
        "audit_pass": max_err_fold < 1e-9 and max_err_spot3_tab < 1e-9,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Audit 3-max terminal chip-EV")
    p.add_argument("--tensor", default="data/aof/cfr_base/trio_eq_tensor.npz")
    p.add_argument("--out", default="data/aof/cfr_base/terminal_audit_report.json")
    args = p.parse_args()
    report = run_audit(tensor_path=Path(args.tensor))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Terminal chip-EV audit ===")
    print(f"Blinds: SB={SB_POST} BB={BB_POST} stack={STACK_BB}")
    for spot in report["spots"]:
        print(f"\nSpot {spot['id']}: {spot['line']}")
        for row in spot["rows"]:
            print(
                f"  {row['seat']}: pencil={row['pencil_bb']:+.4f} "
                f"tabular={row['tabular_bb']:+.4f} oracle={row['oracle_bb']:+.4f} "
                f"d_tab={row['tabular_vs_pencil']:+.4f} d_ora={row['oracle_vs_pencil']:+.4f}"
            )
    print("\nSuspects:")
    for s in report["suspects"]:
        print(f"  {s['name']}: code={s.get('code')} pencil={s.get('pencil', s.get('pencil_spot2_sb'))} "
              f"delta={s.get('delta_vs_pencil'):+.4f}")
    print(
        f"\nmax_fold_err={report['max_abs_fold_terminal_error_bb']:.6f} "
        f"spot3_tab_err={report['max_abs_spot3_tabular_vs_pencil_bb']:.6f} "
        f"pass={report['audit_pass']}"
    )
    print(f"Wrote {out}")
    if not report["audit_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
