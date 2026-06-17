#!/usr/bin/env python3
"""Reconcile contested HU showdown chip-EV: BTN jam 72o, SB fold, BB call AA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import COMBOS_169, LABEL_TO_INDEX
from aof_cfr.pushfold_payoffs import (
    BB_POST,
    POT_HU_STACKS,
    SB_POST,
    STACK_BB,
    build_static_payoffs,
)
from aof_cfr.terminal_ev import showdown_terminal_returns
from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, load_trio_tensor
from scripts.audit_terminal_chip_ev import _card_ids, _first_valid_combo_triple

BTN_HAND = "72o"
BB_HAND = "AA"
# SB folded — pick a bucket that does not block AA/72o combos
SB_HAND = "32o"


def _exact_hu_btn_bb_eq(btn_combo, bb_combo) -> float:
    from aof_equity_grids import _equity_two_combos_exact

    h0, h1 = _card_ids(btn_combo)
    v0, v1 = _card_ids(bb_combo)
    return float(_equity_two_combos_exact(h0, h1, v0, v1))


def _pencil_hu(btn_eq: float) -> dict[str, float]:
    pot = POT_HU_STACKS
    return {
        "BTN": btn_eq * pot - STACK_BB,
        "SB": -SB_POST,
        "BB": (1.0 - btn_eq) * pot - STACK_BB,
        "pot_bb": pot,
        "btn_eq": btn_eq,
        "bb_eq": 1.0 - btn_eq,
    }


def run_audit(*, tensor_path: Path) -> dict[str, object]:
    shares, pair_eq, _ = load_trio_tensor(tensor_path)
    btn_i = LABEL_TO_INDEX[BTN_HAND]
    sb_i = LABEL_TO_INDEX[SB_HAND]
    bb_i = LABEL_TO_INDEX[BB_HAND]

    btn_combo, sb_combo, bb_combo = _first_valid_combo_triple(btn_i, sb_i, bb_i)
    holes = [_card_ids(btn_combo), _card_ids(sb_combo), _card_ids(bb_combo)]
    invested = np.array([STACK_BB, SB_POST, STACK_BB], dtype=np.float64)

    eq_exact = _exact_hu_btn_bb_eq(btn_combo, bb_combo)
    oracle_3way = showdown_terminal_returns(holes, invested)
    # HU oracle: only BTN+BB contest; pot must include SB dead blind (17.5bb)
    invested_hu = np.array([STACK_BB, STACK_BB], dtype=np.float64)
    oracle_hu = showdown_terminal_returns([holes[0], holes[2]], invested_hu)
    pot_oracle_hu = float(invested_hu.sum())
    pencil = _pencil_hu(eq_exact)
    pencil_hu_pot16 = {
        "BTN": eq_exact * pot_oracle_hu - STACK_BB,
        "BB": (1.0 - eq_exact) * pot_oracle_hu - STACK_BB,
    }

    st = build_static_payoffs(shares, pair_eq)
    tab_btn = float(st.btn_bb_call0[btn_i, sb_i, bb_i])
    tab_bb = float(st.btn_bb_call2[btn_i, sb_i, bb_i])
    eq_bucket = float(pair_eq[btn_i, sb_i, bb_i, 0])
    pencil_bucket = _pencil_hu(eq_bucket)

    return {
        "line": "BTN jam 72o, SB fold 32o, BB call AA",
        "pot_hu_stacks_bb": POT_HU_STACKS,
        "stack_committed_bb": STACK_BB,
        "exact_combo": {
            "btn": "".join(btn_combo),
            "sb": "".join(sb_combo),
            "bb": "".join(bb_combo),
            "btn_bb_equity": eq_exact,
            "pencil_bb": pencil,
            "oracle_3way_bb": {
                "BTN": float(oracle_3way[0]),
                "SB": float(oracle_3way[1]),
                "BB": float(oracle_3way[2]),
                "note": "3-way settlement with folded SB still in hole list mis-allocates pot",
            },
            "oracle_hu_2player_bb": {
                "BTN": float(oracle_hu[0]),
                "BB": float(oracle_hu[1]),
                "pot_from_invested_only": pot_oracle_hu,
                "note": "2-player oracle pot=16bb omits SB 0.5bb dead money; tabular uses 17.5bb",
            },
            "oracle_hu_vs_pencil": {
                "BTN": float(oracle_hu[0]) - pencil["BTN"],
                "BB": float(oracle_hu[1]) - pencil["BB"],
            },
            "loser_stack_accounting": {
                "committed_bb": STACK_BB,
                "net_btn_ev_pencil": pencil["BTN"],
                "net_bb_ev_pencil": pencil["BB"],
                "interpretation": (
                    "Jammer does not net -8bb when called; nets equity×pot−8bb (~-6.1bb at 11% eq). "
                    "Full stack is committed but equity share is returned — correct chip EV."
                ),
            },
        },
        "bucket_tensor": {
            "eq_btn_bb_channel": eq_bucket,
            "pencil_from_bucket_eq": pencil_bucket,
            "tabular_btn_bb_call0": tab_btn,
            "tabular_btn_bb_call2": tab_bb,
            "tabular_vs_pencil_btn": tab_btn - pencil_bucket["BTN"],
            "tabular_vs_pencil_bb": tab_bb - pencil_bucket["BB"],
            "tabular_matches_pencil_formula": (
                abs(tab_btn - pencil_bucket["BTN"]) < 1e-9
                and abs(tab_bb - pencil_bucket["BB"]) < 1e-9
            ),
            "bucket_eq_vs_exact_combo": eq_bucket - eq_exact,
        },
        "payoff_layer_pass": (
            abs(tab_btn - pencil_bucket["BTN"]) < 1e-9
            and abs(tab_bb - pencil_bucket["BB"]) < 1e-9
        ),
        "interpretation": (
            "Tabular/payoff layer uses POT_HU_STACKS=17.5 (includes SB dead blind) and "
            "equity×pot−STACK_BB per seat; matches pencil. Earlier audit oracle gap was "
            "oracle settlement (pot=sum(invested) or 3-way with folded SB), not tabular EV."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor", type=Path, default=DEFAULT_TENSOR)
    p.add_argument(
        "--report",
        type=Path,
        default=Path("data/aof/cfr_base/contested_showdown_audit.json"),
    )
    args = p.parse_args()
    out = run_audit(tensor_path=args.tensor)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
