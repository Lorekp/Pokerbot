#!/usr/bin/env python3
"""Merge terminal audit + chip_flow + pair_eq verdict JSON into one report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    base = Path("data/aof/cfr_base")
    audit = json.loads((base / "terminal_audit_report.json").read_text(encoding="utf-8"))
    chip = json.loads((base / "phase_a_1547_verdict_chip_flow.json").read_text(encoding="utf-8"))
    pair_path = base / "phase_a_1547_verdict_pair_eq.json"
    pair = json.loads(pair_path.read_text(encoding="utf-8")) if pair_path.is_file() else None
    probe_path = base / "phase_a_br_position_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.is_file() else None
    field_path = base / "phase_a_br_all_in_field_probe.json"
    field_probe = (
        json.loads(field_path.read_text(encoding="utf-8")) if field_path.is_file() else None
    )
    contested_path = base / "contested_showdown_audit.json"
    contested = (
        json.loads(contested_path.read_text(encoding="utf-8")) if contested_path.is_file() else None
    )

    gap_floor = chip.get("gap_floor_mbb_g", chip["trivial_baselines_mbb_g"]["always_push"] * 0.25)
    out = {
        "terminal_audit": audit,
        "chip_flow_fix": chip,
        "pair_eq_isolated": pair,
        "br_position_probe": probe,
        "br_all_in_field_probe": field_probe,
        "contested_showdown_audit": contested,
        "summary": {
            "terminal_audit_pass": audit.get("audit_pass"),
            "chip_flow_pot_fix": "POT_HU_STACKS=17.5 POT_THREE_STACKS=25.5 (was missing blinds)",
            "stale_cfr_plus_mbb_g": chip.get("stale_manifest_cfr_mbb_g"),
            "cfr_plus_mbb_g_chip_flow": chip["cfr_plus"]["nash_gap_mbb_g"],
            "cfr_plus_delta_vs_stale_mbb_g": chip["cfr_plus"]["nash_gap_mbb_g"]
            - chip.get("stale_manifest_cfr_mbb_g", 0),
            "fp_mbb_g_chip_flow": chip["fp"]["nash_gap_mbb_g"],
            "fp_rounds_chip_flow": chip["fp"]["rounds"],
            "trivial_push_mbb_g": chip["trivial_baselines_mbb_g"]["always_push"],
            "gap_floor_mbb_g": gap_floor,
            "payoff_identity_chip_flow": {
                "cfr_plus": chip["cfr_plus"]["payoff_identity"]["identical"],
                "fp": chip["fp"]["payoff_identity"]["identical"],
            },
            "chart_sanity_chip_flow": {
                "cfr_plus": chip["cfr_plus"]["chart"]["passed"],
                "fp": chip["fp"]["chart"]["passed"],
            },
            "cfr_plus_still_push_heavy": chip["cfr_plus"]["chart"].get(
                "position_order_sample", [{}]
            )[0].get("BTN", 0) > 0.99,
            "overwrite_cfr_base_3p": chip.get("overwrite_cfr_base_3p", False),
            "fp_production_ready": chip.get("fp_production_ready", False),
            "pair_eq_note": (
                "hu_expand vs tensor at solve: <1 mbb/g Nash gap delta on chip-flow-fixed pots; "
                "exact_native 169³ cube not built (row build >>10min). "
                "True 3-way pair-eq needs pair_eq_exact_1693.npz offline build."
            ),
        },
    }
    if pair:
        out["summary"]["pair_eq_delta_vs_chip_flow"] = pair.get("delta_vs_baseline", {})
        out["summary"]["cfr_plus_mbb_g_pair_eq"] = pair["cfr_plus"]["nash_gap_mbb_g"]
        out["summary"]["fp_mbb_g_pair_eq"] = pair["fp"]["nash_gap_mbb_g"]
    if probe:
        out["summary"]["br_position_probe"] = {
            "verdict": probe["verdict"],
            "position_blind": probe["position_blind"],
            "max_spread": probe["max_spread_across_seats"],
            "br_table": probe["br_table"],
            "stop_payoff_fix": probe["verdict"] != "position_blind_payoff_layer",
        }
    if field_probe:
        out["summary"]["br_all_in_field"] = {
            "verdict": field_probe["verdict"],
            "field100_btn_72o": field_probe["field100"]["br_table"]["BTN"]["72o"],
            "field80_btn_72o": field_probe["field80"]["br_table"]["BTN"]["72o"],
            "redirect": "solver_or_fixed_point_not_contested_ev",
        }
    if contested:
        out["summary"]["contested_showdown"] = {
            "payoff_layer_pass": contested.get("payoff_layer_pass"),
            "tabular_matches_pencil": contested["bucket_tensor"]["tabular_matches_pencil_formula"],
        }

    path = base / "phase_a_1547_verdict.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
