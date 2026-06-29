"""Launch/no-launch recommendation after cheap pilots.



Reads ladder artifacts (abstraction sweep survivors, eval selection reports, pilot

verdicts) and emits a conservative recommendation. Full training still requires the

strict activation gate after a marathon run.



Pilot verdicts MUST live under artifacts_dir/pilots/ with distinct pilot_run_id fields.

Eval-harness selection reports are read separately and cannot double as pilot evidence.



CLI:

    python scripts/emd_full_run_decision.py --artifacts-dir artifacts/emd_pilot_ladder/<stamp>

"""



from __future__ import annotations



import argparse

import json

import math

import sys

from pathlib import Path

from typing import Any



ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "scripts") not in sys.path:

    sys.path.insert(0, str(ROOT / "scripts"))



import _emd_auto_activate_gate as activate  # noqa: E402

import _emd_selection_gate as selection  # noqa: E402



PILOT_SUBDIR = "pilots"

SELECTION_CANDIDATES = (

    "eval_verdict.json",

    "eval_harness_verdict.json",

    "selection_report.json",

)

DEFAULT_MIN_PILOT_PASSES = 2

TIER_RANK = {name: i for i, name in enumerate(("heartbeat", "cheap", "medium", "final"))}

PILOT_SELECTION_TIER = "cheap"





def _load(path: Path) -> dict | None:

    if not path.exists():

        return None

    return json.loads(path.read_text(encoding="utf-8"))





def _find_selection_report(artifacts_dir: Path) -> dict | None:

    for name in SELECTION_CANDIDATES:

        v = _load(artifacts_dir / name)

        if v is None:

            continue

        if "selection" in v:

            return v["selection"]

        if v.get("schema") == "emd_selection_report_v1":

            return v

    for p in sorted(artifacts_dir.glob("*selection_report.json")):

        if PILOT_SUBDIR in p.parts:

            continue

        data = _load(p)

        if data:

            return data

    return None





def _is_pilot_verdict(data: dict, path: Path) -> bool:

    if "emd_per_seat" not in data:

        return False

    if data.get("artifact_role") == "fixed_budget_pilot":

        return True

    if PILOT_SUBDIR in path.parts:

        return True

    return False





def _pilot_verdicts(artifacts_dir: Path) -> list[dict]:

    out: list[dict] = []

    seen_ids: set[str] = set()

    paths: list[Path] = []
    pilots_dir = artifacts_dir / PILOT_SUBDIR
    if pilots_dir.exists():
        paths.extend(sorted(pilots_dir.glob("**/*.json")))

    for p in sorted(set(paths)):

        if "selection_report" in p.name:

            continue

        data = _load(p)

        if not data or not _is_pilot_verdict(data, p):

            continue

        pilot_id_raw = data.get("pilot_run_id")
        pilot_id = pilot_id_raw.strip() if isinstance(pilot_id_raw, str) else ""
        if not pilot_id:
            continue

        if pilot_id in seen_ids:

            continue

        seen_ids.add(pilot_id)

        row = dict(data)

        row["_provenance"] = {"path": str(p), "pilot_run_id": pilot_id}

        out.append(row)

    return out


def _get_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _selection_report_k(report: dict) -> int | None:
    identity = report.get("identity") or {}
    raw = identity.get("k_centroids", report.get("k_centroids"))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _selection_report_policy_blockers(report: dict, *, requested_tier: str) -> list[str]:
    blockers: list[str] = []
    report_tier = str(report.get("tier") or "")
    if requested_tier not in TIER_RANK:
        return ["unknown_selection_tier_policy"]
    if report_tier not in TIER_RANK:
        blockers.append("selection_report_tier_missing_or_unknown")
        report_tier = requested_tier
    elif TIER_RANK[report_tier] < TIER_RANK[requested_tier]:
        blockers.append("selection_report_tier_below_policy")

    if TIER_RANK[report_tier] < TIER_RANK["medium"]:
        return blockers

    th = selection.TIER_DEFAULTS[report_tier]
    paired = report.get("paired_crn") or {}
    lowers: list[float] = []
    for seat_label, min_key in (("sb", "sb_min"), ("bb", "bb_min")):
        row = paired.get(seat_label)
        if not isinstance(row, dict):
            blockers.append("selection_report_paired_crn_{0}_missing".format(seat_label))
            continue
        lower = _get_float(row.get("ci95_lower"))
        half = _get_float(row.get("ci95_half_width"))
        if not math.isfinite(lower):
            blockers.append("selection_report_paired_crn_{0}_ci95_lower_missing".format(seat_label))
        elif lower < th[min_key]:
            blockers.append("selection_report_paired_crn_{0}_lower95_below_policy".format(seat_label))
        else:
            lowers.append(lower)
        if not math.isfinite(half):
            blockers.append("selection_report_paired_crn_{0}_half_width_missing".format(seat_label))
        elif half > th["ci_half_width_max"]:
            blockers.append("selection_report_paired_crn_{0}_half_width_wide".format(seat_label))

    if len(lowers) == 2 and sum(lowers) / 2.0 < th["blind_mean_min"]:
        blockers.append("selection_report_paired_crn_blind_mean_lower95_below_policy")
    return blockers





def build_decision(

    *,

    artifacts_dir: Path,

    abstraction_sweep: dict | None,

    selection_report: dict | None,

    pilot_verdicts: list[dict],

    min_survivors: int = 1,

    min_pilot_passes: int = DEFAULT_MIN_PILOT_PASSES,

    min_distinct_pilots: int = DEFAULT_MIN_PILOT_PASSES,

    selection_tier: str = "medium",

) -> dict[str, Any]:

    blockers: list[str] = []

    evidence: dict[str, Any] = {}



    survivors = (abstraction_sweep or {}).get("survivors") or []
    accepted_ks = selection.accepted_survivor_ks(abstraction_sweep)

    evidence["n_abstraction_survivors"] = len(survivors)
    evidence["accepted_survivor_ks"] = sorted(accepted_ks)

    if len(survivors) < min_survivors:

        blockers.append("insufficient_abstraction_survivors")



    if selection_report is None:

        blockers.append("missing_selection_report")
        selected_k = None

    else:
        selected_k = _selection_report_k(selection_report)

        evidence["selection_pass"] = selection_report.get("selection_pass")

        evidence["acceptance_pass"] = selection_report.get("acceptance_pass")

        evidence["selection_report_tier"] = selection_report.get("tier")

        evidence["required_selection_tier"] = selection_tier
        evidence["selection_k_centroids"] = selected_k

        if accepted_ks:
            if selected_k not in accepted_ks:
                blockers.append("selection_k_not_accepted_survivor")
        elif survivors:
            blockers.append("accepted_survivor_k_missing")

        if not selection_report.get("selection_pass"):

            blockers.append("selection_gate_failed")

            blockers.extend(selection_report.get("selection_reason_codes") or [])

        blockers.extend(
            _selection_report_policy_blockers(selection_report, requested_tier=selection_tier)
        )

        paired = selection_report.get("paired_crn") or {}

        if paired.get("ci_too_wide_for_decision"):

            blockers.append("paired_crn_ci_too_wide")

        action_sets = selection_report.get("action_sets") or {}

        if not action_sets.get("complete"):

            blockers.append("action_sets_incomplete_in_selection_report")



    pilot_pass = 0

    pilot_details: list[dict] = []

    pilot_ids: set[str] = set()
    pilot_allowed_ks = ({selected_k} if selected_k is not None else accepted_ks) or None

    for v in pilot_verdicts:

        pilot_id = str(v.get("_provenance", {}).get("pilot_run_id") or v.get("pilot_run_id") or "")

        if pilot_id:

            pilot_ids.add(pilot_id)

        sel_pass, sel_reasons = selection.evaluate_selection(
            v, tier=PILOT_SELECTION_TIER, allowed_k_centroids=pilot_allowed_ks,
        )

        acc_pass, acc_reasons = activate.evaluate_gate(v, allowed_k_centroids=pilot_allowed_ks)

        row = {

            "pilot_run_id": pilot_id,

            "k_centroids": v.get("k_centroids"),

            "selection_pass": sel_pass,

            "acceptance_pass": acc_pass,

            "selection_tier": PILOT_SELECTION_TIER,

            "selection_reason_codes": sel_reasons,

            "acceptance_reason_codes": acc_reasons,

            "provenance": v.get("_provenance"),

        }

        pilot_details.append(row)

        if sel_pass and acc_pass:

            pilot_pass += 1

    evidence["pilot_verdicts"] = pilot_details

    evidence["n_pilot_pass"] = pilot_pass

    evidence["n_distinct_pilot_ids"] = len(pilot_ids)



    if not pilot_verdicts:

        blockers.append("missing_fixed_budget_pilot_verdict")

    elif len(pilot_ids) < min_distinct_pilots:

        blockers.append("insufficient_distinct_pilot_runs")

    elif pilot_pass < min_pilot_passes:

        blockers.append("fixed_budget_pilot_not_repeatable")



    recommend = len(blockers) == 0

    return {

        "schema": "emd_full_run_decision_v1",

        "recommend_launch": recommend,

        "primary_blocker": blockers[0] if blockers else None,

        "blockers": blockers,

        "evidence": evidence,

        "survivor_candidates": survivors[:3],

        "policy": (

            "Recommend full run only when abstraction survivors exist, eval-harness selection "

            "passes at medium tier with complete action-set coverage and CI lower bounds, and "

            "at least two distinct fixed-budget pilot verdicts under pilots/ pass the cheap "

            "structural selection gate and strict activation gate. Eval-harness verdicts cannot "

            "substitute for pilot artifacts."

        ),

    }





def main(argv: list[str] | None = None) -> int:

    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--artifacts-dir", required=True)

    ap.add_argument("--out", default="")

    ap.add_argument("--selection-tier", default="medium")

    ap.add_argument("--min-pilot-passes", type=int, default=DEFAULT_MIN_PILOT_PASSES)

    args = ap.parse_args(argv)



    art = Path(args.artifacts_dir)

    if not art.is_absolute():

        art = ROOT / art



    decision = build_decision(

        artifacts_dir=art,

        abstraction_sweep=_load(art / "abstraction_sweep.json"),

        selection_report=_find_selection_report(art),

        pilot_verdicts=_pilot_verdicts(art),

        min_pilot_passes=args.min_pilot_passes,

        min_distinct_pilots=args.min_pilot_passes,

        selection_tier=args.selection_tier,

    )



    out = Path(args.out) if args.out else art / "full_run_decision.json"

    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    print(

        "recommend_launch={0} blocker={1}".format(

            decision["recommend_launch"], decision["primary_blocker"],

        )

    )

    return 0 if decision["recommend_launch"] else 1





if __name__ == "__main__":

    sys.exit(main())

