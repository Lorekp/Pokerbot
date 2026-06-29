"""Cheap-pilot selection gate for EMD blueprint eval verdicts.

Separate from scripts/_emd_auto_activate_gate.py (production activation).
Selection gates use looser, tier-aware thresholds for heartbeat/cheap/medium
pilots; acceptance_pass delegates to the strict activate gate.

Module surface is pure stdlib + json. ASCII only.

CLI:
    python scripts/_emd_selection_gate.py check --verdict <path> [--tier cheap]
    python scripts/_emd_selection_gate.py report --verdict <path> [--tier cheap] \\
        [--emd-json partial] [--iter62000-json partial] [--out selection_report.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

import _emd_auto_activate_gate as _activate


# Tier thresholds: looser than production activate gate.
TIER_DEFAULTS: dict[str, dict[str, float]] = {
    "heartbeat": {
        "sb_min": -15.0,
        "bb_min": -15.0,
        "blind_mean_min": -15.0,
        "global_reg_max": 15.0,
        "ci_half_width_max": 40.0,
    },
    "cheap": {
        "sb_min": 0.0,
        "bb_min": 0.0,
        "blind_mean_min": 0.0,
        "global_reg_max": 10.0,
        "ci_half_width_max": 30.0,
    },
    "medium": {
        "sb_min": 5.0,
        "bb_min": 5.0,
        "blind_mean_min": 5.0,
        "global_reg_max": 10.0,
        "ci_half_width_max": 25.0,
    },
    "final": {
        "sb_min": _activate.SB_MIN_DEFAULT,
        "bb_min": _activate.BB_MIN_DEFAULT,
        "blind_mean_min": _activate.BLIND_MEAN_MIN_DEFAULT,
        "global_reg_max": _activate.GLOBAL_REG_MAX_DEFAULT,
        "ci_half_width_max": 20.0,
    },
}

# Required LBR probe menus; a friendly in-tree set must not hide a worse adversarial set.
ACTION_SET_REQUIRED = ("in_tree", "pot_fraction", "off_tree_adversarial")
ACTION_SET_DIVERGENCE_MIN = 10.0  # bb/100 blind-mean spread triggers divergence flag


def accepted_survivor_ks(abstraction_sweep: dict | None) -> set[int]:
    """Return K values that actually survived the accepted abstraction sweep."""
    out: set[int] = set()
    for row in (abstraction_sweep or {}).get("survivors") or []:
        if not isinstance(row, dict):
            continue
        try:
            if row.get("survivor_gate_pass", True) is not False:
                out.add(int(row.get("k")))
        except (TypeError, ValueError):
            continue
    return out


def _get_float(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _seat_gain(verdict: dict, seat: int) -> float:
    emd_ps = verdict.get("emd_per_seat", {}) or {}
    base_ps = verdict.get("iter62000_per_seat", {}) or {}
    sk = str(seat)
    if sk in emd_ps and sk in base_ps:
        emd_v = _get_float(emd_ps.get(sk))
        base_v = _get_float(base_ps.get(sk))
        if math.isfinite(emd_v) and math.isfinite(base_v):
            return abs(base_v) - abs(emd_v)
    if seat in emd_ps and seat in base_ps:
        emd_v = _get_float(emd_ps.get(seat))
        base_v = _get_float(base_ps.get(seat))
        if math.isfinite(emd_v) and math.isfinite(base_v):
            return abs(base_v) - abs(emd_v)
    return float("nan")


def _global_regression(verdict: dict) -> float:
    emd_g = _get_float(verdict.get("emd_pooled_bb_per_100"))
    base_g = _get_float(verdict.get("iter62000_pooled_bb_per_100"))
    if math.isfinite(emd_g) and math.isfinite(base_g):
        return abs(emd_g) - abs(base_g)
    return float("nan")


def _paired_seat_ci(
    emd_partial: dict | None,
    base_partial: dict | None,
    seat: str,
) -> dict[str, float] | None:
    if not emd_partial or not base_partial:
        return None
    e_rows = (emd_partial.get("result") or {}).get("per_seed_per_seat") or []
    b_rows = (base_partial.get("result") or {}).get("per_seed_per_seat") or []
    if not e_rows or len(e_rows) != len(b_rows):
        return None
    diffs = []
    for e, b in zip(e_rows, b_rows):
        if seat in e and seat in b:
            # Paired CRN on magnitude gains (not raw signed LBR).
            diffs.append(abs(float(b[seat])) - abs(float(e[seat])))
    if len(diffs) < 2:
        return None
    mean = sum(diffs) / len(diffs)
    var = sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1)
    stderr = math.sqrt(var / len(diffs))
    half = 1.96 * stderr
    return {
        "n": float(len(diffs)),
        "mean": mean,
        "stderr": stderr,
        "ci95_half_width": half,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
    }


def _evaluate_identity(
    verdict: dict,
    *,
    k_max: int,
    allowed_k_centroids: set[int] | None = None,
) -> list[str]:
    """Train/serve identity checks for selection tiers (cheap+)."""
    reasons: list[str] = []
    k_policy_error = verdict.get("k_policy_error")
    if k_policy_error:
        reasons.append(str(k_policy_error))
    k_centroids = verdict.get("k_centroids")
    if k_centroids is None:
        reasons.append("k_centroids_missing")
    else:
        try:
            k_int = int(k_centroids)
        except (TypeError, ValueError):
            reasons.append("k_centroids_invalid")
            k_int = -1
        if allowed_k_centroids is not None:
            if k_int not in {int(k) for k in allowed_k_centroids}:
                reasons.append("k_centroids_mismatch")
        elif k_int != k_max:
            reasons.append("k_centroids_mismatch")
    if not _activate._is_literal_true(verdict.get("sha256_stamped")):
        reasons.append("sha256_stamped=False")
    if not str(verdict.get("emd_frozen_s3", "")).startswith("s3://"):
        reasons.append("emd_frozen_s3_missing")
    if not str(verdict.get("emd_frozen_table_id", "")):
        reasons.append("emd_frozen_table_id_missing")
    frozen_sha = str(verdict.get("emd_frozen_sha256", ""))
    if len(frozen_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in frozen_sha):
        reasons.append("emd_frozen_sha256_missing")
    if not _activate._is_literal_true(verdict.get("serve_artifact_match")):
        reasons.append("serve_artifact_match=False")
    return reasons


def _action_set_blind_mean_gain(entry: dict) -> float:
    if "blind_mean_magnitude_gain" in entry:
        return _get_float(entry.get("blind_mean_magnitude_gain"))
    sb = _get_float(entry.get("sb_magnitude_gain"))
    bb = _get_float(entry.get("bb_magnitude_gain"))
    if math.isfinite(sb) and math.isfinite(bb):
        return (sb + bb) / 2.0
    emd_ps = entry.get("emd_per_seat") or {}
    base_ps = entry.get("iter62000_per_seat") or {}
    if emd_ps and base_ps:
        v = {"emd_per_seat": emd_ps, "iter62000_per_seat": base_ps}
        sb_g = _seat_gain(v, 1)
        bb_g = _seat_gain(v, 2)
        if math.isfinite(sb_g) and math.isfinite(bb_g):
            return (sb_g + bb_g) / 2.0
    return float("nan")


def summarize_action_sets(verdict: dict) -> dict[str, Any]:
    """Worst-case LBR across probe action sets (in-tree, pot-fraction, off-tree)."""
    raw = verdict.get("lbr_action_sets") or {}
    per_set: dict[str, Any] = {}
    blind_means: list[float] = []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        sb = _seat_gain(
            {"emd_per_seat": entry.get("emd_per_seat"), "iter62000_per_seat": entry.get("iter62000_per_seat")},
            1,
        )
        bb = _seat_gain(
            {"emd_per_seat": entry.get("emd_per_seat"), "iter62000_per_seat": entry.get("iter62000_per_seat")},
            2,
        )
        blind = _action_set_blind_mean_gain(entry)
        if math.isfinite(blind):
            blind_means.append(blind)
        per_set[name] = {
            "sb_magnitude_gain": sb,
            "bb_magnitude_gain": bb,
            "blind_mean_magnitude_gain": blind,
            "bet_candidates": entry.get("bet_candidates"),
            "method": entry.get("method"),
        }
    missing = [k for k in ACTION_SET_REQUIRED if k not in per_set]
    invalid = [
        k for k in ACTION_SET_REQUIRED
        if k in per_set and not math.isfinite(per_set[k]["blind_mean_magnitude_gain"])
    ]
    spread = (max(blind_means) - min(blind_means)) if len(blind_means) >= 2 else 0.0
    worst_name = None
    worst_blind = float("nan")
    if blind_means:
        worst_blind = min(blind_means)
        for name, row in per_set.items():
            if row.get("blind_mean_magnitude_gain") == worst_blind:
                worst_name = name
                break
    return {
        "per_set": per_set,
        "required_sets": list(ACTION_SET_REQUIRED),
        "missing_sets": missing,
        "invalid_sets": invalid,
        "complete": len(missing) == 0 and len(invalid) == 0,
        "worst_case_blind_mean_gain": worst_blind,
        "worst_case_set": worst_name,
        "best_case_blind_mean_gain": max(blind_means) if blind_means else float("nan"),
        "spread_bb_per_100": spread,
        "divergence_flag": spread >= ACTION_SET_DIVERGENCE_MIN,
    }


def _evaluate_ci(
    *,
    tier: str,
    th: dict[str, float],
    paired_sb: dict[str, float] | None,
    paired_bb: dict[str, float] | None,
) -> list[str]:
    """CI-backed selection: tier mins must hold at the lower 95% bound when CRN exists."""
    if tier == "heartbeat":
        return []
    reasons: list[str] = []
    for seat_label, paired, min_key in (
        ("sb", paired_sb, "sb_min"),
        ("bb", paired_bb, "bb_min"),
    ):
        if paired is None:
            if tier in ("medium", "final"):
                reasons.append("paired_crn_{0}_missing".format(seat_label))
            continue
        if paired["ci95_half_width"] > th["ci_half_width_max"]:
            reasons.append("paired_crn_{0}_half_width_wide".format(seat_label))
        lower = paired.get("ci95_lower")
        if lower is not None and lower < th[min_key]:
            reasons.append("paired_crn_{0}_lower95_below_tier_min".format(seat_label))
    if tier in ("medium", "final") and paired_sb and paired_bb:
        mean_lo = (paired_sb["ci95_lower"] + paired_bb["ci95_lower"]) / 2.0
        if mean_lo < th["blind_mean_min"]:
            reasons.append("paired_crn_blind_mean_lower95_below_tier_min")
    return reasons


def _evaluate_action_sets(verdict: dict, *, tier: str, th: dict[str, float]) -> list[str]:
    if tier == "heartbeat":
        return []
    summary = summarize_action_sets(verdict)
    reasons: list[str] = []
    if not summary["complete"]:
        reasons.append("action_sets_incomplete")
        reasons.extend("action_set_missing_{0}".format(m) for m in summary["missing_sets"])
        reasons.extend("action_set_invalid_{0}".format(m) for m in summary["invalid_sets"])
        return reasons
    worst = summary["worst_case_blind_mean_gain"]
    if math.isnan(worst) or worst < th["blind_mean_min"]:
        reasons.append("worst_case_action_set_below_tier_min")
    return reasons


def evaluate_selection(
    verdict: dict,
    *,
    tier: str = "cheap",
    k_max: int = _activate.K_MAX_DEFAULT,
    allowed_k_centroids: set[int] | None = None,
    require_k: bool = False,
    emd_partial: dict | None = None,
    base_partial: dict | None = None,
) -> tuple[bool, list[str]]:
    """Looser pilot selection gate. Returns (passes, reason_codes)."""
    if tier not in TIER_DEFAULTS:
        raise ValueError("unknown tier {0!r}".format(tier))
    th = TIER_DEFAULTS[tier]
    reasons: list[str] = []

    if not verdict.get("anchor_ok"):
        reasons.append("anchor_ok=False")

    emd_ps = verdict.get("emd_per_seat", {}) or {}
    base_ps = verdict.get("iter62000_per_seat", {}) or {}
    if not emd_ps or not base_ps:
        reasons.append("raw_per_seat_missing")

    if tier != "heartbeat":
        reasons.extend(_evaluate_identity(verdict, k_max=k_max, allowed_k_centroids=allowed_k_centroids))
    elif require_k:
        k = verdict.get("k_centroids")
        try:
            k_int = int(k) if k is not None else -1
        except (TypeError, ValueError):
            k_int = -1
        if allowed_k_centroids is not None:
            if k_int not in {int(k) for k in allowed_k_centroids}:
                reasons.append("k_centroids_mismatch")
        elif k_int != k_max:
            reasons.append("k_centroids_mismatch")

    sb = _seat_gain(verdict, 1)
    bb = _seat_gain(verdict, 2)
    if math.isnan(sb) or sb < th["sb_min"]:
        reasons.append("sb_gain_below_tier_min")
    if math.isnan(bb) or bb < th["bb_min"]:
        reasons.append("bb_gain_below_tier_min")

    blind_mean = (sb + bb) / 2.0 if math.isfinite(sb) and math.isfinite(bb) else float("nan")
    if math.isnan(blind_mean) or blind_mean < th["blind_mean_min"]:
        reasons.append("blind_mean_below_tier_min")

    g_reg = _global_regression(verdict)
    if math.isnan(g_reg) or g_reg > th["global_reg_max"]:
        reasons.append("global_regression_above_tier_max")

    if tier in ("medium", "final") and not verdict.get("sign_correct_verified"):
        reasons.append("sign_correct_not_verified")

    paired_sb = _paired_seat_ci(emd_partial, base_partial, "1")
    paired_bb = _paired_seat_ci(emd_partial, base_partial, "2")
    reasons.extend(_evaluate_ci(tier=tier, th=th, paired_sb=paired_sb, paired_bb=paired_bb))
    reasons.extend(_evaluate_action_sets(verdict, tier=tier, th=th))

    return (len(reasons) == 0, reasons)


def build_selection_report(
    verdict: dict,
    *,
    tier: str = "cheap",
    emd_partial: dict | None = None,
    base_partial: dict | None = None,
    k_max: int = _activate.K_MAX_DEFAULT,
    allowed_k_centroids: set[int] | None = None,
) -> dict[str, Any]:
    """Build selection_report.json payload from verdict + optional partials."""
    sb = _seat_gain(verdict, 1)
    bb = _seat_gain(verdict, 2)
    blind_mean = (sb + bb) / 2.0 if math.isfinite(sb) and math.isfinite(bb) else float("nan")
    g_reg = _global_regression(verdict)

    sel_pass, sel_reasons = evaluate_selection(
        verdict,
        tier=tier,
        k_max=k_max,
        allowed_k_centroids=allowed_k_centroids,
        emd_partial=emd_partial,
        base_partial=base_partial,
    )
    acc_pass, acc_reasons = _activate.evaluate_gate(
        verdict, k_max=k_max, allowed_k_centroids=allowed_k_centroids,
    )

    paired_sb = _paired_seat_ci(emd_partial, base_partial, "1")
    paired_bb = _paired_seat_ci(emd_partial, base_partial, "2")
    action_sets = summarize_action_sets(verdict)

    th = TIER_DEFAULTS[tier]
    ci_wide = False
    for paired in (paired_sb, paired_bb):
        if paired and paired["ci95_half_width"] > th["ci_half_width_max"]:
            ci_wide = True

    report: dict[str, Any] = {
        "schema": "emd_selection_report_v1",
        "tier": tier,
        "selection_pass": sel_pass,
        "selection_reason_codes": sel_reasons,
        "acceptance_pass": acc_pass,
        "acceptance_reason_codes": acc_reasons,
        "metrics": {
            "sb_magnitude_gain": sb,
            "bb_magnitude_gain": bb,
            "mean_blind_magnitude_gain": blind_mean,
            "global_magnitude_regression": g_reg,
            "worst_case_blind_mean_gain": action_sets.get("worst_case_blind_mean_gain"),
            "emd_per_seat": verdict.get("emd_per_seat"),
            "iter62000_per_seat": verdict.get("iter62000_per_seat"),
            "emd_pooled_bb_per_100": verdict.get("emd_pooled_bb_per_100"),
            "iter62000_pooled_bb_per_100": verdict.get("iter62000_pooled_bb_per_100"),
        },
        "calibration": {
            "anchor_ok": verdict.get("anchor_ok"),
            "emd_anchor_bb_per_100": verdict.get("emd_anchor_bb_per_100"),
            "iter62000_anchor_bb_per_100": verdict.get("iter62000_anchor_bb_per_100"),
        },
        "identity": {
            "k_centroids": verdict.get("k_centroids"),
            "expected_k_centroids": (
                sorted(int(k) for k in allowed_k_centroids)
                if allowed_k_centroids is not None
                else [int(k_max)]
            ),
            "k_policy": (
                "accepted_abstraction_survivor"
                if allowed_k_centroids is not None
                else "exact_k_max_default"
            ),
            "sha256_stamped": verdict.get("sha256_stamped"),
            "emd_frozen_sha256": verdict.get("emd_frozen_sha256"),
            "emd_frozen_s3": verdict.get("emd_frozen_s3"),
            "emd_frozen_table_id": verdict.get("emd_frozen_table_id"),
            "serve_artifact_match": verdict.get("serve_artifact_match"),
        },
        "paired_crn": {
            "sb": paired_sb,
            "bb": paired_bb,
            "ci_too_wide_for_decision": ci_wide,
            "variance_note": (
                "Raw hand counts are not AIVAT-adjusted; use paired CRN magnitude-gain "
                "CRN deltas and per-seat magnitude gains for pilot decisions."
            ),
        },
        "action_sets": action_sets,
        "harness": {
            "seeds": verdict.get("seeds"),
            "hands_per_seed": verdict.get("hands_per_seed"),
            "crn": verdict.get("crn"),
            "emd_method": verdict.get("emd_method"),
            "iter62000_method": verdict.get("iter62000_method"),
        },
        "legacy_flags": {
            "emd_better": verdict.get("emd_better"),
            "sign_correct_verified": verdict.get("sign_correct_verified"),
        },
    }
    return report


def stamp_primary_action_set(verdict: dict) -> dict:
    """Ensure lbr_action_sets.in_tree mirrors the headline per-seat LBR (eval combine helper)."""
    out = dict(verdict)
    sets = dict(out.get("lbr_action_sets") or {})
    if "in_tree" not in sets:
        sets["in_tree"] = {
            "emd_per_seat": out.get("emd_per_seat"),
            "iter62000_per_seat": out.get("iter62000_per_seat"),
            "method": out.get("emd_method") or out.get("harness"),
            "bet_candidates": "default_in_tree",
        }
    out["lbr_action_sets"] = sets
    return out


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cli_check(args: argparse.Namespace) -> int:
    verdict = _load_json(args.verdict)
    emd_p = _load_json(args.emd_json) if getattr(args, "emd_json", "") else None
    base_p = _load_json(args.iter62000_json) if getattr(args, "iter62000_json", "") else None
    passes, reasons = evaluate_selection(
        verdict,
        tier=args.tier,
        k_max=args.k_max,
        require_k=args.require_k,
        emd_partial=emd_p,
        base_partial=base_p,
    )
    if passes:
        print("PASS")
        return 0
    print("FAIL: {0}".format("; ".join(reasons)))
    return 1


def _cli_report(args: argparse.Namespace) -> int:
    verdict = _load_json(args.verdict)
    emd_p = _load_json(args.emd_json) if args.emd_json else None
    base_p = _load_json(args.iter62000_json) if args.iter62000_json else None
    report = build_selection_report(
        verdict, tier=args.tier, emd_partial=emd_p, base_partial=base_p, k_max=args.k_max,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print("wrote {0}".format(args.out))
    print(
        "selection_pass={0} acceptance_pass={1} tier={2}".format(
            report["selection_pass"], report["acceptance_pass"], report["tier"],
        )
    )
    return 0 if report["selection_pass"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("check")
    pc.add_argument("--verdict", required=True)
    pc.add_argument("--tier", default="cheap", choices=list(TIER_DEFAULTS))
    pc.add_argument("--k-max", type=int, default=_activate.K_MAX_DEFAULT)
    pc.add_argument("--require-k", action="store_true")
    pc.add_argument("--emd-json", default="")
    pc.add_argument("--iter62000-json", default="")
    pc.set_defaults(func=_cli_check)
    pr = sub.add_parser("report")
    pr.add_argument("--verdict", required=True)
    pr.add_argument("--emd-json", default="")
    pr.add_argument("--iter62000-json", default="")
    pr.add_argument("--out", default="")
    pr.add_argument("--tier", default="cheap", choices=list(TIER_DEFAULTS))
    pr.add_argument("--k-max", type=int, default=_activate.K_MAX_DEFAULT)
    pr.set_defaults(func=_cli_report)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
