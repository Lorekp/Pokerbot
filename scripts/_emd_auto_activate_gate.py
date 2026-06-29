"""Pluribus-grade auto-activate gate for EMD blueprint verdicts.

Single source of truth for the v2 babysitter + watcher's
auto-activate decision (scripts/emd_pipeline_babysitter_v2.ps1,
scripts/watch_emd_pipeline_v2.ps1). Mirrors the PowerShell gate
function exactly so the PowerShell side can shell out to this
helper and CI / pytest can verify the gate logic without needing
PowerShell on the path.

Background: run_emd_k500_pluribus_20260625 had emd_better=True
on a verdict with a real -140 SB regression (per-seat LBR moved
from -23.85 to -164.32) because the verdict printer was labeling
the literal sign of `delta=base-emd` as "improvement" regardless
of sign. The legacy emd_better/anchor_ok short-circuit was not
safe against this bug class. The v2 gate enforces a full
pluribus-grade checklist (K == K_MAX, anchor in band, both
blind exploitability magnitudes improve by >= min gain, blind-mean
>= min, global magnitude regression <= max, sha256 stamped,
sign-correct verified).

Module surface is pure stdlib + json. ASCII only. No imports
from pokerbot/. Invokable from anywhere with just `python`.

CLI:
    python -m _emd_auto_activate_gate check --verdict <path> [--k-max 200]
    echoes "PASS" or "FAIL: <reasons>" and exits 0 (PASS) or 1 (FAIL).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any


# --- CONSTANTS (mirrors scripts/emd_pipeline_babysitter_v2.ps1) -------------

K_MAX_DEFAULT = 200
ANCHOR_LO_DEFAULT = 110.0
ANCHOR_HI_DEFAULT = 145.0
TARGET_ANCHOR_DEFAULT = 125.0
SB_MIN_DEFAULT = 15.0
BB_MIN_DEFAULT = 15.0
BLIND_MEAN_MIN_DEFAULT = 15.0
GLOBAL_REG_MAX_DEFAULT = 10.0


# --- HELPERS ---------------------------------------------------------------


def _get_float(v: Any) -> float:
    """Coerce to float; return NaN on missing / non-numeric."""
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _is_literal_true(v: Any) -> bool:
    """Return True only if v is the literal boolean True (not truthy).

    Mirrors the PowerShell gate's `($v.sign_correct_verified -is [bool])
    -and ([bool]$v.sign_correct_verified -eq $true)` check.
    """
    return v is True


# --- PUBLIC FUNCTIONS ------------------------------------------------------


def evaluate_gate(
    verdict: dict,
    k_max: int = K_MAX_DEFAULT,
    allowed_k_centroids: set[int] | None = None,
    anchor_lo: float = ANCHOR_LO_DEFAULT,
    anchor_hi: float = ANCHOR_HI_DEFAULT,
    sb_min: float = SB_MIN_DEFAULT,
    bb_min: float = BB_MIN_DEFAULT,
    blind_mean_min: float = BLIND_MEAN_MIN_DEFAULT,
    global_reg_max: float = GLOBAL_REG_MAX_DEFAULT,
) -> tuple[bool, list[str]]:
    """Run the pluribus-grade auto-activate gate against a verdict.

    Returns a (passes, reasons) tuple. `passes` is True iff every
    condition holds; `reasons` is the empty list on PASS, otherwise
    a list of human-readable failure reasons in the same order as
    the conditions are evaluated (one reason per failing condition).

    The conditions are:

        1. k_centroids == k_max (defaults: 200), unless an explicit
           accepted-survivor K set is supplied by the caller
        2. emd_anchor_bb_per_100 AND iter62000_anchor_bb_per_100
           in [anchor_lo, anchor_hi]
        3. SB magnitude gain >= sb_min
        4. BB magnitude gain >= bb_min
        5. mean blind magnitude gain >= blind_mean_min
        6. global magnitude regression <= global_reg_max (positive = regression)
        7. sha256_stamped is the literal True
        8. sign_correct_verified is the literal True
        9. frozen table identity is stamped
        10. serve_artifact_match is the literal True

    A missing / non-numeric numeric field is treated as a failure
    for the corresponding condition (NaN never satisfies a >= or <=
    check). The sha256_stamped and sign_correct_verified checks
    require the literal True (truthy is not enough).
    """
    reasons: list[str] = []

    # 1) K identity. Default is the historic production K_MAX exact match.
    # A caller may pass an accepted-survivor set from an abstraction sweep to
    # evaluate fixed-budget pilots for a selected non-200 K without weakening
    # the default gate into "K <= 200".
    k_centroids = verdict.get("k_centroids", None)
    if k_centroids is None:
        reasons.append(
            "k_centroids missing (the eval must stamp the K it trained on)"
        )
    else:
        try:
            k_int = int(k_centroids)
        except (TypeError, ValueError):
            reasons.append(
                "k_centroids={0!r} is not an integer".format(k_centroids)
            )
            k_int = -1
        if allowed_k_centroids is not None:
            allowed = {int(k) for k in allowed_k_centroids}
            if k_int not in allowed:
                reasons.append(
                    "K={0} not in accepted survivor K set {1}".format(k_int, sorted(allowed))
                )
        elif k_int != k_max:
            reasons.append(
                "K={0} != K_MAX={1}".format(k_int, k_max)
            )

    # 2) Anchor in band on BOTH emd + iter62000
    anchor_emd = _get_float(verdict.get("emd_anchor_bb_per_100", None))
    anchor_iter = _get_float(verdict.get("iter62000_anchor_bb_per_100", None))
    if (
        math.isnan(anchor_emd)
        or math.isnan(anchor_iter)
        or anchor_emd < anchor_lo
        or anchor_emd > anchor_hi
        or anchor_iter < anchor_lo
        or anchor_iter > anchor_hi
    ):
        reasons.append(
            "anchor out of band (emd={0:.1f}, iter62000={1:.1f}, "
            "allowed {2:.0f}..{3:.0f})".format(
                anchor_emd, anchor_iter, anchor_lo, anchor_hi
            )
        )

    # 3-6) Magnitude-correct eval metrics. Per-seat/global LBR can be negative,
    # so production acceptance must use distance from zero. The raw per-seat and
    # pooled values are required; old verdicts that only carry stamped deltas fail
    # closed because stale deltas were the K500 bug class.
    emd_ps = verdict.get("emd_per_seat", {}) or {}
    base_ps = verdict.get("iter62000_per_seat", {}) or {}

    def _seat_gain(seat: int) -> float:
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

    sb_gain = _seat_gain(1)
    if math.isnan(sb_gain) or sb_gain < sb_min:
        reasons.append(
            "SB gain {0:.1f} < {1:.0f}".format(sb_gain, sb_min)
        )

    # 4) BB gain >= bb_min
    bb_gain = _seat_gain(2)
    if math.isnan(bb_gain) or bb_gain < bb_min:
        reasons.append(
            "BB gain {0:.1f} < {1:.0f}".format(bb_gain, bb_min)
        )

    # 5) Blind-mean gain >= blind_mean_min
    blind_mean = (sb_gain + bb_gain) / 2.0 if math.isfinite(sb_gain) and math.isfinite(bb_gain) else float("nan")
    if math.isnan(blind_mean) or blind_mean < blind_mean_min:
        reasons.append(
            "blind-mean gain {0:.1f} < {1:.0f}".format(
                blind_mean, blind_mean_min
            )
        )

    # 6) Global regression <= global_reg_max (positive = regression)
    emd_global = _get_float(verdict.get("emd_pooled_bb_per_100", None))
    base_global = _get_float(verdict.get("iter62000_pooled_bb_per_100", None))
    if math.isfinite(emd_global) and math.isfinite(base_global):
        global_reg = abs(emd_global) - abs(base_global)
    else:
        global_reg = float("nan")
    if math.isnan(global_reg) or global_reg > global_reg_max:
        reasons.append(
            "global magnitude regression {0:.1f} > {1:.0f} bb/100 (positive = regression)".format(
                global_reg, global_reg_max
            )
        )

    # 7) SHA256 stamped (literal True)
    if not _is_literal_true(verdict.get("sha256_stamped", None)):
        reasons.append("sha256_stamped=False")

    # 8) Sign-correct verified (literal True)
    if not _is_literal_true(verdict.get("sign_correct_verified", None)):
        reasons.append("sign_correct_verified is not the literal True")

    # 9) Frozen identity fields are mandatory for auditability. The exact S3
    # key/table id disambiguates runs that contain multiple .emdfrzn blobs.
    if not str(verdict.get("emd_frozen_s3", "")).startswith("s3://"):
        reasons.append("emd_frozen_s3 missing")
    if not str(verdict.get("emd_frozen_table_id", "")):
        reasons.append("emd_frozen_table_id missing")
    frozen_sha = str(verdict.get("emd_frozen_sha256", ""))
    if len(frozen_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in frozen_sha):
        reasons.append("emd_frozen_sha256 missing")

    # 10) The eval verdict must prove it was run against the same frozen table
    # identity stamped into the staged serve artifact.
    if not _is_literal_true(verdict.get("serve_artifact_match", None)):
        reasons.append("serve_artifact_match is not the literal True")

    return (len(reasons) == 0, reasons)


# --- CLI ------------------------------------------------------------------


def _load_verdict(path: str) -> dict:
    with open(path, "r", encoding="ascii") as f:
        return json.load(f)


def _cli_check(args: argparse.Namespace) -> int:
    try:
        verdict = _load_verdict(args.verdict)
    except (OSError, ValueError) as exc:
        print(
            "FAIL: cannot load verdict JSON ({0}): {1}".format(
                args.verdict, exc
            )
        )
        return 1
    kwargs: dict[str, Any] = {}
    if args.k_max is not None:
        kwargs["k_max"] = args.k_max
    if args.anchor_lo is not None:
        kwargs["anchor_lo"] = args.anchor_lo
    if args.anchor_hi is not None:
        kwargs["anchor_hi"] = args.anchor_hi
    if args.sb_min is not None:
        kwargs["sb_min"] = args.sb_min
    if args.bb_min is not None:
        kwargs["bb_min"] = args.bb_min
    if args.blind_mean_min is not None:
        kwargs["blind_mean_min"] = args.blind_mean_min
    if args.global_reg_max is not None:
        kwargs["global_reg_max"] = args.global_reg_max
    passes, reasons = evaluate_gate(verdict, **kwargs)
    if passes:
        print("PASS")
        return 0
    print("FAIL: {0}".format("; ".join(reasons)))
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="_emd_auto_activate_gate",
        description=(
            "Pluribus-grade auto-activate gate (single source of truth, "
            "mirrors scripts/emd_pipeline_babysitter_v2.ps1 Test-AutoActivateGate)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser(
        "check",
        help=(
            "Print PASS or FAIL: <reasons> for a verdict JSON. "
            "Exit 0 on PASS, 1 on FAIL."
        ),
    )
    p_check.add_argument(
        "--verdict", required=True,
        help="Path to the verdict.json produced by eval_emd_vs_iter62000.py.",
    )
    p_check.add_argument("--k-max", type=int, default=None)
    p_check.add_argument("--anchor-lo", type=float, default=None)
    p_check.add_argument("--anchor-hi", type=float, default=None)
    p_check.add_argument("--sb-min", type=float, default=None)
    p_check.add_argument("--bb-min", type=float, default=None)
    p_check.add_argument("--blind-mean-min", type=float, default=None)
    p_check.add_argument("--global-reg-max", type=float, default=None)
    p_check.set_defaults(func=_cli_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
