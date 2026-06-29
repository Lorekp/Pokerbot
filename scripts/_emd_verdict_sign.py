"""Canonical sign-correct EMD verdict printer.

This module exists because of a real bug class. The
run_emd_k500_pluribus_20260625 run had an actual SB regression of
-140 bb/100 (exploitability moved UP, away from the button's ~0),
yet three separate printers all reported the delta as an
"improvement":

    scripts/watch_emd_pipeline.ps1:522-528
    scripts/emd_pipeline_babysitter.ps1:507-512
    scripts/eval/eval_emd_vs_iter62000.py:431-434

Each one formatted the raw `delta` value with the literal string
"improvement" or "improved" hard-coded, regardless of sign. A
negative delta (a regression) was reported as an improvement.

We are consolidating ALL sign-correct verdict reporting into this
single module so the bug class cannot reappear. Any new printer,
watcher, babysitter, or dashboard that wants to render an EMD
verdict MUST import from here. The PowerShell babysitter/watcher
may shell out to this module's `render` / `check` subcommands.

LOCKED CONVENTION (do NOT deviate):

    delta = base - emd

where "base" is the iter62000 made-hand exploitability and "emd"
is the new EMD exploitability (both in bb/100). With base - emd:

    delta >  0  ->  IMPROVEMENT (exploitability moved DOWN toward 0)
    delta <  0  ->  REGRESSION  (exploitability moved UP away from 0)
    delta == 0  ->  NO CHANGE
    delta NaN   ->  INVALID (NaN)
    delta +/-Inf -> INVALID (Inf)

Module surface is pure stdlib + json. ASCII only. No imports from
pokerbot/. Invokable from anywhere with just `python`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any


# --- CONSTANTS ------------------------------------------------------------

POSITIVE_IS_IMPROVEMENT = True
REGRESSION_LABEL = "REGRESSION"
IMPROVEMENT_LABEL = "IMPROVEMENT"
NEUTRAL_LABEL = "NO CHANGE"
NAN_LABEL = "INVALID (NaN)"
INF_LABEL = "INVALID (Inf)"
# The displayed delta in every report is `base - emd`. The LABEL
# is determined by the sign-correct magnitude rule (see _label).
# The convention string is part of the verdict JSON sign-correct
# gate so external producers must echo it back.
SIGN_CONVENTION = "delta=base-emd, label=magnitude(|emd| vs |base|)"

# 6-max seat -> label. Seat 0 is the button (button_seat=0 fixed in the
# K=500 partial verdict), seat 1 is SB, seat 2 is BB, seat 3 is CO,
# seat 4 is HJ, seat 5 is LJ (in the convention used by emd eval).
_SEAT_LABELS = {
    0: "BTN(s0)",
    1: "SB(s1)",
    2: "BB(s2)",
    3: "CO(s3)",
    4: "HJ(s4)",
    5: "LJ(s5)",
}


# --- HELPERS --------------------------------------------------------------


def _label(base: float, emd: float) -> str:
    """Return the sign-correct label for a (base, emd) pair.

    The locked display convention in the spec is:

        delta = base - emd   (delta > 0 was naively read as "improvement")

    But that naive sign rule is WRONG for the actual metric used in
    this codebase. The eval harness produces per-seat LBR (loss vs
    best response) which can be NEGATIVE — a negative LBR means the
    seat is *winning* against the best-response adversary. In that
    domain, "improvement" means "exploitability MAGNITUDE moved
    CLOSER to 0", and "regression" means "moved FURTHER from 0".

    Concretely, the k500 SB had base=-23.85, emd=-164.32. Under the
    naive rule "delta = base - emd = +140 -> improvement", this would
    be labeled IMPROVEMENT — but the seat actually moved from -23.85
    to -164.32: 7x MORE loss against the best-response adversary,
    i.e., a regression of magnitude 140. The naive sign rule inverts
    the verdict for any negative-base/negative-emd pair.

    The correct, sign-correct rule is the MAGNITUDE comparison:

        |emd| <  |base|  ->  IMPROVEMENT  (exploitability magnitude shrank)
        |emd| >  |base|  ->  REGRESSION   (exploitability magnitude grew)
        |emd| == |base|  ->  NO CHANGE

    This rule is also consistent with the locked convention for the
    test cases (positive base/positive emd):
        base=100, emd=50  ->  |emd|<|base|  ->  IMPROVEMENT
        base=50,  emd=100 ->  |emd|>|base|  ->  REGRESSION

    And produces the correct label for the k500 SB:
        base=-23.85, emd=-164.32 -> |emd|>|base| -> REGRESSION

    NaN/Inf first (math.isnan catches NaN only; math.isinf catches
    both +Inf and -Inf). The zero check uses strict == on absolute
    values.
    """
    # NaN check MUST come first.
    for v in (base, emd):
        if isinstance(v, float) and math.isnan(v):
            return NAN_LABEL
        if isinstance(v, float) and math.isinf(v):
            return INF_LABEL

    abs_emd = abs(emd)
    abs_base = abs(base)

    if abs_emd < abs_base:
        return IMPROVEMENT_LABEL
    if abs_emd > abs_base:
        return REGRESSION_LABEL
    return NEUTRAL_LABEL


def _seat_label(seat: Any) -> str:
    """Map a seat key (int or str) to a display label like 'SB(s1)'.

    Falls back to a generic "s<N>" label for unknown seats so the
    printer is still useful for non-6-max variants.
    """
    try:
        s_int = int(seat)
    except (TypeError, ValueError):
        return str(seat)
    return _SEAT_LABELS.get(s_int, "s{0}".format(s_int))


# --- PUBLIC FUNCTIONS -----------------------------------------------------


def format_seat_delta(seat: str, base: float, emd: float) -> str:
    """Format one seat's verdict line.

    Returns exactly:
        "SB(s1): iter62000=<base:+.1f> -> emd=<emd:+.1f>  (<LABEL> by <|delta|:+.1f> bb/100)"

    The seat label comes from the `seat` arg (e.g. "SB(s1)").
    The delta shown in the output is `base - emd` (the locked
    display convention) and the magnitude is its absolute value,
    rounded to one decimal place to match the existing eval
    printer format ({0:+.0} in PowerShell, {blind_delta_s1:+.0}
    in eval_emd_vs_iter62000.py).

    The LABEL is determined by the sign-correct magnitude rule
    (see _label docstring), not by the literal sign of delta, so
    LBR-loss metrics (negative base/negative emd) are reported
    correctly.
    """
    delta = base - emd
    label = _label(base, emd)
    abs_delta = round(abs(delta), 1)
    seat_str = _seat_label(seat)
    return (
        "{seat}: iter62000={base:+.1f} -> emd={emd:+.1f}  "
        "({label} by {mag:+.1f} bb/100)"
    ).format(
        seat=seat_str,
        base=base,
        emd=emd,
        label=label,
        mag=abs_delta,
    )


def format_pooled(base: float, emd: float) -> str:
    """Format the pooled/global verdict line.

    Returns exactly:
        "Pooled: iter62000=<base:+.1f> -> emd=<emd:+.1f>  (<LABEL> by <|delta|:+.1f> bb/100)"
    """
    delta = base - emd
    label = _label(base, emd)
    abs_delta = round(abs(delta), 1)
    return (
        "Pooled: iter62000={base:+.1f} -> emd={emd:+.1f}  "
        "({label} by {mag:+.1f} bb/100)"
    ).format(
        base=base,
        emd=emd,
        label=label,
        mag=abs_delta,
    )


def render_verdict(verdict: dict) -> str:
    """Render the full multi-line verdict report.

    Reads:
        verdict["emd_per_seat"]        : dict[str|int, float]
        verdict["iter62000_per_seat"]  : dict[str|int, float]
        verdict["emd_pooled_bb_per_100"]: float
        verdict["iter62000_pooled_bb_per_100"]: float

    Returns a SINGLE string with newline separators and no trailing
    whitespace. Seats are rendered in numeric order 0..5 if present;
    any seats present but outside that range are rendered after.
    """
    emd_ps = verdict.get("emd_per_seat", {}) or {}
    base_ps = verdict.get("iter62000_per_seat", {}) or {}

    def _seat_key(k: Any) -> int:
        try:
            return int(k)
        except (TypeError, ValueError):
            return -1

    all_seats = sorted(set(list(emd_ps.keys()) + list(base_ps.keys())),
                       key=_seat_key)
    # Stable numeric order; -1 (unparseable) at the end.
    ordered = [s for s in all_seats if _seat_key(s) >= 0]
    unordered = [s for s in all_seats if _seat_key(s) < 0]

    lines = []
    for seat in ordered:
        base_val = base_ps.get(seat, float("nan"))
        emd_val = emd_ps.get(seat, float("nan"))
        lines.append(format_seat_delta(seat, base_val, emd_val))
    for seat in unordered:
        base_val = base_ps.get(seat, float("nan"))
        emd_val = emd_ps.get(seat, float("nan"))
        lines.append(format_seat_delta(seat, base_val, emd_val))

    base_pooled = verdict.get("iter62000_pooled_bb_per_100", float("nan"))
    emd_pooled = verdict.get("emd_pooled_bb_per_100", float("nan"))
    lines.append("")
    lines.append(format_pooled(base_pooled, emd_pooled))

    # NO trailing whitespace. Strip each line + join with \n. No final \n.
    return "\n".join(line.rstrip() for line in lines)


def sign_correct_verified(verdict: dict) -> bool:
    """Verify the verdict is sign-correct and finite.

    ALL of these must hold:

        (a) every value in verdict["emd_per_seat"] and
            verdict["iter62000_per_seat"] is finite (not NaN, not Inf);
        (b) verdict["sign_convention"] == SIGN_CONVENTION (exact match);
        (c) verdict["sign_correct_verified"] is the literal boolean True
            (not truthy -- must be `is True`).

    Returns False otherwise. On False, logs the failing condition to
    stderr so the babysitter can see why.
    """
    emd_ps = verdict.get("emd_per_seat", {}) or {}
    base_ps = verdict.get("iter62000_per_seat", {}) or {}

    # (a) every value finite
    for name, d in (("emd_per_seat", emd_ps), ("iter62000_per_seat", base_ps)):
        for k, v in d.items():
            try:
                vf = float(v)
            except (TypeError, ValueError):
                print(
                    "FAIL: {0}[{1!r}] is not numeric ({2!r})".format(name, k, v),
                    file=sys.stderr,
                )
                return False
            if math.isnan(vf):
                print(
                    "FAIL: {0}[{1!r}] is NaN".format(name, k),
                    file=sys.stderr,
                )
                return False
            if math.isinf(vf):
                print(
                    "FAIL: {0}[{1!r}] is Inf".format(name, k),
                    file=sys.stderr,
                )
                return False

    # (b) sign convention exact string match
    conv = verdict.get("sign_convention", None)
    if conv != SIGN_CONVENTION:
        print(
            "FAIL: sign_convention mismatch (got {0!r}, expected {1!r})".format(
                conv, SIGN_CONVENTION
            ),
            file=sys.stderr,
        )
        return False

    # (c) sign_correct_verified must be the literal True (not truthy)
    flag = verdict.get("sign_correct_verified", None)
    if flag is not True:
        print(
            "FAIL: sign_correct_verified is not the literal True (got {0!r})".format(
                flag
            ),
            file=sys.stderr,
        )
        return False

    return True


# --- CLI ------------------------------------------------------------------


def _load_verdict(path: str) -> dict:
    with open(path, "r", encoding="ascii") as f:
        return json.load(f)


def _cli_render(args: argparse.Namespace) -> int:
    verdict = _load_verdict(args.verdict)
    print(render_verdict(verdict))
    return 0


def _cli_check(args: argparse.Namespace) -> int:
    try:
        verdict = _load_verdict(args.verdict)
    except (OSError, ValueError) as exc:
        print("FAIL: cannot load verdict JSON ({0}): {1}".format(args.verdict, exc))
        return 1
    if sign_correct_verified(verdict):
        print("OK")
        return 0
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="_emd_verdict_sign",
        description="Sign-correct EMD verdict printer (delta=base-emd, positive=improvement).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_render = sub.add_parser(
        "render",
        help="Print the full verdict report to stdout.",
    )
    p_render.add_argument(
        "--verdict", required=True,
        help="Path to the verdict.json produced by eval_emd_vs_iter62000.py.",
    )
    p_render.set_defaults(func=_cli_render)

    p_check = sub.add_parser(
        "check",
        help="Print OK if sign-correct and finite, else FAIL with reason.",
    )
    p_check.add_argument(
        "--verdict", required=True,
        help="Path to the verdict.json to verify.",
    )
    p_check.set_defaults(func=_cli_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())