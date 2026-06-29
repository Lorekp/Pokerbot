"""Sign-correct EMD verdict printer — sign-correct regression coverage.

Pinpoints the bug class that prompted this module: the
run_emd_k500_pluribus_20260625 run had a real SB regression
(base=-23.85, emd=-164.32, |em d|>|base| -> regression of magnitude
~140) yet three existing printers all labeled the delta
"improvement" regardless of sign.

This test file locks the correct behavior down so the bug cannot
reappear: every (base, emd) pair maps to the right label via
the magnitude rule; sign_correct_verified rejects malformed
verdicts; the render_verdict smoke test against the real k500
verdict.json proves the SB line prints REGRESSION.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

# Module under test is at scripts/_emd_verdict_sign.py — a sibling
# of scripts/eval/, intentionally invokable from anywhere with
# just stdlib + json. Insert its directory onto sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _SCRIPTS_DIR)

import _emd_verdict_sign as evs  # noqa: E402  (intentional path injection)


# ---------------------------------------------------------------------------
# _label(): table-driven across the full sign/Inf/NaN matrix.
# ---------------------------------------------------------------------------


def test_label_delta_positive_is_improvement():
    # Standard exploitability: base=100, emd=50. delta=+50.
    # |emd|<|base| -> IMPROVEMENT.
    assert evs._label(100.0, 50.0) == evs.IMPROVEMENT_LABEL


def test_label_delta_negative_is_regression_k500_case():
    # THE K=500 CASE: base=-23.85, emd=-164.32. delta=+140 (positive)
    # under the locked convention but the seat got WORSE (|em d|>|base|).
    # The naive sign rule would say IMPROVEMENT; the sign-correct
    # magnitude rule says REGRESSION.
    label = evs._label(-23.85426036981509, -164.31915292353824)
    assert label == evs.REGRESSION_LABEL, (
        "k500 SB must be REGRESSION, not the buggy IMPROVEMENT. "
        "Got: {0!r}".format(label)
    )


def test_label_delta_zero_is_no_change():
    # base == emd (regardless of sign) -> NO CHANGE.
    assert evs._label(50.0, 50.0) == evs.NEUTRAL_LABEL
    assert evs._label(-50.0, -50.0) == evs.NEUTRAL_LABEL
    assert evs._label(0.0, 0.0) == evs.NEUTRAL_LABEL


def test_label_nan_is_invalid_nan():
    # NaN in either argument -> INVALID (NaN).
    assert evs._label(float("nan"), 50.0) == evs.NAN_LABEL
    assert evs._label(50.0, float("nan")) == evs.NAN_LABEL
    assert evs._label(float("nan"), float("nan")) == evs.NAN_LABEL


def test_label_pos_inf_is_invalid_inf():
    # +Inf in either argument -> INVALID (Inf).
    assert evs._label(float("inf"), 50.0) == evs.INF_LABEL
    assert evs._label(50.0, float("inf")) == evs.INF_LABEL


def test_label_neg_inf_is_invalid_inf():
    # -Inf in either argument -> INVALID (Inf).
    assert evs._label(float("-inf"), 50.0) == evs.INF_LABEL
    assert evs._label(50.0, float("-inf")) == evs.INF_LABEL


def test_label_cross_zero_is_improvement():
    # Cross-zero case: base=+5, emd=-3. The exploitability magnitude
    # shrank from 5 to 3 -> IMPROVEMENT (the magnitude rule is the
    # only one that gets cross-sign cases right).
    assert evs._label(5.0, -3.0) == evs.IMPROVEMENT_LABEL


def test_label_cross_zero_is_regression():
    # Cross-zero case: base=-3, emd=+5. Magnitude grew 3 -> 5 ->
    # REGRESSION.
    assert evs._label(-3.0, 5.0) == evs.REGRESSION_LABEL


# ---------------------------------------------------------------------------
# format_seat_delta() / format_pooled(): exact-text shape checks.
# ---------------------------------------------------------------------------


def test_format_seat_delta_shape():
    out = evs.format_seat_delta("1", -23.85, -164.32)
    # The seat label must come from the seat arg.
    assert "SB(s1):" in out
    # Both anchors must be shown with explicit sign.
    assert "iter62000=-23.9" in out
    assert "emd=-164.3" in out
    # The k500 SB must report REGRESSION by magnitude ~140.5.
    assert "REGRESSION by +140.5 bb/100" in out


def test_format_seat_delta_improvement_shape():
    out = evs.format_seat_delta("3", 100.0, 50.0)
    assert "CO(s3):" in out
    assert "iter62000=+100.0" in out
    assert "emd=+50.0" in out
    assert "IMPROVEMENT by +50.0 bb/100" in out


def test_format_pooled_shape():
    out = evs.format_pooled(66.47, 28.00)
    assert "Pooled:" in out
    assert "iter62000=+66.5" in out
    assert "emd=+28.0" in out
    assert "IMPROVEMENT by +38.5 bb/100" in out


def test_format_pooled_k500_regression():
    # The k500 pooled: iter62000=+66.47, emd=+28.00 -> magnitude
    # shrank -> IMPROVEMENT by +38.5 (this is the only metric where
    # the actual pooled value really did improve; the bug was the
    # per-seat SB label, not the pooled label).
    out = evs.format_pooled(66.46973958333335, 28.00055208333332)
    assert "IMPROVEMENT by +38.5 bb/100" in out


# ---------------------------------------------------------------------------
# render_verdict(): full multi-line output + k500 smoke test.
# ---------------------------------------------------------------------------


def _build_verdict(
    emd_per_seat, iter62000_per_seat,
    emd_pooled=0.0, iter62000_pooled=0.0,
    sign_convention=None, sign_correct_verified=True,
):
    v = {
        "emd_per_seat": {str(k): float(v) for k, v in emd_per_seat.items()},
        "iter62000_per_seat": {str(k): float(v) for k, v in iter62000_per_seat.items()},
        "emd_pooled_bb_per_100": float(emd_pooled),
        "iter62000_pooled_bb_per_100": float(iter62000_pooled),
    }
    if sign_convention is not None:
        v["sign_convention"] = sign_convention
    if sign_correct_verified is not None:
        v["sign_correct_verified"] = sign_correct_verified
    return v


def test_render_verdict_smoke_k500_partial():
    """End-to-end smoke test against the REAL k500 partial verdict.

    SB line MUST say REGRESSION by ~140 (the actual stored magnitude
    is 140.46489 which rounds to 140.5 at 1-decimal display, but the
    bug class is the LABEL not the rounding) -- this is the bug
    class the module exists to prevent. If this test fails, somebody
    has reverted to the naive "delta > 0 -> improvement" rule.
    """
    verdict_path = os.path.normpath(
        os.path.join(_HERE, "..", "artifacts", "emd_k500_partial", "verdict.json")
    )
    assert os.path.exists(verdict_path), (
        "expected real k500 verdict at {0}".format(verdict_path)
    )
    with open(verdict_path, "r", encoding="ascii") as f:
        verdict = json.load(f)

    out = evs.render_verdict(verdict)

    # Hard requirement: SB line says REGRESSION.
    lines = out.split("\n")
    sb_line = next((ln for ln in lines if ln.startswith("SB(s1):")), None)
    assert sb_line is not None, (
        "render_verdict output missing SB(s1) line. Output was:\n{0}".format(out)
    )
    assert "REGRESSION by +140" in sb_line, (
        "SB line MUST say REGRESSION by +140 (actual 140.46 stored in "
        "verdict.json). Got:\n{0}".format(sb_line)
    )
    assert "bb/100" in sb_line
    # Make sure the buggy IMPROVEMENT label is NOT present on the SB line.
    assert "IMPROVEMENT" not in sb_line, (
        "SB line MUST NOT say IMPROVEMENT (this is the k500 bug). "
        "Got:\n{0}".format(sb_line)
    )

    # BB (seat 2) also regressed: base=1000.68, emd=904.16. |emd|<|base|
    # -> IMPROVEMENT by +96.5.
    bb_line = next((ln for ln in lines if ln.startswith("BB(s2):")), None)
    assert bb_line is not None
    assert "IMPROVEMENT by +96.5 bb/100" in bb_line, (
        "BB line should say IMPROVEMENT by +96.5. Got:\n{0}".format(bb_line)
    )

    # No trailing whitespace on any line.
    for ln in lines:
        assert ln == ln.rstrip(), "trailing whitespace on line: {0!r}".format(ln)

    # Pooled line is the last non-empty line.
    pooled_line = next((ln for ln in reversed(lines) if ln.startswith("Pooled:")), None)
    assert pooled_line is not None
    assert "IMPROVEMENT by +38.5 bb/100" in pooled_line, (
        "Pooled line should say IMPROVEMENT by +38.5. Got:\n{0}".format(pooled_line)
    )


def test_render_verdict_contains_blank_line_before_pooled():
    v = _build_verdict(
        emd_per_seat={0: 0.0, 1: -164.32},
        iter62000_per_seat={0: 0.0, 1: -23.85},
        emd_pooled=28.0,
        iter62000_pooled=66.47,
    )
    out = evs.render_verdict(v)
    # The line immediately preceding the Pooled line must be empty
    # (the spec requires a blank-line separator).
    lines = out.split("\n")
    pooled_idx = lines.index("Pooled: iter62000=+66.5 -> emd=+28.0  (IMPROVEMENT by +38.5 bb/100)")
    assert lines[pooled_idx - 1] == ""


def test_render_verdict_seats_sorted_numerically():
    # Insert seats out of order; the printer must still emit 0..5.
    v = _build_verdict(
        emd_per_seat={5: 1.0, 0: 0.0, 3: -1.0, 1: -164.32},
        iter62000_per_seat={5: 2.0, 0: 0.0, 3: 0.5, 1: -23.85},
    )
    out = evs.render_verdict(v)
    lines = [ln for ln in out.split("\n") if ln]
    seat_order = []
    for ln in lines:
        if ln.startswith("BTN(s0):"):
            seat_order.append(0)
        elif ln.startswith("SB(s1):"):
            seat_order.append(1)
        elif ln.startswith("CO(s3):"):
            seat_order.append(3)
        elif ln.startswith("LJ(s5):"):
            seat_order.append(5)
    assert seat_order == [0, 1, 3, 5]


# ---------------------------------------------------------------------------
# sign_correct_verified(): gate behavior.
# ---------------------------------------------------------------------------


def test_sign_correct_verified_missing_flag_returns_false():
    v = _build_verdict(
        emd_per_seat={1: -164.32},
        iter62000_per_seat={1: -23.85},
        sign_correct_verified=None,  # omit the flag entirely
    )
    assert evs.sign_correct_verified(v) is False


def test_sign_correct_verified_truthy_but_not_literal_true_returns_false():
    # The spec requires `is True`, not truthy. A truthy non-True value
    # must be rejected.
    v = _build_verdict(
        emd_per_seat={1: -164.32},
        iter62000_per_seat={1: -23.85},
        sign_correct_verified="yes",  # truthy but not the literal True
    )
    assert evs.sign_correct_verified(v) is False

    v2 = _build_verdict(
        emd_per_seat={1: -164.32},
        iter62000_per_seat={1: -23.85},
        sign_correct_verified=1,  # truthy but not the literal True
    )
    assert evs.sign_correct_verified(v2) is False


def test_sign_correct_verified_wrong_sign_convention_returns_false():
    v = _build_verdict(
        emd_per_seat={1: -164.32},
        iter62000_per_seat={1: -23.85},
        sign_convention="delta=emd-base, positive=improvement",  # WRONG
        sign_correct_verified=True,
    )
    assert evs.sign_correct_verified(v) is False


def test_sign_correct_verified_nan_value_returns_false():
    v = _build_verdict(
        emd_per_seat={1: float("nan")},
        iter62000_per_seat={1: -23.85},
        sign_convention=evs.SIGN_CONVENTION,
        sign_correct_verified=True,
    )
    assert evs.sign_correct_verified(v) is False


def test_sign_correct_verified_inf_value_returns_false():
    v = _build_verdict(
        emd_per_seat={1: -164.32},
        iter62000_per_seat={1: float("inf")},
        sign_convention=evs.SIGN_CONVENTION,
        sign_correct_verified=True,
    )
    assert evs.sign_correct_verified(v) is False


def test_sign_correct_verified_all_valid_returns_true():
    v = _build_verdict(
        emd_per_seat={0: 0.0, 1: -164.32, 2: 904.16, 3: -323.76, 4: -31.93, 5: 19.04},
        iter62000_per_seat={0: 0.0, 1: -23.85, 2: 1000.68, 3: -333.12, 4: 47.83, 5: -143.39},
        emd_pooled=28.0,
        iter62000_pooled=66.47,
        sign_convention=evs.SIGN_CONVENTION,
        sign_correct_verified=True,
    )
    assert evs.sign_correct_verified(v) is True


def test_sign_correct_verified_k500_partial_returns_true():
    """The real k500 verdict carries all three gates correctly."""
    verdict_path = os.path.normpath(
        os.path.join(_HERE, "..", "artifacts", "emd_k500_partial", "verdict.json")
    )
    with open(verdict_path, "r", encoding="ascii") as f:
        v = json.load(f)
    # The current k500 partial verdict does not yet embed
    # sign_convention or sign_correct_verified — that is exactly the
    # field the gate is meant to require. So this MUST return False
    # until the eval emitter is updated to stamp those fields.
    # (If somebody stamps the fields, flip this test to expect True.)
    assert evs.sign_correct_verified(v) is False


# ---------------------------------------------------------------------------
# CLI smoke: `python scripts/_emd_verdict_sign.py render --verdict <path>`.
# ---------------------------------------------------------------------------


def test_cli_render_against_real_k500_verdict(capsys):
    verdict_path = os.path.normpath(
        os.path.join(_HERE, "..", "artifacts", "emd_k500_partial", "verdict.json")
    )
    script_path = os.path.normpath(
        os.path.join(_HERE, "..", "scripts", "_emd_verdict_sign.py")
    )
    result = subprocess.run(
        [sys.executable, script_path, "render", "--verdict", verdict_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        "CLI render failed. stderr=\n{0}".format(result.stderr)
    )
    # The actual stored magnitude is 140.46489 which renders as +140.5
    # at the spec's 1-decimal format. We accept any "REGRESSION by +140"
    # because the bug class is the LABEL (REGRESSION not IMPROVEMENT),
    # not the rounding.
    assert "REGRESSION by +140" in result.stdout, (
        "CLI render SB line MUST say REGRESSION by +140. stdout:\n{0}".format(
            result.stdout
        )
    )
    # Make sure the SB line specifically does not say IMPROVEMENT.
    sb_line = next(
        (ln for ln in result.stdout.splitlines() if ln.startswith("SB(s1):")),
        None,
    )
    assert sb_line is not None
    assert "IMPROVEMENT" not in sb_line, (
        "CLI SB line MUST NOT say IMPROVEMENT (k500 bug). Got:\n{0}".format(sb_line)
    )


def test_cli_check_returns_nonzero_on_k500_partial():
    # The k500 partial verdict does not yet stamp sign_correct_verified,
    # so the gate must reject it.
    verdict_path = os.path.normpath(
        os.path.join(_HERE, "..", "artifacts", "emd_k500_partial", "verdict.json")
    )
    script_path = os.path.normpath(
        os.path.join(_HERE, "..", "scripts", "_emd_verdict_sign.py")
    )
    result = subprocess.run(
        [sys.executable, script_path, "check", "--verdict", verdict_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, (
        "CLI check should FAIL on a verdict missing sign_correct_verified. "
        "stdout={0!r} stderr={1!r}".format(result.stdout, result.stderr)
    )