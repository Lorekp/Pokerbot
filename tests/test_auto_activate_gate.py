"""Pluribus-grade auto-activate gate regression tests (v2 babysitter + watcher).

Background: scripts/emd_pipeline_babysitter_v2.ps1 and
scripts/watch_emd_pipeline_v2.ps1 ship a NEW auto-activate gate that
replaces the legacy emd_better && anchor_ok short-circuit. The gate
enforces a full pluribus-grade checklist:

    1. k_centroids == K_MAX (200)
    2. emd_anchor_bb_per_100 AND iter62000_anchor_bb_per_100 in [110, 145]
    3. blind_delta_s1 >= 15   (SB gain)
    4. blind_delta_s2 >= 15   (BB gain)
    5. mean_blind_gain >= 15  (blind mean)
    6. global_regression <= 10 (positive = regression)
    7. sha256_stamped is the literal True
    8. sign_correct_verified is the literal True

The k500 incident (run_emd_k500_pluribus_20260625) is the canonical
counterexample: K=500 overfit, the per-seat printer sign-flipped the
SB regression (-23.85 -> -164.32) as an "improvement", emd_better
was True, and the legacy short-circuit would have auto-activated a
blueprint that was actually 7x WORSE in the SB. These tests pin down
that the v2 gate refuses the k500 verdict (and any future verdict
where the literal-positive 'improvement' label masks a real
regression on the per-seat LBR).

We test the Python helper (scripts/_emd_auto_activate_gate.py) via
subprocess and via direct import. The PowerShell v2 scripts shell
out to the SAME helper, so the gate logic is single-sourced.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the script under test. Resolved absolute so the subprocess inherits
# the repo layout regardless of pytest's CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_GATE_PY = _REPO_ROOT / "scripts" / "_emd_auto_activate_gate.py"
_K500_VERDICT = _REPO_ROOT / "artifacts" / "emd_k500_partial" / "verdict.json"

# Re-export the helper for direct-import tests. The helper is pure stdlib + json.
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import _emd_auto_activate_gate as _gate  # noqa: E402


# --- HELPERS ---------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run scripts/_emd_auto_activate_gate.py with the given CLI args.

    Uses the same Python interpreter as the test (sys.executable).
    """
    return subprocess.run(
        [sys.executable, str(_GATE_PY), "check", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _make_clean_verdict() -> dict:
    """A synthetic verdict that passes every gate condition.

    Each numeric is set to comfortably pass: K=200, anchors in band,
    SB/BB/mean gains well above the 15 bb/100 floor, global regression
    well below the 10 bb/100 cap, sign-correct verified, SHA256 stamped.
    """
    return {
        "k_centroids": 200,
        "emd_anchor_bb_per_100": 125.0,
        "iter62000_anchor_bb_per_100": 125.0,
        "anchor_target": 125.0,
        "emd_per_seat": {"1": 70.0, "2": 80.0},
        "iter62000_per_seat": {"1": 100.0, "2": 110.0},
        "emd_pooled_bb_per_100": 90.0,
        "iter62000_pooled_bb_per_100": 100.0,
        "blind_delta_s1": 999.0,  # stale stamped fields must be ignored
        "blind_delta_s2": 999.0,
        "mean_blind_gain": 999.0,
        "global_regression": 999.0,
        "sha256_stamped": True,
        "sign_correct_verified": True,
        "sign_convention": "delta=base-emd, label=magnitude(|emd| vs |base|)",
        "emd_frozen_s3": "s3://bucket/run/frozen/table123.emdfrzn",
        "emd_frozen_table_id": "table123",
        "emd_frozen_sha256": "a" * 64,
        "serve_artifact_match": True,
    }


# --- DIRECT IMPORT TESTS --------------------------------------------------


def test_gate_passes_clean_verdict():
    """A synthetic clean verdict must pass every condition.

    Mirrors the v2 babysitter line at scripts/emd_pipeline_babysitter_v2.ps1:653
    (Test-AutoActivateGate -> all conditions pass -> flip live serve pointer).
    """
    passes, reasons = _gate.evaluate_gate(_make_clean_verdict())
    assert passes is True, (
        "Clean verdict must pass. Got reasons: {0!r}".format(reasons)
    )
    assert reasons == [], "PASS must return an empty reasons list"


def test_gate_accepts_k160_only_with_accepted_survivor_policy():
    v = _make_clean_verdict()
    v["k_centroids"] = 160

    passes, reasons = _gate.evaluate_gate(v, allowed_k_centroids={160})

    assert passes is True
    assert reasons == []


def test_gate_rejects_k_not_in_accepted_survivor_policy():
    v = _make_clean_verdict()
    v["k_centroids"] = 120

    passes, reasons = _gate.evaluate_gate(v, allowed_k_centroids={160})

    assert passes is False
    assert any("accepted survivor K set" in r for r in reasons)


def test_gate_fails_on_missing_sha256():
    """sha256_stamped=False must block the gate.

    Mirrors the v2 babysitter line at scripts/emd_pipeline_babysitter_v2.ps1:389
    ('GATE FAIL: sha256_stamped=False (the frozen table SHA256 sidecar is
    missing -- re-run precompute_emd_frozen_lookup.py)').
    """
    v = _make_clean_verdict()
    v["sha256_stamped"] = False
    passes, reasons = _gate.evaluate_gate(v)
    assert passes is False
    assert any("sha256_stamped" in r for r in reasons), (
        "Missing SHA256 must name sha256_stamped in the reasons. Got: {0!r}".format(reasons)
    )


def test_gate_fails_on_sign_not_verified():
    """sign_correct_verified=False (or missing) must block the gate.

    The lock is the LITERAL True (truthy is not enough -- this is the
    gate that prevents the k500 sign-flip bug from re-introducing
    itself). Mirrors the v2 babysitter check at
    scripts/emd_pipeline_babysitter_v2.ps1:399.
    """
    v = _make_clean_verdict()
    v["sign_correct_verified"] = False
    passes, reasons = _gate.evaluate_gate(v)
    assert passes is False
    assert any("sign_correct_verified" in r for r in reasons), (
        "Missing sign-correct verify must be reported. Got: {0!r}".format(reasons)
    )


def test_gate_rejects_truthy_but_not_literal_true():
    """A non-bool truthy value (e.g. 1, "true", 0.5) must NOT pass the literal-True
    checks. The k500 bug class is exactly this -- a printer that
    re-labels the literal sign of the delta; if the gate ever widens
    to truthy, the bug returns. Lock the strictness.
    """
    v = _make_clean_verdict()
    v["sign_correct_verified"] = 1
    v["sha256_stamped"] = 1
    passes, reasons = _gate.evaluate_gate(v)
    assert passes is False
    # Both must be reported.
    assert any("sign_correct_verified" in r for r in reasons)
    assert any("sha256_stamped" in r for r in reasons)


def test_gate_recomputes_and_rejects_k500_raw_positive_sb_delta():
    """The gate must reject the real K500 stale-verdict shape.

    K500 stamped ``blind_delta_s1=+140`` even though the underlying per-seat
    values moved from -23.85 to -164.32, so exploitability magnitude got worse
    by 140 bb/100. The gate must recompute from per-seat values and fail even
    if the raw field still says positive.
    """
    k500_replay = {
        "k_centroids": 200,  # patched to 200 (otherwise K!=K_MAX fires first)
        "emd_anchor_bb_per_100": 125.0,
        "iter62000_anchor_bb_per_100": 125.0,
        "anchor_target": 125.0,
        "emd_per_seat": {
            "1": -164.31915292353824,
            "2": 904.162856071964,
        },
        "iter62000_per_seat": {
            "1": -23.85426036981509,
            "2": 1000.6844077961019,
        },
        "blind_delta_s1": 140.46,  # stale raw field that used to pass
        "blind_delta_s2": 96.52,
        "mean_blind_gain": 118.49,  # mean still positive due to the BB gain
        "global_regression": -38.47,  # pooled went DOWN (less exploitability)
        "sha256_stamped": True,
        "sign_correct_verified": True,
        "sign_convention": "delta=base-emd, label=magnitude(|emd| vs |base|)",
    }
    passes, reasons = _gate.evaluate_gate(k500_replay)
    assert passes is False, (
        "k500 replay with stale positive SB delta must FAIL the v2 gate. "
        "Reasons: {0!r}".format(reasons)
    )
    sb_reasons = [r for r in reasons if r.startswith("SB gain")]
    assert sb_reasons, (
        "Gate must fail on the SB-gain condition. Got: {0!r}".format(reasons)
    )
    assert "-140" in sb_reasons[0] or "-140.5" in sb_reasons[0], (
        "SB-gain reason must include the recomputed negative magnitude. "
        "Got: {0!r}".format(sb_reasons[0])
    )
    assert "15" in sb_reasons[0], (
        "SB-gain reason must include the floor (15.0) for diagnosability. "
        "Got: {0!r}".format(sb_reasons[0])
    )


def test_gate_fails_when_k_centroids_is_500():
    """A k_centroids=500 verdict must fail the gate even if every OTHER
    condition passes. The K=500 ban is the headline fix for the
    k500 incident. Mirrors the v2 babysitter line at
    scripts/emd_pipeline_babysitter_v2.ps1:337 ('GATE FAIL: K=<k>
    != K_MAX=<KMax>').
    """
    v = _make_clean_verdict()
    v["k_centroids"] = 500
    passes, reasons = _gate.evaluate_gate(v)
    assert passes is False
    assert any("K=500" in r or "K_MAX" in r for r in reasons), (
        "K=500 must be reported as a gate failure. Got: {0!r}".format(reasons)
    )


def test_gate_fails_when_k_centroids_missing():
    """A verdict without k_centroids must fail (the field is required so
    the gate can verify the production K cap). Missing != 200, so
    the K check must report a clear reason.
    """
    v = _make_clean_verdict()
    del v["k_centroids"]
    passes, reasons = _gate.evaluate_gate(v)
    assert passes is False
    assert any("k_centroids" in r for r in reasons), (
        "Missing k_centroids must be reported. Got: {0!r}".format(reasons)
    )


def test_gate_fails_on_anchor_out_of_band():
    """Anchor out of the [110, 145] band on EITHER emd or iter62000
    must fail the gate. Mirrors scripts/emd_pipeline_babysitter_v2.ps1:347.
    """
    v = _make_clean_verdict()
    v["emd_anchor_bb_per_100"] = 50.0  # way below 110
    passes, reasons = _gate.evaluate_gate(v)
    assert passes is False
    assert any("anchor" in r for r in reasons), (
        "Out-of-band anchor must be reported. Got: {0!r}".format(reasons)
    )


def test_gate_fails_on_bb_below_min():
    """BB gain below 15 bb/100 must fail. Mirrors the SB equivalent test
    on scripts/emd_pipeline_babysitter_v2.ps1:362.
    """
    v = _make_clean_verdict()
    v["emd_per_seat"]["2"] = 100.0
    v["iter62000_per_seat"]["2"] = 110.0
    passes, reasons = _gate.evaluate_gate(v)
    assert passes is False
    assert any("BB gain" in r for r in reasons), (
        "BB-below-min must be reported. Got: {0!r}".format(reasons)
    )


def test_gate_fails_on_global_regression_exceeding_cap():
    """global_regression > 10 bb/100 (positive = regression) must fail.
    Mirrors scripts/emd_pipeline_babysitter_v2.ps1:379.
    """
    v = _make_clean_verdict()
    v["emd_pooled_bb_per_100"] = 125.0
    v["iter62000_pooled_bb_per_100"] = 100.0
    passes, reasons = _gate.evaluate_gate(v)
    assert passes is False
    assert any("global" in r and "regression" in r for r in reasons), (
        "Global regression over cap must be reported. Got: {0!r}".format(reasons)
    )


def test_gate_fails_when_raw_blind_values_missing():
    """Raw per-seat values are mandatory; stamped gains are not trusted.

    This fails closed on stale verdicts that carry only blind_delta fields.
    """
    v = _make_clean_verdict()
    del v["emd_per_seat"]
    passes, reasons = _gate.evaluate_gate(v)
    assert passes is False
    assert any("SB gain" in r for r in reasons), (
        "Missing raw SB values must be reported. Got: {0!r}".format(reasons)
    )
    assert any("BB gain" in r for r in reasons), (
        "Missing raw BB values must be reported. Got: {0!r}".format(reasons)
    )


# --- SUBPROCESS / CLI TESTS -----------------------------------------------


def test_cli_k500_replay_fails_with_sb_reason():
    """The CLI must refuse the k500 replay and surface the SB-gain
    reason in stderr/stdout even when the stale raw SB delta is positive.
    """
    if not _K500_VERDICT.exists():
        pytest.skip("k500 partial verdict not present at {0}".format(_K500_VERDICT))
    # Build a k500-replay verdict: load the actual k500 partial and patch only
    # the non-SB blockers so this specifically tests magnitude recomputation.
    raw = json.loads(_K500_VERDICT.read_text(encoding="ascii"))
    raw["k_centroids"] = 200
    raw["blind_delta_s1"] = 140.46
    # The k500 partial is missing sha256_stamped + sign_correct_verified;
    # backfill them with the locked-convention values so the SB check
    # is the one that fires.
    raw["sign_convention"] = "delta=base-emd, label=magnitude(|emd| vs |base|)"
    raw["sign_correct_verified"] = True
    raw["sha256_stamped"] = True
    replay_path = _REPO_ROOT / "artifacts" / "_test_k500_replay_verdict.json"
    replay_path.write_text(json.dumps(raw, indent=2), encoding="ascii")
    try:
        result = _run_cli("--verdict", str(replay_path))
        assert result.returncode == 1, (
            "k500 replay must FAIL the gate. Got rc={0}\nstdout: {1}\nstderr: {2}".format(
                result.returncode, result.stdout, result.stderr
            )
        )
        combined = result.stdout + result.stderr
        assert "SB gain" in combined, (
            "CLI output must name the failing SB-gain condition for diagnosability. "
            "Got: {0!r}".format(combined)
        )
        assert "-140" in combined, (
            "CLI output must include the negative magnitude. Got: {0!r}".format(combined)
        )
        assert "15" in combined, (
            "CLI output must include the floor (15). Got: {0!r}".format(combined)
        )
    finally:
        if replay_path.exists():
            replay_path.unlink()


def test_cli_clean_verdict_passes():
    """The CLI must print PASS and exit 0 for a clean verdict. This is
    the v2 babysitter's 'all conditions pass -> flip live serve pointer'
    success path (scripts/emd_pipeline_babysitter_v2.ps1:653).
    """
    v = _make_clean_verdict()
    p = _REPO_ROOT / "artifacts" / "_test_clean_verdict.json"
    p.write_text(json.dumps(v, indent=2), encoding="ascii")
    try:
        result = _run_cli("--verdict", str(p))
        assert result.returncode == 0, (
            "Clean verdict must PASS the gate. Got rc={0}\nstdout: {1}\nstderr: {2}".format(
                result.returncode, result.stdout, result.stderr
            )
        )
        assert "PASS" in result.stdout
    finally:
        if p.exists():
            p.unlink()


def test_cli_missing_sha256_fails():
    """sha256_stamped=False must cause the CLI to fail and name the
    sha256_stamped reason. Mirrors the watcher panel
    'auto-activate gate: FAIL: ...sha256_stamped=False' in
    scripts/watch_emd_pipeline_v2.ps1.
    """
    v = _make_clean_verdict()
    v["sha256_stamped"] = False
    p = _REPO_ROOT / "artifacts" / "_test_nosha256_verdict.json"
    p.write_text(json.dumps(v, indent=2), encoding="ascii")
    try:
        result = _run_cli("--verdict", str(p))
        assert result.returncode == 1
        combined = result.stdout + result.stderr
        assert "sha256_stamped" in combined
    finally:
        if p.exists():
            p.unlink()


def test_cli_sign_not_verified_fails():
    """sign_correct_verified=False must cause the CLI to fail and name
    the sign_correct_verified reason. Mirrors the watcher panel
    'auto-activate gate: FAIL: ...sign_correct_verified...' in
    scripts/watch_emd_pipeline_v2.ps1.
    """
    v = _make_clean_verdict()
    v["sign_correct_verified"] = False
    p = _REPO_ROOT / "artifacts" / "_test_nosignck_verdict.json"
    p.write_text(json.dumps(v, indent=2), encoding="ascii")
    try:
        result = _run_cli("--verdict", str(p))
        assert result.returncode == 1
        combined = result.stdout + result.stderr
        assert "sign_correct_verified" in combined
    finally:
        if p.exists():
            p.unlink()
