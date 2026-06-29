"""Selection gate + known-good/known-bad control tests for cheap pilots."""



from __future__ import annotations



import json

import subprocess

import sys

from pathlib import Path



import pytest



_REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import _emd_selection_gate as sel  # noqa: E402

import _emd_auto_activate_gate as act  # noqa: E402



_SEL_PY = _REPO_ROOT / "scripts" / "_emd_selection_gate.py"





def _action_sets(

    emd_ps: dict | None = None,

    base_ps: dict | None = None,

    *,

    off_tree_emd: dict | None = None,

) -> dict:

    emd_ps = emd_ps or {"1": 70.0, "2": 80.0}

    base_ps = base_ps or {"1": 100.0, "2": 110.0}

    off_emd = off_tree_emd or {"1": 72.0, "2": 82.0}

    return {

        "in_tree": {"emd_per_seat": dict(emd_ps), "iter62000_per_seat": dict(base_ps), "bet_candidates": [2, 3, 4]},

        "pot_fraction": {"emd_per_seat": dict(emd_ps), "iter62000_per_seat": dict(base_ps), "bet_candidates": [2, 3]},

        "off_tree_adversarial": {

            "emd_per_seat": dict(off_emd),

            "iter62000_per_seat": dict(base_ps),

            "bet_candidates": [5, 6],

        },

    }





def _clean_verdict() -> dict:

    v = {

        "k_centroids": 200,

        "anchor_ok": True,

        "emd_anchor_bb_per_100": 125.0,

        "iter62000_anchor_bb_per_100": 125.0,

        "emd_per_seat": {"1": 70.0, "2": 80.0},

        "iter62000_per_seat": {"1": 100.0, "2": 110.0},

        "emd_pooled_bb_per_100": 90.0,

        "iter62000_pooled_bb_per_100": 100.0,

        "blind_delta_s1": 999.0,

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

        "seeds": [0, 1],

        "hands_per_seed": 1000,

        "crn": True,

        "lbr_action_sets": _action_sets(),

    }

    return v





def _k500_bad_verdict() -> dict:

    v = _clean_verdict()

    v["emd_per_seat"] = {"1": -164.32, "2": 904.16}

    v["iter62000_per_seat"] = {"1": -23.85, "2": 1000.68}

    v["blind_delta_s1"] = 140.46

    v["mean_blind_gain"] = 118.49

    v["lbr_action_sets"] = _action_sets(

        emd_ps={"1": -164.32, "2": 904.16},

        base_ps={"1": -23.85, "2": 1000.68},

        off_tree_emd={"1": -170.0, "2": 910.0},

    )

    return v





def _paired_partials_tight() -> tuple[dict, dict]:

    emd_p = {

        "result": {

            "per_seed_per_seat": [{"1": 70.0, "2": 80.0}, {"1": 72.0, "2": 78.0}],

        }

    }

    base_p = {

        "result": {

            "per_seed_per_seat": [{"1": 100.0, "2": 110.0}, {"1": 98.0, "2": 112.0}],

        }

    }

    return emd_p, base_p





def test_selection_passes_clean_at_cheap_tier():

    report = sel.build_selection_report(_clean_verdict(), tier="cheap")

    assert report["selection_pass"] is True

    assert report["acceptance_pass"] is True

    assert report["action_sets"]["complete"] is True


def test_selection_accepts_k160_only_when_abstraction_survivor_allows_it():

    v = _clean_verdict()

    v["k_centroids"] = 160

    report = sel.build_selection_report(v, tier="cheap", allowed_k_centroids={160})

    assert report["selection_pass"] is True

    assert report["acceptance_pass"] is True

    assert report["identity"]["k_policy"] == "accepted_abstraction_survivor"

    assert report["identity"]["expected_k_centroids"] == [160]


def test_selection_rejects_k_not_in_abstraction_survivors():

    v = _clean_verdict()

    v["k_centroids"] = 120

    report = sel.build_selection_report(v, tier="cheap", allowed_k_centroids={160})

    assert report["selection_pass"] is False

    assert "k_centroids_mismatch" in report["selection_reason_codes"]


def test_selection_rejects_k500_at_all_tiers():

    v = _k500_bad_verdict()

    for tier in ("heartbeat", "cheap", "medium", "final"):

        report = sel.build_selection_report(v, tier=tier)

        assert report["selection_pass"] is False, tier

        assert "sb_gain_below_tier_min" in report["selection_reason_codes"]





def test_selection_passes_synthetic_good_but_activation_requires_stricter():

    v = _clean_verdict()

    v["emd_per_seat"] = {"1": 95.0, "2": 100.0}

    v["iter62000_per_seat"] = {"1": 100.0, "2": 110.0}

    v["lbr_action_sets"] = _action_sets(emd_ps={"1": 95.0, "2": 100.0})

    report = sel.build_selection_report(v, tier="cheap")

    assert report["selection_pass"] is True

    assert report["acceptance_pass"] is False

    assert any("SB gain" in r for r in report["acceptance_reason_codes"])





def test_selection_fails_missing_raw_seats():

    v = _clean_verdict()

    del v["emd_per_seat"]

    report = sel.build_selection_report(v, tier="cheap")

    assert report["selection_pass"] is False

    assert "raw_per_seat_missing" in report["selection_reason_codes"]





def test_selection_fails_stale_positive_sb_delta():

    v = _k500_bad_verdict()

    passes, reasons = sel.evaluate_selection(v, tier="cheap")

    assert passes is False

    assert "sb_gain_below_tier_min" in reasons





def test_selection_fails_identity_bad_controls_on_cheap():

    v = _clean_verdict()

    v["k_centroids"] = 500

    v["sha256_stamped"] = False

    v["emd_frozen_s3"] = ""

    v["serve_artifact_match"] = False

    report = sel.build_selection_report(v, tier="cheap")

    assert report["selection_pass"] is False

    codes = report["selection_reason_codes"]

    assert "k_centroids_mismatch" in codes

    assert "sha256_stamped=False" in codes

    assert "emd_frozen_s3_missing" in codes

    assert "serve_artifact_match=False" in codes





def test_selection_fails_ci_lower95_on_medium():

    v = _clean_verdict()

    emd_p = {

        "result": {

            "per_seed_per_seat": [

                {"1": 95.0, "2": 80.0},

                {"1": 95.0, "2": 80.0},

                {"1": 110.0, "2": 80.0},

            ],

        }

    }

    base_p = {

        "result": {

            "per_seed_per_seat": [

                {"1": 100.0, "2": 110.0},

                {"1": 100.0, "2": 110.0},

                {"1": 100.0, "2": 110.0},

            ],

        }

    }

    report = sel.build_selection_report(v, tier="medium", emd_partial=emd_p, base_partial=base_p)

    assert report["selection_pass"] is False

    assert "paired_crn_sb_lower95_below_tier_min" in report["selection_reason_codes"]





def test_selection_fails_incomplete_action_sets():

    v = _clean_verdict()

    v["lbr_action_sets"] = {"in_tree": v["lbr_action_sets"]["in_tree"]}

    report = sel.build_selection_report(v, tier="cheap")

    assert report["selection_pass"] is False

    assert "action_sets_incomplete" in report["selection_reason_codes"]





def test_selection_fails_required_action_set_without_metric():

    v = _clean_verdict()

    v["lbr_action_sets"]["off_tree_adversarial"] = {}

    report = sel.build_selection_report(v, tier="cheap")

    assert report["selection_pass"] is False

    assert "action_sets_incomplete" in report["selection_reason_codes"]

    assert "action_set_invalid_off_tree_adversarial" in report["selection_reason_codes"]




def test_adversarial_probe_negative_ci_cannot_pass_medium():

    """Reproduce adversarial probe: point estimate ok, lower 95% bound negative."""

    v = _clean_verdict()

    emd_p = {

        "result": {

            "per_seed_per_seat": [{"1": 70.0, "2": 80.0}, {"1": 120.0, "2": 80.0}],

        }

    }

    base_p = {

        "result": {

            "per_seed_per_seat": [{"1": 100.0, "2": 110.0}, {"1": 100.0, "2": 110.0}],

        }

    }

    report = sel.build_selection_report(v, tier="medium", emd_partial=emd_p, base_partial=base_p)

    sb = report["paired_crn"]["sb"]

    assert sb is not None

    assert sb["mean"] == 5.0

    assert sb["ci95_lower"] < 0.0

    assert report["selection_pass"] is False





def test_activation_fails_sha_mismatch():

    v = _clean_verdict()

    v["sha256_stamped"] = False

    assert act.evaluate_gate(v)[0] is False





def test_activation_fails_serve_mismatch():

    v = _clean_verdict()

    v["serve_artifact_match"] = False

    assert act.evaluate_gate(v)[0] is False





def test_activation_fails_k500_k_centroids():

    v = _clean_verdict()

    v["k_centroids"] = 500

    assert act.evaluate_gate(v)[0] is False





def test_selection_report_includes_paired_crn_when_partials_present():

    emd_p, base_p = _paired_partials_tight()

    report = sel.build_selection_report(

        _clean_verdict(), tier="cheap", emd_partial=emd_p, base_partial=base_p,

    )

    assert report["paired_crn"]["sb"] is not None

    assert report["paired_crn"]["bb"] is not None

    assert "ci95_lower" in report["paired_crn"]["sb"]





def test_worst_case_action_set_used_when_divergence():

    v = _clean_verdict()

    v["lbr_action_sets"] = _action_sets(off_tree_emd={"1": 95.0, "2": 100.0})

    summary = sel.summarize_action_sets(v)

    assert summary["divergence_flag"] is True

    assert summary["worst_case_blind_mean_gain"] < summary["best_case_blind_mean_gain"]





def test_off_tree_worst_case_below_tier_min_fails_selection():

    v = _clean_verdict()

    v["lbr_action_sets"] = _action_sets(off_tree_emd={"1": 110.0, "2": 120.0})

    report = sel.build_selection_report(v, tier="cheap")

    assert report["action_sets"]["complete"] is True

    assert report["action_sets"]["worst_case_set"] == "off_tree_adversarial"

    assert report["selection_pass"] is False

    assert "worst_case_action_set_below_tier_min" in report["selection_reason_codes"]




def test_cli_report_writes_json(tmp_path):

    vpath = tmp_path / "verdict.json"

    out = tmp_path / "selection_report.json"

    vpath.write_text(json.dumps(_clean_verdict()), encoding="utf-8")

    proc = subprocess.run(

        [sys.executable, str(_SEL_PY), "report", "--verdict", str(vpath), "--out", str(out)],

        capture_output=True,

        text=True,

        timeout=30,

    )

    assert proc.returncode == 0

    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["schema"] == "emd_selection_report_v1"

    assert data["selection_pass"] is True

