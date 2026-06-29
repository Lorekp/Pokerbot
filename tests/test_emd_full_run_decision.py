"""Full-run decision script tests."""



from __future__ import annotations



import json

import sys

from pathlib import Path



_REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import emd_full_run_decision as frd  # noqa: E402





def _survivor_sweep() -> dict:

    return {

        "survivors": [{"k": 160, "n_runouts": 200, "score": 0.01, "assignment_stability_mean": 0.9}],

    }





def _good_selection(*, tier: str = "medium") -> dict:

    return {

        "schema": "emd_selection_report_v1",

        "tier": tier,

        "selection_pass": True,

        "acceptance_pass": False,

        "selection_reason_codes": [],

        "paired_crn": {

            "sb": {

                "mean": 22.0,

                "ci95_lower": 20.0,

                "ci95_upper": 24.0,

                "ci95_half_width": 2.0,

            },

            "bb": {

                "mean": 24.0,

                "ci95_lower": 21.0,

                "ci95_upper": 27.0,

                "ci95_half_width": 3.0,

            },

            "ci_too_wide_for_decision": False,

        },

        "action_sets": {"complete": True, "worst_case_blind_mean_gain": 25.0},

        "identity": {"k_centroids": 160, "expected_k_centroids": [160]},

    }





def _good_pilot_verdict(*, pilot_run_id: str) -> dict:

    return {

        "artifact_role": "fixed_budget_pilot",

        "pilot_run_id": pilot_run_id,

        "k_centroids": 160,

        "anchor_ok": True,

        "emd_anchor_bb_per_100": 125.0,

        "iter62000_anchor_bb_per_100": 125.0,

        "emd_per_seat": {"1": 70.0, "2": 80.0},

        "iter62000_per_seat": {"1": 100.0, "2": 110.0},

        "emd_pooled_bb_per_100": 90.0,

        "iter62000_pooled_bb_per_100": 100.0,

        "sha256_stamped": True,

        "sign_correct_verified": True,

        "emd_frozen_s3": "s3://b/t",

        "emd_frozen_table_id": "t",

        "emd_frozen_sha256": "a" * 64,

        "serve_artifact_match": True,

        "lbr_action_sets": {

            "in_tree": {"emd_per_seat": {"1": 70.0, "2": 80.0}, "iter62000_per_seat": {"1": 100.0, "2": 110.0}},

            "pot_fraction": {"emd_per_seat": {"1": 70.0, "2": 80.0}, "iter62000_per_seat": {"1": 100.0, "2": 110.0}},

            "off_tree_adversarial": {"emd_per_seat": {"1": 72.0, "2": 82.0}, "iter62000_per_seat": {"1": 100.0, "2": 110.0}},

        },

    }





def _write_pilot(art: Path, pilot_run_id: str) -> None:

    pilots = art / "pilots"

    pilots.mkdir(parents=True, exist_ok=True)

    path = pilots / f"verdict_{pilot_run_id}.json"

    path.write_text(json.dumps(_good_pilot_verdict(pilot_run_id=pilot_run_id)), encoding="utf-8")





def test_decision_blocks_without_survivors(tmp_path):

    d = frd.build_decision(

        artifacts_dir=tmp_path,

        abstraction_sweep={"survivors": []},

        selection_report=_good_selection(),

        pilot_verdicts=[_good_pilot_verdict(pilot_run_id="p1")],

    )

    assert d["recommend_launch"] is False

    assert "insufficient_abstraction_survivors" in d["blockers"]





def test_decision_blocks_without_selection(tmp_path):

    d = frd.build_decision(

        artifacts_dir=tmp_path,

        abstraction_sweep=_survivor_sweep(),

        selection_report=None,

        pilot_verdicts=[_good_pilot_verdict(pilot_run_id="p1")],

    )

    assert d["recommend_launch"] is False

    assert "missing_selection_report" in d["blockers"]





def test_decision_blocks_single_pilot(tmp_path):

    d = frd.build_decision(

        artifacts_dir=tmp_path,

        abstraction_sweep=_survivor_sweep(),

        selection_report=_good_selection(),

        pilot_verdicts=[_good_pilot_verdict(pilot_run_id="p1")],

        min_pilot_passes=2,

    )

    assert d["recommend_launch"] is False

    assert "insufficient_distinct_pilot_runs" in d["blockers"]





def test_decision_recommends_with_two_pilots(tmp_path):

    d = frd.build_decision(

        artifacts_dir=tmp_path,

        abstraction_sweep=_survivor_sweep(),

        selection_report=_good_selection(),

        pilot_verdicts=[

            _good_pilot_verdict(pilot_run_id="p1"),

            _good_pilot_verdict(pilot_run_id="p2"),

        ],

        min_pilot_passes=2,

        selection_tier="cheap",

    )

    assert d["recommend_launch"] is True

    assert d["blockers"] == []





def test_decision_blocks_cheap_selection_report_for_medium_policy(tmp_path):

    d = frd.build_decision(

        artifacts_dir=tmp_path,

        abstraction_sweep=_survivor_sweep(),

        selection_report=_good_selection(tier="cheap"),

        pilot_verdicts=[

            _good_pilot_verdict(pilot_run_id="p1"),

            _good_pilot_verdict(pilot_run_id="p2"),

        ],

        min_pilot_passes=2,

        selection_tier="medium",

    )

    assert d["recommend_launch"] is False

    assert "selection_report_tier_below_policy" in d["blockers"]




def test_decision_blocks_medium_selection_report_missing_ci(tmp_path):

    report = _good_selection(tier="medium")

    del report["paired_crn"]["sb"]

    d = frd.build_decision(

        artifacts_dir=tmp_path,

        abstraction_sweep=_survivor_sweep(),

        selection_report=report,

        pilot_verdicts=[

            _good_pilot_verdict(pilot_run_id="p1"),

            _good_pilot_verdict(pilot_run_id="p2"),

        ],

        min_pilot_passes=2,

        selection_tier="medium",

    )

    assert d["recommend_launch"] is False

    assert "selection_report_paired_crn_sb_missing" in d["blockers"]




def test_decision_blocks_medium_selection_report_bad_ci_policy(tmp_path):

    report = _good_selection(tier="medium")

    report["paired_crn"]["sb"]["ci95_lower"] = 0.0

    report["paired_crn"]["bb"]["ci95_half_width"] = 100.0

    d = frd.build_decision(

        artifacts_dir=tmp_path,

        abstraction_sweep=_survivor_sweep(),

        selection_report=report,

        pilot_verdicts=[

            _good_pilot_verdict(pilot_run_id="p1"),

            _good_pilot_verdict(pilot_run_id="p2"),

        ],

        min_pilot_passes=2,

        selection_tier="medium",

    )

    assert d["recommend_launch"] is False

    assert "selection_report_paired_crn_sb_lower95_below_policy" in d["blockers"]

    assert "selection_report_paired_crn_bb_half_width_wide" in d["blockers"]




def test_decision_recommends_with_medium_or_final_selection_report(tmp_path):

    for tier in ("medium", "final"):

        d = frd.build_decision(

            artifacts_dir=tmp_path,

            abstraction_sweep=_survivor_sweep(),

            selection_report=_good_selection(tier=tier),

            pilot_verdicts=[

                _good_pilot_verdict(pilot_run_id="p1"),

                _good_pilot_verdict(pilot_run_id="p2"),

            ],

            min_pilot_passes=2,

            selection_tier=tier,

        )

        assert d["recommend_launch"] is True, tier

        assert d["blockers"] == []


def test_decision_blocks_selection_k_not_accepted_survivor(tmp_path):

    report = _good_selection(tier="medium")

    report["identity"]["k_centroids"] = 200

    d = frd.build_decision(

        artifacts_dir=tmp_path,

        abstraction_sweep=_survivor_sweep(),

        selection_report=report,

        pilot_verdicts=[

            _good_pilot_verdict(pilot_run_id="p1"),

            _good_pilot_verdict(pilot_run_id="p2"),

        ],

        min_pilot_passes=2,

        selection_tier="medium",

    )

    assert d["recommend_launch"] is False

    assert "selection_k_not_accepted_survivor" in d["blockers"]


def test_decision_blocks_pilot_k_not_selected_survivor(tmp_path):

    bad_pilot = _good_pilot_verdict(pilot_run_id="p2")

    bad_pilot["k_centroids"] = 200

    d = frd.build_decision(

        artifacts_dir=tmp_path,

        abstraction_sweep=_survivor_sweep(),

        selection_report=_good_selection(tier="medium"),

        pilot_verdicts=[

            _good_pilot_verdict(pilot_run_id="p1"),

            bad_pilot,

        ],

        min_pilot_passes=2,

        selection_tier="medium",

    )

    assert d["recommend_launch"] is False

    assert "fixed_budget_pilot_not_repeatable" in d["blockers"]

    assert any("K=200 not in accepted survivor K set" in r for r in d["evidence"]["pilot_verdicts"][1]["selection_reason_codes"] + d["evidence"]["pilot_verdicts"][1].get("acceptance_reason_codes", []))




def test_root_verdict_not_used_as_pilot(tmp_path):

    art = tmp_path / "ladder"

    art.mkdir()

    (art / "abstraction_sweep.json").write_text(json.dumps(_survivor_sweep()), encoding="utf-8")

    (art / "eval_verdict.json").write_text(

        json.dumps({"selection": _good_selection()}), encoding="utf-8",

    )

    (art / "verdict.json").write_text(json.dumps(_good_pilot_verdict(pilot_run_id="root")), encoding="utf-8")

    pilots = frd._pilot_verdicts(art)

    assert pilots == []





def test_pilot_verdict_requires_explicit_pilot_run_id(tmp_path):

    art = tmp_path / "ladder"

    pilots_dir = art / "pilots"

    pilots_dir.mkdir(parents=True)

    verdict = _good_pilot_verdict(pilot_run_id="p1")

    del verdict["pilot_run_id"]

    (pilots_dir / "verdict_p1.json").write_text(json.dumps(verdict), encoding="utf-8")

    pilots = frd._pilot_verdicts(art)

    assert pilots == []




def test_cli_writes_json(tmp_path):

    art = tmp_path / "ladder"

    art.mkdir()

    (art / "abstraction_sweep.json").write_text(json.dumps(_survivor_sweep()), encoding="utf-8")

    (art / "eval_verdict.json").write_text(

        json.dumps({"selection": _good_selection()}), encoding="utf-8",

    )

    _write_pilot(art, "p1")

    out = art / "full_run_decision.json"

    rc = frd.main(["--artifacts-dir", str(art), "--out", str(out), "--min-pilot-passes", "2"])

    assert rc == 1

    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["schema"] == "emd_full_run_decision_v1"

    assert "insufficient_distinct_pilot_runs" in data["blockers"]

