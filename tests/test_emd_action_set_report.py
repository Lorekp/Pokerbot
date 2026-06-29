from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import _emd_selection_gate as sel  # noqa: E402
import emd_action_set_report as action_sets  # noqa: E402


def _partial(
    label: str,
    seat_base: float,
    *,
    action_set_name: str | None = None,
    bet_candidate_spec: str | None = None,
    bet_candidates: list[int] | None = None,
) -> dict:
    payload = {
        "label": label,
        "result": {
            "label": label,
            "method": "native_lbr6_" + label,
            "seeds": [101, 102, 103],
            "hands_per_seed": 1000,
            "per_seed_per_seat": [
                {"1": seat_base + 0.0, "2": seat_base + 10.0},
                {"1": seat_base + 2.0, "2": seat_base + 12.0},
                {"1": seat_base + 4.0, "2": seat_base + 14.0},
            ],
            "per_seat": {
                "1": {"bb_per_100": seat_base + 2.0},
                "2": {"bb_per_100": seat_base + 12.0},
            },
        },
    }
    if action_set_name is not None:
        payload["action_set_name"] = action_set_name
    if bet_candidate_spec is not None:
        payload["bet_candidate_spec"] = bet_candidate_spec
    if bet_candidates is not None:
        payload["bet_candidates"] = bet_candidates
    return payload


def _clean_verdict(lbr_action_sets: dict) -> dict:
    return {
        "k_centroids": 200,
        "anchor_ok": True,
        "emd_anchor_bb_per_100": 125.0,
        "iter62000_anchor_bb_per_100": 125.0,
        "emd_per_seat": {"1": 70.0, "2": 80.0},
        "iter62000_per_seat": {"1": 100.0, "2": 110.0},
        "emd_pooled_bb_per_100": 90.0,
        "iter62000_pooled_bb_per_100": 100.0,
        "sha256_stamped": True,
        "sign_correct_verified": True,
        "emd_frozen_s3": "s3://bucket/run/frozen/table123.emdfrzn",
        "emd_frozen_table_id": "table123",
        "emd_frozen_sha256": "a" * 64,
        "serve_artifact_match": True,
        "lbr_action_sets": lbr_action_sets,
    }


def test_action_set_entry_schema_from_real_partials():
    entry = action_sets.build_action_set_entry(
        name="pot_fraction",
        emd_partial=_partial("emd", 70.0),
        base_partial=_partial("iter62000", 100.0),
        bet_candidates=[0.33, 0.5, 0.75],
        emd_path="emd.json",
        base_path="base.json",
    )

    assert entry["emd_per_seat"] == {"1": 72.0, "2": 82.0}
    assert entry["iter62000_per_seat"] == {"1": 102.0, "2": 112.0}
    assert entry["bet_candidates"] == [0.33, 0.5, 0.75]
    assert entry["provenance"]["crn"] is True


def test_generated_action_sets_are_accepted_by_selection_gate():
    emd = _partial("emd", 70.0)
    base = _partial("iter62000", 100.0)
    generated = {
        "in_tree": action_sets.build_action_set_entry(
            name="in_tree",
            emd_partial=emd,
            base_partial=base,
            bet_candidates=["default_in_tree"],
            emd_path="emd_in_tree.json",
            base_path="base_in_tree.json",
        ),
        "pot_fraction": action_sets.build_action_set_entry(
            name="pot_fraction",
            emd_partial=emd,
            base_partial=base,
            bet_candidates=[0.33, 0.5, 0.75],
            emd_path="emd_pot.json",
            base_path="base_pot.json",
        ),
        "off_tree_adversarial": action_sets.build_action_set_entry(
            name="off_tree_adversarial",
            emd_partial=emd,
            base_partial=base,
            bet_candidates=[1.25, 2.0, "all_in"],
            emd_path="emd_off.json",
            base_path="base_off.json",
        ),
    }

    report = sel.build_selection_report(_clean_verdict(generated), tier="cheap")

    assert report["action_sets"]["complete"] is True
    assert report["selection_pass"] is True


def test_placeholder_action_set_fails_selection_gate():
    generated = {
        "in_tree": {"emd_per_seat": {"1": 70.0, "2": 80.0}, "iter62000_per_seat": {"1": 100.0, "2": 110.0}},
        "pot_fraction": {},
        "off_tree_adversarial": {"bet_candidates": [1.25, 2.0]},
    }

    report = sel.build_selection_report(_clean_verdict(generated), tier="cheap")

    assert report["selection_pass"] is False
    assert "action_sets_incomplete" in report["selection_reason_codes"]
    assert "action_set_invalid_pot_fraction" in report["selection_reason_codes"]
    assert "action_set_invalid_off_tree_adversarial" in report["selection_reason_codes"]


def test_cli_writes_action_set_json(tmp_path):
    emd_path = tmp_path / "emd.json"
    base_path = tmp_path / "base.json"
    out_path = tmp_path / "action_sets.json"
    emd_path.write_text(json.dumps(_partial("emd", 70.0)), encoding="utf-8")
    base_path.write_text(json.dumps(_partial("iter62000", 100.0)), encoding="utf-8")

    rc = action_sets.main([
        "--action-set", "pot_fraction", str(emd_path), str(base_path), "0.33,0.5,0.75",
        "--out", str(out_path),
    ])

    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert sorted(payload) == ["pot_fraction"]
    assert payload["pot_fraction"]["provenance"]["emd_json"] == str(emd_path)


def test_auto_packaging_uses_matching_partial_stamped_spec(tmp_path):
    spec = "0.3333333333333333,0.5,0.75,1.0,all_in"
    emd_path = tmp_path / "emd.json"
    base_path = tmp_path / "base.json"
    out_path = tmp_path / "action_sets.json"
    emd_path.write_text(
        json.dumps(_partial("emd", 70.0, action_set_name="pot_fraction",
                            bet_candidate_spec=spec, bet_candidates=[3, 4, 6, 7, 10])),
        encoding="utf-8",
    )
    base_path.write_text(
        json.dumps(_partial("iter62000", 100.0, action_set_name="pot_fraction",
                            bet_candidate_spec=spec, bet_candidates=[3, 4, 6, 7, 10])),
        encoding="utf-8",
    )

    rc = action_sets.main([
        "--action-set", "pot_fraction", str(emd_path), str(base_path),
        "--out", str(out_path),
    ])

    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    entry = payload["pot_fraction"]
    assert entry["bet_candidates"] == [1.0 / 3, 0.5, 0.75, 1.0, "all_in"]
    assert entry["provenance"]["bet_candidates_source"] == "partial_bet_candidate_spec"


def test_manual_tokens_must_match_partial_stamped_spec(tmp_path):
    spec = "0.3333333333333333,0.5,0.75,1.0,all_in"
    emd_path = tmp_path / "emd.json"
    base_path = tmp_path / "base.json"
    emd_path.write_text(
        json.dumps(_partial("emd", 70.0, action_set_name="pot_fraction",
                            bet_candidate_spec=spec, bet_candidates=[3, 4, 6, 7, 10])),
        encoding="utf-8",
    )
    base_path.write_text(
        json.dumps(_partial("iter62000", 100.0, action_set_name="pot_fraction",
                            bet_candidate_spec=spec, bet_candidates=[3, 4, 6, 7, 10])),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manual bet candidates"):
        action_sets.build_action_sets([
            ["pot_fraction", str(emd_path), str(base_path), "0.33,0.5,0.75"],
        ])


def test_emd_base_stamped_spec_mismatch_fails(tmp_path):
    emd_path = tmp_path / "emd.json"
    base_path = tmp_path / "base.json"
    emd_path.write_text(
        json.dumps(_partial("emd", 70.0, action_set_name="off_tree_adversarial",
                            bet_candidate_spec="0.25,0.9,1.25,2.5,all_in",
                            bet_candidates=[3, 7, 8, 9, 10])),
        encoding="utf-8",
    )
    base_path.write_text(
        json.dumps(_partial("iter62000", 100.0, action_set_name="off_tree_adversarial",
                            bet_candidate_spec="0.25,0.9,1.25,2.0,all_in",
                            bet_candidates=[3, 7, 8, 9, 10])),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bet_candidate_spec differ"):
        action_sets.build_action_sets([
            ["off_tree_adversarial", str(emd_path), str(base_path)],
        ])
