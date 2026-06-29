from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "eval"))

import eval_emd_vs_iter62000 as eval_emd  # noqa: E402


def _partial(*, sb: float, bb: float, pooled: float) -> dict:
    return {
        "anchor": {
            "anchor_ok": True,
            "always_fold_bb_per_100": 125.0,
        },
        "result": {
            "per_seat": {
                "1": {"bb_per_100": sb},
                "2": {"bb_per_100": bb},
            },
            "pooled_bb_per_100": pooled,
            "seeds": [0],
            "hands_per_seed": 1,
            "value_only": True,
            "method": "native_lbr6_emd_frozen",
            "num_buckets": 200,
        },
    }


def test_combine_stamps_magnitude_correct_k500_negative_control(tmp_path):
    """K500's stale raw SB sign must not reappear in newly combined verdicts."""
    emd_json = tmp_path / "emd.json"
    base_json = tmp_path / "base.json"
    out_json = tmp_path / "verdict.json"
    emd_json.write_text(
        json.dumps(_partial(sb=-164.31915292353824, bb=904.162856071964, pooled=28.00055208333332)),
        encoding="utf-8",
    )
    base_json.write_text(
        json.dumps(_partial(sb=-23.85426036981509, bb=1000.6844077961019, pooled=66.46973958333335)),
        encoding="utf-8",
    )

    rc = eval_emd._combine(
        argparse.Namespace(
            emd_json=str(emd_json),
            iter62000_json=str(base_json),
            out=str(out_json),
            min_blind_gain=15.0,
            global_slack=10.0,
            num_buckets=200,
            emd_frozen="",
            s3_out="",
        )
    )

    assert rc == 0
    verdict = json.loads(out_json.read_text(encoding="utf-8"))
    assert verdict["blind_delta_s1"] < -140.0
    assert verdict["blind_delta_s2"] > 96.0
    assert verdict["mean_blind_gain"] < 0.0
    assert verdict["global_regression"] < 0.0
    assert verdict["blinds_improved"] is False
    assert verdict["emd_better"] is False


def test_combine_stamps_frozen_identity_and_serve_match(tmp_path):
    emd_json = tmp_path / "emd.json"
    base_json = tmp_path / "base.json"
    out_json = tmp_path / "verdict.json"
    frozen = tmp_path / "table123.emdfrzn"
    serve = tmp_path / "serve.pkl"
    frozen.write_bytes(b"frozen-table")
    sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    Path(str(frozen) + ".sha256").write_text(sha + "\n", encoding="ascii")
    frozen_s3 = "s3://bucket/run/frozen/table123.emdfrzn"
    with serve.open("wb") as f:
        pickle.dump(
            {
                "strat_sum": {},
                "abstraction": {
                    "postflop_emd_frozen_sha256": sha,
                    "postflop_emd_table_id": "table123",
                    "postflop_emd_frozen_s3": frozen_s3,
                },
            },
            f,
        )
    emd_json.write_text(json.dumps(_partial(sb=70.0, bb=80.0, pooled=90.0)), encoding="utf-8")
    base_json.write_text(json.dumps(_partial(sb=100.0, bb=110.0, pooled=100.0)), encoding="utf-8")

    rc = eval_emd._combine(
        argparse.Namespace(
            emd_json=str(emd_json),
            iter62000_json=str(base_json),
            out=str(out_json),
            min_blind_gain=15.0,
            global_slack=10.0,
            num_buckets=200,
            emd_frozen=str(frozen),
            emd_frozen_s3=frozen_s3,
            emd_table_id="table123",
            serve_blueprint=str(serve),
            serve_blueprint_s3="s3://bucket/run/serve/candidate.pkl",
            s3_out="",
        )
    )

    assert rc == 0
    verdict = json.loads(out_json.read_text(encoding="utf-8"))
    assert verdict["sha256_stamped"] is True
    assert verdict["emd_frozen_sha256"] == sha
    assert verdict["emd_frozen_s3"] == frozen_s3
    assert verdict["emd_frozen_table_id"] == "table123"
    assert verdict["serve_artifact_match"] is True


def test_combine_selection_accepts_k160_from_abstraction_survivor(tmp_path):
    emd_json = tmp_path / "emd.json"
    base_json = tmp_path / "base.json"
    out_json = tmp_path / "verdict.json"
    frozen = tmp_path / "table160.emdfrzn"
    serve = tmp_path / "serve.pkl"
    action_sets = tmp_path / "action_sets.json"
    sweep = tmp_path / "abstraction_sweep.json"
    frozen.write_bytes(b"frozen-table-k160")
    sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    Path(str(frozen) + ".sha256").write_text(sha + "\n", encoding="ascii")
    frozen_s3 = "s3://bucket/run/frozen/table160.emdfrzn"
    with serve.open("wb") as f:
        pickle.dump(
            {
                "strat_sum": {},
                "abstraction": {
                    "postflop_emd_frozen_sha256": sha,
                    "postflop_emd_table_id": "table160",
                    "postflop_emd_frozen_s3": frozen_s3,
                },
            },
            f,
        )
    emd_json.write_text(json.dumps(_partial(sb=70.0, bb=80.0, pooled=90.0)), encoding="utf-8")
    base_json.write_text(json.dumps(_partial(sb=100.0, bb=110.0, pooled=100.0)), encoding="utf-8")
    action_sets.write_text(
        json.dumps({
            "pot_fraction": {
                "emd_per_seat": {"1": 70.0, "2": 80.0},
                "iter62000_per_seat": {"1": 100.0, "2": 110.0},
            },
            "off_tree_adversarial": {
                "emd_per_seat": {"1": 72.0, "2": 82.0},
                "iter62000_per_seat": {"1": 100.0, "2": 110.0},
            },
        }),
        encoding="utf-8",
    )
    sweep.write_text(json.dumps({"survivors": [{"k": 160, "survivor_gate_pass": True}]}), encoding="utf-8")

    rc = eval_emd._combine(
        argparse.Namespace(
            emd_json=str(emd_json),
            iter62000_json=str(base_json),
            out=str(out_json),
            min_blind_gain=15.0,
            global_slack=10.0,
            num_buckets=160,
            emd_frozen=str(frozen),
            emd_frozen_s3=frozen_s3,
            emd_table_id="table160",
            serve_blueprint=str(serve),
            serve_blueprint_s3="s3://bucket/run/serve/candidate.pkl",
            s3_out="",
            selection_tier="cheap",
            action_sets_json=str(action_sets),
            abstraction_sweep=str(sweep),
            expected_k_centroids=160,
        )
    )

    assert rc == 0
    report = json.loads((tmp_path / "verdict_selection_report.json").read_text(encoding="utf-8"))
    verdict = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["selection_pass"] is True
    assert report["acceptance_pass"] is True
    assert verdict["expected_k_centroids"] == 160
    assert verdict["accepted_survivor_ks"] == [160]
    assert report["identity"]["k_centroids"] == 160
    assert report["identity"]["expected_k_centroids"] == [160]


def test_combine_expected_k_without_abstraction_sweep_fails_closed(tmp_path):
    emd_json = tmp_path / "emd.json"
    base_json = tmp_path / "base.json"
    out_json = tmp_path / "verdict.json"
    frozen = tmp_path / "table160.emdfrzn"
    serve = tmp_path / "serve.pkl"
    action_sets = tmp_path / "action_sets.json"
    frozen.write_bytes(b"frozen-table-k160")
    sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    Path(str(frozen) + ".sha256").write_text(sha + "\n", encoding="ascii")
    frozen_s3 = "s3://bucket/run/frozen/table160.emdfrzn"
    with serve.open("wb") as f:
        pickle.dump(
            {
                "strat_sum": {},
                "abstraction": {
                    "postflop_emd_frozen_sha256": sha,
                    "postflop_emd_table_id": "table160",
                    "postflop_emd_frozen_s3": frozen_s3,
                },
            },
            f,
        )
    emd_json.write_text(json.dumps(_partial(sb=70.0, bb=80.0, pooled=90.0)), encoding="utf-8")
    base_json.write_text(json.dumps(_partial(sb=100.0, bb=110.0, pooled=100.0)), encoding="utf-8")
    action_sets.write_text(
        json.dumps({
            "pot_fraction": {
                "emd_per_seat": {"1": 70.0, "2": 80.0},
                "iter62000_per_seat": {"1": 100.0, "2": 110.0},
            },
            "off_tree_adversarial": {
                "emd_per_seat": {"1": 72.0, "2": 82.0},
                "iter62000_per_seat": {"1": 100.0, "2": 110.0},
            },
        }),
        encoding="utf-8",
    )

    rc = eval_emd._combine(
        argparse.Namespace(
            emd_json=str(emd_json),
            iter62000_json=str(base_json),
            out=str(out_json),
            min_blind_gain=15.0,
            global_slack=10.0,
            num_buckets=160,
            emd_frozen=str(frozen),
            emd_frozen_s3=frozen_s3,
            emd_table_id="table160",
            serve_blueprint=str(serve),
            serve_blueprint_s3="s3://bucket/run/serve/candidate.pkl",
            s3_out="",
            selection_tier="cheap",
            action_sets_json=str(action_sets),
            abstraction_sweep="",
            expected_k_centroids=160,
        )
    )

    assert rc == 0
    report = json.loads((tmp_path / "verdict_selection_report.json").read_text(encoding="utf-8"))
    verdict = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["selection_pass"] is False
    assert report["acceptance_pass"] is False
    assert "expected_k_requires_abstraction_sweep" in report["selection_reason_codes"]
    assert report["identity"]["k_policy"] == "exact_k_max_default"
    assert report["identity"]["expected_k_centroids"] == [200]
    assert verdict["k_policy_error"] == "expected_k_requires_abstraction_sweep"
