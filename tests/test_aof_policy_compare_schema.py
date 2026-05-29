"""Schema for policy version comparison JSON."""

from __future__ import annotations

from aof_validation.snapshot_schema import check_policy_version_comparison


def test_policy_compare_schema_minimal() -> None:
    data = {
        "schema_version": 1,
        "seed": 42,
        "num_hands": 100,
        "policies": [
            {
                "name": "v1_teacher_v1",
                "checkpoint": "ckpt_v1.pt",
                "obs_version": 1,
                "teacher_version": 1,
                "bb_per_100": 0.0,
                "action_agreement_with_teacher": 0.5,
                "spot_kind_counts": {"open": 10},
                "teacher_path_counts": {"static_profile_open": 10},
                "facing_two_decisions": {
                    "count": 0,
                    "call_rate": 0.0,
                    "avg_teacher_edge_bb": 0.0,
                },
                "call_behind_decisions": {
                    "count": 0,
                    "call_rate": 0.0,
                    "avg_teacher_edge_bb": 0.0,
                },
            }
        ],
        "head_to_head": [],
    }
    check_policy_version_comparison(data)
