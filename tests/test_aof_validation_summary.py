"""Schema / smoke tests for validation artifact summarizer inputs."""

from __future__ import annotations

import json
from pathlib import Path

from aof_validation.snapshot_schema import (
    check_call_behind_snapshot,
    check_multiway_equity_snapshot,
    check_open_multiway_snapshot,
    check_policy_version_comparison,
    check_range_sensitivity_snapshot,
    check_teacher_mismatch_report,
)


def test_summarize_accepts_minimal_artifacts(tmp_path: Path) -> None:
    mw = tmp_path / "mw.json"
    mw.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "t",
                "config": {"samples": 100, "seed": 42},
                "spots": [
                    {
                        "name": "x",
                        "hero_cards": "AsAd",
                        "villain_ranges": ["KK", "QQ"],
                        "exact_equity": 0.8,
                        "mc_equity": 0.79,
                        "abs_delta": 0.01,
                        "method": "exact_vs_mc",
                        "samples": 100,
                        "seed": 42,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    check_multiway_equity_snapshot(json.loads(mw.read_text(encoding="utf-8")))

    om = {
        "schema_version": 1,
        "spots": [
            {
                "spot": "AJo BTN 8bb",
                "new_action": "ALL IN",
                "new_ev": 1.0,
                "legacy_shortcut_ev": 0.5,
                "delta_bb": 0.5,
                "action_flip": False,
                "edge_bb": 0.5,
            }
        ],
    }
    check_open_multiway_snapshot(om)

    cb = {
        "schema_version": 1,
        "spots": [
            {
                "spot": "AA BB vs BTN",
                "new_ev_call": 1.0,
                "legacy_ev_call": 0.9,
                "delta_bb": 0.1,
            }
        ],
    }
    check_call_behind_snapshot(cb)

    rs = {
        "schema_version": 2,
        "profiles": ["chart_default", "tight_archive"],
        "spots": [
            {
                "spot": "AJo BTN 8bb",
                "chart_action": "ALL IN",
                "tight_action": "FOLD",
                "chart_ev": 1.0,
                "tight_ev": 0.5,
                "delta_bb": -0.5,
                "action_flip": True,
            }
        ],
    }
    check_range_sensitivity_snapshot(rs)

    tm = {
        "schema_version": 1,
        "obs_version": 1,
        "teacher_version": 1,
        "num_hands": 10,
        "spot_kind_counts": {"open": 5},
        "teacher_path_counts": {"static_profile_facing_two_missing": 2},
        "terminal_counts": {},
    }
    check_teacher_mismatch_report(tm)

    cmp = {
        "schema_version": 1,
        "seed": 42,
        "num_hands": 100,
        "policies": [
            {
                "name": "a",
                "checkpoint": "x.pt",
                "obs_version": 1,
                "teacher_version": 1,
                "bb_per_100": 0.0,
                "action_agreement_with_teacher": 0.9,
                "spot_kind_counts": {},
                "teacher_path_counts": {},
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
    check_policy_version_comparison(cmp)
