"""JSON schema checks for AoF snapshot artifacts."""

from __future__ import annotations

from typing import Any


def require_keys(obj: dict[str, Any], keys: tuple[str, ...], *, label: str = "root") -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise AssertionError(f"{label} missing keys: {missing}")


def check_multiway_equity_snapshot(data: dict[str, Any]) -> None:
    require_keys(data, ("schema_version", "generated_at", "config", "spots"))
    assert data["schema_version"] == 1
    for i, spot in enumerate(data["spots"]):
        require_keys(
            spot,
            (
                "name",
                "hero_cards",
                "villain_ranges",
                "exact_equity",
                "mc_equity",
                "abs_delta",
                "method",
                "samples",
                "seed",
            ),
            label=f"spots[{i}]",
        )


def check_open_multiway_snapshot(data: dict[str, Any]) -> None:
    require_keys(data, ("schema_version", "spots"))
    assert data["schema_version"] == 1
    for i, spot in enumerate(data["spots"]):
        require_keys(
            spot,
            (
                "spot",
                "new_action",
                "new_ev",
                "legacy_shortcut_ev",
                "delta_bb",
                "action_flip",
                "edge_bb",
            ),
            label=f"spots[{i}]",
        )


def check_call_behind_snapshot(data: dict[str, Any]) -> None:
    require_keys(data, ("schema_version", "spots"))
    assert data["schema_version"] == 1
    for i, spot in enumerate(data["spots"]):
        require_keys(
            spot,
            (
                "spot",
                "new_ev_call",
                "legacy_ev_call",
                "delta_bb",
            ),
            label=f"spots[{i}]",
        )


def check_range_sensitivity_snapshot(data: dict[str, Any]) -> None:
    require_keys(data, ("schema_version", "profiles", "spots"))
    ver = data["schema_version"]
    assert ver in (1, 2)
    for i, spot in enumerate(data["spots"]):
        if ver == 2:
            require_keys(
                spot,
                (
                    "spot",
                    "chart_action",
                    "tight_action",
                    "chart_ev",
                    "tight_ev",
                    "delta_bb",
                    "action_flip",
                ),
                label=f"spots[{i}]",
            )
        else:
            require_keys(
                spot,
                (
                    "spot",
                    "default_action",
                    "chart_action",
                    "default_ev",
                    "chart_ev",
                    "delta_bb",
                    "action_flip",
                ),
                label=f"spots[{i}]",
            )


def check_policy_version_comparison(data: dict[str, Any]) -> None:
    require_keys(data, ("schema_version", "seed", "num_hands", "policies", "head_to_head"))
    assert data["schema_version"] == 1
    for i, pol in enumerate(data["policies"]):
        require_keys(
            pol,
            (
                "name",
                "checkpoint",
                "obs_version",
                "teacher_version",
                "bb_per_100",
                "action_agreement_with_teacher",
                "spot_kind_counts",
                "teacher_path_counts",
                "facing_two_decisions",
                "call_behind_decisions",
            ),
            label=f"policies[{i}]",
        )
        for sub in ("facing_two_decisions", "call_behind_decisions"):
            require_keys(
                pol[sub],
                ("count", "call_rate", "avg_teacher_edge_bb"),
                label=f"policies[{i}].{sub}",
            )


def check_static_range_audit(data: dict[str, Any]) -> None:
    require_keys(
        data,
        (
            "schema_version",
            "generated_at",
            "static_profile_only",
            "not_solver_backed_34max",
            "by_table_size",
        ),
    )
    assert data["schema_version"] == 1
    assert data["static_profile_only"] is True


def check_eval_report(data: dict[str, Any]) -> None:
    require_keys(
        data,
        (
            "schema_version",
            "obs_version",
            "teacher_version",
            "bb_per_100",
            "hands",
            "checkpoint_meta",
        ),
    )
    assert data["schema_version"] == 1


def check_aof_eval_report(data: dict[str, Any]) -> None:
    """Schema v1 for ``aof_rl.eval`` harness JSON reports."""
    require_keys(
        data,
        (
            "schema_version",
            "generated_at",
            "eval_id",
            "policy",
            "config",
            "aggregate",
            "by_seat",
            "by_table_size",
            "calibration",
        ),
    )
    assert data["schema_version"] == 1
    require_keys(data["aggregate"], ("hands", "bb_per_hand", "bb_per_hand_ci95", "bb_per_100"), label="aggregate")
    require_keys(data["policy"], ("name", "obs_version"), label="policy")
    require_keys(data["calibration"], ("thin_edge_buckets",), label="calibration")
    for i, b in enumerate(data["calibration"]["thin_edge_buckets"]):
        require_keys(
            b,
            (
                "label",
                "count",
                "disagreement_rate",
                "mean_q_error_bb",
                "mean_uncertainty_bb",
            ),
            label=f"calibration.thin_edge_buckets[{i}]",
        )
    if data.get("hu_exploitability") is not None:
        require_keys(
            data["hu_exploitability"],
            (
                "mode",
                "exploitability_bb_per_hand",
                "sb_br_gain_bb",
                "bb_br_gain_bb",
                "hands_enumerated",
            ),
            label="hu_exploitability",
        )
    if data.get("paired_comparison") is not None:
        require_keys(
            data["paired_comparison"],
            (
                "policy_a",
                "policy_b",
                "hands",
                "mean_delta_bb_per_hand",
                "delta_bb_per_hand_ci95",
            ),
            label="paired_comparison",
        )


def check_br_exploit_report(data: dict[str, Any]) -> None:
    """Schema v1 for ``aof_rl.exploit`` BR / exploitability JSON reports."""
    require_keys(
        data,
        (
            "schema_version",
            "generated_at",
            "eval_id",
            "policy",
            "config",
            "league_context",
        ),
    )
    assert data["schema_version"] == 1
    require_keys(
        data["policy"],
        ("name", "obs_version"),
        label="policy",
    )
    if data.get("hu_exploitability") is not None:
        require_keys(
            data["hu_exploitability"],
            (
                "mode",
                "br_source",
                "exploitability_bb_per_hand",
                "sb_br_gain_bb",
                "bb_br_gain_bb",
                "hands_enumerated",
            ),
            label="hu_exploitability",
        )
    for key in ("multiway_deviation_oracle", "multiway_deviation_learned"):
        if not data.get(key):
            continue
        for ts, rows in data[key].items():
            for i, row in enumerate(rows):
                require_keys(
                    row,
                    (
                        "seat",
                        "br_source",
                        "deviation_gain_bb_per_hand",
                        "hands",
                    ),
                    label=f"{key}[{ts}][{i}]",
                )
    for i, d in enumerate(data.get("learned_br_diagnostics") or []):
        require_keys(
            d,
            (
                "seat",
                "table_size",
                "steps",
                "gain_before_bb",
                "gain_after_bb",
                "improved",
            ),
            label=f"learned_br_diagnostics[{i}]",
        )
    for i, r in enumerate(data.get("restricted_subgames") or []):
        require_keys(
            r,
            (
                "game_id",
                "table_size",
                "max_deviation_gain_bb",
                "per_seat_gain_bb",
                "br_source",
            ),
            label=f"restricted_subgames[{i}]",
        )


def check_teacher_mismatch_report(data: dict[str, Any]) -> None:
    require_keys(
        data,
        (
            "schema_version",
            "obs_version",
            "teacher_version",
            "num_hands",
            "spot_kind_counts",
            "teacher_path_counts",
            "terminal_counts",
        ),
    )
    assert data["schema_version"] == 1
