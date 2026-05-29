"""Determinism tests for non-AoF solver label artifact serialization."""

from __future__ import annotations

import json

from pokerbot.solver.label_diff import (
    DiffMode,
    _canonical_json,
    canonicalize_action_distribution,
    canonicalize_manifest,
    canonicalize_solver_response,
)


def test_action_distribution_canonicalization_is_idempotent() -> None:
    dist = {"bet_0.50": 0.3, "check": 0.55, "all_in": 0.15}
    once = canonicalize_action_distribution(dist)
    twice = canonicalize_action_distribution(once)
    assert once == twice
    assert list(once.keys()) == sorted(once.keys())


def test_solver_response_canonicalization_sorts_distribution() -> None:
    resp = {
        "converged": True,
        "action_distribution": {"all_in": 0.1, "check": 0.6, "bet_0.50": 0.3},
        "available_actions": ["check", "bet_0.50", "all_in"],
    }
    canon = canonicalize_solver_response(resp)
    keys = list(canon["action_distribution"].keys())
    assert keys == sorted(keys)


def test_manifest_canonicalization_stable_across_dict_insertion_order() -> None:
    manifest_a = {
        "seed": 42,
        "range_scope_distribution": {"z=1": 2, "a=1": 3},
        "sample_labels": [
            {
                "spot_id": "b",
                "board": ["Ks", "8h", "3d"],
                "action_distribution": {"bet_0.50": 0.5, "check": 0.5},
            },
            {
                "spot_id": "a",
                "board": ["Td", "9d", "6h"],
                "action_distribution": {"check": 1.0},
            },
        ],
        "generated_at": "ignore-me",
    }
    manifest_b = {
        "generated_at": "also-ignore",
        "seed": 42,
        "sample_labels": [
            {
                "spot_id": "a",
                "board": ["Td", "9d", "6h"],
                "action_distribution": {"check": 1.0},
            },
            {
                "spot_id": "b",
                "board": ["Ks", "8h", "3d"],
                "action_distribution": {"check": 0.5, "bet_0.50": 0.5},
            },
        ],
        "range_scope_distribution": {"a=1": 3, "z=1": 2},
    }
    ca = canonicalize_manifest(manifest_a, DiffMode.SEMANTIC)
    cb = canonicalize_manifest(manifest_b, DiffMode.SEMANTIC)
    assert _canonical_json(ca) == _canonical_json(cb)


def test_repeated_canonical_json_identical() -> None:
    obj = {"b": 2, "a": {"y": 1, "x": [3, 2, 1]}}
    assert _canonical_json(obj) == _canonical_json(obj)
