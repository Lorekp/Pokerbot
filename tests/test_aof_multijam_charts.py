"""SECOND_JAM / THIRD_JAM chart coverage and lookup."""

from __future__ import annotations

import pytest

from aof_ranges import (
    CALL_RANGE_STR,
    SECOND_JAM_RANGE_STR,
    THIRD_JAM_RANGE_STR,
    has_second_jam_chart,
    has_third_jam_chart,
    lookup_facing_three_ranges,
    lookup_second_shover_facing_jam,
    lookup_third_shover_facing_two_jams,
)
from scripts.aof_static_range_audit import _facing_three_triples, _facing_two_pairs


def test_facing_two_pairs_4max_complete() -> None:
    pairs = _facing_two_pairs(4)
    assert pairs == {("BTN", "UTG"), ("SB", "UTG"), ("SB", "BTN")}
    depth = 8
    for p in pairs:
        assert p in SECOND_JAM_RANGE_STR[depth]


def test_facing_three_triple_4max_complete() -> None:
    triples = _facing_three_triples(4)
    assert triples == {("SB", "UTG", "BTN")}
    assert ("SB", "UTG", "BTN") in THIRD_JAM_RANGE_STR[8]


def test_second_jam_wider_than_continue() -> None:
    from pokerkit import parse_range

    key = ("SB", "BTN")
    call_n = len(list(parse_range(CALL_RANGE_STR[8][key])))
    jam_n = len(list(parse_range(SECOND_JAM_RANGE_STR[8][key])))
    assert jam_n >= call_n


def test_third_jam_canonical_not_fallback() -> None:
    _rng, src, _prov = lookup_third_shover_facing_two_jams(
        "UTG", "BTN", "SB", 8.0, table_size=4
    )
    assert src == "conditional_two_jams"


def test_lookup_facing_three_no_fallback() -> None:
    res = lookup_facing_three_ranges("UTG", "BTN", "SB", 8.0, table_size=4)
    assert res.fallback_used is False
    assert "third_jam_chart" in res.range_note
    assert "fallback_call_chart" not in res.range_note


def test_second_shover_uses_second_jam_chart() -> None:
    assert has_second_jam_chart("UTG", "BTN", 8.0, table_size=4)
    _rng, src, _ = lookup_second_shover_facing_jam("UTG", "BTN", 8.0, table_size=4)
    assert src == "conditional"


def test_strict_third_jam_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AOF_STRICT_RANGES", "1")
    monkeypatch.setattr(
        "aof_ranges._third_jam_map",
        lambda _stack: {},
    )
    from aof_ranges import lookup_range, RangeIntent

    with pytest.raises(KeyError, match="THIRD_JAM"):
        lookup_range(
            RangeIntent.THIRD_SHOVER_FACING_TWO_JAMS,
            "SB",
            8.0,
            table_size=4,
            first_shover="UTG",
            second_shover="BTN",
            allow_fallback=False,
        )
