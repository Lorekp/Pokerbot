"""Tests for label completion ledger and chain-log bootstrap."""

from __future__ import annotations

from pathlib import Path

import pytest

from pokerbot.solver.label_completion import (
    CompletionLedger,
    bootstrap_and_validate,
    parse_chain_log,
    validate_spot_key,
)


SAMPLE_LOG = """\
=== prod600 starting ===
[1] solving sb_vs_bb_flop|Td9d6h|hero=1
  OK converged=True exploit=0.123% iters=500 time=95.0s
=== prod6000 starting ===
[1] solving sb_vs_bb_flop|Ah7c2s|hero=1
  OK converged=True exploit=0.111% iters=400 time=90.0s
[2] solving sb_vs_bb_flop|Ah7c2s|hero=2
  OK converged=True exploit=0.222% iters=410 time=91.0s
[3] solving btn_vs_bb_flop|Ks8h3d|hero=2
  OK converged=True exploit=0.333% iters=420 time=92.0s
"""


def test_validate_spot_key_accepts_canonical_format():
    assert validate_spot_key("sb_vs_bb_flop|Td9d6h|hero=1")
    assert validate_spot_key("btn_vs_bb_flop|Ah7c2s|hero=4")
    assert validate_spot_key("bunching_postflop|Ks8h3d|hero=2")
    assert not validate_spot_key("invalid|Td9d6h|hero=1")
    assert not validate_spot_key("sb_vs_bb_flop|Td9d6h|hero=bad")


def test_parse_chain_log_pairs_solving_with_ok(tmp_path: Path):
    log = tmp_path / "chain.log"
    log.write_text(SAMPLE_LOG, encoding="utf-8")
    result = parse_chain_log(log, section_marker="prod6000 starting", expected_count=3)
    assert result.valid_count == 3
    assert result.ok
    keys = [e.spot_key for e in result.entries]
    assert "sb_vs_bb_flop|Ah7c2s|hero=1" in keys
    assert "sb_vs_bb_flop|Ah7c2s|hero=2" in keys
    assert "btn_vs_bb_flop|Ks8h3d|hero=2" in keys


def test_parse_chain_log_skip_does_not_mark_done(tmp_path: Path):
    log = tmp_path / "chain.log"
    log.write_text(
        "=== prod6000 starting ===\n"
        "[1] solving sb_vs_bb_flop|Ks8h3d|hero=1\n"
        "  SKIP not converged: 1.0% pot (target 0.5%) after 10000 iters, 3600.0s\n"
        "[2] solving btn_vs_bb_flop|Ks8h3d|hero=2\n"
        "  OK converged=True exploit=0.1% iters=100 time=10.0s\n",
        encoding="utf-8",
    )
    result = parse_chain_log(log, section_marker="prod6000 starting")
    assert result.valid_count == 1
    assert result.unpaired_solving == ["sb_vs_bb_flop|Ks8h3d|hero=1"]
    assert "sb_vs_bb_flop|Ks8h3d|hero=1" not in [e.spot_key for e in result.entries]


def test_parse_chain_log_flags_unpaired_solving(tmp_path: Path):
    log = tmp_path / "chain.log"
    log.write_text(
        "=== prod6000 starting ===\n"
        "[1] solving sb_vs_bb_flop|Ah7c2s|hero=1\n"
        "[2] solving sb_vs_bb_flop|Ks8h3d|hero=2\n"
        "  OK converged=True exploit=0.1% iters=100 time=10.0s\n",
        encoding="utf-8",
    )
    result = parse_chain_log(log, section_marker="prod6000 starting")
    assert result.unpaired_solving == ["sb_vs_bb_flop|Ah7c2s|hero=1"]
    assert result.valid_count == 1
    assert not result.ok


def test_bootstrap_expected_count_mismatch(tmp_path: Path):
    log = tmp_path / "chain.log"
    log.write_text(SAMPLE_LOG, encoding="utf-8")
    result = bootstrap_and_validate(
        log,
        ledger_path=tmp_path / "ledger.jsonl",
        expected_count=999,
        section_marker="prod6000 starting",
    )
    assert not result.ok


def test_completion_ledger_append_and_load(tmp_path: Path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = CompletionLedger(path=ledger_path)
    ledger.append("sb_vs_bb_flop|Td9d6h|hero=1", exploitability_pct=0.1, iterations_used=100)
    loaded = CompletionLedger.load(ledger_path)
    assert loaded.contains("sb_vs_bb_flop|Td9d6h|hero=1")
