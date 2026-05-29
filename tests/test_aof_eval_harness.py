"""Tests for AoF RL evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("phevaluator")
pytest.importorskip("torch")

from aof_equity import warmup
from aof_rl.env import reset_hand_from_deal_index, shuffle_deck
from aof_rl.eval.deal_stream import DealStream
from aof_rl.eval.eval_schema import validate_report
from aof_rl.eval.harness import AofEvalHarness, EvalConfig
from aof_rl.eval.hu_exploitability import compute_hu_exploitability
from aof_rl.eval.policy_adapter import ScriptedEvalPolicy
from aof_rl.teacher import NashTeacherCache
from aof_validation.snapshot_schema import check_aof_eval_report

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "aof_eval"
GOLDEN = FIXTURE_DIR / "golden_report.json"


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


def test_shuffle_deck_reproducible() -> None:
    assert shuffle_deck(42) == shuffle_deck(42)
    assert shuffle_deck(42) != shuffle_deck(43)


def test_deal_stream_same_holes() -> None:
    ds = DealStream(hands=3, base_seed=7, table_sizes=(2,))
    deals = list(ds.iter_deals())
    again = list(DealStream(hands=3, base_seed=7, table_sizes=(2,)).iter_deals())
    assert len(deals) == 3
    for a, b in zip(deals, again):
        assert a.holes == b.holes
        assert a.table_size == b.table_size


def test_reset_hand_from_deal_index() -> None:
    s1 = reset_hand_from_deal_index(0, table_size=3, base_seed=99)
    s2 = reset_hand_from_deal_index(0, table_size=3, base_seed=99)
    assert s1.holes == s2.holes


def test_eval_report_schema_scripted() -> None:
    report = AofEvalHarness(
        EvalConfig(
            eval_id="test_scripted",
            hands=8,
            seeds=[11],
            table_sizes=(2,),
            profile="fast",
            policy_mode="scripted",
            compute_hu_exploitability=False,
            compute_multiway=False,
            device="cpu",
        )
    ).run()
    data = report.to_dict()
    check_aof_eval_report(data)
    validate_report(data)
    assert data["aggregate"]["hands"] > 0


def test_paired_delta_zero_same_scripted() -> None:
    cfg = EvalConfig(
        eval_id="paired_same",
        hands=12,
        seeds=[5],
        table_sizes=(2,),
        profile="fast",
        policy_mode="scripted",
        paired=True,
        compute_hu_exploitability=False,
        compute_multiway=False,
        device="cpu",
    )
    report = AofEvalHarness(cfg).run()
    assert report.paired_comparison is not None
    assert abs(report.paired_comparison.mean_delta_bb_per_hand) < 1e-6


def test_hu_exploitability_scripted_positive() -> None:
    loose = ScriptedEvalPolicy(call_prob=0.95, seed=1)
    nit = ScriptedEvalPolicy(call_prob=0.05, seed=2)
    cache = NashTeacherCache(teacher_version=2, use_fast=True)
    ex_loose = compute_hu_exploitability(
        loose,
        cache,
        mode="bucketed_169",
        equity_samples=200,
        seed=3,
    )
    ex_nit = compute_hu_exploitability(
        nit,
        cache,
        mode="bucketed_169",
        equity_samples=200,
        seed=3,
    )
    assert ex_loose.exploitability_bb_per_hand > 0.05
    assert ex_nit.exploitability_bb_per_hand > 0.05


def test_snapshot_golden_scripted(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    report = AofEvalHarness(
        EvalConfig(
            eval_id="golden_scripted",
            hands=4,
            seeds=[0],
            table_sizes=(2,),
            profile="fast",
            policy_mode="scripted",
            compute_hu_exploitability=True,
            hu_equity_samples=100,
            compute_multiway=False,
            device="cpu",
            out_json=str(out),
        )
    ).run()
    data = json.loads(out.read_text(encoding="utf-8"))
    check_aof_eval_report(data)
    assert data["eval_id"] == "golden_scripted"
    assert data["aggregate"]["hands"] == 8
    if GOLDEN.is_file():
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        assert golden["eval_id"] == data["eval_id"]
        assert golden["aggregate"]["hands"] == data["aggregate"]["hands"]
    else:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        FIXTURE_DIR.joinpath("golden_report.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
