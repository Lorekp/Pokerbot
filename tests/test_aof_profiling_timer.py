"""Tests for AoF league profiling timers."""

from __future__ import annotations

import time

import pytest

from aof_rl.profiling.league_profiler import get_profiler, set_profiler
from aof_rl.profiling.timer import TimingAccumulator


def test_timing_accumulator_sorted_order() -> None:
    acc = TimingAccumulator()
    acc.add("slow", 3.0)
    acc.add("fast", 1.0)
    acc.add("mid", 2.0)
    rows = acc.sorted_report()
    assert rows[0]["name"] == "slow"
    assert rows[0]["rank"] == 1
    pct_sum = sum(r["pct"] for r in rows)
    assert 99.0 <= pct_sum <= 101.0


def test_measure_context() -> None:
    acc = TimingAccumulator()
    with acc.measure("block"):
        time.sleep(0.01)
    assert acc.buckets["block"].calls == 1
    assert acc.buckets["block"].seconds >= 0.005


def test_merge() -> None:
    a = TimingAccumulator()
    b = TimingAccumulator()
    a.add("x", 1.0, calls=2)
    b.add("x", 2.0, calls=1)
    b.add("y", 0.5)
    a.merge(b)
    assert a.buckets["x"].seconds == 3.0
    assert a.buckets["x"].calls == 3


def test_update_snapshot_deltas() -> None:
    acc = TimingAccumulator()
    acc.begin_update(0, "hu_only")
    acc.add("update.rollout", 1.0, calls=1)
    acc.add("update.ppo.train", 0.2, calls=4)
    acc.end_update()

    acc.begin_update(1, "hu_only")
    acc.add("update.rollout", 0.5, calls=1)
    acc.add("update.ppo.train", 0.1, calls=4)
    acc.end_update()

    assert len(acc.update_snapshots) == 2
    snap0 = acc.update_snapshots[0]
    assert snap0["update"] == 0
    assert snap0["stage"] == "hu_only"
    assert snap0["buckets"]["update.rollout"]["seconds"] == 1.0
    assert snap0["buckets"]["update.ppo.train"]["seconds"] == 0.2
    assert snap0["top_buckets"][0]["name"] == "update.rollout"

    snap1 = acc.update_snapshots[1]
    assert snap1["buckets"]["update.rollout"]["seconds"] == 0.5
    assert "update.ppo.train" in snap1["buckets"]


def test_spot_snapshot_deltas() -> None:
    acc = TimingAccumulator()
    acc.begin_spot(0, "sb_vs_bb|Td9d6h|hero=4", "sb_vs_bb_flop")
    acc.add("spot.solve_subprocess", 2.0, calls=1)
    acc.add("spot.encode_row", 0.05, calls=1)
    acc.end_spot()

    acc.begin_spot(1, "sb_vs_bb|Td9d6h|hero=5", "sb_vs_bb_flop")
    acc.add("spot.solve_subprocess", 1.5, calls=1)
    acc.end_spot()

    assert len(acc.spot_snapshots) == 2
    snap0 = acc.spot_snapshots[0]
    assert snap0["spot"] == 0
    assert snap0["spot_key"] == "sb_vs_bb|Td9d6h|hero=4"
    assert snap0["buckets"]["spot.solve_subprocess"]["seconds"] == 2.0
    assert snap0["top_buckets"][0]["name"] == "spot.solve_subprocess"

    snap1 = acc.spot_snapshots[1]
    assert snap1["buckets"]["spot.solve_subprocess"]["seconds"] == 1.5


@pytest.mark.slow
def test_profile_train_emits_buckets(tmp_path) -> None:
    from aof_rl.config import load_train_config
    from aof_rl.league_trainer import train_ppo_league

    cfg = load_train_config("configs/aof_v3_ppo_league_smoke.yaml")
    cfg.train.profile = True
    cfg.train.total_updates = 1
    cfg.train.batch_hands = 4
    cfg.train.device = "cpu"
    cfg.train.allow_cpu = True
    cfg.train.torch_compile = False
    cfg.checkpoint_path = str(tmp_path / "prof_smoke.pt")
    cfg.league.adversarial_br_pool = False

    acc = TimingAccumulator()
    set_profiler(acc)
    try:
        train_ppo_league(cfg)
    finally:
        set_profiler(None)

    names = set(acc.buckets)
    assert "update.rollout" in names or "startup.warmup" in names
