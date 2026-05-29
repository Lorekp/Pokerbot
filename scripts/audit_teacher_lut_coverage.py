#!/usr/bin/env python3
"""Gate 0: measure LUT coverage vs rollout teacher-call volume (target >=90%)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from aof_rl.batched_rollout import collect_update_rollout_batched
from aof_rl.config import load_train_config
from aof_rl.curriculum import CurriculumScheduler
from aof_rl.league import LeagueManager
from aof_rl.policy_v3 import AoFPolicyNetV3
from aof_rl.profiling.league_profiler import set_profiler
from aof_rl.profiling.timer import TimingAccumulator
from aof_rl.rollout import RolloutConfig
from aof_rl.teacher import NashTeacherCache
from aof_rl.teacher_lut.context import load_ctx_rows_jsonl
from aof_rl.teacher_lut.tabulated import V1_COVERED_PATHS

V1_PATH_NAMES = frozenset(V1_COVERED_PATHS)


def _summarize_prof(prof: TimingAccumulator) -> dict:
    path_secs: dict[str, float] = {}
    path_calls: dict[str, int] = {}
    for name, stats in prof.buckets.items():
        if name.startswith("teacher.path."):
            path_secs[name] = stats.seconds
            path_calls[name] = stats.calls
    counter_calls = {
        k.replace("teacher.path.calls.", ""): v
        for k, v in prof.counters.items()
        if k.startswith("teacher.path.calls.")
    }
    hero_teacher_sec = prof.buckets.get("rollout.hero_teacher", None)
    hero_teacher_sec = hero_teacher_sec.seconds if hero_teacher_sec else 0.0

    lut_hit_sec = path_secs.get("teacher.path.lut_hit", 0.0)
    lut_hit_calls = counter_calls.get("lut_hit", path_calls.get("teacher.path.lut_hit", 0))
    lut_ctx_miss = prof.counters.get("teacher.path.calls.lut_ctx_miss", 0)

    legacy_sec = sum(
        s for k, s in path_secs.items() if k != "teacher.path.lut_hit"
    )
    legacy_calls = sum(
        c
        for k, c in path_calls.items()
        if k != "teacher.path.lut_hit"
    )
    total_teacher_sec = lut_hit_sec + legacy_sec
    total_teacher_calls = lut_hit_calls + legacy_calls

    v1_legacy_sec = sum(
        path_secs.get(f"teacher.path.{p}", 0.0)
        for p in V1_PATH_NAMES
    )
    v1_eligible_calls = sum(
        counter_calls.get(p, path_calls.get(f"teacher.path.{p}", 0)) for p in V1_PATH_NAMES
    )

    hit_rate_sec = (
        lut_hit_sec / total_teacher_sec if total_teacher_sec > 0 else 0.0
    )
    hit_rate_calls = (
        lut_hit_calls / total_teacher_calls if total_teacher_calls > 0 else 0.0
    )
    non_v1_calls = total_teacher_calls - lut_hit_calls - lut_ctx_miss
    non_v1_sec = legacy_sec - v1_legacy_sec

    return {
        "path_seconds": path_secs,
        "path_calls": path_calls,
        "counter_calls": counter_calls,
        "lut_hit_sec": lut_hit_sec,
        "lut_hit_calls": lut_hit_calls,
        "lut_ctx_miss": lut_ctx_miss,
        "legacy_sec": legacy_sec,
        "legacy_calls": legacy_calls,
        "total_teacher_sec": total_teacher_sec,
        "total_teacher_calls": total_teacher_calls,
        "hero_teacher_sec": hero_teacher_sec,
        "v1_legacy_sec": v1_legacy_sec,
        "v1_eligible_calls": v1_eligible_calls,
        "hit_rate_by_seconds": hit_rate_sec,
        "hit_rate_by_calls": hit_rate_calls,
        "non_v1_calls": non_v1_calls,
        "non_v1_legacy_sec": non_v1_sec,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/aof_v3_ppo_league.yaml")
    p.add_argument("--census", default="data/aof/teacher_lut_ctx_rows_8bb.jsonl")
    p.add_argument("--lut", default="data/aof/teacher_lut_v2_8bb.npz")
    p.add_argument("--updates", type=int, default=15)
    p.add_argument("--target", type=float, default=0.90)
    p.add_argument("--out", type=Path, default=Path("artifacts/audit/lut_coverage.json"))
    args = p.parse_args()

    census_path = _ROOT / args.census
    lut_path = _ROOT / args.lut
    if not census_path.is_file():
        print(f"ERROR: census missing: {census_path}", file=sys.stderr)
        sys.exit(1)
    if not lut_path.is_file():
        print(f"ERROR: LUT missing: {lut_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_ctx_rows_jsonl(census_path)
    spot_hist = Counter(r.spot_kind for r in rows)
    table_hist = Counter(r.table_size for r in rows)
    print(f"Census: {len(rows)} ctx rows")
    print(f"  spot_kind: {dict(spot_hist)}")
    print(f"  table_size: {dict(table_hist)}")
    print(f"  V1 paths: {sorted(V1_PATH_NAMES)}")

    cfg = load_train_config(_ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: running audit on CPU (slower)", flush=True)

    prof = TimingAccumulator()
    set_profiler(prof)
    prof.start_wall()

    encoder_cfg = cfg.model.to_encoder_config()
    model = AoFPolicyNetV3(encoder_cfg).to(device)
    model.eval()
    tcfg = cfg.train
    rollout_cfg = RolloutConfig(
        stack_bb=tcfg.stack_bb,
        teacher_samples=tcfg.teacher_samples,
        payoff_samples=tcfg.payoff_samples,
        seed=tcfg.seed,
        teacher_version=cfg.teacher_version,
        use_fast_teacher=tcfg.use_fast_teacher,
        teacher_margin_bb=tcfg.teacher_margin_bb,
        teacher_min_samples=tcfg.teacher_min_samples,
    )
    league = LeagueManager(cfg.league, encoder_cfg, device, rollout_cfg)
    scheduler = CurriculumScheduler(cfg.curriculum)
    cache = NashTeacherCache(
        use_fast=rollout_cfg.use_fast_teacher,
        teacher_margin_bb=tcfg.teacher_margin_bb,
        teacher_min_samples=tcfg.teacher_min_samples,
        teacher_version=cfg.teacher_version,
        teacher_lut_path=str(lut_path),
        teacher_samples=tcfg.teacher_samples,
        stack_bb=tcfg.stack_bb,
    )
    import random

    rng = random.Random(tcfg.seed)
    for update in range(args.updates):
        cache.clear()
        collect_update_rollout_batched(
            model,
            device,
            cache,
            rollout_cfg,
            league,
            scheduler,
            rng,
            update,
            n_parallel=tcfg.parallel_hands,
            total_hands=tcfg.batch_hands,
        )

    prof.stop_wall()
    set_profiler(None)
    summary = _summarize_prof(prof)
    summary["census_rows"] = len(rows)
    summary["updates"] = args.updates
    summary["target_hit_rate"] = args.target
    # V1 LUT table: pass on call coverage + zero ctx misses.
    # facing_two/three (deferred) may dominate wall time despite few calls.
    summary["verdict"] = (
        "PASS"
        if summary["hit_rate_by_calls"] >= args.target
        and summary["lut_ctx_miss"] == 0
        else "FAIL"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Rollout sample: {args.updates} updates x {tcfg.batch_hands} hands")
    print(f"  teacher calls (path buckets): {summary['total_teacher_calls']}")
    print(f"  lut_hit calls: {summary['lut_hit_calls']} ({100*summary['hit_rate_by_calls']:.1f}%)")
    print(f"  lut_hit wall (teacher.path): {100*summary['hit_rate_by_seconds']:.1f}%")
    print(f"  lut_ctx_miss (V1 path, ctx not in table): {summary['lut_ctx_miss']}")
    print(f"  rollout.hero_teacher total: {summary['hero_teacher_sec']:.3f}s")
    print()
    print("Per-path calls (counter):")
    for k, v in sorted(summary["counter_calls"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print()
    print(
        f"  non-V1 calls (e.g. facing_two): {summary['non_v1_calls']} "
        f"({100*summary['non_v1_legacy_sec']/max(summary['total_teacher_sec'],1e-9):.1f}% teacher wall)"
    )
    print(
        f"VERDICT: {summary['verdict']} "
        f"(target >= {100*args.target:.0f}% lut_hit by calls, 0 ctx_miss)"
    )
    print(f"Wrote {args.out}")
    sys.exit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
