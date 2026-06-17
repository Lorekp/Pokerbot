#!/usr/bin/env python3
"""
Local proxy for the cloud target-scale merge gate (handoff Step 3).

Runs the bulk-sync iteration through the object-store data path
(`run_iteration_via_store_coordinated`) at the largest scale that is locally tractable, with the
real EC2-spot failure modes induced (kill→reclaim, out-of-order gather, duplicate delivery), and
asserts the result is bit-identical to the single-threaded reference. This is the trustworthy
reference the handoff calls for: the cloud coordinator runs the SAME logic at 1080 visits / 8
workers, so passing here at many shards (the "concat ordering with many shards returning out of
order" risk) is the local evidence that the merge is scale-stable before any EC2 spend.

The single-threaded reference is the bottleneck, so scale is chosen for tractability (shallower
`--depth`, moderate `--traversals`) while pushing `--shards` high to stress the merge ordering.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer
from pokerbot.training.sd_cfr_nlhe_cloud import (
    LocalDirObjectStore,
    run_iteration_via_store_coordinated,
)
from pokerbot.training.sd_cfr_nlhe_distributed import compare_trainer_states


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traversals", type=int, default=32)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", default="logs/sd_cfr_hu_store_merge_gate_work")
    p.add_argument("--output", default="logs/sd_cfr_hu_store_merge_gate.json")
    args = p.parse_args()

    cfg = SDCFRNLHEConfig(
        iterations=1,
        traversals_per_iter=args.traversals,
        deal_sweeps=1,
        train_batch_size=args.batch,
        sft_checkpoint=None,
        seed=args.seed,
        max_traverse_depth=args.depth,
    )
    visits = args.traversals * 2
    print(
        f"Store merge gate: {visits} visits, {args.shards} shards, depth={args.depth}, "
        f"induced kill+shuffle+duplicate",
        flush=True,
    )

    t0 = time.perf_counter()
    ref = SDCFRNLHETrainer(cfg)
    np.random.seed(cfg.seed)
    loss_ref = ref.train_iteration(1.0)
    ref_wall = round(time.perf_counter() - t0, 1)
    print(f"  single-threaded reference: loss={loss_ref:.6f} wall={ref_wall}s", flush=True)

    t1 = time.perf_counter()
    dist = SDCFRNLHETrainer(cfg)
    np.random.seed(cfg.seed)
    store = LocalDirObjectStore(Path(args.work_dir))
    killed = frozenset({args.shards // 2}) if args.shards >= 2 else frozenset()
    duplicate = frozenset({0})
    loss_dist = run_iteration_via_store_coordinated(
        dist,
        1.0,
        num_shards=args.shards,
        store=store,
        killed_shard_ids=killed,
        shuffle_gather=True,
        duplicate_shard_ids=duplicate,
    )
    dist_wall = round(time.perf_counter() - t1, 1)
    print(f"  store-path (coordinated): loss={loss_dist:.6f} wall={dist_wall}s", flush=True)

    cmp = compare_trainer_states(ref, dist)
    passed = bool(
        cmp["cum_regret_bit_identical"]
        and cmp["strat_sum_bit_identical"]
        and cmp["params_match"]
        and abs(loss_ref - loss_dist) < 1e-5
    )
    result = {
        "gate": "store_path_merge_equals_single_threaded",
        "passed": passed,
        "visits": visits,
        "shards": args.shards,
        "induced": {"killed_shard_ids": sorted(killed), "shuffle_gather": True,
                    "duplicate_shard_ids": sorted(duplicate)},
        "max_cum_regret_diff": cmp["max_cum_regret_diff"],
        "max_strat_sum_diff": cmp["max_strat_sum_diff"],
        "params_match": cmp["params_match"],
        "loss_ref": loss_ref,
        "loss_dist": loss_dist,
        "wall_sec_ref": ref_wall,
        "wall_sec_dist": dist_wall,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote {out} -> {'PASS' if passed else 'FAIL'}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
