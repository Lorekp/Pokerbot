#!/usr/bin/env python3
"""Measure encode duplication: cache hits vs misses, distinct infosets vs encode calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from pokerbot.engine.jax_engine import init_hand
from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer

ROOT = Path(__file__).resolve().parents[1]


def _run_probe(*, traversals: int, deal_sweeps: int, seed: int) -> dict:
    cfg = SDCFRNLHEConfig(
        iterations=1,
        traversals_per_iter=traversals,
        deal_sweeps=deal_sweeps,
        sft_checkpoint=None,
        seed=seed,
        max_traverse_depth=64,
    )
    trainer = SDCFRNLHETrainer(cfg)
    trainer._begin_iteration_regret()
    distinct_fp: set[tuple] = set()
    occ = jnp.asarray(trainer.occupancy, dtype=bool)
    visits = traversals * max(1, deal_sweeps) * trainer._n_live

    for _ in range(max(1, deal_sweeps)):
        for _ in range(traversals):
            trainer.key, k1, k2 = jax.random.split(trainer.key, 3)
            state = init_hand(
                k1,
                button=jnp.int32(cfg.button_seat),
                occupancy=occ,
            )
            for seat in trainer._live_seats:
                trainer.traverse(state, seat, k2)

    for fp in trainer._bundle_cache.values():
        distinct_fp.add(fp.fingerprint())

    stats = dict(trainer._encode_stats)
    misses = stats["cache_misses"]
    hits = stats["cache_hits"]
    calls = stats["encode_calls"]
    total_lookups = hits + misses
    dup_ratio = 1.0 - (misses / max(total_lookups, 1))
    per_visit_encodes = misses / max(visits, 1)

    return {
        "visits": visits,
        "encode_lookups": total_lookups,
        "encode_calls_actual": calls,
        "cache_hits": hits,
        "cache_misses": misses,
        "duplicate_lookup_fraction": round(dup_ratio, 4),
        "distinct_infosets_encoded": len(distinct_fp),
        "distinct_structural_keys": len(trainer._bundle_cache),
        "encodes_per_visit": round(per_visit_encodes, 1),
        "pre_fix_calls_per_node_estimate": "2-3 _bundle per decision node (sigma+key+bundle)",
        "interpretation": (
            "cache_misses = actual encode_nlhe_bundle_fast calls; "
            "cache_hits = avoided re-encode on same structural infoset within iteration"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traversals", type=int, default=4)
    p.add_argument("--deal-sweeps", type=int, default=2)
    p.add_argument("--output", default="logs/sd_cfr_encode_duplication.json")
    args = p.parse_args()

    report = {
        "without_cache_hypothesis": {
            "note": "Before per-node dedup + memo, each node called _bundle 2-3×",
            "expected_inflation_factor": "2-3× on encode_calls vs distinct nodes",
        },
        "with_bundle_cache": _run_probe(
            traversals=args.traversals, deal_sweeps=args.deal_sweeps, seed=0
        ),
    }
    w = report["with_bundle_cache"]
    print(json.dumps(report, indent=2), flush=True)
    print(
        f"\nEncode: {w['encode_calls_actual']} actual encodes, "
        f"{w['cache_hits']} cache hits, "
        f"{w['duplicate_lookup_fraction']*100:.1f}% duplicate lookups avoided, "
        f"{w['encodes_per_visit']:.1f} encodes/visit, "
        f"{w['distinct_infosets_encoded']} distinct infosets",
        flush=True,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
