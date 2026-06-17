#!/usr/bin/env python3
"""Benchmark SD-CFR NLHE traversal throughput (env + net + encoding)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from pokerbot.engine.jax_weva_encode import encode_nlhe_weva_bundle
from pokerbot.encoding.weva_encoder import WEVAMapper
from pokerbot.training.nlhe_collision_check import _walk_hu_decision_nodes
from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer

ROOT = Path(__file__).resolve().parents[1]


def _profile_encode(n_nodes: int = 200) -> dict:
    mapper = WEVAMapper.default()
    nodes = _walk_hu_decision_nodes(n_hands=max(8, n_nodes // 8), seed=0)[:n_nodes]
    t0 = time.perf_counter()
    for state, seat in nodes:
        encode_nlhe_weva_bundle(state, seat, mapper)
    elapsed = time.perf_counter() - t0
    return {
        "encode_nodes": len(nodes),
        "encode_total_sec": round(elapsed, 3),
        "encode_ms_per_node": round((elapsed / max(len(nodes), 1)) * 1000, 3),
        "encode_nodes_per_sec": round(len(nodes) / max(elapsed, 1e-9), 2),
    }


def main() -> None:
    for tag, occ in (
        ("hu", (True, True, False, False, False, False)),
        ("6max", (True, True, True, True, True, True)),
    ):
        cfg = SDCFRNLHEConfig(
            iterations=1,
            traversals_per_iter=4,
            occupancy=occ,
            max_traverse_depth=24,
        )
        trainer = SDCFRNLHETrainer(cfg)
        t0 = time.perf_counter()
        trainer.train_iteration()
        elapsed = time.perf_counter() - t0
        visits = cfg.traversals_per_iter * cfg.deal_sweeps * len(trainer._live_seats)
        print(f"{tag}: {visits} traverser visits in {elapsed:.2f}s ({visits/elapsed:.2f} visits/s)")

    encode_stats = _profile_encode()
    print(
        f"encode: {encode_stats['encode_nodes']} nodes in {encode_stats['encode_total_sec']}s "
        f"({encode_stats['encode_ms_per_node']} ms/node)",
        flush=True,
    )

    report = {"note": "One train_iteration per occupancy; encode profile included", **encode_stats}
    out = ROOT / "logs" / "sd_cfr_nlhe_throughput.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
