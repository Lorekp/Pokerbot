#!/usr/bin/env python3
"""Pre-HU-run validation: SFT init parity, Gate A on init policy, encode throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pokerbot.training.nlhe_gate_a import run_gate_a_report
from pokerbot.training.nlhe_spot_replay import load_heldout_records, verify_parquet_encode_replay
from pokerbot.training.nlhe_sft_init import verify_sft_forward_parity
from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer
from pokerbot.engine.jax_weva_encode import encode_nlhe_weva_bundle
from pokerbot.engine.jax_weva_fast import encode_nlhe_bundle_fast, default_weva_bucket_table
from pokerbot.encoding.weva_encoder import WEVAMapper
from pokerbot.training.nlhe_spot_replay import jax_state_from_parquet_record

ROOT = Path(__file__).resolve().parents[1]


def profile_encode_throughput(*, n_nodes: int = 200) -> dict:
    """Compare slow bridge path vs fast in-graph path."""
    from pokerbot.training.nlhe_collision_check import _walk_hu_decision_nodes

    mapper = WEVAMapper.default()
    table = default_weva_bucket_table()
    nodes = _walk_hu_decision_nodes(n_hands=max(8, n_nodes // 8), seed=0)[:n_nodes]
    if not nodes:
        return {"error": "no nodes sampled"}

    def bench(fn):
        for st, seat in nodes[:10]:
            fn(st, seat)
        t0 = time.perf_counter()
        for st, seat in nodes:
            fn(st, seat)
        elapsed = time.perf_counter() - t0
        return elapsed

    slow_t = bench(lambda s, seat: encode_nlhe_weva_bundle(s, seat, mapper))
    fast_t = bench(lambda s, seat: encode_nlhe_bundle_fast(s, seat, weva_table=table))
    return {
        "nodes": len(nodes),
        "slow_ms_per_encode": round(slow_t / len(nodes) * 1000, 3),
        "fast_ms_per_encode": round(fast_t / len(nodes) * 1000, 3),
        "speedup": round(slow_t / max(fast_t, 1e-9), 2),
        "fast_encodes_per_sec": round(len(nodes) / max(fast_t, 1e-9), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="NLHE pre-train validation (SFT init + Gate A + encode profile)")
    p.add_argument("--sft-checkpoint", default="checkpoints/policy_sft_prod6000.pt")
    p.add_argument("--output", default="logs/nlhe_pretrain_validation.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--encode-nodes", type=int, default=200)
    args = p.parse_args()

    report: dict = {"sft_checkpoint": args.sft_checkpoint}
    sft_path = Path(args.sft_checkpoint)

    records = load_heldout_records()
    replay_checks = [verify_parquet_encode_replay(r) for r in records]
    report["parquet_jax_replay"] = {
        "n_rows": len(replay_checks),
        "all_ok": all(all(c[k] for k in ("weva_ok", "cards_ok", "actions_ok")) for c in replay_checks),
        "failures": [c for c in replay_checks if not all(c[k] for k in ("weva_ok", "cards_ok", "actions_ok"))],
    }

    if sft_path.exists():
        report["sft_weight_parity"] = verify_sft_forward_parity(sft_path, records=records, device=args.device)
        cfg = SDCFRNLHEConfig(iterations=0, traversals_per_iter=0, sft_checkpoint=str(sft_path))
        trainer = SDCFRNLHETrainer(cfg)
        gate_a = run_gate_a_report(trainer=trainer, sft_path=str(sft_path), device=args.device)
        report["gate_a_iter0"] = {
            "combined_mean_kl_sd_cfr": gate_a["combined_mean_kl_sd_cfr"],
            "sft_reference_combined_mean": gate_a["sft_reference_combined_mean"],
            "vanilla_mean_kl": gate_a["vanilla"]["mean_kl"],
            "bunching_mean_kl": gate_a["bunching"]["mean_kl"],
            "replay_failures": gate_a["vanilla"]["replay_failures"] + gate_a["bunching"]["replay_failures"],
        }
        # Iter-0 SD-CFR policy should match SFT vs labels
        if gate_a["sft_reference_combined_mean"] is not None:
            delta = abs(gate_a["combined_mean_kl_sd_cfr"] - gate_a["sft_reference_combined_mean"])
            report["gate_a_iter0"]["kl_delta_vs_sft_reference"] = delta
            report["gate_a_iter0"]["init_holds"] = delta < 0.02
    else:
        report["sft_missing"] = f"Checkpoint not found: {sft_path} — copy policy_sft_prod6000.pt before HU run"

    report["encode_throughput"] = profile_encode_throughput(n_nodes=args.encode_nodes)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
