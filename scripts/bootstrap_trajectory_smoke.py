#!/usr/bin/env python3
"""Bootstrap minimal trajectory Parquet for Phase 3 smoke tests."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy
from pokerbot.opponent.scapt_manager import SCAPTConfig, SCAPTManager
from pokerbot.training.mixed_self_play import rollout_opponent_trajectories
from pokerbot.training.phase3 import resolve_hero_checkpoint
from pokerbot.training.trajectory_dataset import compute_trajectory_build_fingerprint


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap trajectory smoke data for Phase 3")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--hands", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="data/trajectories")
    p.add_argument("--no-scapt", action="store_true")
    p.add_argument("--scapt-population", type=int, default=4)
    p.add_argument("--scapt-mutation-sigma", type=float, default=None)
    p.add_argument("--scapt-diversity-floor", type=float, default=None)
    p.add_argument("--require-scapt-diversity", action="store_true")
    p.add_argument("--diversity-probe-states", type=int, default=16)
    args = p.parse_args()

    from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_hero_checkpoint()
    if not ckpt_path.exists():
        raise SystemExit(f"Hero checkpoint not found: {ckpt_path}")

    model = load_pseudo_siamese_policy(ckpt_path, device)

    scapt = None
    scapt_gen = 0
    if not args.no_scapt:
        scapt_cfg = SCAPTConfig(population_size=args.scapt_population)
        if args.scapt_mutation_sigma is not None:
            scapt_cfg.mutation_sigma = args.scapt_mutation_sigma
        scapt = SCAPTManager(scapt_cfg)
        scapt.init_population(model)

    build_fp = compute_trajectory_build_fingerprint(str(ckpt_path), scapt_gen)
    path = rollout_opponent_trajectories(
        model,
        device,
        args.hands,
        Path(args.output),
        scapt=scapt,
        shard_metadata={
            "build_fingerprint": build_fp,
            "hero_checkpoint": str(ckpt_path),
            "scapt_generation": scapt_gen,
            "use_scapt": not args.no_scapt,
            "bootstrap": True,
        },
        scapt_diversity_floor=args.scapt_diversity_floor,
        require_scapt_diversity=args.require_scapt_diversity,
        diversity_probe_states=args.diversity_probe_states,
    )
    print(f"Wrote trajectory smoke shard: {path} ({args.hands} hands)")


if __name__ == "__main__":
    main()
