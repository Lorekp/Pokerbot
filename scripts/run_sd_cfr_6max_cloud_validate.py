#!/usr/bin/env python3
"""Validate50-style gate: short SD-CFR 6-max shard on cloud/local before full blueprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer, save_nlhe_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="SD-CFR 6-max validate50 gate")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--traversals", type=int, default=2)
    p.add_argument("--output", default="checkpoints/sd_cfr_6max_validate50.pkl")
    p.add_argument("--log", default="logs/sd_cfr_6max_validate50.json")
    args = p.parse_args()

    cfg = SDCFRNLHEConfig(
        iterations=args.iterations,
        traversals_per_iter=args.traversals,
        occupancy=(True, True, True, True, True, True),
        max_traverse_depth=24,
    )
    trainer = SDCFRNLHETrainer(cfg)
    losses = trainer.train(log_every=1)
    trainer.assert_profile_ev_in_range(samples=4)
    save_nlhe_checkpoint(trainer, Path(args.output))
    log = {
        "iterations": args.iterations,
        "traversals_per_iter": args.traversals,
        "losses": losses,
        "checkpoint": args.output,
        "completion_authority": "checkpoint artifact exists",
    }
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
