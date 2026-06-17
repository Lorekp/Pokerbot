#!/usr/bin/env python3
"""6-max SD-CFR local smoke (pipeline correctness, small iteration count)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokerbot.training.sd_cfr_nlhe import (
    SDCFRNLHEConfig,
    SDCFRNLHETrainer,
    save_nlhe_checkpoint,
)


def main() -> None:
    p = argparse.ArgumentParser(description="6-max SD-CFR local smoke")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--traversals", type=int, default=2)
    p.add_argument("--output", default="checkpoints/sd_cfr_6max_smoke.pkl")
    p.add_argument("--log", default="logs/sd_cfr_6max_smoke.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = SDCFRNLHEConfig(
        iterations=args.iterations,
        traversals_per_iter=args.traversals,
        seed=args.seed,
        occupancy=(True, True, True, True, True, True),
        max_traverse_depth=24,
        train_batch_size=256,
    )
    trainer = SDCFRNLHETrainer(cfg)
    losses = trainer.train()
    trainer.assert_profile_ev_in_range(samples=4)
    out = Path(args.output)
    save_nlhe_checkpoint(trainer, out)

    log = {
        "iterations": args.iterations,
        "traversals_per_iter": args.traversals,
        "occupancy": "6-max",
        "losses": losses,
        "checkpoint": str(out),
    }
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
