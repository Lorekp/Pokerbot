#!/usr/bin/env python3
"""Distill neural-CFR AverageStrategyNet into WEVA PseudoSiamesePolicy."""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.training.blueprint_distill import BlueprintDistillConfig, distill_blueprint_to_policy


def main() -> None:
    p = argparse.ArgumentParser(description="Distill CFR blueprint to WEVA policy")
    p.add_argument("--cfr-checkpoint", default="checkpoints/history_value_net.pkl")
    p.add_argument("--output", default="checkpoints/policy_cfr_blueprint.pt")
    p.add_argument("--hands", type=int, default=200)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    distill_blueprint_to_policy(
        BlueprintDistillConfig(
            cfr_checkpoint=Path(args.cfr_checkpoint),
            output_path=Path(args.output),
            hands=args.hands,
            epochs=args.epochs,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
