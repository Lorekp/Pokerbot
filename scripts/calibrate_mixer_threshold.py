#!/usr/bin/env python3
"""Calibrate HybridPolicy belief-confidence threshold from held-out showdown trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pokerbot.opponent.mixer_calibration import calibrate_belief_threshold
from pokerbot.opponent.opponent_transformer import OpponentTransformer, OpponentTransformerConfig
from pokerbot.training.trajectory_dataset import OpponentTrajectoryDataset


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate mixer belief-confidence threshold")
    p.add_argument("--transformer-checkpoint", default="checkpoints/opponent_transformer_madt.pt")
    p.add_argument("--parquet-glob", default="data/trajectories/*.parquet")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default="checkpoints/mixer_threshold.json")
    args = p.parse_args()

    ckpt_path = Path(args.transformer_checkpoint)
    if not ckpt_path.exists():
        print(
            f"No transformer checkpoint at {ckpt_path}; keeping provisional threshold 0.35.\n"
            "See docs/phase3-runbook.md § Mixer confidence."
        )
        return

    ds = OpponentTrajectoryDataset(args.parquet_glob)
    if len(ds) == 0:
        print(f"No trajectories at {args.parquet_glob}; keeping provisional threshold 0.35.")
        return

    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = OpponentTransformer(OpponentTransformerConfig()).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_mask: list[torch.Tensor] = []

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=ds.collate)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = batch.pop("belief_targets")
            mask = batch.pop("cards_known_mask")
            batch.pop("policy_targets", None)
            batch.pop("action_known_mask", None)
            batch.pop("causal_mask", None)
            batch.pop("opp_seat_map", None)
            out = model(
                batch["token_types"],
                batch["token_features"],
                batch["attn_mask"],
                batch["hero_seat"],
            )
            all_logits.append(out["belief_logits"].cpu())
            all_targets.append(targets.cpu())
            all_mask.append(mask.cpu())

    belief_logits = torch.cat(all_logits, dim=0)
    belief_targets = torch.cat(all_targets, dim=0)
    cards_known_mask = torch.cat(all_mask, dim=0)

    curve = calibrate_belief_threshold(belief_logits, belief_targets, cards_known_mask)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chosen_threshold": curve.chosen_threshold,
        "calibrated": True,
        "transformer_checkpoint": str(ckpt_path),
        "parquet_glob": args.parquet_glob,
        "curve": curve.as_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Calibrated threshold={curve.chosen_threshold:.3f} → {out_path}")


if __name__ == "__main__":
    main()
