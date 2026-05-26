#!/usr/bin/env python3
"""Evaluate policy vs bots with per-street P&L breakdown."""

import argparse
import json
from pathlib import Path

import torch

from pokerbot.eval.bots import GTOCopyBot
from pokerbot.eval.street_breakdown import evaluate_street_breakdown_vs_bots
from pokerbot.models.policy import PseudoSiamesePolicy


def load_model(checkpoint: str, device: str) -> PseudoSiamesePolicy:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PseudoSiamesePolicy().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Street-level eval vs bot pool")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hands", type=int, default=500)
    p.add_argument("--bots", default="lag,donk", help="Comma-separated bot names")
    p.add_argument("--output", default="eval_street_breakdown.json")
    args = p.parse_args()

    model = load_model(args.checkpoint, args.device)
    bot = GTOCopyBot(model, device=args.device)
    names = [x.strip() for x in args.bots.split(",") if x.strip()]

    results = evaluate_street_breakdown_vs_bots(bot, names, hands_per_bot=args.hands)
    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
