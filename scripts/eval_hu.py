#!/usr/bin/env python3
"""Heads-up bb/100 evaluation vs AlphaNLHoldem (or fallback)."""

import argparse
import random

from pokerbot.engine.hu_env import play_hu_hand
from pokerbot.eval.bots import GTOCopyBot, build_bot
from pokerbot.models.policy import PseudoSiamesePolicy
import torch


def main() -> None:
    p = argparse.ArgumentParser(description="HU eval vs AlphaNLHoldem")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--hands", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    villain = build_bot("alpha_hu", rng=rng)

    if args.checkpoint:
        model = PseudoSiamesePolicy()
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        hero = GTOCopyBot(model, device=args.device)
    else:
        from pokerbot.eval.bots import RandomBot

        hero = RandomBot(rng)

    total = 0.0
    for h in range(args.hands):
        hero_seat = h % 2
        _, deltas = play_hu_hand(hero, villain, rng=rng, hero_seat=hero_seat)
        total += deltas.get(hero_seat, 0.0)

    bb100 = total / args.hands * 100
    print(f"HU vs alpha_hu: {args.hands} hands, bb/100 = {bb100:.2f}")


if __name__ == "__main__":
    main()
