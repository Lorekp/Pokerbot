#!/usr/bin/env python3
"""Full evaluation suite: PokerBench + bot pool + exploitability."""

import argparse
import json
from pathlib import Path

import torch

from pokerbot.data.pokerbench import load_pokerbench
from pokerbot.data.tensor_dataset import _actions_from_text, _position_from_text
from pokerbot.eval.bots import GTOCopyBot, BOT_REGISTRY
from pokerbot.eval.exploitability import estimate_exploitability
from pokerbot.eval.head_to_head import evaluate_vs_bot_pool, run_head_to_head
from pokerbot.eval.pokerbench_eval import evaluate_predictions
from pokerbot.models.policy import PseudoSiamesePolicy


def load_model(checkpoint: str, device: str) -> PseudoSiamesePolicy:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PseudoSiamesePolicy().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate poker policy")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--hands", type=int, default=100)
    p.add_argument("--output", type=str, default="eval_results.json")
    args = p.parse_args()

    device = args.device
    results = {}

    if args.checkpoint:
        model = load_model(args.checkpoint, device)

        def predict_fn(instruction: str) -> str:
            from pokerbot.engine.actions import ACTION_VOCAB
            from pokerbot.data.tensor_dataset import _HOLDING_RE
            from pokerbot.engine.cards import parse_holding
            from pokerbot.encoding.weva_encoder import WEVAMapper

            actions = torch.from_numpy(_actions_from_text(instruction)).unsqueeze(0).to(device)
            position = torch.from_numpy(_position_from_text(instruction)).unsqueeze(0).to(device)
            mapper = WEVAMapper.default()
            weva_id = 0
            holding = _HOLDING_RE.search(instruction)
            if holding:
                parts = holding.group(1).split(" and ")
                rank_map = {
                    "ace": "A", "king": "K", "queen": "Q", "jack": "J", "ten": "T",
                }
                cards_str = ""
                for part in parts[:2]:
                    words = part.strip().lower().replace(" of ", " ").split()
                    r = rank_map.get(words[0], words[0][0].upper() if words else "A")
                    s = words[-1][0].lower() if words else "h"
                    cards_str += f"{r}{s}"
                if len(cards_str) >= 4:
                    hole = parse_holding(cards_str[:4])
                    pos_idx = int(position.argmax().item())
                    weva_id = mapper.lookup(hole, position=pos_idx)
            weva_ids = torch.tensor([weva_id], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(weva_ids, actions, position)
                idx = out.logits.argmax(dim=-1).item()
            label = ACTION_VOCAB[idx]
            if label.startswith("bet_"):
                frac = label.split("_")[1]
                return f"bet {frac}"
            if label == "all_in":
                return "all in"
            return label

        pre = load_pokerbench("preflop", "test", limit=50)
        post = load_pokerbench("postflop", "test", limit=50)
        results["pokerbench_preflop"] = str(evaluate_predictions(pre, predict_fn))
        results["pokerbench_postflop"] = str(evaluate_predictions(post, predict_fn))
        results["pokerbench_overall"] = str(evaluate_predictions(pre + post, predict_fn))

        bot = GTOCopyBot(model, device=device)
        results["vs_bots"] = evaluate_vs_bot_pool(bot, hands_per_bot=args.hands)
        results["exploitability"] = estimate_exploitability(bot, hands=args.hands).__dict__
    else:
        # Engine-only smoke eval
        from pokerbot.eval.bots import NitBot, LAGBot
        from pokerbot.eval.head_to_head import run_head_to_head

        nit, lag = NitBot(), LAGBot()
        r = run_head_to_head(lag, nit, hands=args.hands)
        results["lag_vs_nit_bb100"] = r.bb_per_100

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
