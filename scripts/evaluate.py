#!/usr/bin/env python3
"""Full evaluation suite: PokerBench + bot pool + exploitability."""

import argparse
import json
from pathlib import Path

import torch

from pokerbot.agent import HybridAgent
from pokerbot.data.pokerbench import load_pokerbench
from pokerbot.data.tensor_dataset import _actions_from_text, _position_from_text
from pokerbot.eval.bots import GTOCopyBot
from pokerbot.eval.gate_harness import GATE_OPPONENT_POOL, run_gate_eval
from pokerbot.eval.gate_reference import (
    phase4_gate_passed,
    resolve_baseline_checkpoint,
    resolve_hybrid_baseline,
    resolve_search_baseline,
)
from pokerbot.eval.pokerbench_eval import evaluate_predictions
from pokerbot.eval.search_agent import SearchAgent
from pokerbot.models.policy import PseudoSiamesePolicy


def load_model(checkpoint: str, device: str) -> PseudoSiamesePolicy:
    from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy

    return load_pseudo_siamese_policy(checkpoint, device)


def _resolve_default_agent(use_gate: bool) -> str:
    if use_gate and phase4_gate_passed():
        return "search"
    return "hybrid"


def _build_bot(agent: str, device: str, use_gate: bool, checkpoint: str | None, time_budget_ms: int):
    if agent == "search":
        if use_gate:
            v_path, m_path, _ = resolve_search_baseline(require_gate=False)
        else:
            v_path = Path("checkpoints/history_value_net.pkl")
            m_path = Path("checkpoints/opponent_transformer_madt.pt")
        return SearchAgent(
            value_checkpoint=v_path,
            transformer_checkpoint=m_path,
            device=device,
            time_budget_ms=time_budget_ms,
            use_mock_leaf=not v_path.exists(),
        )

    if use_gate:
        hero_path, tf_path, _ = resolve_hybrid_baseline(require=False)
    else:
        hero_path, _ = resolve_baseline_checkpoint(require=False)
        tf_path = Path("checkpoints/opponent_transformer_madt.pt")

    ckpt = checkpoint or str(hero_path)
    model = load_model(ckpt, device)

    if agent == "gto":
        return GTOCopyBot(model, device=device)
    if agent == "hybrid":
        return HybridAgent(
            model,
            device=device,
            use_transformer=True,
            opponent_transformer_ckpt=str(tf_path) if tf_path.exists() else None,
        )
    raise ValueError(f"Unknown agent: {agent}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate poker policy")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--agent", choices=("gto", "hybrid", "search", "incumbent"), default="incumbent")
    p.add_argument("--use-gate-baselines", action="store_true")
    p.add_argument("--time-budget-ms", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--hands", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="eval_results.json")
    args = p.parse_args()

    device = args.device
    agent = args.agent
    if agent == "incumbent":
        agent = _resolve_default_agent(args.use_gate_baselines)

    results: dict = {"agent": agent, "seed": args.seed, "hands": args.hands}

    bot = _build_bot(agent, device, args.use_gate_baselines, args.checkpoint, args.time_budget_ms)
    gate_metrics = run_gate_eval(lambda: bot, hands=args.hands, seed=args.seed, opponent_pool=GATE_OPPONENT_POOL)
    results.update(gate_metrics)

    if agent in ("gto", "hybrid") and args.checkpoint:
        model = load_model(args.checkpoint, device)

        def predict_fn(instruction: str) -> str:
            from pokerbot.engine.actions import ACTION_VOCAB
            from pokerbot.data.tensor_dataset import _HOLDING_RE
            from pokerbot.engine.cards import parse_holding
            from pokerbot.encoding.weva_encoder import WEVAMapper

            from pokerbot.encoding.spot_scope import encode_spot_scope, fuse_position_scope
            from pokerbot.encoding.tensors import SPEC

            actions = torch.from_numpy(_actions_from_text(instruction)).unsqueeze(0).to(device)
            position = torch.from_numpy(
                fuse_position_scope(_position_from_text(instruction), encode_spot_scope(""))
            ).unsqueeze(0).float().to(device)
            cards = torch.zeros(1, *SPEC.card_shape, device=device)
            mapper = WEVAMapper.default()
            weva_id = 0
            holding = _HOLDING_RE.search(instruction)
            if holding:
                parts = holding.group(1).split(" and ")
                rank_map = {"ace": "A", "king": "K", "queen": "Q", "jack": "J", "ten": "T"}
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
                out = model(weva_ids, actions, position, cards=cards)
                idx = out.logits.argmax(dim=-1).item()
            label = ACTION_VOCAB[idx]
            if label.startswith("bet_"):
                return f"bet {label.split('_')[1]}"
            if label == "all_in":
                return "all in"
            return label

        pre = load_pokerbench("preflop", "test", limit=50)
        post = load_pokerbench("postflop", "test", limit=50)
        results["pokerbench_preflop"] = str(evaluate_predictions(pre, predict_fn))
        results["pokerbench_postflop"] = str(evaluate_predictions(post, predict_fn))
        results["pokerbench_overall"] = str(evaluate_predictions(pre + post, predict_fn))

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
