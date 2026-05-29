#!/usr/bin/env python3
"""Compare PPO vs CFR blueprint on identical 6-max exploitability harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pokerbot.eval.bots import GTOCopyBot
from pokerbot.eval.exploitability import (
    RELATIVE_EXPLOITABILITY_METRIC_NOTE,
    estimate_exploitability,
    require_exploitability_harness_sanity,
)
from pokerbot.eval.head_to_head import evaluate_vs_bot_pool
from pokerbot.eval.jax_blueprint_bot import JAXBlueprintBot
from pokerbot.models.policy import PseudoSiamesePolicy


def _load_policy(checkpoint: Path, device: str) -> PseudoSiamesePolicy:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PseudoSiamesePolicy().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _eval_policy_bot(bot, hands: int, seed: int) -> dict:
    exploit = estimate_exploitability(bot, hands=hands, seed=seed)
    pool = evaluate_vs_bot_pool(lambda s: bot(s), hands_per_bot=max(hands // 3, 1), seed=seed)
    bb100 = sum(pool.values()) / max(len(pool), 1)
    return {
        "exploiter_ev_bb_per_hand": exploit.exploiter_ev_bb_per_hand,
        "relative_exploitability_metric_note": RELATIVE_EXPLOITABILITY_METRIC_NOTE,
        "bb100_vs_pool": bb100,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 PPO vs CFR comparison harness")
    p.add_argument("--ppo-checkpoint", default="checkpoints/policy_selfplay_mixed.pt")
    p.add_argument("--cfr-checkpoint", default="checkpoints/history_value_net.pkl")
    p.add_argument("--cfr-policy-checkpoint", default="checkpoints/policy_cfr_blueprint.pt")
    p.add_argument("--hands", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bb100-tolerance", type=float, default=5.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="eval_phase2_compare.json")
    args = p.parse_args()

    harness_sanity = require_exploitability_harness_sanity(hands=400, seed=args.seed)

    device = args.device
    results: dict = {
        "seed": args.seed,
        "hands": args.hands,
        "harness_sanity_precondition": harness_sanity,
        "relative_exploitability_metric_note": RELATIVE_EXPLOITABILITY_METRIC_NOTE,
    }

    ppo_path = Path(args.ppo_checkpoint)
    if ppo_path.exists():
        ppo_model = _load_policy(ppo_path, device)
        ppo_bot = GTOCopyBot(ppo_model, device=device)
        results["ppo"] = {"checkpoint": str(ppo_path), **_eval_policy_bot(ppo_bot, args.hands, args.seed)}
    else:
        results["ppo"] = {"error": f"missing {ppo_path}"}

    cfr_policy_path = Path(args.cfr_policy_checkpoint)
    if cfr_policy_path.exists():
        cfr_model = _load_policy(cfr_policy_path, device)
        cfr_weva_bot = GTOCopyBot(cfr_model, device=device)
        results["cfr_blueprint"] = {
            "checkpoint": str(cfr_policy_path),
            **_eval_policy_bot(cfr_weva_bot, args.hands, args.seed + 1),
        }
    elif Path(args.cfr_checkpoint).exists():
        jax_bot = JAXBlueprintBot(args.cfr_checkpoint)
        results["cfr_blueprint"] = {
            "checkpoint": str(args.cfr_checkpoint),
            **_eval_policy_bot(jax_bot, args.hands, args.seed + 1),
        }
    else:
        results["cfr_blueprint"] = {"error": "no CFR checkpoint found"}

    winner = "tie"
    gate_passed = False
    if "error" not in results.get("ppo", {}) and "error" not in results.get("cfr_blueprint", {}):
        ppo_m = results["ppo"]
        cfr_m = results["cfr_blueprint"]
        cfr_better_exploit = (
            cfr_m["exploiter_ev_bb_per_hand"]
            < ppo_m["exploiter_ev_bb_per_hand"]
        )
        bb_ok = cfr_m["bb100_vs_pool"] >= ppo_m["bb100_vs_pool"] - args.bb100_tolerance
        if cfr_better_exploit and bb_ok:
            winner = "cfr"
            gate_passed = True
        elif (
            ppo_m["exploiter_ev_bb_per_hand"]
            < cfr_m["exploiter_ev_bb_per_hand"]
            and ppo_m["bb100_vs_pool"] >= cfr_m["bb100_vs_pool"] - args.bb100_tolerance
        ):
            winner = "ppo"

    results["winner"] = winner
    results["gate_passed"] = gate_passed
    results["gate_criteria"] = {
        "relative_exploitability": "lower exploiter_ev_bb_per_hand is better (signed; not clamped)",
        "bb100_vs_pool": f"cfr not worse than ppo - {args.bb100_tolerance} bb/100",
    }

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
