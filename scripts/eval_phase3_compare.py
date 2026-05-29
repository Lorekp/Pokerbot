#!/usr/bin/env python3
"""Compare HybridAgent vs frozen GTOCopyBot on identical 6-max exploitability harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokerbot.agent import HybridAgent
from pokerbot.eval.bots import GTOCopyBot
from pokerbot.eval.gate_harness import GATE_OPPONENT_POOL, run_gate_eval
from pokerbot.eval.gate_reference import (
    DEFAULT_MANIFEST,
    promote_baseline,
    resolve_baseline_checkpoint,
    resolve_hybrid_baseline,
    sha256_file,
)
from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy


def _load_hero_policy(checkpoint: Path, device: str):
    return load_pseudo_siamese_policy(checkpoint, device)


def _eval_bot(bot, hands: int, seed: int) -> dict:
    return run_gate_eval(lambda: bot, hands=hands, seed=seed, opponent_pool=GATE_OPPONENT_POOL)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 HybridAgent vs frozen GTOCopyBot harness")
    p.add_argument("--transformer-checkpoint", default="checkpoints/opponent_transformer_madt.pt")
    p.add_argument("--baseline-checkpoint", default=None, help="Override frozen gate baseline")
    p.add_argument("--gate-manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--require-gate-reference", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--promote-baseline", default=None, help="Promote PATH then run harness")
    p.add_argument("--hands", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bb100-tolerance", type=float, default=5.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="eval_phase3_compare.json")
    p.add_argument("--use-frozen-transformer", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    manifest_path = Path(args.gate_manifest)
    if args.promote_baseline:
        promote_baseline(Path(args.promote_baseline), manifest_path=manifest_path)

    baseline_path, gate_ref = resolve_baseline_checkpoint(
        manifest_path=manifest_path,
        override=Path(args.baseline_checkpoint) if args.baseline_checkpoint else None,
        require=args.require_gate_reference,
    )

    device = args.device
    frozen_hero = _load_hero_policy(baseline_path, device)

    results: dict = {
        "seed": args.seed,
        "hands": args.hands,
        "gate_reference": {
            "hero_checkpoint": str(baseline_path),
            "hero_sha256": gate_ref.get("hero_sha256"),
            "weva_build_fingerprint": gate_ref.get("weva_build_fingerprint"),
            "manifest": str(manifest_path),
            "opponent_pool": list(GATE_OPPONENT_POOL),
        },
    }

    tf_path: Path | None = None
    if args.use_frozen_transformer:
        try:
            _, tf_path, _ = resolve_hybrid_baseline(
                manifest_path=manifest_path,
                require=False,
            )
        except (FileNotFoundError, ValueError):
            tf_path = None
    if tf_path is None:
        tf_path = Path(args.transformer_checkpoint)

    gto_bot = GTOCopyBot(frozen_hero, device=device)
    results["gto_copy"] = _eval_bot(gto_bot, args.hands, args.seed)

    if tf_path.exists():
        hybrid = HybridAgent(
            frozen_hero,
            device=device,
            use_transformer=True,
            opponent_transformer_ckpt=str(tf_path),
        )
        results["hybrid_agent"] = {
            "transformer_checkpoint": str(tf_path),
            "transformer_sha256": sha256_file(tf_path),
            "policy_checkpoint": str(baseline_path),
            **_eval_bot(hybrid, args.hands, args.seed + 1),
        }
    else:
        results["hybrid_agent"] = {"error": f"missing {tf_path}"}

    winner = "tie"
    gate_passed = False
    if "error" not in results.get("hybrid_agent", {}):
        gto_m = results["gto_copy"]
        hy_m = results["hybrid_agent"]
        hybrid_better = (
            hy_m["exploiter_ev_bb_per_hand"]
            < gto_m["exploiter_ev_bb_per_hand"]
        )
        bb_ok = hy_m["bb100_vs_pool"] >= gto_m["bb100_vs_pool"] - args.bb100_tolerance
        if hybrid_better and bb_ok:
            winner = "hybrid"
            gate_passed = True
        elif (
            gto_m["exploiter_ev_bb_per_hand"]
            < hy_m["exploiter_ev_bb_per_hand"]
            and gto_m["bb100_vs_pool"] >= hy_m["bb100_vs_pool"] - args.bb100_tolerance
        ):
            winner = "gto"

    results["winner"] = winner
    results["gate_passed"] = gate_passed
    results["gate_criteria"] = {
        "relative_exploitability": "lower exploiter_ev_bb_per_hand is better (signed; not clamped)",
        "bb100_vs_pool": f"hybrid not worse than gto_copy - {args.bb100_tolerance} bb/100",
        "baseline": "frozen hero policy from gate_reference.json (both arms use same weights)",
    }

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
