#!/usr/bin/env python3
"""Phase 4 gate: SearchAgent vs frozen HybridAgent + GTO safety floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokerbot.agent import HybridAgent
from pokerbot.eval.bots import GTOCopyBot
from pokerbot.eval.gate_harness import GATE_OPPONENT_POOL, run_gate_eval
from pokerbot.eval.gate_reference import (
    DEFAULT_MANIFEST,
    append_search_gate_result,
    load_weva_fingerprint,
    resolve_hybrid_baseline,
    sha256_file,
)
from pokerbot.eval.search_agent import SearchAgent
from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4 SearchAgent vs frozen HybridAgent harness")
    p.add_argument("--value-checkpoint", default="checkpoints/history_value_net.pkl")
    p.add_argument("--madt-checkpoint", default="checkpoints/opponent_transformer_madt.pt")
    p.add_argument("--gate-manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--baseline-checkpoint", default=None)
    p.add_argument("--transformer-baseline", default=None)
    p.add_argument("--require-gate-reference", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hands", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bb100-tolerance", type=float, default=5.0)
    p.add_argument("--gto-safety-tolerance", type=float, default=2.0)
    p.add_argument("--gto-exploit-delta", type=float, default=0.05)
    p.add_argument("--require-gto-safety", action="store_true")
    p.add_argument("--time-budget-ms", type=int, default=2000)
    p.add_argument("--use-mock-leaf", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="eval_phase4_compare.json")
    args = p.parse_args()

    manifest_path = Path(args.gate_manifest)
    hero_path, tf_baseline_path, gate_ref = resolve_hybrid_baseline(
        manifest_path=manifest_path,
        hero_override=Path(args.baseline_checkpoint) if args.baseline_checkpoint else None,
        transformer_override=Path(args.transformer_baseline) if args.transformer_baseline else None,
        require=args.require_gate_reference,
    )

    device = args.device
    frozen_hero = load_pseudo_siamese_policy(hero_path, device)

    hvn_path = Path(args.value_checkpoint)
    madt_path = Path(args.madt_checkpoint)

    results: dict = {
        "seed": args.seed,
        "hands": args.hands,
        "gate_reference": {
            "hero_checkpoint": str(hero_path),
            "hero_sha256": gate_ref.get("hero_sha256"),
            "transformer_checkpoint": str(tf_baseline_path),
            "transformer_sha256": gate_ref.get("transformer_sha256"),
            "weva_build_fingerprint": gate_ref.get("weva_build_fingerprint") or load_weva_fingerprint(),
            "harness_seed": args.seed,
            "opponent_pool": list(GATE_OPPONENT_POOL),
            "manifest": str(manifest_path),
        },
        "search_artifacts": {
            "value_checkpoint": str(hvn_path),
            "value_sha256": sha256_file(hvn_path) if hvn_path.exists() else None,
            "candidate_madt_checkpoint": str(madt_path),
            "candidate_madt_sha256": sha256_file(madt_path) if madt_path.exists() else None,
            "time_budget_ms": args.time_budget_ms,
        },
    }

    gto_bot = GTOCopyBot(frozen_hero, device=device)
    results["gto_copy"] = run_gate_eval(lambda: gto_bot, hands=args.hands, seed=args.seed)

    hybrid = HybridAgent(
        frozen_hero,
        device=device,
        use_transformer=True,
        opponent_transformer_ckpt=str(tf_baseline_path),
    )
    results["hybrid_agent"] = {
        "policy_checkpoint": str(hero_path),
        "transformer_checkpoint": str(tf_baseline_path),
        **run_gate_eval(lambda: hybrid, hands=args.hands, seed=args.seed + 1),
    }

    search = SearchAgent(
        value_checkpoint=hvn_path,
        transformer_checkpoint=madt_path,
        device=device,
        time_budget_ms=args.time_budget_ms,
        use_mock_leaf=args.use_mock_leaf or not hvn_path.exists(),
    )
    search_metrics = run_gate_eval(lambda: search, hands=args.hands, seed=args.seed + 2)
    search_metrics["solve_metadata_sample"] = search.last_solve_metadata
    results["search_agent"] = search_metrics

    hy = results["hybrid_agent"]
    sr = results["search_agent"]
    gto = results["gto_copy"]

    primary_exploit = (
        sr["exploiter_ev_bb_per_hand"]
        < hy["exploiter_ev_bb_per_hand"]
    )
    primary_bb = sr["bb100_vs_pool"] >= hy["bb100_vs_pool"] - args.bb100_tolerance
    gate_passed = primary_exploit and primary_bb

    gto_bb_ok = sr["bb100_vs_pool"] >= gto["bb100_vs_pool"] - args.gto_safety_tolerance
    gto_exploit_ok = (
        sr["exploiter_ev_bb_per_hand"]
        <= gto["exploiter_ev_bb_per_hand"] + args.gto_exploit_delta
    )
    gto_safety_passed = gto_bb_ok and gto_exploit_ok

    results["gate_passed"] = gate_passed
    results["gto_safety_passed"] = gto_safety_passed
    results["winner"] = "search" if gate_passed else ("hybrid" if not primary_exploit else "tie")
    results["gate_criteria"] = {
        "primary_blocking": (
            "search beats frozen HybridAgent "
            "(exploiter_ev_bb_per_hand + bb100 tolerance)"
        ),
        "gto_safety_floor": "non-blocking; search does not bleed EV vs GTOCopyBot",
        "opponent_pool": "fixed static BOT_REGISTRY — no SCAPT league sampling",
        "bb100_tolerance": args.bb100_tolerance,
        "gto_safety_tolerance": args.gto_safety_tolerance,
    }

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    append_search_gate_result(results)
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")

    if args.require_gto_safety and not gto_safety_passed:
        raise SystemExit("GTO safety floor failed (--require-gto-safety)")


if __name__ == "__main__":
    main()
