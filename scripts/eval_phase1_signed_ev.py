#!/usr/bin/env python3
"""Signed exploiter-EV gate eval for prod60 / prod600 / rank baseline (no pass/fail threshold)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pokerbot.eval.abstraction_bots import RankEquityAbstractBot
from pokerbot.eval.bots import GTOCopyBot
from pokerbot.eval.exploitability import (
    SIGNED_EXPLOITER_EV_METRIC_NOTE,
    estimate_exploitability_scoped,
    require_exploitability_harness_sanity,
)
from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy


def _ci95(samples: np.ndarray) -> tuple[float, float]:
    if len(samples) < 2:
        v = float(samples[0]) if len(samples) else 0.0
        return v, v
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_ci_mean(values: np.ndarray, *, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    boots = [float(values[rng.integers(0, n, size=n)].mean()) for _ in range(draws)]
    return _ci95(np.array(boots))


def _eval_policy(
    label: str,
    bot,
    *,
    seeds: list[int],
    hands: int,
) -> dict[str, object]:
    by_seed: dict[str, float] = {}
    for seed in seeds:
        res, _scope = estimate_exploitability_scoped(bot, hands=hands, seed=seed)
        by_seed[str(seed)] = res.exploiter_ev_bb_per_hand
    arr = np.array([by_seed[str(s)] for s in seeds])
    return {
        "label": label,
        "exploiter_ev_bb_per_hand_by_seed": by_seed,
        "mean_exploiter_ev_bb_per_hand": float(arr.mean()),
        "ci95_exploiter_ev_bb_per_hand": list(_ci95(arr)),
        "bootstrap_ci95_mean": list(_bootstrap_ci_mean(arr)),
        "min_seed": float(arr.min()),
        "max_seed": float(arr.max()),
        "interpretation": "lower exploiter_ev_bb_per_hand => target less exploitable by this heuristic ensemble",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Signed exploiter EV eval (no gate pass/fail)")
    p.add_argument("--prod60-checkpoint", default="checkpoints/policy_sft.pt")
    p.add_argument("--prod600-checkpoint", default="checkpoints/policy_sft_prod600.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--hands", type=int, default=400)
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--audit-reference-json", default="eval_phase1_compare_prod600_v2.json")
    p.add_argument("--output", default="eval_phase1_signed_ev.json")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    sanity = require_exploitability_harness_sanity(hands=400, seed=42)

    prod60_model = load_pseudo_siamese_policy(args.prod60_checkpoint, device=args.device)
    prod60_bot = GTOCopyBot(
        prod60_model,
        device=args.device,
        eval_spot_class="hu_postflop_2p",
        require_scope=True,
    )
    prod600_model = load_pseudo_siamese_policy(args.prod600_checkpoint, device=args.device)
    prod600_bot = GTOCopyBot(
        prod600_model,
        device=args.device,
        eval_spot_class="hu_postflop_2p",
        require_scope=True,
    )
    rank_bot = RankEquityAbstractBot()

    prod60_res = _eval_policy("prod60_sft", prod60_bot, seeds=seeds, hands=args.hands)
    prod600_res = _eval_policy("prod600_sft", prod600_bot, seeds=seeds, hands=args.hands)
    baseline_res = _eval_policy("rank_equity_baseline", rank_bot, seeds=seeds, hands=args.hands)

    audit_path = Path(args.audit_reference_json)
    reproducibility: dict[str, object] = {"audit_json": str(audit_path), "checks": []}
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_prod600 = audit["solver_sft_policy"]["br_ev_bb_per_hand_by_seed"]
        audit_baseline = audit["rank_equity_baseline"]["br_ev_bb_per_hand_by_seed"]
        for name, got, audit_map in [
            ("prod600_sft", prod600_res, audit_prod600),
            ("rank_baseline", baseline_res, audit_baseline),
        ]:
            diffs = []
            for s in seeds:
                audit_v = float(audit_map[str(s)])
                new_v = float(got["exploiter_ev_bb_per_hand_by_seed"][str(s)])
                diffs.append(abs(new_v - audit_v))
            max_diff = max(diffs) if diffs else 0.0
            reproducibility["checks"].append(
                {
                    "policy": name,
                    "max_abs_seed_diff_vs_audit_br_ev": max_diff,
                    "matches_within_1e-6": max_diff < 1e-6,
                }
            )
        prod60_audit = None
        v2_path = Path("eval_phase1_compare_v2.json")
        if v2_path.is_file():
            v2 = json.loads(v2_path.read_text(encoding="utf-8"))
            prod60_audit = v2["solver_sft_policy"]["br_ev_bb_per_hand_by_seed"]
            diffs = []
            for s in seeds:
                audit_v = float(prod60_audit[str(s)])
                new_v = float(prod60_res["exploiter_ev_bb_per_hand_by_seed"][str(s)])
                diffs.append(abs(new_v - audit_v))
            max_diff = max(diffs) if diffs else 0.0
            reproducibility["checks"].append(
                {
                    "policy": "prod60_sft",
                    "reference": str(v2_path),
                    "max_abs_seed_diff_vs_audit_br_ev": max_diff,
                    "matches_within_1e-6": max_diff < 1e-6,
                }
            )

    report = {
        "metric": "exploiter_ev_bb_per_hand",
        "sign_convention": (
            "Signed EV of best heuristic exploiter seat (bb/hand). "
            "Positive => exploiter wins. Lower => target less exploitable under this ensemble."
        ),
        "signed_exploiter_ev_metric_note": SIGNED_EXPLOITER_EV_METRIC_NOTE,
        "no_gate_pass_fail": (
            "This run reports signed exploiter EV only. Threshold selection and pass/fail "
            "require a decision about exploiter strength — not set in this task."
        ),
        "harness_sanity_precondition": sanity,
        "hands_per_seed": args.hands,
        "seeds": seeds,
        "policies": {
            "prod60_sft": {
                "checkpoint": args.prod60_checkpoint,
                **prod60_res,
            },
            "prod600_sft": {
                "checkpoint": args.prod600_checkpoint,
                **prod600_res,
            },
            "rank_equity_baseline": baseline_res,
        },
        "audit_reproducibility": reproducibility,
        "audit_reference_means_from_gate_audit_md": {
            "prod60_sft": 73.5,
            "prod600_sft": -14.2,
            "rank_equity_baseline": -16.6,
            "note": "Rounded means from logs/prod600_gate_audit.md; per-seed check is authoritative.",
        },
        "pairwise_signed_ev_margin_baseline_minus_prod600": {
            "by_seed": {
                str(s): float(baseline_res["exploiter_ev_bb_per_hand_by_seed"][str(s)])
                - float(prod600_res["exploiter_ev_bb_per_hand_by_seed"][str(s)])
                for s in seeds
            },
            "mean": float(
                np.mean(
                    [
                        float(baseline_res["exploiter_ev_bb_per_hand_by_seed"][str(s)])
                        - float(prod600_res["exploiter_ev_bb_per_hand_by_seed"][str(s)])
                        for s in seeds
                    ]
                )
            ),
        },
        "exploiter_too_weak_caveat": (
            "Negative exploiter EV on prod600 and baseline means this heuristic ensemble loses "
            "to both policies. Signed-EV ordering (baseline -16.6 vs prod600 -14.2) may reflect "
            "exploiter weakness rather than true policy ordering."
        ),
    }

    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
