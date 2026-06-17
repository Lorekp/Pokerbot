#!/usr/bin/env python3
"""Freeze VR-DeepPDCFR+ baseline on float env before int-chip migration (Step 0)."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        return out.strip()
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Freeze Phase 2 legacy baseline")
    p.add_argument("--output", default="logs/phase2_legacy_baseline.json")
    p.add_argument("--hands", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ppo-checkpoint", default="checkpoints/policy_selfplay_mixed.pt")
    p.add_argument("--cfr-policy-checkpoint", default="checkpoints/policy_cfr_blueprint.pt")
    p.add_argument("--cfr-checkpoint", default="checkpoints/history_value_net.pkl")
    p.add_argument("--eval-output", default="logs/eval_phase2_compare_baseline.json")
    args = p.parse_args()

    eval_cmd = [
        "python",
        str(ROOT / "scripts/eval_phase2_compare.py"),
        "--hands",
        str(args.hands),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--ppo-checkpoint",
        args.ppo_checkpoint,
        "--cfr-policy-checkpoint",
        args.cfr_policy_checkpoint,
        "--cfr-checkpoint",
        args.cfr_checkpoint,
        "--output",
        args.eval_output,
    ]
    subprocess.run(eval_cmd, cwd=ROOT, check=False)

    eval_path = ROOT / args.eval_output
    eval_data = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else {}

    report = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "env_schema": "int_v2_chips",
        "note": "VR-DeepPDCFR+ retired post-migration; SD-CFR compared against this snapshot.",
        "git_commit": _git_commit(),
        "seed": args.seed,
        "hands": args.hands,
        "checkpoints": {
            "ppo": args.ppo_checkpoint,
            "cfr_policy": args.cfr_policy_checkpoint,
            "cfr_hvn": args.cfr_checkpoint,
        },
        "eval_source": args.eval_output,
        "ppo": eval_data.get("ppo"),
        "cfr_blueprint": eval_data.get("cfr_blueprint"),
        "harness_sanity": eval_data.get("harness_sanity_precondition"),
        "exploitability_bb_cfr": eval_data.get("cfr_blueprint", {}).get("exploiter_ev_bb_per_hand"),
        "exploitability_bb_ppo": eval_data.get("ppo", {}).get("exploiter_ev_bb_per_hand"),
        "bb100_vs_pool_cfr": eval_data.get("cfr_blueprint", {}).get("bb100_vs_pool"),
        "bb100_vs_pool_ppo": eval_data.get("ppo", {}).get("bb100_vs_pool"),
    }

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
