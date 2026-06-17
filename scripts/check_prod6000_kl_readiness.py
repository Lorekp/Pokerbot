#!/usr/bin/env python3
"""Check prod6000 btn/bunching accumulation for interim KL trend readiness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pokerbot.solver.label_completion import CompletionLedger


def _spot_type(spot_key: str) -> str:
    return spot_key.split("|", 1)[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Check prod6000 KL trend readiness")
    p.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/solver/train_prod6000_completed.jsonl"),
    )
    p.add_argument("--btn-target", type=int, default=200)
    p.add_argument("--bunching-target", type=int, default=200)
    p.add_argument("--output", type=Path, default=Path("logs/prod6000_kl_trend_status.json"))
    p.add_argument("--run-interim", action="store_true", help="Run interim SFT+KL if ready")
    args = p.parse_args()

    if not args.ledger.is_file():
        report = {
            "ready": False,
            "reason": f"ledger not found: {args.ledger}",
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        sys.exit(0)

    ledger = CompletionLedger.load(args.ledger)
    counts = Counter(_spot_type(k) for k in ledger.completed)
    btn = counts.get("btn_vs_bb_flop", 0)
    bunching = counts.get("bunching_postflop", 0)
    sb = counts.get("sb_vs_bb_flop", 0)

    ready = btn >= args.btn_target and bunching >= args.bunching_target
    report: dict = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "ledger_path": str(args.ledger),
        "total_completed": len(ledger.completed),
        "spot_type_counts": dict(counts),
        "btn_vs_bb_flop": btn,
        "bunching_postflop": bunching,
        "sb_vs_bb_flop": sb,
        "btn_target": args.btn_target,
        "bunching_target": args.bunching_target,
        "ready_for_interim_kl": ready,
        "prior_trend_points": {
            "prod60_combined_heldout_kl": 1.004,
            "prod600_combined_heldout_kl": 0.309,
            "prod600_vanilla_heldout_kl": 0.176,
            "prod600_bunching_heldout_kl": 0.443,
        },
    }

    if not ready:
        btn_need = max(0, args.btn_target - btn)
        bunch_need = max(0, args.bunching_target - bunching)
        report["remaining"] = {"btn": btn_need, "bunching": bunch_need}
        report["estimate"] = "~1-2 days at post-cutover interleaved rate"
    elif args.run_interim:
        report["interim"] = {"status": "not_implemented_on_homeserver_pull"}
        # Interim train requires merged parquet from homeserver shards — run separately when ready.

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
