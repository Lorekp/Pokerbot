#!/usr/bin/env python3
"""Record Phase 2 gate pass and promote CFR blueprint as hero when two seeds pass."""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.eval.gate_reference import (
    DEFAULT_CFR_HERO,
    DEFAULT_MANIFEST,
    PHASE2_GATE_PASSES_REQUIRED,
    record_phase2_gate_pass,
    should_use_cfr_hero,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Promote CFR hero after Phase 2 gate passes")
    p.add_argument("--results", default="eval_phase2_compare.json")
    p.add_argument("--seed", type=int, required=True, help="Harness seed for this pass")
    p.add_argument("--cfr-hero", default=str(DEFAULT_CFR_HERO))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = p.parse_args()

    manifest = record_phase2_gate_pass(
        seed=args.seed,
        results_path=Path(args.results),
        manifest_path=Path(args.manifest),
        cfr_hero_path=Path(args.cfr_hero),
    )
    seeds = manifest.get("phase2_gate_pass_seeds") or []
    print(f"Recorded Phase 2 gate pass (seed={args.seed}); passes={seeds}")
    print(f"  Required distinct seeds: {PHASE2_GATE_PASSES_REQUIRED}")
    if should_use_cfr_hero(Path(args.manifest)):
        print(f"  CFR hero promoted: {manifest.get('cfr_hero_checkpoint')}")
        print(f"  cfr_hero_sha256: {manifest.get('cfr_hero_sha256', '')[:16]}...")
        print(f"  phase2_gate_passed_at: {manifest.get('phase2_gate_passed_at')}")
    else:
        print("  Awaiting additional passing seed before CFR hero cutover.")


if __name__ == "__main__":
    main()
