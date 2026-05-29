#!/usr/bin/env python3
"""Promote search artifacts after Phase 4 gate_passed."""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.eval.gate_reference import DEFAULT_MANIFEST, promote_search_baseline, sha256_file


def main() -> None:
    p = argparse.ArgumentParser(description="Promote search baseline after Phase 4 gate")
    p.add_argument("--results", default="eval_phase4_compare.json")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--value-checkpoint", default=None)
    p.add_argument("--madt-checkpoint", default=None)
    args = p.parse_args()

    manifest = promote_search_baseline(
        Path(args.results),
        manifest_path=Path(args.manifest),
        value_path=Path(args.value_checkpoint) if args.value_checkpoint else None,
        madt_path=Path(args.madt_checkpoint) if args.madt_checkpoint else None,
    )
    print("Promoted search baseline:")
    print(f"  search_value: {manifest.get('search_value_checkpoint')}")
    print(f"  search_value_sha256: {manifest.get('search_value_sha256', '')[:16]}...")
    print(f"  search_madt: {manifest.get('search_madt_checkpoint')}")
    print(f"  phase4_gate_passed_at: {manifest.get('phase4_gate_passed_at')}")
    v_path = Path(manifest["search_value_checkpoint"])
    print(f"  verified: {sha256_file(v_path)[:16]}...")


if __name__ == "__main__":
    main()
