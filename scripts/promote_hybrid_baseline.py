#!/usr/bin/env python3
"""Promote winning MADT transformer to frozen Phase 4 Hybrid gate baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.eval.gate_reference import (
    DEFAULT_MANIFEST,
    DEFAULT_TRANSFORMER_BASELINE,
    promote_hybrid_baseline,
    sha256_file,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Promote MADT transformer to frozen Hybrid gate baseline")
    p.add_argument(
        "--source",
        default="checkpoints/opponent_transformer_madt.pt",
        help="Winning Phase 3 MADT checkpoint",
    )
    p.add_argument("--baseline", default=str(DEFAULT_TRANSFORMER_BASELINE))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = p.parse_args()

    source = Path(args.source)
    manifest = promote_hybrid_baseline(
        source,
        transformer_baseline_path=Path(args.baseline),
        manifest_path=Path(args.manifest),
    )
    tf_path = Path(manifest["transformer_checkpoint"])
    print(f"Promoted Hybrid gate baseline from {source}")
    print(f"  transformer_checkpoint: {tf_path}")
    print(f"  transformer_sha256: {manifest.get('transformer_sha256')}")
    print(f"  phase3_gate_passed_at: {manifest.get('phase3_gate_passed_at')}")
    print(f"  verified_sha: {sha256_file(tf_path)[:16]}...")


if __name__ == "__main__":
    main()
