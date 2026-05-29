#!/usr/bin/env python3
"""Promote a hero checkpoint to the frozen Phase 3 gate baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.eval.gate_reference import DEFAULT_BASELINE, DEFAULT_MANIFEST, promote_baseline
from pokerbot.training.phase3 import resolve_hero_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Promote hero checkpoint to frozen gate baseline")
    p.add_argument(
        "--source",
        default=None,
        help="Source checkpoint (default: resolve hero chain)",
    )
    p.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = p.parse_args()

    source = Path(args.source) if args.source else resolve_hero_checkpoint()
    manifest = promote_baseline(
        source,
        baseline_path=Path(args.baseline),
        manifest_path=Path(args.manifest),
    )
    print(f"Promoted gate baseline from {source}")
    print(f"  hero_checkpoint: {manifest['hero_checkpoint']}")
    print(f"  hero_sha256: {manifest['hero_sha256']}")
    print(f"  weva_build_fingerprint: {manifest.get('weva_build_fingerprint')}")


if __name__ == "__main__":
    main()
