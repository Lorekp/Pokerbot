#!/usr/bin/env python3
"""Re-promote hero + hybrid gate baselines after WEVA rebuild."""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.eval.gate_reference import (
    DEFAULT_MANIFEST,
    load_weva_fingerprint,
    promote_baseline,
    promote_hybrid_baseline,
    save_manifest,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Refresh gate baselines after WEVA index rebuild")
    p.add_argument("--hero-source", required=True, help="Current hero checkpoint to freeze")
    p.add_argument("--transformer-source", default="checkpoints/opponent_transformer_madt.pt")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    fp = load_weva_fingerprint()
    promote_baseline(Path(args.hero_source), manifest_path=manifest_path)
    tf = Path(args.transformer_source)
    if tf.exists():
        promote_hybrid_baseline(tf, manifest_path=manifest_path)
    manifest = __import__("pokerbot.eval.gate_reference", fromlist=["load_manifest"]).load_manifest(manifest_path)
    if manifest and fp:
        manifest["weva_build_fingerprint"] = fp
        save_manifest(manifest, manifest_path)
    print(f"Refreshed gate baselines with WEVA fingerprint: {fp}")
    print(f"  hero: {args.hero_source}")
    if tf.exists():
        print(f"  transformer: {tf}")


if __name__ == "__main__":
    main()
