#!/usr/bin/env python3
"""Validate Phase 4 artifacts before live search deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pokerbot.eval.gate_reference import DEFAULT_MANIFEST, resolve_hybrid_baseline, sha256_file
from pokerbot.search.leaf_evaluator import HistoryValueEvaluator


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 4 artifact validation")
    p.add_argument("--value-checkpoint", default="checkpoints/history_value_net.pkl")
    p.add_argument("--madt-checkpoint", default="checkpoints/opponent_transformer_madt.pt")
    p.add_argument("--gate-manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--allow-mock-leaf", action="store_true")
    p.add_argument("--require-hybrid-baseline", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    hvn = Path(args.value_checkpoint)
    madt = Path(args.madt_checkpoint)

    if not hvn.exists():
        if args.allow_mock_leaf:
            warnings.append(f"HVN missing ({hvn}); live assist would use mock leaf or suppress")
        else:
            errors.append(f"HistoryValueNet missing: {hvn}")
    else:
        ev = HistoryValueEvaluator(hvn)
        if not ev.has_trained_checkpoint:
            errors.append(f"HVN at {hvn} lacks average_strategy_params (untrained?)")
        else:
            print(f"OK: HVN trained checkpoint sha256={sha256_file(hvn)[:16]}...")

    if not madt.exists():
        warnings.append(f"MADT checkpoint missing ({madt}); PBS will use uniform beliefs")
    else:
        print(f"OK: MADT checkpoint sha256={sha256_file(madt)[:16]}...")

    try:
        hero_path, tf_path, manifest = resolve_hybrid_baseline(
            manifest_path=Path(args.gate_manifest),
            require=args.require_hybrid_baseline,
        )
        print(f"OK: Hybrid gate baseline hero={hero_path.name} transformer={tf_path.name}")
        if manifest.get("transformer_sha256"):
            print(f"     transformer_sha256={manifest['transformer_sha256'][:16]}...")
    except (FileNotFoundError, ValueError) as exc:
        if args.require_hybrid_baseline:
            errors.append(str(exc))
        else:
            warnings.append(str(exc))

    for w in warnings:
        print(f"WARN: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    if errors:
        print("\nPhase 4 validation FAILED — live assist would suppress recommendations.")
        return 1

    print("\nPhase 4 validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
