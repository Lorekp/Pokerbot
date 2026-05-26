#!/usr/bin/env python3
"""Build WEVA bucket index from preflop CFR warm-up + MC rollouts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pokerbot.encoding.weva_encoder import (
    WEVABuilder,
    WEVA_DEFAULT_K,
    WEVA_DEFAULT_ROLLOUTS,
    WEVA_WARMUP_ITERATIONS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(description="Build WEVA FAISS k-means index")
    p.add_argument("--output", type=Path, default=Path("data/weva"))
    p.add_argument("--k", type=int, default=WEVA_DEFAULT_K)
    p.add_argument("--w", type=int, default=WEVA_WARMUP_ITERATIONS)
    p.add_argument("--gamma", type=float, default=0.85)
    p.add_argument(
        "--feature-mode",
        choices=("hole_only", "hole_plus_context"),
        default="hole_only",
    )
    p.add_argument("--device", default="cpu", help="Reserved for future GPU rollout batching")
    p.add_argument("--rollouts", type=int, default=WEVA_DEFAULT_ROLLOUTS)
    p.add_argument("--cfr-seed", type=int, default=42)
    p.add_argument("--rollout-seed", type=int, default=42)
    p.add_argument("--max-nodes", type=int, default=4)
    p.add_argument("--max-roots", type=int, default=None, help="Limit roots for dev builds")
    p.add_argument("--force", action="store_true", help="Rebuild even when build_fingerprint matches")
    args = p.parse_args()

    builder = WEVABuilder(
        k=args.k,
        warmup_iterations=args.w,
        gamma=args.gamma,
        feature_mode=args.feature_mode,
        device=args.device,
        num_rollouts=args.rollouts,
        cfr_seed=args.cfr_seed,
        rollout_seed=args.rollout_seed,
        max_nodes=args.max_nodes,
        max_roots=args.max_roots,
    )
    out = builder.save_artifacts(args.output, force=args.force)
    print(f"WEVA index saved to {out} (build_fingerprint={builder.build_fingerprint()}, scope={builder.build_scope()})")


if __name__ == "__main__":
    main()
