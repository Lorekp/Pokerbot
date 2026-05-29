#!/usr/bin/env python3
"""Serve AoF obs-v3 policy inference over FastAPI."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from aof_rl.inference.app import create_app
from aof_rl.inference.engine import ServerConfig


def main() -> None:
    p = argparse.ArgumentParser(description="AoF inference server")
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="League checkpoint (.pt) obs_version=3",
    )
    p.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSON")
    p.add_argument(
        "--reference-histogram",
        type=Path,
        default=None,
        help="OOD reference histogram JSON",
    )
    p.add_argument("--ensemble-checkpoint", type=Path, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--compile", action="store_true", help="Enable torch.compile")
    p.add_argument("--amp", action="store_true", help="Enable CUDA autocast")
    p.add_argument(
        "--allow-side-pots",
        action="store_true",
        help="Allow unequal-stack side-pot states",
    )
    p.add_argument(
        "--skip-manifest-verify",
        action="store_true",
        help="Skip SHA-256 manifest verification",
    )
    p.add_argument("--model-id", default=None)
    args = p.parse_args()

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = ServerConfig(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        reference_histogram_path=args.reference_histogram,
        ensemble_checkpoint_path=args.ensemble_checkpoint,
        device=device,
        use_compile=args.compile,
        use_amp=args.amp,
        allow_side_pots=args.allow_side_pots,
        skip_manifest_verify=args.skip_manifest_verify,
        model_id=args.model_id,
    )
    app = create_app(config=config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
