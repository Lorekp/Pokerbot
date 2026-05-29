#!/usr/bin/env python3
"""Build deterministic interleaved spot_key manifest for cloud/local label workers."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pokerbot.solver.spot_enumerator import enumerate_spot_keys, load_solver_config


def main() -> None:
    p = argparse.ArgumentParser(description="Plan label work manifest")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-spots", type=int, default=None)
    p.add_argument("--run-id", type=str, default=None)
    args = p.parse_args()

    config = load_solver_config(args.config)
    keys = enumerate_spot_keys(config)
    if args.max_spots is not None:
        keys = keys[: args.max_spots]

    run_id = args.run_id or datetime.now(timezone.utc).strftime("labelgen_%Y%m%d_%H%M%S")
    cfg_bytes = args.config.read_bytes()
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config_hash": hashlib.sha256(cfg_bytes).hexdigest()[:16],
        "enumeration_mode": config.enumeration_mode,
        "disable_bunching": config.disable_bunching,
        "spot_keys": keys,
        "entries": {k: {"status": "pending"} for k in keys},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "count": len(keys), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
