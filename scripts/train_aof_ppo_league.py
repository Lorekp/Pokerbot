#!/usr/bin/env python3
"""Train AoF obs-v3 policy via solver-guided PPO league (YAML config)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_rl.config import load_train_config
from aof_rl.league_trainer import train_ppo_league


def main() -> None:
    p = argparse.ArgumentParser(description="AoF PPO league trainer (obs v3)")
    p.add_argument(
        "--config",
        type=str,
        default="configs/aof_v3_ppo_league.yaml",
        help="YAML training config",
    )
    args = p.parse_args()
    cache_dir = _ROOT / ".torch_inductor_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
    cfg = load_train_config(_ROOT / args.config)
    print(
        f"experiment={cfg.experiment} obs_version={cfg.obs_version} "
        f"encoder={cfg.model.encoder_type} updates={cfg.train.total_updates}",
        flush=True,
    )
    train_ppo_league(cfg)
    print(f"Saved checkpoint: {cfg.checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
