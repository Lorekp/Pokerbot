"""Parquet dataset for opponent transformer trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from pokerbot.opponent.sequence_encoder import (
    TrajectoryRecord,
    dict_to_trajectory,
    trajectory_to_batch,
    trajectory_to_dict,
)


@dataclass
class TrajectoryDatasetConfig:
    parquet_glob: str = "data/trajectories/*.parquet"
    max_samples: int | None = None


class OpponentTrajectoryDataset(Dataset):
    """Load self-play trajectory shards for opponent transformer training."""

    def __init__(self, parquet_glob: str | Path, max_samples: int | None = None) -> None:
        from glob import glob

        pattern = str(parquet_glob)
        paths = sorted(Path(p) for p in glob(pattern))
        if not paths:
            self.rows: list[dict[str, Any]] = []
        else:
            frames = [pd.read_parquet(p) for p in paths]
            df = pd.concat(frames, ignore_index=True)
            if max_samples:
                df = df.head(max_samples)
            self.rows = df.to_dict(orient="records")

        for row in self.rows:
            for key in ("token_types", "token_features", "actor_seats", "belief_targets", "policy_targets", "cards_known_mask", "action_known_mask"):
                if key in row and isinstance(row[key], str):
                    row[key] = json.loads(row[key])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> TrajectoryRecord:
        return dict_to_trajectory(self.rows[idx])

    def collate(self, batch: list[TrajectoryRecord]) -> dict[str, torch.Tensor]:
        return trajectory_to_batch(batch)


def write_trajectory_shard(
    trajectories: list[TrajectoryRecord],
    path: Path,
) -> None:
    """Write trajectory batch to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [trajectory_to_dict(t) for t in trajectories]
    serializable = []
    for row in rows:
        serializable.append(
            {
                **row,
                "token_types": json.dumps(row["token_types"]),
                "token_features": json.dumps(row["token_features"]),
                "actor_seats": json.dumps(row["actor_seats"]),
                "belief_targets": json.dumps(row["belief_targets"]),
                "policy_targets": json.dumps(row["policy_targets"]),
                "cards_known_mask": json.dumps(row["cards_known_mask"]),
                "action_known_mask": json.dumps(row["action_known_mask"]),
            }
        )
    pd.DataFrame(serializable).to_parquet(path, index=False)


def append_trajectories(
    trajectories: list[TrajectoryRecord],
    output_dir: Path,
    shard_name: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("shard_*.parquet"))
    idx = len(existing)
    path = output_dir / (shard_name or f"shard_{idx:05d}.parquet")
    write_trajectory_shard(trajectories, path)
    return path
