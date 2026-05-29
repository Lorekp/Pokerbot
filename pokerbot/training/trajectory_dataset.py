"""Parquet dataset for opponent transformer trajectories."""

from __future__ import annotations

import hashlib
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


def compute_trajectory_build_fingerprint(hero_checkpoint: str, scapt_generation: int = 0) -> str:
    """Hash hero checkpoint bytes + SCAPT generation for shard cache invalidation."""
    h = hashlib.sha256()
    path = Path(hero_checkpoint)
    if path.exists():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    h.update(str(scapt_generation).encode())
    return h.hexdigest()[:16]


def _validate_trajectory_row(row: dict[str, Any]) -> None:
    """Belief targets must only be set where cards_known_mask is true."""
    belief = row.get("belief_targets")
    cards_known = row.get("cards_known_mask")
    if not isinstance(belief, dict) or not isinstance(cards_known, dict):
        return
    for seat, target in belief.items():
        seat_key = int(seat) if str(seat).isdigit() else seat
        known = cards_known.get(seat_key, cards_known.get(str(seat_key), False))
        if target is not None and target >= 0 and not known:
            raise ValueError(f"belief target for seat {seat} without cards_known_mask")


class OpponentTrajectoryDataset(Dataset):
    """Load self-play trajectory shards for opponent transformer training."""

    def __init__(
        self,
        parquet_glob: str | Path,
        max_samples: int | None = None,
        expected_build_fingerprint: str | None = None,
    ) -> None:
        from glob import glob

        pattern = str(parquet_glob)
        paths = sorted(Path(p) for p in glob(pattern))
        self.rows: list[dict[str, Any]] = []

        if paths:
            frames = [pd.read_parquet(p) for p in paths]
            df = pd.concat(frames, ignore_index=True)
            if expected_build_fingerprint and "build_fingerprint" in df.columns:
                df = df[df["build_fingerprint"] == expected_build_fingerprint]
            if max_samples:
                df = df.head(max_samples)
            self.rows = df.to_dict(orient="records")

        for row in self.rows:
            for key in (
                "token_types",
                "token_features",
                "actor_seats",
                "belief_targets",
                "policy_targets",
                "cards_known_mask",
                "action_known_mask",
            ):
                if key in row and isinstance(row[key], str):
                    row[key] = json.loads(row[key])
            _validate_trajectory_row(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> TrajectoryRecord:
        return dict_to_trajectory(self.rows[idx])

    def collate(self, batch: list[TrajectoryRecord]) -> dict[str, torch.Tensor]:
        return trajectory_to_batch(batch)


def write_trajectory_shard(
    trajectories: list[TrajectoryRecord],
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write trajectory batch to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [trajectory_to_dict(t) for t in trajectories]
    serializable = []
    meta = metadata or {}
    for row in rows:
        serializable.append(
            {
                **row,
                **meta,
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
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("shard_*.parquet"))
    idx = len(existing)
    path = output_dir / (shard_name or f"shard_{idx:05d}.parquet")
    write_trajectory_shard(trajectories, path, metadata=metadata)
    return path
