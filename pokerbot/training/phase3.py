"""Phase 3: opponent transformer training (MADT) + optional mixed self-play."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.opponent.opponent_transformer import OpponentTransformer
from pokerbot.training.opponent_transformer_train import (
    OpponentTransformerTrainConfig,
    train_opponent_transformer,
)


@dataclass
class PretrainConfig:
    limit: int = 5000
    batch_size: int = 16
    epochs: int = 5
    lr: float = 3e-4
    device: str = "cpu"
    checkpoint_dir: Path = Path("checkpoints")
    rollout_hands: int = 256
    parquet_glob: str = "data/trajectories/*.parquet"
    hero_checkpoint: str | None = "checkpoints/policy_distilled.pt"


def pretrain_opponent_transformer(
    config: PretrainConfig | None = None,
    hero_model: PseudoSiamesePolicy | None = None,
) -> OpponentTransformer:
    """Train MADT opponent transformer (replaces legacy hierarchical encoder pretrain)."""
    config = config or PretrainConfig()
    device = torch.device(config.device)

    if hero_model is None and config.hero_checkpoint:
        ckpt_path = Path(config.hero_checkpoint)
        if ckpt_path.exists():
            hero_model = PseudoSiamesePolicy().to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            hero_model.load_state_dict(ckpt["model"])

    return train_opponent_transformer(
        OpponentTransformerTrainConfig(
            parquet_glob=config.parquet_glob,
            rollout_hands=config.rollout_hands,
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            device=config.device,
            checkpoint_dir=config.checkpoint_dir,
            exploitative_limit=config.limit,
            exploitative_weight=0.1,
        ),
        hero_model=hero_model,
    )
