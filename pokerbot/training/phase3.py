"""Phase 3: opponent transformer training (MADT) + SCAPT opponent league rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.opponent.opponent_transformer import OpponentTransformer
from pokerbot.opponent.scapt_manager import SCAPTConfig, SCAPTManager
from pokerbot.training.checkpoint_provenance import warn_if_bc_checkpoint
from pokerbot.training.opponent_transformer_train import (
    OpponentTransformerTrainConfig,
    train_opponent_transformer,
)


def resolve_hero_checkpoint() -> Path:
    """Fallback chain for Step 2 hero policy used in opponent rollouts."""
    for name in (
        "policy_selfplay_mixed.pt",
        "policy_sft.pt",
        "policy_sft_bc.pt",
        "policy_distilled.pt",
    ):
        path = Path("checkpoints") / name
        if path.exists():
            warn_if_bc_checkpoint(path, context="hero rollout fallback")
            return path
    return Path("checkpoints/policy_selfplay_mixed.pt")


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
    hero_checkpoint: str | None = None
    use_scapt: bool = True
    scapt_population_size: int = 16
    scapt_mutation_sigma: float | None = None
    scapt_diversity_floor: float | None = None
    require_scapt_diversity: bool = False
    diversity_probe_states: int = 16


def pretrain_opponent_transformer(
    config: PretrainConfig | None = None,
    hero_model: PseudoSiamesePolicy | None = None,
) -> OpponentTransformer:
    """Train MADT opponent transformer from SCAPT/static league rollouts."""
    config = config or PretrainConfig()
    device = torch.device(config.device)
    hero_ckpt = Path(config.hero_checkpoint) if config.hero_checkpoint else resolve_hero_checkpoint()
    if config.hero_checkpoint:
        warn_if_bc_checkpoint(hero_ckpt, context="Phase 3 hero rollout")

    if hero_model is None and hero_ckpt.exists():
        hero_model = PseudoSiamesePolicy().to(device)
        ckpt = torch.load(hero_ckpt, map_location=device, weights_only=False)
        hero_model.load_state_dict(ckpt["model"])

    scapt: SCAPTManager | None = None
    if config.use_scapt and hero_model is not None:
        scapt_cfg = SCAPTConfig(population_size=config.scapt_population_size)
        if config.scapt_mutation_sigma is not None:
            scapt_cfg.mutation_sigma = config.scapt_mutation_sigma
        scapt = SCAPTManager(scapt_cfg)
        scapt.init_population(hero_model)

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
            use_scapt=config.use_scapt,
            hero_checkpoint=str(hero_ckpt),
            scapt_diversity_floor=config.scapt_diversity_floor,
            require_scapt_diversity=config.require_scapt_diversity,
            diversity_probe_states=config.diversity_probe_states,
        ),
        hero_model=hero_model,
        scapt=scapt,
    )
