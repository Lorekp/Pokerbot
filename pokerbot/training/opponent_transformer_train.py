"""Training loop for MADT opponent transformer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pokerbot.data.exploitative import load_exploitative_dataset
from pokerbot.engine.actions import action_to_index, parse_pokerbench_output
from pokerbot.opponent.belief_loss import LossBreakdown, opponent_transformer_loss
from pokerbot.opponent.opponent_transformer import OpponentTransformer, OpponentTransformerConfig
from pokerbot.opponent.scapt_manager import SCAPTManager
from pokerbot.opponent.sequence_encoder import (
    HandTrajectoryBuilder,
    TrajectoryRecord,
)
from pokerbot.training.mixed_self_play import rollout_opponent_trajectories
from pokerbot.training.trajectory_dataset import OpponentTrajectoryDataset, compute_trajectory_build_fingerprint


@dataclass
class OpponentTransformerTrainConfig:
    parquet_glob: str = "data/trajectories/*.parquet"
    rollout_hands: int = 256
    epochs: int = 5
    batch_size: int = 16
    lr: float = 3e-4
    device: str = "cpu"
    checkpoint_dir: Path = Path("checkpoints")
    lambda_policy: float = 1.0
    lambda_belief: float = 1.0
    lambda_mi: float = 0.1
    exploitative_limit: int = 0
    exploitative_weight: float = 0.1
    grad_clip: float = 1.0
    use_scapt: bool = True
    hero_checkpoint: str = "checkpoints/policy_selfplay_mixed.pt"
    mi_population_samples: int = 4
    scapt_diversity_floor: float | None = None
    require_scapt_diversity: bool = False
    diversity_probe_states: int = 16


def _hero_checkpoint_sha256(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _exploitative_trajectories(limit: int) -> list[TrajectoryRecord]:
    """Build policy-only trajectories from SoelMgd (belief mask zero)."""
    from pokerbot.opponent.sequence_encoder import TokenType, _pad_feature, encode_action_token

    samples = load_exploitative_dataset("train", limit)
    trajectories: list[TrajectoryRecord] = []
    builder = HandTrajectoryBuilder(hero_seat=0)

    for s in samples:
        if not s.parsed_action:
            continue
        try:
            act = parse_pokerbench_output(s.parsed_action)
            label = action_to_index(act)
        except ValueError:
            continue
        builder.reset(hero_seat=0)
        builder.token_types.append(int(TokenType.ACTION))
        builder.token_features.append(_pad_feature(encode_action_token(1, act)))
        builder.actor_seats.append(1)
        builder.policy_targets[1] = label
        builder.action_known[1] = True
        trajectories.append(builder.finalize())

    return trajectories


def _population_policy_logits(
    model: OpponentTransformer,
    batch: dict[str, torch.Tensor],
    scapt: SCAPTManager | None,
    *,
    samples: int,
) -> torch.Tensor | None:
    """Stack (P, B, 5, A) policy logits from primary + SCAPT-perturbed forwards for MI loss."""

    if scapt is None or not scapt.population:
        return None

    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    logits_list: list[torch.Tensor] = []

    with torch.no_grad():
        primary = model(
            batch["token_types"],
            batch["token_features"],
            batch["attn_mask"],
            batch["hero_seat"],
        )["policy_logits"]
        logits_list.append(primary.unsqueeze(0))

        n_extra = min(samples - 1, max(len(scapt.population) - 1, 2))
        for _ in range(n_extra):
            perturbed = scapt._perturb_state_dict(base_sd, sigma=scapt.mutation_sigma)
            model.load_state_dict(perturbed, strict=True)
            out = model(
                batch["token_types"],
                batch["token_features"],
                batch["attn_mask"],
                batch["hero_seat"],
            )
            logits_list.append(out["policy_logits"].unsqueeze(0))

    model.load_state_dict(base_sd, strict=True)

    if len(logits_list) < 2:
        return None
    return torch.cat(logits_list, dim=0)


def train_opponent_transformer(
    config: OpponentTransformerTrainConfig | None = None,
    hero_model=None,
    scapt: SCAPTManager | None = None,
) -> OpponentTransformer:
    config = config or OpponentTransformerTrainConfig()
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    traj_dir = Path("data/trajectories")
    scapt_gen = scapt._generation if scapt is not None else 0
    build_fp = compute_trajectory_build_fingerprint(config.hero_checkpoint, scapt_gen)

    # SCAPT generates opponent-seat policies only; hero seat uses Step 2 checkpoint.
    if config.rollout_hands > 0 and hero_model is not None:
        rollout_opponent_trajectories(
            hero_model,
            device,
            config.rollout_hands,
            traj_dir,
            scapt=scapt if config.use_scapt else None,
            shard_metadata={
                "build_fingerprint": build_fp,
                "hero_checkpoint": config.hero_checkpoint,
                "hero_sha256": _hero_checkpoint_sha256(config.hero_checkpoint),
                "scapt_generation": scapt_gen,
                "use_scapt": config.use_scapt,
            },
            scapt_diversity_floor=config.scapt_diversity_floor,
            require_scapt_diversity=config.require_scapt_diversity,
            diversity_probe_states=config.diversity_probe_states,
        )
        if scapt is not None:
            scapt.advance_generation()
            scapt.save()

    ds = OpponentTrajectoryDataset(config.parquet_glob, expected_build_fingerprint=build_fp)
    if len(ds) == 0 and config.exploitative_limit > 0:
        aux = _exploitative_trajectories(config.exploitative_limit)
        if aux:
            from pokerbot.training.trajectory_dataset import append_trajectories

            append_trajectories(
                aux,
                traj_dir,
                shard_name="soelmgd_aux.parquet",
                metadata={"build_fingerprint": build_fp, "source": "soelmgd_aux"},
            )
            ds = OpponentTrajectoryDataset(config.parquet_glob, expected_build_fingerprint=build_fp)

    if len(ds) == 0:
        raise ValueError("No trajectory data available for opponent transformer training")

    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=ds.collate,
    )

    model = OpponentTransformer(OpponentTransformerConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(config.epochs):
        model.train()
        totals = LossBreakdown(
            total=torch.tensor(0.0),
            belief=torch.tensor(0.0),
            policy=torch.tensor(0.0),
            mi=torch.tensor(0.0),
            belief_count=0.0,
            policy_count=0.0,
        )
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = {
                "belief_targets": batch.pop("belief_targets"),
                "policy_targets": batch.pop("policy_targets"),
                "cards_known_mask": batch.pop("cards_known_mask"),
                "action_known_mask": batch.pop("action_known_mask"),
            }
            batch.pop("causal_mask", None)
            batch.pop("opp_seat_map", None)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(
                    batch["token_types"],
                    batch["token_features"],
                    batch["attn_mask"],
                    batch["hero_seat"],
                )
                pop_logits = None
                if config.use_scapt and scapt is not None:
                    pop_logits = _population_policy_logits(
                        model,
                        batch,
                        scapt,
                        samples=config.mi_population_samples,
                    )
                breakdown = opponent_transformer_loss(
                    out,
                    targets,
                    lambda_policy=config.lambda_policy,
                    lambda_belief=config.lambda_belief,
                    lambda_mi=config.lambda_mi,
                    population_policy_logits=pop_logits,
                )

            scaler.scale(breakdown.total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(opt)
            scaler.update()

            totals = LossBreakdown(
                total=totals.total + breakdown.total.detach(),
                belief=totals.belief + breakdown.belief.detach(),
                policy=totals.policy + breakdown.policy.detach(),
                mi=totals.mi + breakdown.mi.detach(),
                belief_count=totals.belief_count + breakdown.belief_count,
                policy_count=totals.policy_count + breakdown.policy_count,
            )
            n_batches += 1

        if n_batches:
            print(
                f"Epoch {epoch + 1}/{config.epochs}: "
                f"loss={totals.total.item() / n_batches:.4f} "
                f"belief={totals.belief.item() / n_batches:.4f} "
                f"policy={totals.policy.item() / n_batches:.4f} "
                f"mi={totals.mi.item() / n_batches:.4f} "
                f"belief_n={totals.belief_count:.0f} policy_n={totals.policy_count:.0f}"
            )

    ckpt_path = config.checkpoint_dir / "opponent_transformer_madt.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": OpponentTransformerConfig().__dict__,
            "source": "madt_pretrain",
            "hero_checkpoint": config.hero_checkpoint,
            "hero_sha256": _hero_checkpoint_sha256(config.hero_checkpoint),
            "scapt_generation": scapt_gen,
            "use_scapt": config.use_scapt,
            "build_fingerprint": build_fp,
        },
        ckpt_path,
    )
    print(f"Saved {ckpt_path}")
    return model
