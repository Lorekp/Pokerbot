"""YAML-driven training config for AoF PPO league."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aof_rl.curriculum import CurriculumConfig, CurriculumStage
from aof_rl.league import LeagueConfig
from aof_rl.policy_v3 import EncoderConfig, EncoderType
from aof_rl.ppo_v3 import PPOv3Config


@dataclass
class ModelConfig:
    encoder_type: EncoderType = "set_transformer_lite"
    d_model: int = 96
    n_blocks: int = 2
    n_heads: int = 4
    n_inducing: int = 8
    trunk_hidden: int = 128

    def to_encoder_config(self) -> EncoderConfig:
        return EncoderConfig(
            encoder_type=self.encoder_type,
            d_model=self.d_model,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            n_inducing=self.n_inducing,
            trunk_hidden=self.trunk_hidden,
        )


@dataclass
class TrainConfig:
    algo: str = "ppo_league"
    gamma: float = 1.0
    gae_lambda: float = 1.0
    lr: float = 3e-4
    clip_ratio: float = 0.1
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    q_coef: float = 0.5
    teacher_bc_coef: float = 0.25
    uncertainty_coef: float = 0.1
    batch_hands: int = 128
    epochs_per_update: int = 4
    normalize_returns: bool = True
    normalize_advantages: bool = True
    mixed_precision: bool = False
    total_updates: int = 200
    teacher_samples: int = 300
    payoff_samples: int = 1200
    use_fast_teacher: bool = True
    teacher_margin_bb: float = 0.4
    teacher_min_samples: int = 200
    stack_bb: float = 8.0
    seed: int = 42
    device: str = "cuda"
    allow_cpu: bool = False
    torch_compile: bool = False
    profile: bool = False
    deterministic_validation: bool = False
    save_checkpoint_on_best: bool = True
    min_bb_hand_improvement: float = 0.0
    use_teacher_lut: bool = False
    teacher_lut_path: str = "data/aof/teacher_lut_v2_8bb.npz"
    use_batched_rollout: bool = True
    parallel_hands: int = 64
    torch_compile_villain: bool = False
    torch_compile_hero: bool = False
    store_obs_on_cpu: bool = False
    payoff_hu_mode: str = "table"
    hu_showdown_table_path: str = "data/aof/hu_showdown_equity.npz"

    def to_ppo_config(self) -> PPOv3Config:
        return PPOv3Config(
            clip_eps=self.clip_ratio,
            value_coef=self.value_coef,
            q_coef=self.q_coef,
            teacher_coef=self.teacher_bc_coef,
            entropy_coef=self.entropy_coef,
            uncertainty_coef=self.uncertainty_coef,
            ppo_epochs=self.epochs_per_update,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize_advantages=self.normalize_advantages,
        )


@dataclass
class ValidationConfig:
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    paired_eval_hands: int = 0
    br_exploit_hands: int = 0
    deterministic: bool = True


@dataclass
class TrainExperimentConfig:
    experiment: str = "aof_v3_ppo_league"
    obs_version: int = 3
    teacher_version: int = 2
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint_path: str = "checkpoints/aof_rl/policy_league_v3.pt"
    checkpoint_best_path: str | None = None
    checkpoint_dir: str = "checkpoints/aof_rl"


def _parse_curriculum(raw: dict[str, Any]) -> CurriculumConfig:
    stages_raw = raw.get("stages")
    if not stages_raw:
        names = [
            "stage_1",
            "stage_2",
            "stage_3",
            "stage_4",
            "stage_5",
        ]
        mapping = {
            "stage_1": ("hu_only", (2,), "open_or_facing"),
            "stage_2": ("three_handed_one_shover", (3,), "max_one_shover"),
            "stage_3": ("multiway_shovers", (3,), "min_two_shovers"),
            "stage_4": ("four_handed_full", (4,), "full"),
            "stage_5": ("rake_randomized", (2, 3, 4), "full"),
        }
        stages = []
        for key in names:
            if key in raw:
                name, sizes, spot = mapping[key]
                updates = int(raw.get(f"{key}_updates", 50))
                rake = 0.05 if "rake" in name else 0.0
                stages.append(
                    CurriculumStage(
                        name,
                        sizes,
                        spot,  # type: ignore[arg-type]
                        flat_rake_bb=rake,
                        range_randomization="rake" in name,
                        updates=updates,
                    )
                )
        if stages:
            return CurriculumConfig(stages=stages)
        return CurriculumConfig.default_pipeline()
    stages = []
    for s in stages_raw:
        stages.append(
            CurriculumStage(
                name=str(s["name"]),
                table_sizes=tuple(int(x) for x in s["table_sizes"]),
                spot_constraint=s.get("spot_constraint", "any"),
                flat_rake_bb=float(s.get("flat_rake_bb", 0.0)),
                range_randomization=bool(s.get("range_randomization", False)),
                updates=int(s.get("updates", 50)),
            )
        )
    return CurriculumConfig(stages=stages)


def load_train_config(path: str | Path) -> TrainExperimentConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model_raw = data.get("model") or {}
    train_raw = data.get("train") or {}
    league_raw = data.get("league") or {}
    val_raw = data.get("validation") or {}

    model = ModelConfig(
        encoder_type=model_raw.get("encoder", "set_transformer_lite"),
        d_model=int(model_raw.get("d_model", 96)),
        n_blocks=int(model_raw.get("n_blocks", 2)),
        n_heads=int(model_raw.get("n_heads", 4)),
        n_inducing=int(model_raw.get("n_inducing", 8)),
        trunk_hidden=int(model_raw.get("trunk_hidden", 128)),
    )
    train = TrainConfig(
        lr=float(train_raw.get("lr", 3e-4)),
        clip_ratio=float(train_raw.get("clip_ratio", 0.1)),
        entropy_coef=float(train_raw.get("entropy_coef", 0.01)),
        value_coef=float(train_raw.get("value_coef", 0.5)),
        q_coef=float(train_raw.get("q_coef", 0.5)),
        teacher_bc_coef=float(train_raw.get("teacher_bc_coef", 0.25)),
        uncertainty_coef=float(train_raw.get("uncertainty_coef", 0.1)),
        batch_hands=int(train_raw.get("batch_hands", train_raw.get("hands_per_update", 128))),
        epochs_per_update=int(train_raw.get("epochs_per_update", 4)),
        total_updates=int(train_raw.get("total_updates", data.get("updates", 200))),
        teacher_samples=int(train_raw.get("teacher_samples", 300)),
        payoff_samples=int(train_raw.get("payoff_samples", 1200)),
        use_fast_teacher=bool(train_raw.get("use_fast_teacher", True)),
        gamma=float(train_raw.get("gamma", 1.0)),
        gae_lambda=float(train_raw.get("gae_lambda", 1.0)),
        seed=int(data.get("seed", train_raw.get("seed", 42))),
        device=str(train_raw.get("device", data.get("device", "cuda"))),
        allow_cpu=bool(data.get("allow_cpu", train_raw.get("allow_cpu", False))),
        stack_bb=float(train_raw.get("stack_bb", 8.0)),
        deterministic_validation=bool(
            val_raw.get("deterministic", False)
        ),
        torch_compile=bool(train_raw.get("torch_compile", False)),
        profile=bool(train_raw.get("profile", False)),
        save_checkpoint_on_best=bool(train_raw.get("save_checkpoint_on_best", True)),
        min_bb_hand_improvement=float(train_raw.get("min_bb_hand_improvement", 0.0)),
        use_teacher_lut=bool(train_raw.get("use_teacher_lut", False)),
        teacher_lut_path=str(
            train_raw.get("teacher_lut_path", "data/aof/teacher_lut_v2_8bb.npz")
        ),
        use_batched_rollout=bool(train_raw.get("use_batched_rollout", True)),
        parallel_hands=int(train_raw.get("parallel_hands", 64)),
        torch_compile_villain=bool(train_raw.get("torch_compile_villain", False)),
        torch_compile_hero=bool(train_raw.get("torch_compile_hero", False)),
        store_obs_on_cpu=bool(train_raw.get("store_obs_on_cpu", False)),
        payoff_hu_mode=str(train_raw.get("payoff_hu_mode", "table")),
        hu_showdown_table_path=str(
            train_raw.get(
                "hu_showdown_table_path", "data/aof/hu_showdown_equity.npz"
            )
        ),
    )
    league = LeagueConfig(
        main_pool_size=int(league_raw.get("main_pool_size", 1)),
        archive_top_k=int(league_raw.get("archive_top_k", 8)),
        scripted_pool=bool(league_raw.get("scripted_pool", True)),
        oracle_pool=bool(league_raw.get("oracle_pool", True)),
        adversarial_br_pool=bool(league_raw.get("adversarial_br_pool", True)),
        snapshot_every_updates=int(league_raw.get("snapshot_every_updates", 50)),
        br_every_updates=int(league_raw.get("best_response_every_updates", 200)),
        br_train_steps=int(league_raw.get("br_train_steps", 32)),
    )
    return TrainExperimentConfig(
        experiment=str(data.get("experiment", "aof_v3_ppo_league")),
        obs_version=int(data.get("obs_version", 3)),
        teacher_version=int(data.get("teacher_version", 2)),
        model=model,
        train=train,
        league=league,
        curriculum=_parse_curriculum(data.get("curriculum") or {}),
        validation=ValidationConfig(
            seeds=list(val_raw.get("seeds", [1, 2, 3])),
            paired_eval_hands=int(val_raw.get("paired_eval_hands", 0)),
            br_exploit_hands=int(val_raw.get("br_exploit_hands", 0)),
            deterministic=bool(val_raw.get("deterministic", True)),
        ),
        checkpoint_path=str(
            data.get("out", data.get("checkpoint_path", "checkpoints/aof_rl/policy_league_v3.pt"))
        ),
        checkpoint_best_path=(
            str(data["checkpoint_best_path"])
            if data.get("checkpoint_best_path")
            else (
                str(data["out_best"])
                if data.get("out_best")
                else None
            )
        ),
        checkpoint_dir=str(data.get("checkpoint_dir", "checkpoints/aof_rl")),
    )
