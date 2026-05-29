"""PSRO-style league opponent pools for AoF PPO v3 training."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import torch

from aof_ranges import active_seats
from aof_rl.policy_v3 import AoFPolicyNetV3, EncoderConfig
from aof_rl.rollout import (
    NeuralPolicyActor,
    OraclePolicyActor,
    PolicyActor,
    RolloutConfig,
    ScriptedPolicyActor,
)


class OpponentPool(str, Enum):
    MAIN = "main"
    ARCHIVE = "archive"
    SCRIPTED = "scripted"
    ORACLE = "oracle"
    ADVERSARIAL_BR = "adversarial_br"


MetaSolver = Literal["uniform", "softmax_elo"]


@dataclass
class LeagueConfig:
    main_pool_size: int = 1
    archive_top_k: int = 8
    scripted_pool: bool = True
    oracle_pool: bool = True
    adversarial_br_pool: bool = True
    snapshot_every_updates: int = 50
    br_every_updates: int = 200
    br_train_steps: int = 32
    br_gain_threshold_bb: float = 0.05
    meta_solver: MetaSolver = "uniform"
    scripted_call_prob: float = 0.55
    pool_weights: dict[str, float] = field(
        default_factory=lambda: {
            "archive": 0.35,
            "scripted": 0.2,
            "oracle": 0.25,
            "adversarial_br": 0.2,
        }
    )


@dataclass
class LeagueMember:
    name: str
    pool: OpponentPool
    model: AoFPolicyNetV3 | None = None
    actor: PolicyActor | None = None
    elo: float = 1000.0
    checkpoint_path: str | None = None


class LeagueManager:
    def __init__(
        self,
        config: LeagueConfig,
        encoder_cfg: EncoderConfig,
        device: torch.device,
        rollout_cfg: RolloutConfig,
    ) -> None:
        self.config = config
        self.encoder_cfg = encoder_cfg
        self.device = device
        self.rollout_cfg = rollout_cfg
        self.archive: list[LeagueMember] = []
        self.adversarial: list[LeagueMember] = []
        self._scripted = ScriptedPolicyActor(
            call_prob=config.scripted_call_prob,
            seed=rollout_cfg.seed,
            policy_key="scripted:scripted",
        )
        self._oracle = OraclePolicyActor(
            rollout_cfg, policy_key="oracle:oracle"
        )

    def _clone_model(self, source: AoFPolicyNetV3) -> AoFPolicyNetV3:
        m = AoFPolicyNetV3(self.encoder_cfg).to(self.device)
        m.load_state_dict(copy.deepcopy(source.state_dict()))
        m.eval()
        return m

    def maybe_snapshot(self, update: int, main: AoFPolicyNetV3, bb_hand: float) -> None:
        if update <= 0 or update % self.config.snapshot_every_updates != 0:
            return
        snap = self._clone_model(main)
        member = LeagueMember(
            name=f"archive_u{update}",
            pool=OpponentPool.ARCHIVE,
            model=snap,
            elo=1000.0 + bb_hand * 100.0,
        )
        self.archive.append(member)
        self.archive.sort(key=lambda m: m.elo, reverse=True)
        del self.archive[self.config.archive_top_k :]

    def register_adversarial(self, model: AoFPolicyNetV3, name: str) -> None:
        self.adversarial.append(
            LeagueMember(
                name=name,
                pool=OpponentPool.ADVERSARIAL_BR,
                model=self._clone_model(model),
            )
        )
        del self.adversarial[4:]

    def _sample_pool(self, rng: random.Random) -> OpponentPool:
        w = self.config.pool_weights
        pools: list[OpponentPool] = []
        weights: list[float] = []
        if self.archive:
            pools.append(OpponentPool.ARCHIVE)
            weights.append(w.get("archive", 0.35))
        if self.config.scripted_pool:
            pools.append(OpponentPool.SCRIPTED)
            weights.append(w.get("scripted", 0.2))
        if self.config.oracle_pool:
            pools.append(OpponentPool.ORACLE)
            weights.append(w.get("oracle", 0.25))
        if self.adversarial and self.config.adversarial_br_pool:
            pools.append(OpponentPool.ADVERSARIAL_BR)
            weights.append(w.get("adversarial_br", 0.2))
        if not pools:
            return OpponentPool.ORACLE
        if self.config.meta_solver == "softmax_elo" and self.archive:
            total = sum(weights) or 1.0
            probs = [x / total for x in weights]
            return rng.choices(pools, weights=probs, k=1)[0]
        total = sum(weights) or 1.0
        probs = [x / total for x in weights]
        return rng.choices(pools, weights=probs, k=1)[0]

    def _actor_for_pool(
        self, pool: OpponentPool, main: AoFPolicyNetV3, rng: random.Random
    ) -> PolicyActor:
        if pool == OpponentPool.SCRIPTED:
            return self._scripted
        if pool == OpponentPool.ORACLE:
            return self._oracle
        if pool == OpponentPool.ADVERSARIAL_BR and self.adversarial:
            member = rng.choice(self.adversarial)
            if member.model is not None:
                return NeuralPolicyActor(
                    member.model,
                    self.device,
                    self.rollout_cfg,
                    policy_key=f"{pool.value}:{member.name}",
                )
        if pool == OpponentPool.ARCHIVE and self.archive:
            member = rng.choice(self.archive)
            if member.model is not None:
                return NeuralPolicyActor(
                    member.model,
                    self.device,
                    self.rollout_cfg,
                    policy_key=f"{pool.value}:{member.name}",
                )
        return NeuralPolicyActor(
            main,
            self.device,
            self.rollout_cfg,
            policy_key="neural:main",
        )

    def sample_seat_policies(
        self,
        table_size: int,
        hero_seat: str,
        main: AoFPolicyNetV3,
        rng: random.Random,
    ) -> dict[str, PolicyActor]:
        policies: dict[str, PolicyActor] = {}
        for seat in active_seats(table_size):
            if seat == hero_seat:
                continue
            pool = self._sample_pool(rng)
            policies[seat] = self._actor_for_pool(pool, main, rng)
        return policies
