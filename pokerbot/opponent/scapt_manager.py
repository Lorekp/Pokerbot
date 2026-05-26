"""Scalable Population Training (SCAPT) manager for diverse opponent leagues."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PopulationMember:
    member_id: int
    policy_state: dict[str, torch.Tensor]
    transformer_state: dict[str, torch.Tensor] | None = None
    fitness_bb100: float = 0.0
    games_played: int = 0
    generation: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SCAPTConfig:
    population_size: int = 16
    mutation_sigma: float = 0.02
    mutation_decay: float = 0.995
    min_sigma: float = 0.005
    checkpoint_dir: Path = Path("checkpoints/scapt_population")


class SCAPTManager:
    """
    Maintains a league of diverse agents via weight perturbation and tracks
    fitness for matchmaking during self-play data collection.
    """

    def __init__(self, config: SCAPTConfig | None = None) -> None:
        self.config = config or SCAPTConfig()
        self.population: list[PopulationMember] = []
        self._generation = 0
        self._sigma = self.config.mutation_sigma
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def mutation_sigma(self) -> float:
        return self._sigma

    def init_population(
        self,
        base_policy: nn.Module,
        base_transformer: nn.Module | None = None,
    ) -> list[PopulationMember]:
        """Seed population from base checkpoints with initial perturbations."""
        self.population.clear()
        base_policy_sd = {k: v.detach().cpu().clone() for k, v in base_policy.state_dict().items()}
        base_tf_sd = None
        if base_transformer is not None:
            base_tf_sd = {k: v.detach().cpu().clone() for k, v in base_transformer.state_dict().items()}

        for i in range(self.config.population_size):
            policy_sd = self._perturb_state_dict(base_policy_sd, sigma=0.0 if i == 0 else self._sigma)
            tf_sd = self._perturb_state_dict(base_tf_sd, sigma=0.0 if i == 0 else self._sigma) if base_tf_sd else None
            self.population.append(
                PopulationMember(
                    member_id=i,
                    policy_state=policy_sd,
                    transformer_state=tf_sd,
                    generation=0,
                )
            )
        return self.population

    def _perturb_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        sigma: float | None = None,
    ) -> dict[str, torch.Tensor]:
        sigma = self._sigma if sigma is None else sigma
        out: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if not v.is_floating_point():
                out[k] = v.clone()
                continue
            noise = torch.randn_like(v) * sigma
            out[k] = v + noise
        return out

    def perturb(self, member: PopulationMember) -> PopulationMember:
        """Create mutated copy of a population member."""
        new_id = len(self.population)
        return PopulationMember(
            member_id=new_id,
            policy_state=self._perturb_state_dict(member.policy_state),
            transformer_state=(
                self._perturb_state_dict(member.transformer_state)
                if member.transformer_state
                else None
            ),
            fitness_bb100=0.0,
            games_played=0,
            generation=self._generation,
        )

    def sample_opponents(self, k: int, rng: random.Random | None = None) -> list[int]:
        """Sample k distinct population member indices for matchmaking."""
        rng = rng or random.Random()
        if not self.population:
            return [0] * k
        n = len(self.population)
        if k >= n:
            return list(range(n))
        return rng.sample(range(n), k)

    def sample_opponent_seats(
        self,
        hero_seat: int,
        num_players: int,
        rng: random.Random | None = None,
    ) -> dict[int, int]:
        """Map non-hero seats to population member indices."""
        rng = rng or random.Random()
        mapping: dict[int, int] = {}
        for seat in range(num_players):
            if seat == hero_seat:
                continue
            mapping[seat] = rng.randrange(len(self.population)) if self.population else 0
        return mapping

    def update_fitness(self, member_id: int, bb_per_100: float, games: int = 1) -> None:
        if member_id < 0 or member_id >= len(self.population):
            return
        m = self.population[member_id]
        total = m.games_played + games
        m.fitness_bb100 = (m.fitness_bb100 * m.games_played + bb_per_100 * games) / total
        m.games_played = total

    def advance_generation(self) -> None:
        """Anneal mutation rate and optionally replace weak members."""
        self._generation += 1
        self._sigma = max(
            self.config.min_sigma,
            self._sigma * self.config.mutation_decay,
        )
        if len(self.population) < 2:
            return
        ranked = sorted(self.population, key=lambda m: m.fitness_bb100, reverse=True)
        elite = ranked[: max(1, len(ranked) // 4)]
        while len(self.population) < self.config.population_size:
            parent = random.choice(elite)
            self.population.append(self.perturb(parent))

    def diversity_features(
        self,
        action_distributions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Summarize behavioral diversity from action distributions.

        action_distributions: (P, B, 5, A) softmax probs from population forwards
        Returns: (B, 5, P) stacked member distributions per opponent slot
        """
        return action_distributions.permute(1, 2, 0, 3).mean(dim=-1)

    def get_member_policy_state(self, member_id: int) -> dict[str, torch.Tensor]:
        if not self.population:
            raise RuntimeError("Population not initialized")
        return self.population[member_id].policy_state

    def save(self, path: Path | None = None) -> Path:
        path = path or self.config.checkpoint_dir / "population.json"
        meta = {
            "generation": self._generation,
            "sigma": self._sigma,
            "members": [
                {
                    "member_id": m.member_id,
                    "fitness_bb100": m.fitness_bb100,
                    "games_played": m.games_played,
                    "generation": m.generation,
                    "policy_path": f"member_{m.member_id}_policy.pt",
                    "transformer_path": (
                        f"member_{m.member_id}_transformer.pt" if m.transformer_state else None
                    ),
                }
                for m in self.population
            ],
        }
        for m in self.population:
            torch.save(m.policy_state, self.config.checkpoint_dir / f"member_{m.member_id}_policy.pt")
            if m.transformer_state:
                torch.save(
                    m.transformer_state,
                    self.config.checkpoint_dir / f"member_{m.member_id}_transformer.pt",
                )
        path.write_text(json.dumps(meta, indent=2))
        return path

    def load(self, path: Path | None = None) -> None:
        path = path or self.config.checkpoint_dir / "population.json"
        if not path.exists():
            return
        meta = json.loads(path.read_text())
        self._generation = meta.get("generation", 0)
        self._sigma = meta.get("sigma", self.config.mutation_sigma)
        self.population.clear()
        for entry in meta.get("members", []):
            mid = entry["member_id"]
            policy_path = self.config.checkpoint_dir / entry["policy_path"]
            policy_sd = torch.load(policy_path, map_location="cpu", weights_only=True)
            tf_sd = None
            if entry.get("transformer_path"):
                tf_path = self.config.checkpoint_dir / entry["transformer_path"]
                if tf_path.exists():
                    tf_sd = torch.load(tf_path, map_location="cpu", weights_only=True)
            self.population.append(
                PopulationMember(
                    member_id=mid,
                    policy_state=policy_sd,
                    transformer_state=tf_sd,
                    fitness_bb100=entry.get("fitness_bb100", 0.0),
                    games_played=entry.get("games_played", 0),
                    generation=entry.get("generation", 0),
                )
            )
