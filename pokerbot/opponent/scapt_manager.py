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


# Tuned on held-out preflop probe states: σ=0.02 → mean pairwise action KL ≈ 0.01 (collapsed);
# σ=0.08 → mean KL ≈ 0.23, cos ≈ 0.84 (distinct, non-degenerate); σ≥0.12 → heavy tail / unstable max KL.
DEFAULT_MUTATION_SIGMA = 0.08
SCAPT_MIN_LEAGUE_ACTION_KL = 0.02


@dataclass
class SCAPTConfig:
    population_size: int = 16
    mutation_sigma: float = DEFAULT_MUTATION_SIGMA
    mutation_decay: float = 0.995
    min_sigma: float = 0.02
    checkpoint_dir: Path = Path("checkpoints/scapt_population")


@dataclass
class LeagueDiversityReport:
    mean_pairwise_action_kl: float
    min_pairwise_action_kl: float
    max_pairwise_action_kl: float
    mean_pairwise_cosine: float
    mutation_sigma: float
    num_members: int
    num_probe_states: int

    def as_metadata(self) -> dict[str, float | int]:
        return {
            "scapt_league_mean_pairwise_action_kl": self.mean_pairwise_action_kl,
            "scapt_league_min_pairwise_action_kl": self.min_pairwise_action_kl,
            "scapt_league_max_pairwise_action_kl": self.max_pairwise_action_kl,
            "scapt_league_mean_pairwise_cosine": self.mean_pairwise_cosine,
            "scapt_mutation_sigma": self.mutation_sigma,
            "scapt_population_size": self.num_members,
            "scapt_diversity_probe_states": self.num_probe_states,
        }


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

    @staticmethod
    def mean_pairwise_action_kl(
        action_distributions: torch.Tensor,
    ) -> tuple[float, float, float, float]:
        """
        Symmetric mean action KL and cosine similarity across population members.

        action_distributions: (P, B, A) softmax probs — one row per league member,
        averaged over B held-out probe states per member.
        """
        import torch.nn.functional as F

        p_count = int(action_distributions.shape[0])
        if p_count < 2:
            return 0.0, 0.0, 0.0, 1.0

        kls: list[float] = []
        cosines: list[float] = []
        for i in range(p_count):
            for j in range(i + 1, p_count):
                pi = action_distributions[i]
                pj = action_distributions[j]
                kl_ij = (pi * (pi.add(1e-8).log() - pj.add(1e-8).log())).sum(dim=-1).mean()
                kl_ji = (pj * (pj.add(1e-8).log() - pi.add(1e-8).log())).sum(dim=-1).mean()
                kls.append(float(0.5 * (kl_ij + kl_ji)))
                cosines.append(float(F.cosine_similarity(pi, pj, dim=-1).mean()))

        return (
            min(kls),
            max(kls),
            sum(kls) / len(kls),
            sum(cosines) / len(cosines),
        )

    def diversity_features(
        self,
        action_distributions: torch.Tensor,
    ) -> dict[str, float]:
        """
        Summarize behavioral diversity from league member action distributions.

        action_distributions: (P, B, A) softmax probs from SCAPT hero-policy members
        on held-out probe states (B=batch of states, A=actions).
        """
        min_kl, max_kl, mean_kl, mean_cos = self.mean_pairwise_action_kl(action_distributions)
        return {
            "min_pairwise_action_kl": min_kl,
            "max_pairwise_action_kl": max_kl,
            "mean_pairwise_action_kl": mean_kl,
            "mean_pairwise_cosine": mean_cos,
        }

    def measure_league_policy_diversity(
        self,
        base_policy: nn.Module,
        device: torch.device,
        *,
        num_probe_states: int = 16,
        rng: random.Random | None = None,
    ) -> LeagueDiversityReport:
        """
        Forward every SCAPT league member on held-out game states and measure spread.
        """
        from pokerbot.engine.game import GameState
        from pokerbot.encoding.tensors import encode_state
        from pokerbot.training.self_play import _legal_mask

        import copy

        if not self.population:
            raise RuntimeError("SCAPT population not initialized")

        rng = rng or random.Random()
        probe_encodings: list[dict] = []
        for _ in range(num_probe_states):
            state = GameState.new_hand(button_seat=rng.randrange(6), rng=rng)
            seat = rng.randrange(6)
            enc = encode_state_for_policy(state, seat, require_scope=False)
            enc["legal_mask"] = _legal_mask(state, seat)
            probe_encodings.append(enc)

        member_probs: list[torch.Tensor] = []
        for member in self.population:
            model = copy.deepcopy(base_policy)
            model.load_state_dict(member.policy_state)
            model.eval()
            state_probs: list[torch.Tensor] = []
            with torch.no_grad():
                for enc in probe_encodings:
                    weva = torch.tensor([enc["weva_id"]], device=device)
                    cards = torch.from_numpy(enc["cards"]).unsqueeze(0).float().to(device)
                    actions = torch.from_numpy(enc["actions"]).unsqueeze(0).float().to(device)
                    pos = torch.from_numpy(enc["policy_position"]).unsqueeze(0).float().to(device)
                    mask = torch.from_numpy(enc["legal_mask"]).unsqueeze(0).bool().to(device)
                    logits = model(weva, actions, pos, mask, cards=cards).logits
                    state_probs.append(torch.softmax(logits, dim=-1).cpu())
            member_probs.append(torch.cat(state_probs, dim=0))

        stacked = torch.stack(member_probs, dim=0)
        features = self.diversity_features(stacked)
        return LeagueDiversityReport(
            mean_pairwise_action_kl=features["mean_pairwise_action_kl"],
            min_pairwise_action_kl=features["min_pairwise_action_kl"],
            max_pairwise_action_kl=features["max_pairwise_action_kl"],
            mean_pairwise_cosine=features["mean_pairwise_cosine"],
            mutation_sigma=float(self._sigma),
            num_members=len(self.population),
            num_probe_states=num_probe_states,
        )

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


def check_league_diversity(
    report: LeagueDiversityReport,
    *,
    floor: float = SCAPT_MIN_LEAGUE_ACTION_KL,
    require: bool = False,
) -> None:
    """Warn or fail when SCAPT league behavioral spread is below floor."""
    if report.mean_pairwise_action_kl >= floor:
        print(
            f"SCAPT league diversity OK: mean pairwise action KL={report.mean_pairwise_action_kl:.4f} "
            f"(min={report.min_pairwise_action_kl:.4f}, max={report.max_pairwise_action_kl:.4f}, "
            f"cos={report.mean_pairwise_cosine:.4f}, sigma={report.mutation_sigma})"
        )
        return

    banner = (
        "\n" + "=" * 72 + "\n"
        f"WARNING: SCAPT league diversity below floor ({report.mean_pairwise_action_kl:.4f} < {floor}).\n"
        "Opponent Transformer may train against a near-homogeneous league.\n"
        f"sigma={report.mutation_sigma}, members={report.num_members}, "
        f"probe_states={report.num_probe_states}\n"
        + "=" * 72 + "\n"
    )
    if require:
        raise RuntimeError(banner.strip())
    print(banner)
