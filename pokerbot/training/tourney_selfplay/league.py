"""League / PSRO manager — AlphaStar roles, PFSP matchmaking, add/prune, frozen past, diversity gate.

The league marshals a roster of typed members (design component #7) and produces the seat→member maps
the hero PPO trainer plays against. It plays NO hands itself (the conflict-free SEAM, design §5): it
only bookkeeps statedicts + payoff stats and calls a ``LeagueMatchBackend.play_match(...)`` — which the
Python reference impl and the Rust sim both implement. This keeps the native crate swap-in zero-API-churn.

Roles (AlphaStar, Vinyals et al. 2019):
  * ``FROZEN_HUMAN``    — a LiftedAgent clone tier (the realistic field floor; NEVER pruned).
  * ``MAIN``            — the hero in training (NEVER pruned).
  * ``MAIN_EXPLOITER``  — trains ONLY vs the current hero (reward = -hero_$).
  * ``LEAGUE_EXPLOITER`` — trains by f_hard PFSP over the whole history.
  * ``PAST_MAIN``       — frozen hero snapshots (the curriculum's "f_hard past snapshots").

PFSP weighting (AlphaStar): given the hero's win-prob ``x`` vs a candidate,
  * ``f_hard(x) = (1-x)^p``   (p=2) — concentrate on the hardest opponents the hero can still beat.
  * ``f_var(x)  = x(1-x)``     — the learning-signal fallback when the hero is winning/losing too hard.

add / prune rules:
  * ADD a frozen copy at win-rate ≥ 0.70 (design D3) OR a step budget.
  * PRUNE over a cap (N≤50, design): never FROZEN_HUMAN/MAIN; otherwise meta-Nash σ* support over the
    empirical payoff matrix M (cheap regret-matching), drop the lowest-support + most-redundant
    PAST_MAIN. Implemented here with a ``nash_baseline``-style regret-matching meta-solve (reused) and a
    cosine-redundancy tiebreak; the elo K-best fallback is the documented degrade.
  * DIVERSITY GATE: ``scapt_manager.measure_league_policy_diversity`` mean pairwise action-KL ≥ 0.02;
    below it, force a reset-to-clone of one exploiter (inject a new style) instead of a near-duplicate.

The payoff matrix M and roster are JSON-serializable (``to_dict``/``from_dict``) so a PSRO round can
checkpoint between the (strengthen / exploit / hero-train / prune) phases.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from pokerbot.opponent.scapt_manager import SCAPT_MIN_LEAGUE_ACTION_KL


class Role(str, Enum):
    FROZEN_HUMAN = "frozen_human"
    MAIN = "main"
    MAIN_EXPLOITER = "main_exploiter"
    LEAGUE_EXPLOITER = "league_exploiter"
    PAST_MAIN = "past_main"


_NEVER_PRUNE = {Role.FROZEN_HUMAN, Role.MAIN}

# AlphaStar PFSP exponent for f_hard.
PFSP_P = 2.0
# Curriculum mixture weights (design Phase E): 0.35 self-play / 0.50 f_hard past / 0.15 human+exploiters.
MIX_SELF_PLAY = 0.35
MIX_PAST = 0.50
MIX_HUMAN_FLOOR = 0.15
# ADD threshold: an exploiter beating its target at this $-ROI win-rate is frozen into the league.
ADD_WIN_RATE = 0.70
# Reset-to-clone probability on ADD (design D3).
RESET_TO_CLONE_PROB = 0.25
# League cap (design): prune PAST_MAIN over this.
LEAGUE_CAP = 50


@dataclass
class LeagueMember:
    """A typed league member. ``clone_tier`` / ``lam`` tag a LiftedAgent; ``state_id`` keys its
    statedict (the actual weights live in the backend / a sidecar, not in this bookkeeping object)."""

    member_id: int
    role: Role
    clone_tier: Optional[str] = None         # for FROZEN_HUMAN / reset-to-clone exploiters
    lam: Optional[float] = None              # the tier's play-λ (None = a learned net member)
    state_id: Optional[str] = None           # opaque handle to the weights (ckpt path / sidecar key)
    parent_id: Optional[int] = None          # provenance (AlphaStar parent)
    games_played: int = 0
    wins: float = 0.0                        # accumulated $-ROI win count (for win-rate)
    metadata: dict = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.games_played) if self.games_played else 0.0

    def to_dict(self) -> dict:
        return {
            "member_id": self.member_id,
            "role": self.role.value,
            "clone_tier": self.clone_tier,
            "lam": self.lam,
            "state_id": self.state_id,
            "parent_id": self.parent_id,
            "games_played": self.games_played,
            "wins": self.wins,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LeagueMember":
        return cls(
            member_id=int(d["member_id"]),
            role=Role(d["role"]),
            clone_tier=d.get("clone_tier"),
            lam=d.get("lam"),
            state_id=d.get("state_id"),
            parent_id=d.get("parent_id"),
            games_played=int(d.get("games_played", 0)),
            wins=float(d.get("wins", 0.0)),
            metadata=d.get("metadata", {}),
        )


def f_hard(x: float, p: float = PFSP_P) -> float:
    """AlphaStar PFSP ``f_hard(x) = (1-x)^p`` — concentrate on the hardest beatable opponents."""
    return float(max(0.0, 1.0 - x) ** p)


def f_var(x: float) -> float:
    """AlphaStar PFSP ``f_var(x) = x(1-x)`` — the variance/learning-signal fallback."""
    return float(max(0.0, x * (1.0 - x)))


class League:
    """The PSRO/AlphaStar league bookkeeper (plays no hands; calls a backend's ``play_match``)."""

    def __init__(self, *, cap: int = LEAGUE_CAP, rng: Optional[random.Random] = None) -> None:
        self.members: list[LeagueMember] = []
        self.cap = cap
        self.rng = rng or random.Random(0)
        self._next_id = 0
        # Empirical payoff matrix M[i][j] = mean hero(i) $-ROI vs member j (PSRO meta-game).
        self._payoff: dict[tuple[int, int], list[float]] = {}

    # -- roster -----------------------------------------------------------------------------------
    def add_member(
        self,
        role: Role,
        *,
        clone_tier: Optional[str] = None,
        lam: Optional[float] = None,
        state_id: Optional[str] = None,
        parent_id: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> LeagueMember:
        m = LeagueMember(
            member_id=self._next_id,
            role=role,
            clone_tier=clone_tier,
            lam=lam,
            state_id=state_id,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self.members.append(m)
        self._next_id += 1
        return m

    def by_role(self, role: Role) -> list[LeagueMember]:
        return [m for m in self.members if m.role == role]

    def get(self, member_id: int) -> LeagueMember:
        for m in self.members:
            if m.member_id == member_id:
                return m
        raise KeyError(member_id)

    # -- bootstrap (design Phase B) ---------------------------------------------------------------
    def seed_human_floor(self, tier_lambdas: dict[str, float], weights: Optional[dict[str, float]] = None) -> list[LeagueMember]:
        """Seed FROZEN_HUMAN LiftedAgent tiers (population-proportional). ``tier_lambdas`` maps a tier
        name to its representative play-λ; ``weights`` (optional) tags the population-proportional mass
        in member metadata for the realistic-field sampler to read."""
        out = []
        for tier, lam in tier_lambdas.items():
            w = (weights or {}).get(tier)
            out.append(
                self.add_member(
                    Role.FROZEN_HUMAN, clone_tier=tier, lam=lam,
                    metadata={"population_weight": w} if w is not None else {},
                )
            )
        return out

    # -- PFSP matchmaking (design Phase E) --------------------------------------------------------
    def pfsp_weights(self, hero_id: int, candidates: list[LeagueMember], mode: str = "f_hard") -> list[float]:
        """PFSP weights for ``candidates`` from the hero's empirical win-prob vs each (payoff matrix)."""
        fn = f_hard if mode == "f_hard" else f_var
        weights = []
        for c in candidates:
            x = self._win_prob(hero_id, c.member_id)
            weights.append(fn(x))
        s = sum(weights)
        if s <= 0:  # all-zero (e.g. no data yet) -> uniform
            return [1.0 / len(candidates)] * len(candidates) if candidates else []
        return [w / s for w in weights]

    def sample_opponents(self, hero_id: int, n_seats: int, mode: str = "mixture") -> list[LeagueMember]:
        """Sample ``n_seats`` opponents for the hero per the curriculum mixture (design Phase E).

        ``mixture`` draws each seat from {self-play 0.35 / past-snapshot f_hard 0.50 / human-floor +
        exploiters 0.15}. ``f_hard`` / ``f_var`` force a single PFSP pool.
        """
        if mode in ("f_hard", "f_var"):
            pool = [m for m in self.members if m.member_id != hero_id]
            if not pool:
                return []
            w = self.pfsp_weights(hero_id, pool, mode=mode)
            return list(self.rng.choices(pool, weights=w, k=n_seats))

        past = self.by_role(Role.PAST_MAIN)
        floor = self.by_role(Role.FROZEN_HUMAN) + self.by_role(Role.MAIN_EXPLOITER) + self.by_role(Role.LEAGUE_EXPLOITER)
        hero = self.get(hero_id)
        seats: list[LeagueMember] = []
        for _ in range(n_seats):
            r = self.rng.random()
            if r < MIX_SELF_PLAY or (not past and not floor):
                seats.append(hero)                       # self-play
            elif r < MIX_SELF_PLAY + MIX_PAST and past:
                w = self.pfsp_weights(hero_id, past, mode="f_hard")
                seats.append(self.rng.choices(past, weights=w, k=1)[0])
            elif floor:
                seats.append(self.rng.choice(floor))     # human floor + exploiters
            elif past:
                w = self.pfsp_weights(hero_id, past, mode="f_hard")
                seats.append(self.rng.choices(past, weights=w, k=1)[0])
            else:
                seats.append(hero)
        return seats

    # -- payoff bookkeeping -----------------------------------------------------------------------
    def record_result(self, hero_id: int, opp_id: int, roi: float, win: bool) -> None:
        """Record one match's hero $-ROI vs an opponent (updates M + the opponent's win-rate book)."""
        self._payoff.setdefault((hero_id, opp_id), []).append(float(roi))
        # An exploiter "wins" when IT beats the hero, i.e. the hero's roi vs it is < 0.
        opp = self.get(opp_id)
        opp.games_played += 1
        if not win:  # hero lost => the opponent beat the hero
            opp.wins += 1.0

    def _win_prob(self, hero_id: int, opp_id: int) -> float:
        """Hero's empirical win-prob vs ``opp`` = P(hero roi > 0) over recorded matches (0.5 prior)."""
        vals = self._payoff.get((hero_id, opp_id))
        if not vals:
            return 0.5
        return sum(1 for v in vals if v > 0) / len(vals)

    def payoff_matrix(self, member_ids: Optional[list[int]] = None) -> np.ndarray:
        """Dense mean payoff matrix M over the given members (default: all)."""
        ids = member_ids or [m.member_id for m in self.members]
        n = len(ids)
        M = np.zeros((n, n), dtype=np.float64)
        index = {mid: i for i, mid in enumerate(ids)}
        for (i_id, j_id), vals in self._payoff.items():
            if i_id in index and j_id in index and vals:
                M[index[i_id], index[j_id]] = sum(vals) / len(vals)
        return M

    # -- add / prune (design Phase D/PSRO prune) --------------------------------------------------
    def maybe_add_frozen(
        self, exploiter_id: int, *, win_rate_threshold: float = ADD_WIN_RATE, step_budget_hit: bool = False
    ) -> Optional[LeagueMember]:
        """ADD a frozen PAST_MAIN copy of an exploiter beating its target at ≥ threshold (or budget)."""
        exp = self.get(exploiter_id)
        if exp.win_rate >= win_rate_threshold or step_budget_hit:
            reset = self.rng.random() < RESET_TO_CLONE_PROB
            return self.add_member(
                Role.PAST_MAIN,
                clone_tier="strong_reg" if reset else exp.clone_tier,
                state_id=None if reset else exp.state_id,
                parent_id=exp.member_id,
                metadata={"reset_to_clone": reset, "from_exploiter": exp.member_id},
            )
        return None

    def snapshot_hero(self, hero_id: int, state_id: str) -> LeagueMember:
        """Snapshot the hero into the league as a frozen PAST_MAIN (design Phase E step budget)."""
        return self.add_member(Role.PAST_MAIN, state_id=state_id, parent_id=hero_id)

    def prune_to_cap(self) -> list[int]:
        """Prune PAST_MAIN over the cap by meta-Nash σ*-support + cosine redundancy (design PSRO prune).

        Never prunes FROZEN_HUMAN / MAIN. Returns the pruned member ids.
        """
        if len(self.members) <= self.cap:
            return []
        prunable = [m for m in self.members if m.role not in _NEVER_PRUNE]
        # Solve the meta-Nash σ* over M (regret-matching), then drop the lowest-support, most-redundant.
        ids = [m.member_id for m in self.members]
        M = self.payoff_matrix(ids)
        sigma = _meta_nash(M)
        support = {mid: sigma[i] for i, mid in enumerate(ids)}
        redundancy = _row_redundancy(M, ids)

        pruned: list[int] = []
        n_to_prune = len(self.members) - self.cap
        # rank prunable by (lowest support, then highest redundancy)
        ranked = sorted(
            prunable,
            key=lambda m: (support.get(m.member_id, 0.0), -redundancy.get(m.member_id, 0.0)),
        )
        for m in ranked[:n_to_prune]:
            self.members.remove(m)
            pruned.append(m.member_id)
        return pruned

    # -- diversity gate (design §6.4) -------------------------------------------------------------
    def diversity_ok(self, mean_pairwise_action_kl: float, *, floor: float = SCAPT_MIN_LEAGUE_ACTION_KL) -> bool:
        """True iff the league's behavioral spread clears the KL floor (design exploiter diversity gate).

        The actual KL is measured by ``scapt_manager.measure_league_policy_diversity`` (or, for the
        LiftedAgent league, by averaging ``pikl.kl_to_anchor`` spread over probe states); this method
        is the cheap threshold check. When False, the caller resets one exploiter to a fresh clone.
        """
        return mean_pairwise_action_kl >= floor

    def force_reset_exploiter(self, *, clone_tier: str = "strong_reg") -> Optional[LeagueMember]:
        """Reset the lowest-win-rate exploiter to a fresh clone init (inject a new style)."""
        exps = self.by_role(Role.MAIN_EXPLOITER) + self.by_role(Role.LEAGUE_EXPLOITER)
        if not exps:
            return None
        victim = min(exps, key=lambda m: m.win_rate)
        victim.clone_tier = clone_tier
        victim.state_id = None
        victim.wins = 0.0
        victim.games_played = 0
        victim.metadata["reset_to_clone"] = True
        return victim

    # -- serialization ----------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "cap": self.cap,
            "next_id": self._next_id,
            "members": [m.to_dict() for m in self.members],
            "payoff": {f"{i}:{j}": v for (i, j), v in self._payoff.items()},
        }

    @classmethod
    def from_dict(cls, d: dict, *, rng: Optional[random.Random] = None) -> "League":
        lg = cls(cap=int(d.get("cap", LEAGUE_CAP)), rng=rng)
        lg._next_id = int(d.get("next_id", 0))
        lg.members = [LeagueMember.from_dict(x) for x in d.get("members", [])]
        for key, v in d.get("payoff", {}).items():
            i, j = key.split(":")
            lg._payoff[(int(i), int(j))] = list(v)
        return lg

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path, *, rng: Optional[random.Random] = None) -> "League":
        return cls.from_dict(json.loads(Path(path).read_text()), rng=rng)


def _meta_nash(M: np.ndarray, iters: int = 200) -> np.ndarray:
    """A cheap symmetric meta-Nash σ* over the empirical payoff matrix M via regret-matching.

    Treats the population game as symmetric (row = column strategies); returns a mixture over members.
    This is the PSRO prune signal (a diversity-preserving support estimate), NOT a convergence
    certificate (multiplayer has no Nash guarantee — design risk #8). Falls back to uniform on a
    degenerate M.
    """
    n = M.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    regret = np.zeros(n)
    sigma_sum = np.zeros(n)
    sigma = np.ones(n) / n
    A = M - M.T  # zero-sum antisymmetric payoff (row vs mixture)
    for _ in range(iters):
        util = A @ sigma                       # per-strategy utility vs current mixture
        ev = float(sigma @ util)
        regret += np.maximum(util - ev, 0.0)
        pos = np.maximum(regret, 0.0)
        z = pos.sum()
        sigma = (pos / z) if z > 0 else np.ones(n) / n
        sigma_sum += sigma
    avg = sigma_sum / sigma_sum.sum() if sigma_sum.sum() > 0 else np.ones(n) / n
    return avg


def _row_redundancy(M: np.ndarray, ids: list[int]) -> dict[int, float]:
    """Max cosine similarity of each member's payoff row to any other row (the redundancy tiebreak)."""
    n = M.shape[0]
    out: dict[int, float] = {}
    for i in range(n):
        best = 0.0
        ri = M[i]
        ni = np.linalg.norm(ri)
        for j in range(n):
            if i == j:
                continue
            rj = M[j]
            nj = np.linalg.norm(rj)
            if ni > 0 and nj > 0:
                cos = float(ri @ rj / (ni * nj))
                best = max(best, cos)
        out[ids[i]] = best
    return out
