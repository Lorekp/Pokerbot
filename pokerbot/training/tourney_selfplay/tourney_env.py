"""TourneySelfPlayEnv — the clean boundary the hero trains against the league with $-ROI reward.

This is the env interface (design components #10/#11 + §5 seam) the hero PPO trainer consumes. It ties
together:
  * a :class:`League` (the opponent roster + PFSP matchmaking),
  * a :class:`LeagueMatchBackend` (the Python ref impl now; the Rust sim later — PLUGGABLE),
  * the $-ROI reward (``eval_score.score_with_cv`` via :mod:`tournament_reward`),

behind a single, backend-agnostic API:

    env = TourneySelfPlayEnv(league, backend=ReferenceLeagueMatch(), payouts=..., n_players=...)
    rollout = env.collect_round(hero_spec, n_matches=K, seed=...)
    #   -> RoundRollout(cv_rewards, hero_finishes, hero_payouts, matchups, mean_cv, roi)

The hero trainer turns ``rollout.cv_rewards`` into GAE advantages via
``tournament_reward.returns_and_advantages``, updates the hero net, and periodically snapshots the hero
into the league (``league.snapshot_hero``). The env NEVER touches ``battery_native``; swapping the
backend (``backend=NativeMttSim(...)``) changes nothing else — that is the conflict-free seam.

THE EXACT INTERFACE THE RUST SIM MUST IMPLEMENT is :class:`league_match.LeagueMatchBackend`
(``play_match`` returning a :class:`league_match.MatchResult`); the env consumes only that protocol.
The native ``agent_specs`` (is_neural + net_id), the flat encode buffer, ``step_until_inference`` /
``apply_neural_actions`` / ``drain_results`` are the Rust IMPLEMENTATION of ``play_match`` — invisible
to this env, which sees only the (seat→member, n_hands, settle_seed) → MatchResult contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from pokerbot.training.tourney_selfplay.league import League, Role
from pokerbot.training.tourney_selfplay.league_match import (
    LeagueMatchBackend,
    MatchResult,
    MemberSpec,
    ReferenceLeagueMatch,
)


@dataclass
class RoundRollout:
    """One round of hero-vs-league matches (the trainer's experience batch)."""

    cv_rewards: list[float]                       # flat per-hand CV'd $-deltas (the GAE reward series)
    hero_finishes: list[int]
    hero_payouts: list[float]
    matchups: list[tuple[int, int]]               # (hero_id, opponent_id) per match
    n_matches: int

    @property
    def mean_cv(self) -> float:
        return float(np.mean(self.cv_rewards)) if self.cv_rewards else 0.0

    @property
    def mean_finish(self) -> float:
        return float(np.mean(self.hero_finishes)) if self.hero_finishes else 0.0

    @property
    def mean_payout(self) -> float:
        return float(np.mean(self.hero_payouts)) if self.hero_payouts else 0.0

    def roi(self, buy_in_share: float) -> float:
        return (self.mean_payout - buy_in_share) / buy_in_share if buy_in_share > 0 else 0.0


class TourneySelfPlayEnv:
    """The hero-vs-league $-ROI training env (backend-pluggable: Python ref now, Rust sim later)."""

    def __init__(
        self,
        league: League,
        *,
        backend: Optional[LeagueMatchBackend] = None,
        payouts: Sequence[float],
        n_players: int,
        start_chips: Optional[int] = None,
        fmt=None,
        n_seats: int = 6,
    ) -> None:
        self.league = league
        self.backend: LeagueMatchBackend = backend or ReferenceLeagueMatch()
        self.payouts = list(payouts)
        self.n_players = int(n_players)
        self.start_chips = start_chips
        self.fmt = fmt
        self.n_seats = n_seats

    @property
    def buy_in_share(self) -> float:
        return sum(self.payouts) / self.n_players if self.n_players else 0.0

    def play_one(
        self, hero_spec: MemberSpec, opponent_specs: list[MemberSpec], *, seed: int
    ) -> MatchResult:
        """Play ONE tournament: hero + opponents → a MatchResult (via the pluggable backend)."""
        seat_to_member: dict[int, MemberSpec] = {0: hero_spec}
        for i, opp in enumerate(opponent_specs, start=1):
            seat_to_member[i] = opp
        return self.backend.play_match(
            seat_to_member,
            self.n_players,
            settle_seed=seed,
            payouts=self.payouts,
            start_chips=self.start_chips,
            fmt=self.fmt,
        )

    def collect_round(
        self,
        hero_spec: MemberSpec,
        opponent_builder,
        *,
        n_matches: int,
        seed: int = 0,
        mode: str = "mixture",
        n_opponents: int = 5,
    ) -> RoundRollout:
        """Collect ``n_matches`` hero-vs-league matches and aggregate them into a :class:`RoundRollout`.

        ``opponent_builder(member) -> MemberSpec`` turns a sampled :class:`LeagueMember` into a
        ``MemberSpec`` (the league only knows ids + tiers; the builder attaches the decide_fn — kept as
        a callback so this env stays free of the LiftedAgent/torch dependency). PFSP sampling is the
        league's job (``league.sample_opponents``); this env only orchestrates the matches + reward.
        """
        cv_rewards: list[float] = []
        finishes: list[int] = []
        payouts_realized: list[float] = []
        matchups: list[tuple[int, int]] = []

        for m in range(n_matches):
            members = self.league.sample_opponents(hero_spec.member_id, n_opponents, mode=mode)
            opp_specs = [opponent_builder(mem) for mem in members]
            result = self.play_one(hero_spec, opp_specs, seed=seed + m)
            # record reward + per-opponent payoff into the league (drives next-round PFSP)
            opp_id = opp_specs[0].member_id if opp_specs else hero_spec.member_id
            deltas = result.cv_dollar_deltas.get(opp_id, [])
            cv_rewards.extend(deltas)
            finishes.append(result.hero_finish)
            payouts_realized.append(result.hero_payout)
            matchups.append((hero_spec.member_id, opp_id))
            roi = (result.hero_payout - self.buy_in_share)
            self.league.record_result(hero_spec.member_id, opp_id, roi, win=roi > 0)

        return RoundRollout(
            cv_rewards=cv_rewards,
            hero_finishes=finishes,
            hero_payouts=payouts_realized,
            matchups=matchups,
            n_matches=n_matches,
        )

    def eval_vs_human_floor(
        self, hero_spec: MemberSpec, opponent_builder, *, n_matches: int, seed: int = 0
    ) -> RoundRollout:
        """The anti-drift metric (design §6.3): hero $-ROI VS THE FROZEN_HUMAN floor, tracked SEPARATELY.

        Forces a field of FROZEN_HUMAN members only — if the hero stops printing $ vs the human floor,
        drift is caught even if its league win-rate looks fine.
        """
        floor = self.league.by_role(Role.FROZEN_HUMAN)
        if not floor:
            raise ValueError("no FROZEN_HUMAN floor to evaluate against")
        cv_rewards: list[float] = []
        finishes: list[int] = []
        payouts_realized: list[float] = []
        matchups: list[tuple[int, int]] = []
        rng = np.random.default_rng(seed)
        for m in range(n_matches):
            members = [floor[int(rng.integers(len(floor)))] for _ in range(5)]
            opp_specs = [opponent_builder(mem) for mem in members]
            result = self.play_one(hero_spec, opp_specs, seed=seed + m)
            opp_id = opp_specs[0].member_id if opp_specs else hero_spec.member_id
            cv_rewards.extend(result.cv_dollar_deltas.get(opp_id, []))
            finishes.append(result.hero_finish)
            payouts_realized.append(result.hero_payout)
            matchups.append((hero_spec.member_id, opp_id))
        return RoundRollout(
            cv_rewards=cv_rewards,
            hero_finishes=finishes,
            hero_payouts=payouts_realized,
            matchups=matchups,
            n_matches=n_matches,
        )
