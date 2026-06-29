"""LeagueMatch protocol — the conflict-free SEAM between the league and the (Python ref | Rust) sim.

The league NEVER plays hands (design §5). It marshals a ``seat → member`` map and calls
``LeagueMatchBackend.play_match(...)``, which returns the per-member CV'd $-ROI deltas. The SAME
protocol is implemented by:
  * :class:`ReferenceLeagueMatch` — the Python reference impl HERE, over ``mtt_lifecycle`` +
    ``eval_score`` (a drop-in, runnable now);
  * ``tools/mtt_sim_native`` (the follow-on Rust crate) — the fast impl, byte-identical PyO3 boundary.

Because both implement the identical signature, the native crate swaps in with zero API churn and zero
contact with ``battery_native`` (the ownership firewall).

THE PROTOCOL (the exact method the Rust sim must implement):

    play_match(seat_to_member: dict[int, MemberSpec],
               n_hands: int,
               settle_seed: int,
               *, payouts, start_chips, fmt) -> MatchResult

    where MatchResult carries, per seat/member:
        * cv_dollar_deltas: list[float]  — per-terminal-hand ``eval_score.score_with_cv`` $-deltas
        * hero_finish: int               — the hero seat's finish rank (1 == won)
        * hero_payout: float             — the hero seat's realized prize
        * mean_cv: float                 — mean of the CV deltas (the headline $-ROI signal)

A ``MemberSpec`` is the league's opaque description of a seat occupant: a ``decide_fn`` (the
LiftedAgent's serve policy adapted to the ``mtt_lifecycle`` ``SpotContext``) + the member id. The
reference impl uses the lifecycle's pluggable ``DecideFn`` seam (``decide_fn`` for the hero seat,
``villain_decide_fn`` for the field) — the lifecycle already supports a heterogeneous field via those
two callables, and a richer per-seat field is the Rust sim's job (the reference impl approximates a
heterogeneous field by the hero-vs-field split, which is all ``mtt_lifecycle`` exposes today).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from pokerbot.tournament import eval_score as _eval_score
from pokerbot.tournament import mtt_lifecycle as _life

DecideFn = _life.DecideFn  # SpotContext -> SpotDecision


@dataclass
class MemberSpec:
    """A seat occupant: the league member id + its lifecycle ``decide_fn``.

    The league builds this from a :class:`LiftedAgent` (or a hero net) via
    :func:`pokerbot.tournament.realistic_field.lifted_agent_decide_fn` — kept out of this module so the
    protocol stays dependency-light (the Rust sim only needs the member id + an agent_spec, not a
    Python callable).
    """

    member_id: int
    decide_fn: DecideFn
    is_hero: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class MatchResult:
    """Per-member CV'd $-ROI results from one ``play_match`` call (the protocol return type)."""

    # member_id -> list of per-hand CV $-deltas the hero seat realized (the variance-reduced signal).
    cv_dollar_deltas: dict[int, list[float]]
    hero_member_id: int
    hero_finish: int
    hero_payout: float
    n_players: int
    levels_played: int = 0

    def mean_cv(self, member_id: Optional[int] = None) -> float:
        mid = member_id if member_id is not None else self.hero_member_id
        vals = self.cv_dollar_deltas.get(mid, [])
        return float(np.mean(vals)) if vals else 0.0


@runtime_checkable
class LeagueMatchBackend(Protocol):
    """The protocol both the Python ref impl and the Rust sim implement (the swap-in seam)."""

    def play_match(
        self,
        seat_to_member: dict[int, MemberSpec],
        n_hands: int,
        settle_seed: int,
        *,
        payouts: Sequence[float],
        start_chips: Optional[int] = None,
        fmt=None,
    ) -> MatchResult:
        ...


class ReferenceLeagueMatch:
    """The Python reference backend over ``mtt_lifecycle`` + ``eval_score`` (drop-in, runnable now).

    Plays ONE tournament with the hero seat's ``decide_fn`` and the field's ``decide_fn`` (the
    lifecycle's two-callable heterogeneous-field seam), scores the hero's per-spot chip deltas to
    CV'd $-deltas via ``eval_score.score_with_cv``, and returns a :class:`MatchResult`. ``settle_seed``
    is the CRN hook (paired arms share runouts).
    """

    def __init__(self, *, seats_per_table: int = 9, realistic: bool = True) -> None:
        self.seats_per_table = seats_per_table
        self.realistic = realistic

    def play_match(
        self,
        seat_to_member: dict[int, MemberSpec],
        n_hands: int,                       # interpreted as n_players for the lifecycle (one MTT)
        settle_seed: int,
        *,
        payouts: Sequence[float],
        start_chips: Optional[int] = None,
        fmt=None,
    ) -> MatchResult:
        if fmt is None:
            from pokerbot.tournament.payouts import TournamentFormat

            fmt = TournamentFormat.NORMAL

        hero_spec = next((m for m in seat_to_member.values() if m.is_hero), None)
        if hero_spec is None:
            raise ValueError("play_match needs exactly one hero MemberSpec (is_hero=True)")
        # The field decide_fn: any one non-hero member (the lifecycle field is a single villain_decide_fn;
        # a true per-seat field is the Rust sim's job). Pick the most common non-hero member.
        field_specs = [m for m in seat_to_member.values() if not m.is_hero]
        field_fn: DecideFn = field_specs[0].decide_fn if field_specs else hero_spec.decide_fn
        field_id = field_specs[0].member_id if field_specs else hero_spec.member_id

        n_players = int(n_hands)
        rng = random.Random(settle_seed)
        result = _life.simulate_tournament_lifecycle(
            n_players,
            list(payouts),
            hero_spec.decide_fn,
            field_fn,
            seats_per_table=self.seats_per_table,
            rng=rng,
            fmt=fmt,
            start_chips=start_chips,
            realistic=self.realistic,
            trace=True,
        )

        # Score the per-spot trace to CV'd $-deltas for the hero seat (field index of the hero is 0 in
        # the lifecycle's identity-tracked balancing — the hero is player 0 at table 0 seat 0).
        cv_deltas: list[float] = []
        for spot in result.trace:
            if not spot.hero_involved:
                continue
            pre = list(spot.pre_stacks)
            n = len(pre)
            if n == 0:
                continue
            hero_idx = 0 if 0 < n else None
            if hero_idx is None:
                continue
            chip_delta = spot.chip_delta[hero_idx] if hero_idx < len(spot.chip_delta) else 0.0
            # realized $-delta for the hero from this confrontation's chip delta, then CV-subtract the
            # pre-hand ICM baseline. We use score_chip_to_dollar for the realized $, score_with_cv for
            # the baseline subtraction (mean-preserving, design risk #6).
            realized = float(
                _eval_score.score_chip_to_dollar(pre, _hero_only_delta(chip_delta, hero_idx, n),
                                                 list(spot.payouts), fmt)[hero_idx]
            )
            cv_deltas.append(_eval_score.score_with_cv(realized, pre, list(spot.payouts), hero_idx))

        return MatchResult(
            cv_dollar_deltas={field_id: cv_deltas},
            hero_member_id=hero_spec.member_id,
            hero_finish=result.hero_finish,
            hero_payout=result.hero_payout,
            n_players=result.players,
            levels_played=result.levels_played,
        )


def _hero_only_delta(hero_chip_delta: float, hero_idx: int, n: int) -> list[float]:
    """A per-seat chip-delta vector carrying only the hero's delta (the rest absorbed by the field).

    ``icm_delta`` needs a full per-seat vector; for the hero's CV signal we only need the hero index
    correct, so we put ``hero_chip_delta`` at ``hero_idx`` and spread its negative over the others (a
    zero-sum vector — the field's exact distribution does not affect the hero's ICM delta sign).
    """
    v = [0.0] * n
    v[hero_idx] = float(hero_chip_delta)
    if n > 1:
        share = -float(hero_chip_delta) / (n - 1)
        for i in range(n):
            if i != hero_idx:
                v[i] = share
    return v


# -- a trivial in-process backend for unit tests (no engine) ---------------------------------------


class StubLeagueMatch:
    """A deterministic stub backend for protocol/league unit tests (no engine, no randomness).

    ``play_match`` returns a fixed small positive hero CV signal scaled by the hero's metadata
    ``strength`` minus the field's, so the league's add/prune/PFSP logic can be exercised without
    playing real tournaments. Implements the :class:`LeagueMatchBackend` protocol.
    """

    def play_match(
        self,
        seat_to_member: dict[int, MemberSpec],
        n_hands: int,
        settle_seed: int,
        *,
        payouts: Sequence[float],
        start_chips: Optional[int] = None,
        fmt=None,
    ) -> MatchResult:
        hero = next(m for m in seat_to_member.values() if m.is_hero)
        field = next((m for m in seat_to_member.values() if not m.is_hero), hero)
        edge = float(hero.metadata.get("strength", 0.5)) - float(field.metadata.get("strength", 0.5))
        rng = np.random.default_rng(settle_seed)
        deltas = (edge + 0.01 * rng.standard_normal(max(1, n_hands))).tolist()
        return MatchResult(
            cv_dollar_deltas={field.member_id: deltas},
            hero_member_id=hero.member_id,
            hero_finish=1 if edge > 0 else int(payouts and len(payouts) or 2),
            hero_payout=float(max(0.0, edge)) * (payouts[0] if payouts else 1.0),
            n_players=len(seat_to_member),
        )
