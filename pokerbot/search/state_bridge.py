"""Bridge Python GameState to JAX EnvState for TurboReBeL search."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.actions import action_to_index
from pokerbot.engine.game import GameState, NUM_PLAYERS, Street
from pokerbot.engine.jax_engine import (
    BB_CHIPS,
    CHIP_UNIT,
    EnvState,
    MAX_HISTORY,
    SEAT_EMPTY,
    legal_action_mask,
)
from pokerbot.engine.side_pots_jax import SEAT_ALLIN, SEAT_FOLDED, SEAT_LIVE
from pokerbot.search.pbs_tracker import PBSTracker

_log = logging.getLogger(__name__)

STREET_MAP = {
    Street.PREFLOP: 0,
    Street.FLOP: 1,
    Street.TURN: 2,
    Street.RIVER: 3,
    Street.SHOWDOWN: 4,
}

# Largest pot/commit inconsistency we treat as benign float noise. The live feed reports pot_bb and
# per-seat street commitments in whole 0.01bb chips, so any REAL reconstruction error is >= one chip
# (1/CHIP_UNIT bb); we tolerate strictly less than that so genuine rounding never trips the alarm.
_POT_DEAD_TOL_BB: float = 0.5 / CHIP_UNIT


@dataclass
class PotConservationBreach:
    """A LOUD, structured signal that the live reconstruction produced ``pot_bb < sum(committed_street)``.

    Ground-rule 5 (no silent clamps that hide bugs): the old code did ``pot_dead = max(pot_dead, 0)`` and
    SHRANK the dead-money pot to absorb the inconsistency, masking a reconstruction bug on the LIVE serve
    path. We now emit this record instead -- the state stays usable (we still floor pot_dead at 0 so the
    int-chip engine never sees a negative dead pot, i.e. no crash) but the breach is VISIBLE and auditable.
    """

    pot_bb: float          # the feed-reported total pot
    committed_street_bb: float  # sum of per-seat street commitments the reconstruction derived
    pot_dead_bb: float     # the (negative) raw dead-money pot = pot_bb - committed_street_bb
    deficit_bb: float      # how far pot_bb fell short of committed_street_bb (positive magnitude)
    street: str
    hero_seat: int


# Module-level audit trail + optional sink so the live integrity plumbing (pokerbot/live/state_invariants
# -> an I3 pot-conservation check) and the decision log can CONSUME breaches without this module importing
# the live stack. The counter makes the breach rate queryable in tests / monitors; the sink is an optional
# observer callback. Default sink None => the loud logging.warning is the only side effect (do-no-harm).
_pot_breach_count: int = 0
_pot_breach_sink: "Callable[[PotConservationBreach], None] | None" = None


def set_pot_breach_sink(sink: "Callable[[PotConservationBreach], None] | None") -> None:
    """Register (or clear with None) an observer called on every pot-conservation breach. Observe-only —
    it must not mutate the serve path. Used by the live decision log / state-invariant I3 consumer."""
    global _pot_breach_sink
    _pot_breach_sink = sink


def pot_breach_count() -> int:
    """Total pot-conservation breaches recorded this process (monitor / test introspection)."""
    return _pot_breach_count


def _record_pot_breach(breach: PotConservationBreach) -> None:
    global _pot_breach_count
    _pot_breach_count += 1
    # LOUD: a warning (not debug) so it surfaces in the live log instead of being silently absorbed.
    _log.warning(
        "[state_bridge] POT-CONSERVATION BREACH: pot_bb=%.4f < sum(committed_street)=%.4f "
        "(deficit=%.4f bb, raw pot_dead=%.4f) street=%s hero_seat=%d -- reconstruction inconsistency; "
        "flooring pot_dead at 0 to keep the state usable but NOT silently shrinking it away.",
        breach.pot_bb, breach.committed_street_bb, breach.deficit_bb, breach.pot_dead_bb,
        breach.street, breach.hero_seat,
    )
    if _pot_breach_sink is not None:
        try:
            _pot_breach_sink(breach)
        except Exception:  # noqa: BLE001 — a buggy observer must never break the live serve path
            _log.exception("[state_bridge] pot-breach sink raised; ignoring")


def _card_to_idx(card) -> int:
    return card.index if hasattr(card, "index") else int(card)


def _bb_to_chips(x: float) -> int:
    return int(round(x * CHIP_UNIT))


def game_state_to_env(state: GameState, hero_seat: int) -> EnvState:
    """Convert production GameState to JAX EnvState for search stepping."""
    # ALWAYS emit NUM_PLAYERS-wide per-seat arrays, padding absent seats (a short-handed live table that
    # lists fewer than 6 players) as SEAT_EMPTY with 0 chips. The EnvState fields must be uniformly 6-wide:
    # the strict native engine (_to_flat) requires it, and a ragged (len(players),) array silently corrupts
    # downstream indexing even on the pure-Python path. This is also the short-handed (5/4/3-handed, HU)
    # live path — empty seats beyond the seated count.
    holes = np.full((NUM_PLAYERS, 2), -1, dtype=np.int32)
    seat_status = np.full(NUM_PLAYERS, SEAT_EMPTY, dtype=np.int8)
    stacks = np.zeros(NUM_PLAYERS, dtype=np.int32)
    committed_street = np.zeros(NUM_PLAYERS, dtype=np.int32)
    committed_total = np.zeros(NUM_PLAYERS, dtype=np.int32)
    for i, p in enumerate(state.players):
        if i >= NUM_PLAYERS:
            break
        if p.hole:
            holes[i, 0] = _card_to_idx(p.hole[0])
            holes[i, 1] = _card_to_idx(p.hole[1])
        if p.folded:
            seat_status[i] = SEAT_FOLDED
        elif p.all_in:
            seat_status[i] = SEAT_ALLIN
        else:
            seat_status[i] = SEAT_LIVE
        stacks[i] = _bb_to_chips(p.stack_bb)
        committed_street[i] = _bb_to_chips(p.committed_street_bb)
        committed_total[i] = _bb_to_chips(p.committed_total_bb)

    board = np.full(5, -1, dtype=np.int32)
    for i, c in enumerate(state.board[:5]):
        board[i] = _card_to_idx(c)

    hist_seat = np.full(MAX_HISTORY, -1, dtype=np.int32)
    hist_action = np.full(MAX_HISTORY, -1, dtype=np.int32)
    hist_amount = np.zeros(MAX_HISTORY, dtype=np.int32)
    hist_len = 0
    for i, (seat, action) in enumerate(state.action_history()):
        if i >= MAX_HISTORY:
            break
        hist_seat[i] = seat
        hist_action[i] = action_to_index(action)
        hist_amount[i] = _bb_to_chips(action.amount_bb)
        hist_len = i + 1

    # Dead-money pot = total pot minus the chips still on the table this street. A correct reconstruction
    # always has pot_bb >= sum(committed_street); pot_bb < sum(committed_street) means the live feed's pot
    # total and the per-seat street commitments DISAGREE -> a reconstruction bug. The OLD code did
    # ``pot_dead = max(pot_dead, 0)`` and SILENTLY shrank the dead pot to swallow that inconsistency
    # (ground-rule 5 violation: a clamp that hides a bug). We now VALIDATE: if pot_bb falls short by more
    # than half-a-chip of float noise we emit a LOUD, structured PotConservationBreach (warning + counter +
    # optional sink) and only THEN floor pot_dead at 0 to keep the int-chip engine from crashing on a
    # negative dead pot. The state stays usable; the inconsistency is no longer invisible.
    committed_street_sum_bb = sum(p.committed_street_bb for p in state.players)
    pot_dead = _bb_to_chips(state.pot_bb - committed_street_sum_bb)
    deficit_bb = committed_street_sum_bb - state.pot_bb
    if deficit_bb > _POT_DEAD_TOL_BB:
        _record_pot_breach(PotConservationBreach(
            pot_bb=float(state.pot_bb),
            committed_street_bb=float(committed_street_sum_bb),
            pot_dead_bb=float(state.pot_bb - committed_street_sum_bb),
            deficit_bb=float(deficit_bb),
            street=getattr(state.street, "name", str(state.street)),
            hero_seat=int(hero_seat),
        ))
    pot_dead_chips = max(pot_dead, 0)  # floored AFTER the loud signal: keep usable, never silent.

    # Per-street "has acted" flags. The jax engine's _street_complete / reopen logic keys on `acted`
    # (round closes only when every still-actable seat has acted + matched). GameState tracks this in
    # `_acted_this_street` (reset to {raiser} after a full raise, exactly like the jax engine). Leaving
    # it all-False — as the original bridge did — makes the solver believe NO opponent has acted yet, so
    # from any non-first-to-act hero seat it searches a subgame with phantom extra betting rounds (the
    # already-acted opponents are expected to act again), corrupting the EVs out of position.
    acted = np.zeros(NUM_PLAYERS, dtype=bool)
    for s in getattr(state, "_acted_this_street", set()):
        acted[s] = True

    # Reopen/min-raise context. These were hard-coded (min_raise=1bb, last_raise_full=True), which let the
    # engine grant already-acted seats phantom re-raise rights after a short all-in under-raise. Read the
    # real values from the GameState when present (the live reconstruction supplies them); engine-built
    # GameStates lack the attrs and fall back to the prior constants (unchanged behavior).
    _mrb = getattr(state, "min_raise_bb", None)
    min_raise_chips = _bb_to_chips(_mrb) if (_mrb and _mrb > 0) else BB_CHIPS
    last_raise_full = bool(getattr(state, "last_raise_full", True))

    return EnvState(
        holes=jnp.asarray(holes),
        board=jnp.asarray(board),
        seat_status=jnp.asarray(seat_status),
        stacks=jnp.asarray(stacks),
        committed_street=jnp.asarray(committed_street),
        committed_total=jnp.asarray(committed_total),
        street=jnp.int32(STREET_MAP.get(state.street, 4)),
        pot_dead=jnp.int32(pot_dead_chips),
        current_bet=jnp.int32(_bb_to_chips(state.current_bet_bb)),
        min_raise=jnp.int32(min_raise_chips),
        button=jnp.int32(state.button_seat),
        action_seat=jnp.int32(state.action_seat),
        last_aggressor=jnp.int32(state.last_aggressor if state.last_aggressor is not None else -1),
        acted=jnp.asarray(acted),
        last_raise_full=jnp.bool_(last_raise_full),
        deck=jnp.arange(52, dtype=jnp.int32),
        deck_ptr=jnp.int32(0),
        rng_key=jnp.zeros(2, dtype=jnp.uint32),
        hist_seat=jnp.asarray(hist_seat),
        hist_action=jnp.asarray(hist_action),
        hist_amount=jnp.asarray(hist_amount),
        hist_len=jnp.int32(hist_len),
        done=jnp.bool_(state.street == Street.SHOWDOWN or state.active_player_count() <= 1),
        rewards=jnp.zeros(NUM_PLAYERS, dtype=jnp.int32),
        pending_deal=jnp.bool_(False),
    )


def env_to_numpy(env: EnvState) -> EnvState:
    """Materialize JAX arrays to host for Python CFR traversal."""
    return jax.tree.map(lambda x: np.asarray(x), env)


@dataclass
class SubgameRoot:
    """Root node for localized TurboReBeL solve."""

    env: EnvState
    hero_seat: int
    pbs_vector: np.ndarray
    legal_mask: np.ndarray
    # Optional seat -> OpponentStats (from the persistent player DB) for opponent-conditioned
    # safe exploitation in the solve. None (default) => GTO/blueprint behavior, unchanged. The
    # exploitation logic that consumes this is built later (see docs/exploitation-design.md).
    opponent_profiles: dict | None = None


def build_subgame_root(
    state: GameState,
    pbs: PBSTracker,
    hero_seat: int,
    opponent_profiles: dict | None = None,
) -> SubgameRoot:
    env = game_state_to_env(state, hero_seat)
    mask = np.asarray(legal_action_mask(env, jnp.int32(hero_seat)), dtype=bool)
    return SubgameRoot(
        env=env,
        hero_seat=hero_seat,
        pbs_vector=pbs.get_public_belief_state(),
        legal_mask=mask,
        opponent_profiles=opponent_profiles,
    )
