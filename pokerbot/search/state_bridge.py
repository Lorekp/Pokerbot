"""Bridge Python GameState to JAX EnvState for TurboReBeL search."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.actions import action_to_index
from pokerbot.engine.game import GameState, NUM_PLAYERS, Street
from pokerbot.engine.jax_engine import MAX_HISTORY, EnvState, legal_action_mask
from pokerbot.search.pbs_tracker import PBSTracker

STREET_MAP = {
    Street.PREFLOP: 0,
    Street.FLOP: 1,
    Street.TURN: 2,
    Street.RIVER: 3,
    Street.SHOWDOWN: 4,
}


def _card_to_idx(card) -> int:
    return card.index if hasattr(card, "index") else int(card)


def game_state_to_env(state: GameState, hero_seat: int) -> EnvState:
    """Convert production GameState to JAX EnvState for search stepping."""
    holes = np.full((NUM_PLAYERS, 2), -1, dtype=np.int32)
    for i, p in enumerate(state.players):
        if p.hole:
            holes[i, 0] = _card_to_idx(p.hole[0])
            holes[i, 1] = _card_to_idx(p.hole[1])

    board = np.full(5, -1, dtype=np.int32)
    for i, c in enumerate(state.board[:5]):
        board[i] = _card_to_idx(c)

    stacks = np.array([p.stack_bb for p in state.players], dtype=np.float32)
    committed_street = np.array([p.committed_street_bb for p in state.players], dtype=np.float32)
    committed_total = np.array([p.committed_total_bb for p in state.players], dtype=np.float32)
    folded = np.array([p.folded for p in state.players], dtype=bool)
    all_in = np.array([p.all_in for p in state.players], dtype=bool)

    hist_seat = np.full(MAX_HISTORY, -1, dtype=np.int32)
    hist_action = np.zeros(MAX_HISTORY, dtype=np.int32)
    hist_amount = np.zeros(MAX_HISTORY, dtype=np.float32)
    hist_len = 0
    for i, (seat, action) in enumerate(state.action_history()):
        if i >= MAX_HISTORY:
            break
        hist_seat[i] = seat
        hist_action[i] = action_to_index(action)
        hist_amount[i] = action.amount_bb
        hist_len = i + 1

    deck = np.arange(52, dtype=np.int32)
    deck_ptr = np.int32(0)

    return EnvState(
        holes=jnp.asarray(holes),
        board=jnp.asarray(board),
        stacks=jnp.asarray(stacks),
        committed_street=jnp.asarray(committed_street),
        committed_total=jnp.asarray(committed_total),
        folded=jnp.asarray(folded),
        all_in=jnp.asarray(all_in),
        street=jnp.int32(STREET_MAP.get(state.street, 4)),
        pot=jnp.float32(state.pot_bb),
        current_bet=jnp.float32(state.current_bet_bb),
        button=jnp.int32(state.button_seat),
        action_seat=jnp.int32(state.action_seat),
        last_aggressor=jnp.int32(state.last_aggressor if state.last_aggressor is not None else -1),
        acted=jnp.zeros(NUM_PLAYERS, dtype=bool),
        deck=jnp.asarray(deck),
        deck_ptr=deck_ptr,
        hist_seat=jnp.asarray(hist_seat),
        hist_action=jnp.asarray(hist_action),
        hist_amount=jnp.asarray(hist_amount),
        hist_len=jnp.int32(hist_len),
        done=jnp.bool_(state.street == Street.SHOWDOWN or state.active_player_count() <= 1),
        rewards=jnp.zeros(NUM_PLAYERS, dtype=jnp.float32),
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


def build_subgame_root(state: GameState, pbs: PBSTracker, hero_seat: int) -> SubgameRoot:
    env = game_state_to_env(state, hero_seat)
    mask = np.asarray(legal_action_mask(env, jnp.int32(hero_seat)), dtype=bool)
    return SubgameRoot(
        env=env,
        hero_seat=hero_seat,
        pbs_vector=pbs.get_public_belief_state(),
        legal_mask=mask,
    )
