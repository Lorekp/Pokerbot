"""Legal action mask derivation for normalized AoF game states."""

from __future__ import annotations

import torch

from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD
from aof_rl.game_state import AoFGameState, seats_by_id
from aof_rl.obs_v3 import NUM_ACTIONS


def legal_mask_for_game(game: AoFGameState) -> torch.Tensor:
    """
    Return [fold, all_in] legality for the hero decision.

    Hero must be pending with positive remaining stack for both actions.
    """
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    by_id = seats_by_id(game)
    hero = by_id.get(game.hero_seat_id)
    if hero is None:
        return mask
    if hero.state != "pending":
        return mask
    if hero.stack_remaining_bb <= 1e-9:
        return mask
    mask[ACTION_FOLD] = 1.0
    mask[ACTION_ALL_IN] = 1.0
    return mask


def apply_legal_mask_to_obs(obs, legal: torch.Tensor):
    """Return ObsV3 copy with updated legal_mask."""
    from aof_rl.obs_v3 import ObsV3

    return ObsV3(
        hero_hand=obs.hero_hand,
        global_feats=obs.global_feats,
        seat_tokens=obs.seat_tokens,
        seat_mask=obs.seat_mask,
        history_tokens=obs.history_tokens,
        history_mask=obs.history_mask,
        legal_mask=legal.to(dtype=obs.legal_mask.dtype, device=obs.legal_mask.device),
    )
