"""CFR blueprint bot using AverageStrategyNet in the PyTorch GameState engine."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from pokerbot.engine.actions import Action
from pokerbot.engine.game import GameState
from pokerbot.engine.jax_engine import encode_info_state, legal_action_mask
from pokerbot.search.state_bridge import game_state_to_env
from pokerbot.training.neural_cfr import AverageStrategyNet, load_network_params
from pokerbot.training.self_play import _index_to_action


class JAXBlueprintBot:
    """Wraps neural-CFR average strategy for head-to-head eval in ``game.py``."""

    def __init__(self, checkpoint: str | Path, hidden_dim: int | None = None) -> None:
        payload = load_network_params(checkpoint)
        self.hidden_dim = int(hidden_dim or payload.get("hidden_dim", 256))
        self.params = payload["average_strategy_params"]
        self.module = AverageStrategyNet(self.hidden_dim)
        # Skew guard: the served encode_info_state layout must match what the net was trained on.
        trained_suits = payload.get("info_suits")
        if trained_suits is not None:
            import pokerbot.engine.jax_engine as _je

            if bool(trained_suits) != bool(_je._INFO_SUITS):
                raise RuntimeError(
                    f"info-state encoding skew: checkpoint trained with POKERBOT_INFO_SUITS="
                    f"{int(bool(trained_suits))} but serving with {int(bool(_je._INFO_SUITS))}. "
                    f"Set POKERBOT_INFO_SUITS to match BEFORE importing pokerbot."
                )

    def action_probs(self, state: GameState, seat: int | None = None) -> jnp.ndarray:
        seat = state.action_seat if seat is None else seat
        env = game_state_to_env(state, seat)
        mask = legal_action_mask(env, jnp.int32(seat))
        info = encode_info_state(env, jnp.int32(seat))
        probs = self.module.apply(self.params, info, mask)
        return probs * mask

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        probs = self.action_probs(state, seat)
        idx = int(jnp.argmax(probs).item())
        return _index_to_action(state, seat, idx)
