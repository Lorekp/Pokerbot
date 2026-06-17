"""HU local best-response proxy for SD-CFR (directional exploitability signal)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand, is_chance_node, legal_action_mask, sample_chance, step_state
from pokerbot.training.sd_cfr_nlhe import SDCFRNLHETrainer


@dataclass
class HULBRResult:
    exploiter_ev_bb_per_hand: float
    hands: int
    note: str = "Heuristic LBR proxy — not certified Nash exploitability."


class _AggressiveExploiter:
    """Simple raise-heavy response for directional signal."""

    def pick(self, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return 0
        for prefer in (10, 9, 8, 7, 6, 5, 2, 1, 0):  # prefer large bets / call
            if prefer in legal:
                return int(prefer)
        return int(legal[0])


def hu_lbr_proxy_ev(
    trainer: SDCFRNLHETrainer,
    *,
    hands: int = 32,
    seed: int = 0,
    max_depth: int = 48,
) -> HULBRResult:
    """Exploiter EV (bb/hand) vs SD-CFR average strategy in HU rollouts."""
    occ = jnp.asarray(trainer.occupancy, dtype=bool)
    key = jax.random.PRNGKey(seed)
    exploiter = _AggressiveExploiter()
    # Use the ACTUAL live seats (the HU config sits at the 6-max BTN/BB seats 3/5, not 0/1) — keying
    # hero/villain to 0/1 made seat==hero never match, so the blueprint never acted and rewards[hero]
    # read an empty seat (always 0). Alternate which live seat is the blueprint to cancel position.
    live = list(trainer._live_seats)
    total = 0.0
    for h in range(hands):
        key, k = jax.random.split(key)
        state = init_hand(k, occupancy=occ)
        hero = live[h % 2]
        villain = live[1 - (h % 2)]
        depth = 0
        while depth < max_depth and not bool(state.done):
            while bool(is_chance_node(state)):
                state = sample_chance(state)
            if bool(state.done):
                break
            seat = int(state.action_seat)
            mask = np.asarray(legal_action_mask(state), dtype=bool)
            if seat == hero:
                sigma = trainer.average_strategy_policy(state, seat)
                legal = np.flatnonzero(mask)
                probs = sigma[legal] / max(sigma[legal].sum(), 1e-12)
                key, sub = jax.random.split(key)
                pick = int(jax.random.choice(sub, legal.shape[0], p=jnp.asarray(probs, dtype=jnp.float32)))
                action = int(legal[pick])
            else:
                action = exploiter.pick(mask)
            state = step_state(state, jnp.int32(action))
            depth += 1
        if bool(state.done):
            r = float(np.asarray(state.rewards[hero], dtype=np.float64) / CHIP_UNIT)
            total += r
    ev = total / max(hands, 1)
    return HULBRResult(
        exploiter_ev_bb_per_hand=ev,
        hands=hands,
    )
