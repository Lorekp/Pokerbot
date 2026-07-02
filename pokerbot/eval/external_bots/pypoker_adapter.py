"""Adapters for the four PyPokerEngine example players.

sys.path bootstraps the third_party/pypokerengine tree (no pip install needed).

Each make_*_policy() returns a PolicyFn: (env, seat) -> np.ndarray[11].

The external bots operate via:
    declare_action(valid_actions, hole_card, round_state) -> (action_str, amount)

where:
  hole_card     : list of suit-first strings ['SA', 'DK']   (PPE format)
  valid_actions : [{'action':'fold','amount':0},
                   {'action':'call','amount': bet-TO-total},
                   {'action':'raise','amount':{'min': min-raise-TO, 'max': shove-TO}}]
  round_state   : dict; HonestPlayer reads only round_state['community_card']
                  (list of suit-first strings)

All outputs are routed through common.finalize_action + common.safe_policy.
The global `random` module is seeded via common.stable_seed before each declare_action
so sampling is CRN-deterministic.
"""
from __future__ import annotations

import os
import random
import sys
from typing import Callable

import numpy as np

from pokerbot.eval.external_bots.common import (
    build_view,
    card_id_to_ppe,
    card_id_to_str,
    finalize_action,
    safe_policy,
    stable_seed,
)
from pokerbot.engine.np_engine import (
    SEAT_LIVE,
    SEAT_ALLIN,
    legal_action_mask_np,
    NUM_ACTIONS,
)

# ---------------------------------------------------------------------------
# One-time sys.path bootstrap for the third_party pypokerengine tree.
# We use the path relative to this file's location which is inside the repo.
# ---------------------------------------------------------------------------

def _bootstrap_ppe_path() -> None:
    """Add third_party/pypokerengine to sys.path exactly once."""
    here = os.path.dirname(os.path.abspath(__file__))
    # Walk up: pokerbot/eval/external_bots -> pokerbot/eval -> pokerbot -> repo_root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    ppe_path = os.path.join(repo_root, "third_party", "pypokerengine")
    if ppe_path not in sys.path:
        sys.path.insert(0, ppe_path)


_bootstrap_ppe_path()


# ---------------------------------------------------------------------------
# Helper: build the valid_actions list PPE bots expect
# ---------------------------------------------------------------------------

def _build_valid_actions(view) -> list[dict]:
    """Build the PPE valid_actions structure from a GenericView.

    PPE convention:
      index 0 -> fold   (amount = 0)
      index 1 -> call   (amount = total chips needed to call, i.e. bet-TO-total; 0 if check)
      index 2 -> raise  (amount = {'min': min raise-TO, 'max': stack+paid (shove-TO)})

    The 'call' amount mirrors what PPE uses: the total committed_street target after calling,
    i.e. current_bet (PPE ignores the committed already in; bots subtract internally if needed,
    but the canonical PPE API passes the TO amount).

    If to_call == 0 (free check), call amount = 0, fold amount = 0 (FoldMan needs amount=0).
    'raise' min = current_bet + min_raise (total), max = hero stack + committed_street (shove).
    """
    to_call = view.to_call
    call_amount = int(view.current_bet)  # bet-TO total for call
    if to_call == 0:
        call_amount = 0

    # min raise-TO: total street commitment target = current_bet + min_raise
    # If min_raise makes no sense (stack too small), set to -1 like PPE does
    min_raise_total = int(view.current_bet) + int(view.min_raise)
    shove_to = int(view.hero_stack) + int(view.hero_committed_street)
    if view.hero_stack <= 0 or min_raise_total > shove_to:
        min_rt = -1
        max_rt = -1
    else:
        min_rt = min_raise_total
        max_rt = shove_to

    return [
        {"action": "fold",  "amount": 0},
        {"action": "call",  "amount": call_amount},
        {"action": "raise", "amount": {"min": min_rt, "max": max_rt}},
    ]


def _build_round_state(view) -> dict:
    """Build a minimal round_state dict satisfying HonestPlayer's needs.

    HonestPlayer only reads round_state['community_card'] — a list of suit-first strings.
    """
    return {
        "community_card": view.board_ppe,
        "street": _STREET_NAMES[min(view.street, 3)],
        "pot": {"main": {"amount": view.pot_before}},
    }


_STREET_NAMES = ["preflop", "flop", "turn", "river"]


# ---------------------------------------------------------------------------
# PPE output -> finalize_action translation
# ---------------------------------------------------------------------------

def _translate_ppe_output(env, seat: int, view, action_str: str, amount) -> np.ndarray:
    """Convert PPE declare_action output to our 11-vector via finalize_action.

    action_str : 'fold' | 'call' | 'raise'
    amount     : chips (scalar for fold/call, raise-TO for raise)
    """
    if action_str == "fold":
        return finalize_action(env, seat, "fold")
    if action_str == "call":
        # 'call' with amount=0 when to_call=0 => check
        if view.to_call == 0:
            return finalize_action(env, seat, "check")
        return finalize_action(env, seat, "call")
    if action_str == "raise":
        amt = int(amount) if amount is not None else -1
        if amt < 0 or amt >= view.hero_stack + view.hero_committed_street:
            return finalize_action(env, seat, "all_in")
        return finalize_action(env, seat, "raise", raise_to=amt)
    # Unknown: fall back to call/check
    if view.to_call > 0:
        return finalize_action(env, seat, "call")
    return finalize_action(env, seat, "check")


# ---------------------------------------------------------------------------
# Factory: FoldMan
# ---------------------------------------------------------------------------

def make_fold_man_policy() -> Callable:
    """Always folds (or checks when a fold would be free)."""
    from examples.players.fold_man import FoldMan  # noqa: PLC0415

    bot = FoldMan()

    def _policy(env, seat: int) -> np.ndarray:
        seat = int(seat)
        view = build_view(env, seat)
        rng_seed = stable_seed(env, seat)
        random.seed(rng_seed)

        valid_actions = _build_valid_actions(view)
        hole_card = list(view.hero_hole_ppe)
        round_state = _build_round_state(view)

        action_str, amount = bot.declare_action(valid_actions, hole_card, round_state)
        return _translate_ppe_output(env, seat, view, action_str, amount)

    _policy.__name__ = "fold_man_policy"
    return safe_policy(_policy)


# ---------------------------------------------------------------------------
# Factory: RandomPlayer
# ---------------------------------------------------------------------------

def make_random_policy() -> Callable:
    """Random uniform over fold/call/raise (PPE's 1/3-1/3-1/3 default)."""
    from examples.players.random_player import RandomPlayer  # noqa: PLC0415

    bot = RandomPlayer()

    def _policy(env, seat: int) -> np.ndarray:
        seat = int(seat)
        view = build_view(env, seat)
        rng_seed = stable_seed(env, seat)
        random.seed(rng_seed)

        valid_actions = _build_valid_actions(view)
        hole_card = list(view.hero_hole_ppe)
        round_state = _build_round_state(view)

        action_str, amount = bot.declare_action(valid_actions, hole_card, round_state)
        return _translate_ppe_output(env, seat, view, action_str, amount)

    _policy.__name__ = "random_policy"
    return safe_policy(_policy)


# ---------------------------------------------------------------------------
# Factory: FishPlayer (always call/check)
# ---------------------------------------------------------------------------

def make_fish_policy() -> Callable:
    """Always calls (or checks when free to do so)."""
    from examples.players.fish_player import FishPlayer  # noqa: PLC0415

    bot = FishPlayer()

    def _policy(env, seat: int) -> np.ndarray:
        seat = int(seat)
        view = build_view(env, seat)
        rng_seed = stable_seed(env, seat)
        random.seed(rng_seed)

        valid_actions = _build_valid_actions(view)
        hole_card = list(view.hero_hole_ppe)
        round_state = _build_round_state(view)

        action_str, amount = bot.declare_action(valid_actions, hole_card, round_state)
        return _translate_ppe_output(env, seat, view, action_str, amount)

    _policy.__name__ = "fish_policy"
    return safe_policy(_policy)


# ---------------------------------------------------------------------------
# Factory: HonestPlayer (equity-based call/fold)
# ---------------------------------------------------------------------------

def make_honest_policy(nb_simulation: int = 100) -> Callable:
    """Equity-based call/fold.

    Monkeypatches HonestPlayer.NB_SIMULATION = nb_simulation (default 100 for speed).
    Sets bot.nb_player = n_live at each call so the equity calc uses the correct
    number of active players.
    """
    import examples.players.honest_player as _hp_module  # noqa: PLC0415
    from examples.players.honest_player import HonestPlayer  # noqa: PLC0415

    # Monkeypatch the module-level constant before the bot reads it
    _hp_module.NB_SIMULATION = nb_simulation
    bot = HonestPlayer()

    def _policy(env, seat: int) -> np.ndarray:
        seat = int(seat)
        view = build_view(env, seat)
        rng_seed = stable_seed(env, seat)
        random.seed(rng_seed)

        # Set nb_player from live seat count (HonestPlayer reads self.nb_player in declare_action)
        bot.nb_player = view.n_live

        valid_actions = _build_valid_actions(view)
        hole_card = list(view.hero_hole_ppe)
        round_state = _build_round_state(view)

        action_str, amount = bot.declare_action(valid_actions, hole_card, round_state)
        return _translate_ppe_output(env, seat, view, action_str, amount)

    _policy.__name__ = "honest_policy"
    return safe_policy(_policy)
