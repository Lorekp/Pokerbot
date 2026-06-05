"""HU AoF push/fold OpenSpiel game (169-bucket chance deal, tabular CFR+)."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pyspiel

from aof_cfr.buckets import LABELS_169
from aof_cfr.hu_payoff_matrix import build_hu_bucket_equity_matrix, load_equity_matrix
from aof_cfr.infoset import encode_infoset_key, history_after_action
from aof_cfr.terminal_ev import (
    hu_fold_sb_utility,
    hu_push_fold_sb_utility,
    hu_showdown_sb_utility,
)
from aof_ranges import active_seats

_NUM_PLAYERS = 2
_N_BUCKETS = 169
_STACK_BB = 8.0
_SB_POSTED = 0.5
_BB_POSTED = 1.0

_EQ_CACHE: np.ndarray | None = None


class Action(IntEnum):
    FOLD = 0
    ALL_IN = 1


def _load_eq() -> np.ndarray:
    global _EQ_CACHE
    if _EQ_CACHE is not None:
        return _EQ_CACHE
    from pathlib import Path

    cache = Path("data/aof/cfr_base/hu_eq_matrix.npz")
    if cache.is_file():
        _EQ_CACHE = load_equity_matrix(cache)
    else:
        _EQ_CACHE = build_hu_bucket_equity_matrix()
    return _EQ_CACHE


_GAME_TYPE = pyspiel.GameType(
    short_name="python_aof_hu_pushfold",
    long_name="Python AoF HU Push/Fold (8bb)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={},
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=2,
    max_chance_outcomes=_N_BUCKETS * _N_BUCKETS,
    num_players=_NUM_PLAYERS,
    min_utility=-_STACK_BB,
    max_utility=_STACK_BB,
    utility_sum=0.0,
    max_game_length=4,
)

_SEATS = active_seats(2)


class AoFHUPushfoldGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})
        self._eq = _load_eq()

    def new_initial_state(self):
        return AoFHUPushfoldState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return _NullObserver()


class _NullObserver:
    def __init__(self, params=None):
        self.tensor = np.zeros(1, np.float32)
        self.dict = {}

    def set_from(self, state, player):
        pass

    def string_from(self, state, player):
        return state.information_state_string(player)


class AoFHUPushfoldState(pyspiel.State):
    def __init__(self, game: AoFHUPushfoldGame):
        super().__init__(game)
        self._deal_done = False
        self._sb_bucket = -1
        self._bb_bucket = -1
        self._history = ""
        self._terminal = False
        self._returns: list[float] | None = None
        self._cur_player = 0

    def _sb_pos(self) -> str:
        return _SEATS[0]

    def _bb_pos(self) -> str:
        return _SEATS[1]

    def current_player(self):
        if self._terminal:
            return pyspiel.PlayerId.TERMINAL
        if not self._deal_done:
            return pyspiel.PlayerId.CHANCE
        return self._cur_player

    def _legal_actions(self, player):
        return [Action.FOLD, Action.ALL_IN]

    def chance_outcomes(self):
        assert not self._deal_done
        p = 1.0 / (_N_BUCKETS * _N_BUCKETS)
        return [(i * _N_BUCKETS + j, p) for i in range(_N_BUCKETS) for j in range(_N_BUCKETS)]

    def _apply_action(self, action):
        if not self._deal_done:
            idx = int(action)
            self._sb_bucket = idx // _N_BUCKETS
            self._bb_bucket = idx % _N_BUCKETS
            self._deal_done = True
            self._cur_player = 0
            return

        eq = self.get_game()._eq
        if self._cur_player == 0:
            if action == Action.FOLD:
                sb_u = hu_fold_sb_utility(sb_posted=_SB_POSTED)
                self._returns = [sb_u, -sb_u]
                self._terminal = True
                return
            self._history = history_after_action(self._history or "-", "A")
            self._cur_player = 1
            return

        # BB acts after SB push
        if action == Action.FOLD:
            sb_u = hu_push_fold_sb_utility(sb_posted=_SB_POSTED, bb_posted=_BB_POSTED)
        else:
            sb_u = hu_showdown_sb_utility(
                self._sb_bucket,
                self._bb_bucket,
                eq,
                stack_bb=_STACK_BB,
                sb_posted=_SB_POSTED,
                bb_posted=_BB_POSTED,
            )
        self._history = history_after_action(self._history, "A" if action == Action.ALL_IN else "F")
        self._returns = [sb_u, -sb_u]
        self._terminal = True

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            idx = int(action)
            i, j = idx // _N_BUCKETS, idx % _N_BUCKETS
            return f"Deal:{LABELS_169[i]}/{LABELS_169[j]}"
        return "FOLD" if action == Action.FOLD else "ALL_IN"

    def is_terminal(self):
        return self._terminal

    def returns(self):
        if not self._terminal or self._returns is None:
            return [0.0, 0.0]
        return self._returns

    def information_state_string(self, player):
        if player == 0:
            bucket = LABELS_169[self._sb_bucket] if self._sb_bucket >= 0 else "??"
            hist = self._history if self._history else "-"
            return encode_infoset_key(self._sb_pos(), bucket, hist)
        bucket = LABELS_169[self._bb_bucket] if self._bb_bucket >= 0 else "??"
        hist = self._history if self._history else "-"
        return encode_infoset_key(self._bb_pos(), bucket, hist)

    def __str__(self):
        return f"HU({self._sb_bucket},{self._bb_bucket},{self._history})"


pyspiel.register_game(_GAME_TYPE, AoFHUPushfoldGame)
