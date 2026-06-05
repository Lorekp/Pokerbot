"""N-max AoF push/fold OpenSpiel game with combo-level chance deals (MCCFR)."""

from __future__ import annotations

import copy
from enum import IntEnum

import numpy as np
import pyspiel

from aof_cfr.buckets import LABELS_169
from aof_cfr.combo_deal import bucket_from_card_ids
from aof_cfr.infoset import encode_infoset_key, history_after_action
from aof_cfr.terminal_ev import fold_terminal_returns, showdown_terminal_returns
from aof_ranges import active_seats

_STACK_BB = 8.0
_FOLD = 0
_ALL_IN = 1
_PAIR_CACHE: dict[int, list[tuple[int, float]]] = {}


def _dead_mask(dead: np.ndarray) -> int:
    m = 0
    for c in np.nonzero(dead)[0]:
        m |= 1 << int(c)
    return m


def _encode_pair(c0: int, c1: int) -> int:
    if c0 > c1:
        c0, c1 = c1, c0
    return c0 * 52 + c1


def _decode_pair(action: int) -> tuple[int, int]:
    c1 = int(action) % 52
    c0 = int(action) // 52
    if c0 >= c1:
        raise ValueError(f"invalid pair action {action}")
    return c0, c1


def _pair_chance_outcomes(dead: np.ndarray) -> list[tuple[int, float]]:
    key = _dead_mask(dead)
    cached = _PAIR_CACHE.get(key)
    if cached is not None:
        return cached
    rem = [c for c in range(52) if not dead[c]]
    n_pairs = len(rem) * (len(rem) - 1) // 2
    p = 1.0 / n_pairs
    outs = [_encode_pair(rem[i], rem[j]) for i in range(len(rem)) for j in range(i + 1, len(rem))]
    cached = [(a, p) for a in outs]
    _PAIR_CACHE[key] = cached
    return cached


class Action(IntEnum):
    FOLD = 0
    ALL_IN = 1


def _game_type(num_players: int) -> pyspiel.GameType:
    return pyspiel.GameType(
        short_name=f"python_aof_pushfold_{num_players}p",
        long_name=f"Python AoF Push/Fold {num_players}-max (8bb)",
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
        information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.ZERO_SUM,
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        max_num_players=num_players,
        min_num_players=num_players,
        provides_information_state_string=True,
        provides_information_state_tensor=False,
        provides_observation_string=True,
        provides_observation_tensor=False,
        parameter_specification={},
    )


def _game_info(num_players: int) -> pyspiel.GameInfo:
    return pyspiel.GameInfo(
        num_distinct_actions=2,
        max_chance_outcomes=1326,
        num_players=num_players,
        min_utility=-_STACK_BB,
        max_utility=_STACK_BB,
        utility_sum=0.0,
        max_game_length=num_players + num_players + 2,
    )


class AoFPushfoldGame(pyspiel.Game):
    def __init__(self, game_type, game_info, params=None):
        super().__init__(game_type, game_info, params or {})
        self._num_players = game_info.num_players
        self._seats = active_seats(self._num_players)

    def new_initial_state(self):
        return AoFPushfoldState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        from aof_cfr.game_hu import _NullObserver

        return _NullObserver()


class AoFPushfoldState(pyspiel.State):
    def __init__(self, game: AoFPushfoldGame):
        super().__init__(game)
        n = game.num_players()
        self._seats = game._seats
        self._holes = np.full((n, 2), -1, dtype=np.int32)
        self._hole_buckets = [""] * n
        self._dead = np.zeros(52, dtype=np.bool_)
        self._deal_round = 0
        self._folded: set[int] = set()
        self._acted: set[int] = set()
        self._invested = np.zeros(n, dtype=np.float64)
        self._post_blinds()
        self._history = "-"
        self._terminal = False
        self._returns: list[float] | None = None
        self._actor: int | None = None
        if self._deal_round >= n:
            self._actor = self._next_actor()

    def _post_blinds(self) -> None:
        for i, seat in enumerate(self._seats):
            if seat == "SB":
                self._invested[i] = min(0.5, _STACK_BB)
            elif seat == "BB":
                self._invested[i] = min(1.0, _STACK_BB)

    def _next_actor(self) -> int | None:
        n = self.get_game().num_players()
        for i in range(n):
            if i in self._folded or i in self._acted:
                continue
            return i
        return None

    def _alive(self) -> list[int]:
        return [i for i in range(self.get_game().num_players()) if i not in self._folded]

    def current_player(self):
        if self._terminal:
            return pyspiel.PlayerId.TERMINAL
        if self._deal_round < self.get_game().num_players():
            return pyspiel.PlayerId.CHANCE
        if self._actor is None:
            return pyspiel.PlayerId.TERMINAL
        return self._actor

    def _legal_actions(self, player):
        return [_FOLD, _ALL_IN]

    def chance_outcomes(self):
        return _pair_chance_outcomes(self._dead)

    def clone(self):
        c = AoFPushfoldState(self.get_game())
        c._holes = self._holes.copy()
        c._hole_buckets = list(self._hole_buckets)
        c._dead = self._dead.copy()
        c._deal_round = self._deal_round
        c._folded = set(self._folded)
        c._acted = set(self._acted)
        c._invested = self._invested.copy()
        c._history = self._history
        c._terminal = self._terminal
        c._returns = list(self._returns) if self._returns is not None else None
        c._actor = self._actor
        return c

    def _finalize(self) -> None:
        alive = self._alive()
        if len(alive) == 1:
            rets = fold_terminal_returns(self._folded, alive, self._invested)
        else:
            hole_ids = [
                (int(self._holes[i, 0]), int(self._holes[i, 1])) for i in alive
            ]
            inv = np.array([self._invested[i] for i in alive], dtype=np.float64)
            shares = showdown_terminal_returns(hole_ids, inv)
            rets = [-float(x) for x in self._invested]
            for j, seat in enumerate(alive):
                rets[seat] = shares[j]
        self._returns = rets
        self._terminal = True
        self._actor = None

    def _apply_action(self, action):
        n = self.get_game().num_players()
        if self._deal_round < n:
            c0, c1 = _decode_pair(int(action))
            if self._dead[c0] or self._dead[c1]:
                raise ValueError(f"invalid deal action {action}")
            pi = self._deal_round
            self._holes[pi, 0] = c0
            self._holes[pi, 1] = c1
            self._dead[c0] = True
            self._dead[c1] = True
            self._hole_buckets[pi] = bucket_from_card_ids(c0, c1)
            self._deal_round += 1
            if self._deal_round >= n:
                self._actor = self._next_actor()
            return

        seat = self._actor
        assert seat is not None
        self._acted.add(seat)
        tok = "F" if action == _FOLD else "A"
        self._history = history_after_action(self._history, tok)

        if action == _FOLD:
            self._folded.add(seat)
        else:
            self._invested[seat] = _STACK_BB

        alive = self._alive()
        if len(alive) == 1:
            self._finalize()
            return

        self._actor = self._next_actor()
        if self._actor is None:
            self._finalize()

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal:{action}"
        return "FOLD" if action == _FOLD else "ALL_IN"

    def is_terminal(self):
        return self._terminal

    def returns(self):
        if not self._terminal or self._returns is None:
            return [0.0] * self.get_game().num_players()
        return self._returns

    def information_state_string(self, player):
        seat = self._seats[player]
        bucket = self._hole_buckets[player] if self._hole_buckets[player] else "??"
        return encode_infoset_key(seat, bucket, self._history)

    def __str__(self):
        return f"Pushfold(h={self._holes.tolist()},hist={self._history})"


def _register(num_players: int) -> None:
    gt = _game_type(num_players)
    gi = _game_info(num_players)

    class _Game(AoFPushfoldGame):
        def __init__(self, params=None):
            super().__init__(gt, gi, params)

    pyspiel.register_game(gt, _Game)


for _n in (3, 4):
    _register(_n)
