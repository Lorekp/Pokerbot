"""Tabular policies for Phase 1 WEVA vs rank/equity baseline gate."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import GameState
from pokerbot.encoding.weva_encoder import WEVAMapper, _preflop_strength


def _hand_strength(state: GameState, seat: int) -> float:
    player = state.players[seat]
    if player.hole is None:
        return 0.5
    r1, r2 = int(player.hole[0].rank), int(player.hole[1].rank)
    hi, lo = max(r1, r2), min(r1, r2)
    suited = int(player.hole[0].suit) == int(player.hole[1].suit)
    pair = r1 == r2
    if pair:
        base = 0.55 + (hi - 2) / 30.0
    else:
        base = (hi + lo) / 30.0 + (0.05 if suited else 0.0)
    if state.street.name != "PREFLOP" and state.board:
        board_ranks = [int(c.rank) for c in state.board]
        if pair and hi in board_ranks:
            base += 0.15
        if hi == max(board_ranks, default=0):
            base += 0.08
    return float(np.clip(base, 0.05, 0.95))


class RankEquityAbstractBot:
    """Rank/equity baseline: no WEVA buckets, strength heuristics only."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def __call__(self, state: GameState) -> Action:
        legal = state.legal_actions()
        seat = state.action_seat
        strength = _hand_strength(state, seat)
        if state.street.name == "PREFLOP":
            strength = _preflop_strength(state.players[seat].hole) if state.players[seat].hole else strength

        if strength < 0.35:
            return Action(ActionType.FOLD) if Action(ActionType.FOLD) in legal else legal[0]
        if strength > 0.72:
            bets = [a for a in legal if a.action_type in (ActionType.RAISE, ActionType.BET)]
            if bets:
                return bets[len(bets) // 2]
        if strength > 0.50:
            return next((a for a in legal if a.action_type in (ActionType.CALL, ActionType.CHECK)), legal[0])
        if Action(ActionType.FOLD) in legal and self.rng.random() < 0.6:
            return Action(ActionType.FOLD)
        return next((a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)), legal[0])


class WEVAAbstractBot:
    """WEVA abstraction policy: bucket strength from cached CFR/rollout features."""

    def __init__(self, mapper: WEVAMapper, rng: random.Random | None = None) -> None:
        self.mapper = mapper
        self.rng = rng or random.Random()
        self._bucket_strength = self._build_bucket_strength(mapper)

    @classmethod
    def from_weva_dir(cls, weva_dir: Path, rng: random.Random | None = None) -> WEVAAbstractBot:
        return cls(WEVAMapper.from_directory(weva_dir), rng=rng)

    def _build_bucket_strength(self, mapper: WEVAMapper) -> np.ndarray:
        k = mapper.index.k
        strengths = np.full(k, 0.5, dtype=np.float64)
        if mapper.features is None:
            return strengths
        feats = mapper.features.astype(np.float64)
        # First depth-weighted EV dimension as bucket proxy (normalized per build).
        raw = feats[:, 0] if feats.ndim == 2 else feats
        lo, hi = float(raw.min()), float(raw.max())
        norm = (raw - lo) / max(hi - lo, 1e-6)
        for i in range(len(norm)):
            bid = int(mapper.index.assign(feats[i : i + 1])[0])
            strengths[bid] = max(strengths[bid], norm[i])
        for bid in range(k):
            if strengths[bid] == 0.5:
                mask = mapper.index.assign(mapper.index.centroids) == bid
                if mask.any():
                    strengths[bid] = float(norm[mask].mean()) if len(norm) == len(mask) else 0.5
        return np.clip(strengths, 0.05, 0.95)

    def _bucket_strength_for_state(self, state: GameState, seat: int) -> float:
        hero = state.players[seat]
        if hero.hole is None:
            return 0.5
        bid = self.mapper.lookup(hero.hole, position=int(hero.position))
        base = float(self._bucket_strength[bid])
        if state.street.name != "PREFLOP":
            return 0.65 * base + 0.35 * _hand_strength(state, seat)
        return base

    def __call__(self, state: GameState) -> Action:
        legal = state.legal_actions()
        seat = state.action_seat
        strength = self._bucket_strength_for_state(state, seat)

        if strength < 0.38:
            return Action(ActionType.FOLD) if Action(ActionType.FOLD) in legal else legal[0]
        if strength > 0.70:
            bets = [a for a in legal if a.action_type in (ActionType.RAISE, ActionType.BET)]
            if bets:
                return bets[min(len(bets) - 1, len(bets) // 2 + 1)]
        if strength > 0.52:
            return next((a for a in legal if a.action_type in (ActionType.CALL, ActionType.CHECK)), legal[0])
        if Action(ActionType.FOLD) in legal and self.rng.random() < 0.55:
            return Action(ActionType.FOLD)
        return next((a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)), legal[0])
