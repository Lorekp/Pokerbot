"""Public Belief State tracker with Bayesian hole-card range updates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.engine.cards import Card
from pokerbot.engine.game import NUM_PLAYERS
from pokerbot.encoding.weva_encoder import enumerate_hole_combos
from pokerbot.opponent.sequence_encoder import (
    NUM_BELIEF_COMBOS,
    cards_to_combo_index,
)

_COMBO_CARD_INDICES: list[tuple[int, int]] = [
    (c0.index, c1.index) for c0, c1 in enumerate_hole_combos()
]


def _numpy_removal_mask(hero_hole: tuple[Card, Card] | None, board: list[Card]) -> np.ndarray:
    dead = {c.index for c in board}
    if hero_hole:
        dead.update(c.index for c in hero_hole)
    mask = np.ones(NUM_BELIEF_COMBOS, dtype=bool)
    for i, (c0, c1) in enumerate(_COMBO_CARD_INDICES):
        if c0 in dead or c1 in dead:
            mask[i] = False
    return mask


@dataclass
class PBSTracker:
    """Maintains per-player hole-card belief distributions over 1326 combos."""

    hero_seat: int
    beliefs: np.ndarray = field(default_factory=lambda: np.zeros((NUM_PLAYERS, NUM_BELIEF_COMBOS)))
    dead_cards: set[int] = field(default_factory=set)
    _combo_mask: np.ndarray = field(default_factory=lambda: np.ones(NUM_BELIEF_COMBOS, dtype=bool))

    def __post_init__(self) -> None:
        if self.beliefs.shape != (NUM_PLAYERS, NUM_BELIEF_COMBOS):
            self.beliefs = np.ones((NUM_PLAYERS, NUM_BELIEF_COMBOS), dtype=np.float32) / NUM_BELIEF_COMBOS
        self._renormalize_all()

    @classmethod
    def init_uniform(cls, hero_seat: int, hero_hole: tuple[Card, Card] | None, board: list[Card]) -> PBSTracker:
        tracker = cls(hero_seat=hero_seat)
        tracker._apply_dead_cards(hero_hole, board)
        tracker._set_hero_delta(hero_hole)
        return tracker

    @classmethod
    def init_from_transformer(
        cls,
        belief_logits: np.ndarray,
        hero_seat: int,
        hero_hole: tuple[Card, Card] | None,
        board: list[Card],
    ) -> PBSTracker:
        """
        Initialize from MADT belief logits.

        belief_logits: (5, 1326) for non-hero opponents in fixed order.
        """
        tracker = cls.init_uniform(hero_seat, hero_hole, board)
        opps = tracker._opponent_seats()
        for oi, seat in enumerate(opps):
            if oi >= belief_logits.shape[0]:
                break
            logits = belief_logits[oi].astype(np.float64)
            logits = np.where(tracker._combo_mask, logits, -1e9)
            probs = np.exp(logits - logits.max())
            probs /= probs.sum() + 1e-12
            tracker.beliefs[seat] = probs.astype(np.float32)
        tracker._set_hero_delta(hero_hole)
        return tracker

    def _opponent_seats(self) -> list[int]:
        return [s for s in range(NUM_PLAYERS) if s != self.hero_seat]

    def _apply_dead_cards(self, hero_hole: tuple[Card, Card] | None, board: list[Card]) -> None:
        self.dead_cards = {c.index for c in board}
        if hero_hole:
            self.dead_cards.add(hero_hole[0].index)
            self.dead_cards.add(hero_hole[1].index)
        self._combo_mask = _numpy_removal_mask(hero_hole, board)
        for seat in range(NUM_PLAYERS):
            self.beliefs[seat] = np.where(self._combo_mask, self.beliefs[seat], 0.0)
            self._renormalize_seat(seat)

    def _set_hero_delta(self, hero_hole: tuple[Card, Card] | None) -> None:
        if hero_hole is None:
            return
        idx = cards_to_combo_index(hero_hole)
        self.beliefs[self.hero_seat] = 0.0
        self.beliefs[self.hero_seat, idx] = 1.0

    def _renormalize_seat(self, seat: int) -> None:
        row = self.beliefs[seat]
        if seat == self.hero_seat:
            return
        s = row.sum()
        if s > 0:
            self.beliefs[seat] = row / s
        else:
            legal = self._combo_mask.astype(np.float32)
            self.beliefs[seat] = legal / (legal.sum() + 1e-12)

    def _renormalize_all(self) -> None:
        for seat in range(NUM_PLAYERS):
            self._renormalize_seat(seat)

    def update_action(
        self,
        seat: int,
        action_idx: int,
        policy_likelihood: np.ndarray | None = None,
    ) -> None:
        """
        Bayes update: P(h|a) ∝ P(a|h) · P(h).

        policy_likelihood: optional (1326,) probability of action_idx per combo.
        """
        if seat == self.hero_seat:
            return
        if policy_likelihood is None:
            policy_likelihood = np.full(NUM_BELIEF_COMBOS, 1.0 / max(NUM_ACTIONS, 1), dtype=np.float32)
        else:
            policy_likelihood = np.clip(policy_likelihood.astype(np.float32), 1e-9, 1.0)

        prior = self.beliefs[seat]
        post = prior * policy_likelihood
        post = np.where(self._combo_mask, post, 0.0)
        self.beliefs[seat] = post
        self._renormalize_seat(seat)
        self._enforce_cross_seat_consistency()

    def _enforce_cross_seat_consistency(self) -> None:
        """Zero combos whose cards conflict with another seat's dominant combo (joint PBS v1)."""
        for seat in self._opponent_seats():
            row = self.beliefs[seat]
            if row.max() < 0.85:
                continue
            dom = int(row.argmax())
            c0, c1 = _COMBO_CARD_INDICES[dom]
            for other in self._opponent_seats():
                if other == seat:
                    continue
                for i, (a, b) in enumerate(_COMBO_CARD_INDICES):
                    if a in (c0, c1) or b in (c0, c1):
                        self.beliefs[other, i] = 0.0
                self._renormalize_seat(other)

    def update_board(self, new_card: Card) -> None:
        """Zero combos conflicting with newly revealed board card."""
        self.dead_cards.add(new_card.index)
        for i, (c0, c1) in enumerate(_COMBO_CARD_INDICES):
            if c0 == new_card.index or c1 == new_card.index:
                self._combo_mask[i] = False
        for seat in range(NUM_PLAYERS):
            self.beliefs[seat] = np.where(self._combo_mask, self.beliefs[seat], 0.0)
            if seat != self.hero_seat:
                self._renormalize_seat(seat)

    def get_public_belief_state(self) -> np.ndarray:
        """Flattened PBS vector for search root initialization."""
        opps = self._opponent_seats()
        rows = [self.beliefs[s] for s in opps]
        return np.concatenate(rows).astype(np.float32)

    def opponent_ranges(self) -> np.ndarray:
        """(5, 1326) opponent belief rows."""
        opps = self._opponent_seats()
        return np.stack([self.beliefs[s] for s in opps], axis=0)

    def sample_opponent_holes(
        self,
        rng: np.random.Generator,
        *,
        max_resample: int = 8,
    ) -> dict[int, tuple[int, int]]:
        """Sample one world line for SSMIG (card indices per opponent)."""
        for _attempt in range(max_resample):
            samples: dict[int, tuple[int, int]] = {}
            used = set(self.dead_cards)
            failed = False
            for seat in self._opponent_seats():
                probs = self.beliefs[seat].copy()
                for i, (c0, c1) in enumerate(_COMBO_CARD_INDICES):
                    if c0 in used or c1 in used:
                        probs[i] = 0.0
                total = probs.sum()
                if total < 1e-12:
                    failed = True
                    break
                probs /= total
                idx = int(rng.choice(NUM_BELIEF_COMBOS, p=probs))
                c0, c1 = _COMBO_CARD_INDICES[idx]
                samples[seat] = (c0, c1)
                used.add(c0)
                used.add(c1)
            if not failed:
                return samples
        # Fallback: uniform over legal combos per seat
        samples = {}
        used = set(self.dead_cards)
        for seat in self._opponent_seats():
            legal = [i for i, (c0, c1) in enumerate(_COMBO_CARD_INDICES) if c0 not in used and c1 not in used]
            if not legal:
                legal = [i for i in range(NUM_BELIEF_COMBOS) if self._combo_mask[i]]
            idx = int(rng.choice(legal))
            c0, c1 = _COMBO_CARD_INDICES[idx]
            samples[seat] = (c0, c1)
            used.update((c0, c1))
        return samples

    def action_likelihood_from_policy(
        self,
        action_idx: int,
        policy_probs: np.ndarray,
        *,
        transformer=None,
        trajectory=None,
        hero_seat: int | None = None,
        opponent_seat: int | None = None,
        device=None,
        board: list | None = None,
        top_k: int = 64,
        full_1326: bool = False,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Combo-stratified likelihood when transformer context is provided;
        otherwise uniform scalar proxy (legacy fallback).
        """
        if transformer is not None and trajectory is not None and hero_seat is not None and opponent_seat is not None:
            from pokerbot.search.belief_likelihood import combo_action_likelihood_from_transformer

            return combo_action_likelihood_from_transformer(
                transformer,
                trajectory,
                hero_seat,
                opponent_seat,
                action_idx,
                self._combo_mask,
                self.beliefs[opponent_seat],
                device,
                board=board,
                top_k=top_k,
                full_1326=full_1326,
                rng=rng,
            )
        p = float(np.clip(policy_probs[action_idx], 1e-6, 1.0))
        return np.full(NUM_BELIEF_COMBOS, p, dtype=np.float32)
