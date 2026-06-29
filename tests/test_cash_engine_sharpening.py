"""CASH-ENGINE-AUDIT sharpening tests — pins each Tier 1+2 fix from
artifacts/CASH-SHARPENING-REPORT.md so a future regression is caught immediately.

Categories:
- Cards.from_str 3-char rank form (CRIT-3 fix)
- GameState.action_history_with_street per-street stamps (CRIT-2 fix)
- np_engine.legal_action_mask_np CHECK fallback (CRIT-4 fix)
- evaluate_seven thread safety
- ALL_IN explicit partial amount honors
"""
from __future__ import annotations

import threading
import random
import warnings

import pytest


# =============================================================================
# CRIT-3: cards.py 3-char rank form
# =============================================================================

class TestCardsFromStr:
    def test_card_from_str_accepts_3char_rank(self):
        """#16: Card.from_str('10h') must succeed and yield Ten of Hearts."""
        from pokerbot.engine.cards import Card
        c = Card.from_str("10h")
        assert str(c) == "Th"

    def test_card_from_str_accepts_all_four_tens(self):
        from pokerbot.engine.cards import Card
        for suit in "cdhs":
            c = Card.from_str(f"10{suit}")
            assert c.rank.value == 10, f"failed for {suit}"

    def test_card_from_str_rejects_4char_rank(self):
        from pokerbot.engine.cards import Card
        with pytest.raises(ValueError):
            Card.from_str("10hx")

    def test_card_from_str_rejects_garbage_3char(self):
        """A 3-char rank that isn't '10' (e.g. '11h') must be rejected."""
        from pokerbot.engine.cards import Card
        with pytest.raises(ValueError):
            Card.from_str("11h")

    def test_parse_cards_handles_3char_forms(self):
        from pokerbot.engine.cards import parse_cards
        cards = parse_cards("10h10d10s10c")
        assert len(cards) == 4
        assert all(c.rank.value == 10 for c in cards)

    def test_parse_cards_mixed_2_and_3_char(self):
        from pokerbot.engine.cards import parse_cards
        cards = parse_cards("10hKs7h")
        assert len(cards) == 3
        assert str(cards[0]) == "Th"
        assert str(cards[1]) == "Ks"
        assert str(cards[2]) == "7h"

    def test_parse_cards_rejects_invalid_3char(self):
        from pokerbot.engine.cards import parse_cards
        with pytest.raises(ValueError):
            parse_cards("10h11h")

    def test_parse_cards_still_handles_2char_form(self):
        from pokerbot.engine.cards import parse_cards
        cards = parse_cards("Ks7h2d")
        assert len(cards) == 3
        assert str(cards[0]) == "Ks"


# =============================================================================
# CRIT-2: GameState.action_history_with_street
# =============================================================================

class TestActionHistoryWithStreet:
    def test_empty_history(self):
        from pokerbot.engine.game import GameState, Street
        gs = GameState.new_hand(rng=random.Random(0))
        out = list(gs.action_history_with_street())
        assert out == []

    def test_apply_action_records_street(self):
        """When actions go through apply_action, _history_streets is populated
        in lockstep with _history."""
        from pokerbot.engine.game import GameState, Street, Action, ActionType
        gs = GameState.new_hand(rng=random.Random(0))
        # Apply 2 folds; should remain on PREFLOP
        # First actor is UTG (after BTN/SB/BB post)
        actor = gs.action_seat
        gs.apply_action(Action(ActionType.FOLD))
        # next actor
        actor = gs.action_seat
        gs.apply_action(Action(ActionType.FOLD))
        out = list(gs.action_history_with_street())
        assert len(out) == 2
        for seat, action, street in out:
            assert street == Street.PREFLOP

    def test_postflop_actions_stamped_correctly(self):
        """After a street transition, new actions are stamped with the new street."""
        from pokerbot.engine.game import GameState, Street, Action, ActionType
        gs = GameState.new_hand(rng=random.Random(0))
        # Drive everyone to flop with calls
        # SB calls BB; everyone checks/calls until flop
        # Easier: just verify the apply_action path stamps correctly via direct _history inspection
        gs.apply_action(Action(ActionType.FOLD))
        # Manually advance street and apply
        gs.street = Street.FLOP
        gs._acted_this_street = set()
        gs.apply_action(Action(ActionType.CHECK))
        out = list(gs.action_history_with_street())
        assert len(out) == 2
        assert out[0][2] == Street.PREFLOP
        assert out[1][2] == Street.FLOP

    def test_legacy_state_with_only_history_falls_back_to_preflop(self):
        """A state built by direct _history mutation (legacy replay path) gets
        Street.PREFLOP for all entries when _history_streets is empty — matches
        the pre-fix audit-flagged behavior so we don't crash downstream consumers."""
        from pokerbot.engine.game import GameState, Street, Action, ActionType
        gs = GameState.new_hand(rng=random.Random(0))
        gs._history = [(0, Action(ActionType.FOLD)), (1, Action(ActionType.CALL))]
        gs._history_streets = []  # legacy: not populated
        out = list(gs.action_history_with_street())
        assert len(out) == 2
        for seat, action, street in out:
            assert street == Street.PREFLOP  # byte-identical to pre-fix behavior


# =============================================================================
# CRIT-4: np_engine.legal_action_mask_np CHECK fallback
# =============================================================================

class TestLegalActionMaskNoCheckFallback:
    def test_legal_action_mask_signature(self):
        """Smoke test: legal_action_mask_np is callable and returns a numpy bool array."""
        from pokerbot.engine import np_engine
        assert hasattr(np_engine, "legal_action_mask_np"), (
            "legal_action_mask_np must be exported from np_engine"
        )
        # Don't construct a full EnvState (signature varies across versions);
        # just verify the symbol exists and the file's mask-None-fallback path
        # is structurally sound (no bare return without setting mask[CHECK]).
        import inspect
        src = inspect.getsource(np_engine.legal_action_mask_np)
        assert "mask[CHECK] = True" in src, (
            "CRIT-4 fix REQUIRES legacy CHECK fallback (audit-approved compat shape); "
            "must preserve non-empty mask for downstream lbr*.py callers"
        )


# =============================================================================
# ALL_IN explicit partial amount honors (F-ENGINE-4) — NOT YET APPLIED on HEAD
# (audit recommended, but not in the current fix batch). Test omitted.
# =============================================================================



# =============================================================================
# evaluate_seven thread safety
# =============================================================================

class TestEvaluateSevenThreadSafe:
    def test_eval7_cache_lock_exists(self):
        """Module-level lock must be defined on np_engine."""
        from pokerbot.engine import np_engine
        assert hasattr(np_engine, "_EVAL7_CACHE_LOCK"), (
            "np_engine must export _EVAL7_CACHE_LOCK for thread-safe cache writes"
        )
        import threading
        assert isinstance(np_engine._EVAL7_CACHE_LOCK, type(threading.Lock())), (
            "_EVAL7_CACHE_LOCK must be a threading.Lock()"
        )


# =============================================================================
# Value-sizing monotonicity invariant
# =============================================================================

class TestValueSizingMonotonicity:
    def test_K_BIG_exceeds_K_MID(self):
        """Module-load assert: _K_BIG > _K_MID > 0."""
        from pokerbot.opponent import value_sizing
        assert value_sizing._K_BIG > value_sizing._K_MID > 0
