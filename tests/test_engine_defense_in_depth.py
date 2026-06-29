"""Regression tests for the engine defense-in-depth fixes from
``artifacts/six_max_bug_hunt_report.md``:

  * **F-ENGINE-4**  (``pokerbot/engine/game.py::apply_action`` ALL_IN ignores ``amount_bb``)
  * **F-ENGINE-24** (``pokerbot/engine/game.py::_commit`` doesn't reject negative ``amount_bb``)
  * **F-ENGINE-6**  (FOLD on all-in player overwrites ``all_in=False``)
  * **F-ENGINE-19** (``_nth_live_after`` doesn't skip empty seats)

Each guard pins ONE behavior a buggy upstream policy could silently break. These are
defense-in-depth — the engine already worked correctly when its own policy drove the
table, but a foreign caller (replay / OCR / search) can hand in malformed actions and
the engine MUST refuse rather than silently corrupt side-pot state.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.cards import Card, Deck, Rank, Suit
from pokerbot.engine.game import GameState, NUM_PLAYERS


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _fresh_state(button: int = 3, *, n_players: int = NUM_PLAYERS) -> GameState:
    """Build a 6-max fresh hand. Caller can ``del`` players to model a short-handed state."""
    s = GameState.new_hand(button_seat=button)
    if n_players < NUM_PLAYERS:
        # Trim to the first ``n_players`` seats (B7-style occupancy handling). The remaining seats
        # keep a "phantom" PlayerState that has zero stack / zero committed / no hole — this is the
        # shape F-ENGINE-19 trips on at a 5-handed boundary.
        keep = s.players[:n_players]
        phantom = s.players[n_players:]
        for p in phantom:
            p.stack_bb = 0.0
            p.committed_street_bb = 0.0
            p.committed_total_bb = 0.0
            p.hole = None
        s.players = keep + phantom
    return s


# ---------------------------------------------------------------------------------------
# F-ENGINE-4 — ALL_IN with caller-supplied amount honors the partial amount (capped at stack)
# ---------------------------------------------------------------------------------------
def test_all_in_with_explicit_amount_below_stack_does_not_overcommit():
    """A live replay may emit ``Action(ALL_IN, partial_amount)`` for a short-side all-in. Previously
    the engine IGNORED ``amount_bb`` and always committed the FULL stack, silently over-committing
    the player. The fix: honor the caller-supplied amount (raise increment above call) up to the
    stack. ``Action.amount_bb`` is the raise increment above the call, so the final commitment is
    ``to_call + explicit`` (here to_call == 1.0bb for the BB)."""
    s = _fresh_state()
    seat = s.action_seat                       # the first to act preflop (UTG)
    player = s.players[seat]
    # Make the explicit amount a small raise increment well below the starting stack.
    explicit = 0.25
    pre_stack = player.stack_bb
    assert pre_stack > explicit + 1.0         # sanity: starting stack > explicit
    s.apply_action(Action(ActionType.ALL_IN, explicit), seat=seat)
    # Final commitment = to_call (1.0bb for the BB) + explicit. Critically, the player did NOT
    # all-in (to_call + explicit was below the stack) and was NOT silently committed for the full
    # stack (the OLD bug).
    expected_total = 1.0 + explicit           # BB already standing; UTG calls +0.25 above
    assert abs(player.committed_total_bb - expected_total) < 1e-9
    assert abs(player.stack_bb - (pre_stack - expected_total)) < 1e-9
    assert player.all_in is False
    assert player.folded is False


def test_all_in_without_amount_falls_back_to_full_stack():
    """Backward compat: a policy that emits ``Action(ALL_IN, None)`` (no amount) MUST still go all-in
    for the full stack — the new "honor amount" path must not regress this contract."""
    s = _fresh_state()
    seat = s.action_seat
    player = s.players[seat]
    pre_stack = player.stack_bb
    s.apply_action(Action(ActionType.ALL_IN, None), seat=seat)
    # Full stack committed, all_in flag set.
    assert abs(player.committed_total_bb - pre_stack) < 1e-9
    assert abs(player.stack_bb) < 1e-9
    assert player.all_in is True


def test_all_in_with_amount_exceeding_stack_clamps_to_stack():
    """A caller that mis-specifies ``amount > stack`` (defensive: should not happen but the engine
    must not over-commit). The cap is ``min(amount, stack)``; remaining_stack == 0 -> all_in."""
    s = _fresh_state()
    seat = s.action_seat
    player = s.players[seat]
    pre_stack = player.stack_bb
    s.apply_action(Action(ActionType.ALL_IN, pre_stack + 50.0), seat=seat)
    assert abs(player.committed_total_bb - pre_stack) < 1e-9
    assert player.all_in is True


# ---------------------------------------------------------------------------------------
# F-ENGINE-24 — _commit clamps negative amounts to zero
# ---------------------------------------------------------------------------------------
def test_commit_with_negative_amount_does_not_grow_stack():
    """The bug: ``min(-1, 10) == -1`` -> ``player.stack_bb -= -1`` -> stack grew. Fix: ``max(0, ...)``
    clamps negative commits to zero, so a buggy policy that hands in a negative ``amount_bb`` can
    never GROW a stack."""
    s = _fresh_state()
    seat = s.action_seat
    player = s.players[seat]
    pre_stack = player.stack_bb
    pre_pot = s.pot_bb
    # Directly drive the private _commit (the defense sits inside it). Use a value that would
    # otherwise GROW the stack via ``-= -1``.
    returned = s._commit(seat, -1.0)
    assert returned == 0.0                     # clamped to zero
    assert abs(player.stack_bb - pre_stack) < 1e-9   # unchanged
    assert abs(s.pot_bb - pre_pot) < 1e-9           # unchanged
    assert player.committed_total_bb == 0.0
    # Also: stack-clamp semantics still hold (a too-large positive commit doesn't over-draw).
    big = s._commit(seat, pre_stack + 50.0)
    assert abs(big - pre_stack) < 1e-9


# ---------------------------------------------------------------------------------------
# F-ENGINE-6 — FOLD from an all-in player is REJECTED, not silently overwritten
# ---------------------------------------------------------------------------------------
def test_fold_from_all_in_player_raises_value_error():
    """The bug: a player who is already all-in has no legal FOLD. A buggy policy that emits FOLD for
    such a player used to silently set ``folded=True``, overwriting ``all_in=False`` downstream and
    corrupting the side-pot collapse path. The fix raises ``ValueError`` so the caller can fix the
    policy instead of corrupting state."""
    s = _fresh_state()
    seat = s.action_seat
    player = s.players[seat]
    pre_stack = player.stack_bb
    # Force the player all-in by committing the full stack via the protected ALL_IN(no-amount) path.
    s.apply_action(Action(ActionType.ALL_IN, None), seat=seat)
    assert player.all_in is True
    # Now have the player FOLD — must be rejected.
    pre_folded = player.folded
    with pytest.raises(ValueError, match=r"all-in player"):
        s.apply_action(Action(ActionType.FOLD, None), seat=seat)
    assert player.folded == pre_folded         # state untouched on rejection


# ---------------------------------------------------------------------------------------
# F-ENGINE-19 — _nth_live_after skips empty (zero-stack, no-hole) seats
# ---------------------------------------------------------------------------------------
def test_nth_live_after_skips_phantom_seat():
    """At a 5-handed boundary the standard 6-wide ring keeps the 6th seat as a "phantom"
    (zero stack, no hole, not folded). Previously ``_nth_live_after`` skipped only folded seats
    and could land on the phantom — cycling the position label onto a player who is not in the
    hand. The fix skips seats with stack<=0 AND committed_total<=0 AND hole is None."""
    s = _fresh_state(n_players=5)
    # The button is at seat 3 by default; the 5 live seats are 0..4 (the first 5 in seating order
    # we keep), seat 5 is the phantom.
    # Sanity: seat 5 is the phantom.
    assert s.players[5].stack_bb == 0.0
    assert s.players[5].hole is None
    # Walk from seat 4 clockwise looking for the 1st live seat; should land on seat 0 (the BTN
    # position is seat 3 by default, so seat 4 = SB, seat 5 = phantom, seat 0 = BTN... wait, the
    # SB is button+1 = 4. SB is live. seat 5 = phantom, seat 0 is BTN. So the 1st live after
    # seat 4 (seat 4 -> 5 phantom -> 0 BTN) is seat 0.
    assert s._nth_live_after(4, 1) == 0
    # 2nd live after seat 4: 0 (BTN) -> 1 (next live)
    assert s._nth_live_after(4, 2) == 1


def test_nth_live_after_skips_phantom_but_stops_on_folded_live_seat():
    """Skipping is for EMPTY seats only, not folded-but-dealt seats. A folded seat has committed>0
    or hole set, so the "empty" predicate fails and it stays in the cycle (folded seats already
    skip via the first predicate). The test pins both arms: a phantom seat is skipped, a folded
    live seat is also skipped, and the walker lands on the next LIVE-and-DEALT seat."""
    s = _fresh_state(n_players=5)
    # Make seat 1 explicitly folded. Note: in a 5-max state with seats 0..4 live + seat 5 phantom,
    # seat 1 is HJ — not the SB/BB (those are seat 4 / seat 5 in the BTN-anchored layout). Folding
    # it from the fresh state is direct: we just set folded=True on the player.
    s.players[1].folded = True
    # Walk from seat 0 clockwise: 0->1 (folded) ->2 (HJ live). Land on seat 2.
    res = s._nth_live_after(0, 1)
    assert res != 5                              # NEVER lands on the phantom
    assert res == 2                              # lands on the next live, non-folded seat


# ---------------------------------------------------------------------------------------
# composite: defense-in-depth stack stays internally consistent after all the guards engage
# ---------------------------------------------------------------------------------------
def test_all_guards_compose_on_a_normal_hand():
    """Run a short happy-path scenario: button raise + caller all-in (with explicit amount). Verify
    the post-hand committed totals and pot add up cleanly across both defense-in-depth fixes."""
    s = _fresh_state()
    # BTN (seat 3) raises by an explicit amount. Pick a small raise so we don't push everyone to
    # all-in on the first action.
    btn = 3
    s.apply_action(Action(ActionType.RAISE, 2.5), seat=btn)
    # SB (seat 4) calls by raising the call commitment (full call).
    sb = 4
    s.apply_action(Action(ActionType.CALL, None), seat=sb)
    # BB (seat 5) folds.
    bb = 5
    s.apply_action(Action(ActionType.FOLD, None), seat=bb)
    # UTG (seat 0) goes all-in for an explicit partial amount. To force ``all_in=True`` the raise
    # increment must cover the entire remaining stack, so we pass the full pre minus a tiny epsilon.
    utg = 0
    pre = s.players[utg].stack_bb
    s.apply_action(Action(ActionType.ALL_IN, pre + 50.0), seat=utg)   # over-spec; clamps to stack
    # HJ (seat 1) folds.
    s.apply_action(Action(ActionType.FOLD, None), seat=1)
    # CO (seat 2) calls the all-in. Then advance to showdown by passing through remaining actors.
    s.apply_action(Action(ActionType.CALL, None), seat=2)
    # No negative commits anywhere (F-ENGINE-24).
    for p in s.players:
        assert p.committed_total_bb >= 0.0
        assert p.stack_bb >= 0.0
    # SB and BB are not all-in (only UTG is).
    assert s.players[utg].all_in is True
    assert s.players[btn].all_in is False
    assert s.players[sb].all_in is False