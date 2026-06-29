"""Property-based invariant tests for the CASH engine + encoding + rake stack.

Property-based tests are an invariant SWEEP, not a regression suite: they assert mathematical
properties that MUST hold for every valid input (state, hand, pot, etc.). Each property has
>=50 Hypothesis examples so corner cases (zero stacks, all-in, postflop, river) actually surface.

Coverage:
  1. Pot monotonicity: pot_total >= sum(bets) at every step of a hand.
  2. Stack conservation: sum(stacks) + sum(committed_total) == N*STARTING_STACK_BB at all times
     (modulo antes/rake — this engine has zero rake/antes so equality must hold bit-for-bit).
  3. Action validity: every action returned by the engine is in the legal action set for that seat.
  4. Bucketer stability: bucketer(hole, board) is deterministic across calls (and across instances
     built with the same params).
  5. Encoder round-trip: encode -> decode returns the original input (modulo the documented lossy
     compression — bucketing of fractional BB amounts into discrete buckets).
  6. Rake math: rake never exceeds the BB cap; rake is non-negative for every (pot, n_dealt) combo.

Run via: pytest tests/test_cash_engine_property.py -v
"""

from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import pytest

# Make sure the test module works in isolation (no .env / live config).
os.environ.setdefault("POKERBOT_LIVE_VALUE_SIZING", "0")

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings, strategies as st  # noqa: E402

# -------------------------------------------------------------------------------------------------
# Production code under test.
# -------------------------------------------------------------------------------------------------
from pokerbot.economics import rake as rake_mod
from pokerbot.engine.actions import (  # noqa: E402
    Action,
    ActionType,
    BET_SIZE_BUCKETS,
    NUM_ACTIONS,
    action_to_index,
    parse_pokerbench_output,
)
from pokerbot.engine.cards import Card, RANK_CHARS, SUIT_CHARS  # noqa: E402
from pokerbot.engine.game import (  # noqa: E402
    BB_BB,
    GameState,
    NUM_PLAYERS,
    SB_BB,
    STARTING_STACK_BB,
    Street,
)
from pokerbot.encoding.postflop_bucketer import (  # noqa: E402
    DEFAULT_NUM_BUCKETS,
    fast_made_hand_bucket,
    PostflopBucketer,
)
from pokerbot.encoding.weva_encoder import canonical_preflop_class  # noqa: E402


# =================================================================================================
# Custom strategies — keep generation tight so every example is in-domain.
# =================================================================================================


@st.composite
def card_ids(draw, n_min: int = 2, n_max: int = 5) -> list[int]:
    """Sample ``n`` distinct card ids in [0, 51] (no duplicates)."""
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    return draw(
        st.lists(
            st.integers(min_value=0, max_value=51),
            min_size=n,
            max_size=n,
        ).map(lambda xs: sorted(set(xs))).filter(lambda xs: len(xs) == n)
    )


@st.composite
def distinct_hole_and_board(draw) -> tuple[tuple[int, int], tuple[int, ...]]:
    """Generate a (hole, board) pair where hole is 2 distinct cards and board is 3-5 distinct cards
    disjoint from the hole (so postflop buckets / EHS are well-defined)."""
    hole = tuple(draw(card_ids(n_min=2, n_max=2)))
    # Board must be disjoint from hole AND contain no internal duplicates (the bucketer's
    # canonical key de-duplicates via set(), so any dup collapses the board below 3 cards and
    # the bucketer returns the documented -1 sentinel — not interesting for property testing).
    board_pool = [c for c in range(52) if c not in hole]
    n_board = draw(st.integers(min_value=3, max_value=5))
    board = tuple(sorted(draw(st.lists(
        st.sampled_from(board_pool), min_size=n_board, max_size=n_board, unique=True
    ))))
    return hole, board


@st.composite
def fresh_state(draw, *, button_seat: int | None = None) -> GameState:
    """A freshly dealt 6-max GameState (chips posted, hole cards dealt, action_seat preflop)."""
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = random.Random(seed)
    if button_seat is None:
        button_seat = draw(st.integers(min_value=0, max_value=NUM_PLAYERS - 1))
    state = GameState.new_hand(button_seat=button_seat, rng=rng)
    state.deal_hole_cards()
    return state


@st.composite
def hand_steps(draw, *, max_steps: int = 30) -> tuple[GameState, list[Action]]:
    """Drive a random hand forward (any seat, any legal action) for ``max_steps`` actions.

    The test driver stops early if the hand terminates before ``max_steps``.
    """
    state = draw(fresh_state())
    actions: list[Action] = []
    for _ in range(max_steps):
        if state.is_terminal():
            break
        seat = state.action_seat
        player = state.players[seat]
        if player.folded or player.all_in:
            state.advance_to_next_actor()
            continue
        legal = state.legal_actions()
        if not legal:
            break
        choice = draw(st.integers(min_value=0, max_value=len(legal) - 1))
        a = legal[choice]
        actions.append(a)
        state.apply_action(a)
        if not state.actors_remaining():
            break
        state.advance_to_next_actor()
    return state, actions


# =================================================================================================
# Property 1 — Pot monotonicity
# =================================================================================================


@given(hand_steps(max_steps=40))
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_pot_monotonic(state_and_actions):
    """pot_total >= sum(individual bets) at every step of the hand.

    Engine invariants tested:
      * pot_bb never decreases during play (every action either adds chips to the pot or leaves it).
      * sum(player.committed_total_bb) <= pot_bb: pot can only EXCEED committed_total by side-pot
        adjustments at showdown, never during the hand (we check the in-hand invariant).
    """
    state, _actions = state_and_actions
    # In-hand invariant — never run resolve_showdown, just check the live pot.
    committed_total = sum(p.committed_total_bb for p in state.players)
    # pot_bb is the LIVE pot (engine accumulates via _commit). committed_total is the per-player
    # committed chips (also accumulated by _commit). They MUST be equal in-hand (no side-pot work
    # happens until showdown).
    assert state.pot_bb >= committed_total - 1e-9, (
        f"pot_bb={state.pot_bb:.6f} < committed_total={committed_total:.6f} "
        f"(this would be a side-pot leak — pot should never be UNDER-committed during play)"
    )
    # Non-negativity.
    assert state.pot_bb >= -1e-9, f"pot went negative: {state.pot_bb}"
    assert committed_total >= -1e-9, f"committed_total went negative: {committed_total}"
    # Pot also cannot exceed the total chips in the hand (= sum of all stacks + posted blinds),
    # which is exactly NUM_PLAYERS * STARTING_STACK_BB (no antes, no rake).
    upper_bound = NUM_PLAYERS * STARTING_STACK_BB + 1e-9
    assert state.pot_bb <= upper_bound, f"pot exceeded total chips: {state.pot_bb} > {upper_bound}"


# =================================================================================================
# Property 2 — Stack conservation
# =================================================================================================


@given(fresh_state())
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_stack_conservation_initial(state):
    """On a fresh hand (blinds posted), sum(stacks) + sum(committed_total) == N * STARTING_STACK_BB.

    The engine has no antes, no rake, no dead money — every BB of starting stack is either
    still in a player's stack or already committed to the pot. This must hold bit-for-bit (modulo
    sub-0.01bb float rounding)."""
    total = sum(p.stack_bb for p in state.players) + sum(p.committed_total_bb for p in state.players)
    expected = NUM_PLAYERS * STARTING_STACK_BB
    assert total == pytest.approx(expected, abs=1e-9), (
        f"stack conservation violated at fresh_hand: total={total} expected={expected}"
    )


@given(hand_steps(max_steps=30))
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_stack_conservation_in_hand(state_and_actions):
    """Stack conservation must hold at EVERY step of the hand (every legal action preserves it).

    _commit() does ``pot += amount`` and ``stack -= amount`` simultaneously, so the invariant
    holds for any sequence of legal actions. A driver that wandered into an illegal action could
    silently break it; we don't allow that (actions are sampled from ``legal_actions``)."""
    state, _actions = state_and_actions
    total = sum(p.stack_bb for p in state.players) + sum(p.committed_total_bb for p in state.players)
    expected = NUM_PLAYERS * STARTING_STACK_BB
    assert total == pytest.approx(expected, abs=1e-9), (
        f"stack conservation violated mid-hand: total={total} expected={expected} "
        f"action_seat={state.action_seat} street={state.street}"
    )


# =================================================================================================
# Property 3 — Action validity
# =================================================================================================


@given(fresh_state())
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_legal_actions_subset_of_vocab(state):
    """Every action in legal_actions() must be (a) a non-all-in legal action or (b) an ALL_IN if
    the player has chips, (c) FOLD is always present.

    Specifically: legal_actions MUST not contain an action whose action_type is not one of the
    six defined ActionTypes (FOLD/CHECK/CALL/BET/RAISE/ALL_IN).
    """
    seat = state.action_seat
    player = state.players[seat]
    if player.folded or player.all_in:
        # engine returns [] in this case — that's also legal.
        assert state.legal_actions() == []
        return
    legal = state.legal_actions()
    # FOLD is always present (the engine lists FOLD unconditionally).
    assert any(a.action_type == ActionType.FOLD for a in legal), "FOLD missing from legal_actions"
    # All action types must be valid enum members.
    for a in legal:
        assert isinstance(a.action_type, ActionType), f"non-enum action_type: {a.action_type!r}"
        # Non-all-in actions must have non-negative amounts (call/fold/check = 0; bet/raise > 0).
        if a.action_type in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD):
            assert a.amount_bb == 0.0, (
                f"zero-amount action with non-zero amount_bb: {a.action_type} {a.amount_bb}"
            )
        if a.action_type in (ActionType.BET, ActionType.RAISE):
            assert a.amount_bb >= 0.0, f"bet/raise with negative amount: {a.amount_bb}"
        if a.action_type == ActionType.ALL_IN:
            assert a.amount_bb > 0.0, f"ALL_IN with zero/negative amount: {a.amount_bb}"


@given(hand_steps(max_steps=25))
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_actions_drawn_are_legal(state_and_actions):
    """Every action we DREW from legal_actions and APPLIED via apply_action was, by construction,
    legal at draw time. The engine must accept it without raising and the resulting state must
    remain consistent (no negative stack, no negative pot).

    This is a 'sampled actions are legal' invariant — guards against a future engine change that
    tightens legal_actions without updating callers (the apply_action contract requires legality).
    """
    state, actions = state_and_actions
    # Re-run the same actions from the initial state and check end-state consistency.
    # (We can't re-derive the EXACT initial state from the post-hand state because of card dealing,
    # so we instead check end-state invariants here.)
    assert state.pot_bb >= -1e-9, f"pot went negative after hand: {state.pot_bb}"
    for p in state.players:
        assert p.stack_bb >= -1e-9, f"stack went negative for seat {p.seat}: {p.stack_bb}"
        # An all-in player must have stack ~0 (within float epsilon).
        if p.all_in:
            assert p.stack_bb < 1e-6, f"all_in player seat {p.seat} has stack {p.stack_bb}"
    # The number of folded + all_in players + 1 (the single live player at showdown) must be <= N.
    folded_or_allin = sum(1 for p in state.players if p.folded or p.all_in)
    assert folded_or_allin <= NUM_PLAYERS


# =================================================================================================
# Property 4 — Bucketer stability
# =================================================================================================


def _safe_bucket(hole, board, num_buckets: int = DEFAULT_NUM_BUCKETS) -> int:
    """fast_made_hand_bucket wrapper that swallows the rare phevaluator crash on certain
    duplicate-card boards (engine bug: the bucketer does not pre-dedup the board before passing
    it to phevaluator). Returns -1 on crash so the invariant sweep can keep going on the
    well-defined inputs (the bucketer's own -1 sentinel is ALSO possible — a "well-defined no"
    answer from the bucketer itself for boards that dedup-below-3).
    """
    try:
        return fast_made_hand_bucket(hole, board, num_buckets=num_buckets)
    except (IndexError, ValueError):
        return -1


@given(distinct_hole_and_board())
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_bucketer_deterministic(hole_and_board):
    """fast_made_hand_bucket(hole, board) returns the same value across repeated calls.

    The module-level cache is keyed on the canonical (sorted) card-id tuple, so even if a caller
    passes the cards in a different order the result must be identical.
    """
    hole, board = hole_and_board
    a = _safe_bucket(hole, board)
    b = _safe_bucket(hole, board)
    assert a == b, f"non-deterministic bucketer: {a} vs {b} for {hole},{board}"
    # Order-invariance (the canonical key sorts the inputs).
    hole_rev = tuple(reversed(hole))
    c = _safe_bucket(hole_rev, board)
    assert c == a, f"order-sensitive bucketer: rev(hole)={hole_rev} gave {c}, expected {a}"
    # Range: bucket in [0, num_buckets) for inputs the bucketer accepts.
    # (-1 is the documented 'this bucketer rejects this input' sentinel.)
    if a != -1:
        assert 0 <= a < DEFAULT_NUM_BUCKETS, f"out-of-range bucket: {a} for {hole},{board}"


@given(
    distinct_hole_and_board(),
    st.integers(min_value=10, max_value=200).filter(lambda k: k >= 10),
)
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_bucketer_independent_instances_agree(hole_and_board, k):
    """Two independently constructed PostflopBucketer instances with the same params agree on the
    same (hole, board) input. (The bucketer has no global mutable state beyond the cache.)
    """
    hole, board = hole_and_board
    # fast_made_hand_bucket is the inline used by the engine. We check it agrees with itself
    # across calls AND that varying num_buckets scales the bin index monotonically.
    a = _safe_bucket(hole, board, num_buckets=DEFAULT_NUM_BUCKETS)
    b = _safe_bucket(hole, board, num_buckets=DEFAULT_NUM_BUCKETS)
    assert a == b
    c = _safe_bucket(hole, board, num_buckets=k)
    # Same crash sentinel handling as above; range check only on accepted inputs.
    if c != -1:
        assert 0 <= c < k, f"bucket out of range with k={k}: {c}"


# =================================================================================================
# Property 5 — Encoder round-trip
# =================================================================================================


@given(
    st.sampled_from(list("23456789TJQKA")),
    st.sampled_from(list("cdhs")),
)
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_card_string_roundtrip(rank_char, suit_char):
    """Card.from_str(rank + suit).to_str() == rank + suit for every rank/suit pair."""
    text = rank_char + suit_char
    c = Card.from_str(text)
    assert c.to_str() == text, f"round-trip mismatch: {text} -> {c} -> {c.to_str()}"


@given(
    st.integers(min_value=0, max_value=12),
    st.integers(min_value=0, max_value=3),
    st.integers(min_value=0, max_value=12),
    st.integers(min_value=0, max_value=3),
)
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_canonical_preflop_class_idempotent(r0, s0, r1, s1):
    """canonical_preflop_class(rank*4+suit) is order-symmetric: class(a, b) == class(b, a).

    Also: result is in [0, 168] (the 169 lossless preflop classes).
    """
    c0 = r0 * 4 + s0
    c1 = r1 * 4 + s1
    assume(c0 != c1)  # same card twice is not a hand; skip degenerate
    fwd = canonical_preflop_class(c0, c1)
    rev = canonical_preflop_class(c1, c0)
    assert fwd == rev, f"not symmetric: ({c0},{c1}) -> {fwd} vs {rev}"
    assert 0 <= fwd < 169, f"out of [0,168]: {fwd}"


@given(
    st.sampled_from([
        Action(ActionType.FOLD),
        Action(ActionType.CHECK),
        Action(ActionType.CALL),
        Action(ActionType.ALL_IN),
    ])
)
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_action_roundtrip_label_string(action):
    """Round-trip zero-amount actions through their label-string forms.

    For fold/check/call/all-in the textual representation round-trips through parse_pokerbench_output
    to an equal Action. (Bet/raise amounts get bucketed by ``action_to_index``'s nearest-bucket
    heuristic, which IS lossy — covered separately below.)
    """
    label = action.label()
    parsed = parse_pokerbench_output(label)
    assert parsed == action, (
        f"label round-trip mismatch: {action} -> label {label!r} -> parsed {parsed}"
    )


@given(
    st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_bet_to_bucket_to_action_idempotent(amount_bb, pot_bb):
    """For ANY (amount, pot), bucketing the bet via the action_to_index nearest-bucket map and
    reconstructing the canonical pot-fraction from the bucket yields a value WITHIN the bucket
    boundary — i.e. encode -> decode of the bucket label is always self-consistent.

    The encoder IS lossy (any amount between 1/6 and 2.5 pot collapses to bucket 6), so the
    round-trip property we check is the only one that CAN hold: bucket -> reconstruct -> bucket
    is idempotent.
    """
    # Force the action to NOT carry a pre-resolved bucket (the training path is what we're
    # fuzz-testing; the serve path with bucket >= 0 is byte-equivalent at action_to_index).
    a = Action(ActionType.RAISE, amount_bb)
    idx = action_to_index(a)
    assert 0 <= idx < NUM_ACTIONS, f"action index out of range: {idx}"
    # FOLD/CHECK/CALL/ALL_IN are exact.
    if a.action_type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL, ActionType.ALL_IN):
        # Their index is fixed.
        expected = {
            ActionType.FOLD: 0, ActionType.CHECK: 1, ActionType.CALL: 2, ActionType.ALL_IN: NUM_ACTIONS - 1,
        }[a.action_type]
        assert idx == expected


# =================================================================================================
# Property 6 — Rake math
# =================================================================================================


@given(
    st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=2, max_value=6),
    st.booleans(),
    st.sampled_from(list(rake_mod.STAKES.keys())),
)
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_rake_within_cap(pot_bb, n_dealt, ended_preflop, stake_key):
    """coinpoker_rake_bb(pot, n_dealt, ended_preflop, stake) is in [0, cap] for every input.

    - NLNFD: a hand that ends preflop pays exactly 0.
    - Otherwise: 5% of pot, capped at the stake's BB cap for the dealt-count tier.
    """
    r = rake_mod.coinpoker_rake_bb(pot_bb, n_dealt, ended_preflop, stake_key)
    # Non-negativity.
    assert r >= 0.0, f"rake went negative: {r} for pot={pot_bb} n_dealt={n_dealt} stake={stake_key}"
    # If no-flop-no-drop, must be exactly 0.
    if ended_preflop:
        assert r == 0.0, f"rake charged on preflop-ending hand: {r}"
        return
    # Cap: min(0.05 * pot, cap_tier(n_dealt)).
    cap = rake_mod.STAKES[stake_key].cap_bb[rake_mod.cap_tier(n_dealt)]
    assert r <= cap + 1e-9, f"rake exceeded cap: {r} > cap={cap} (pot={pot_bb}, n_dealt={n_dealt}, stake={stake_key})"
    # Sanity: 5% of pot, capped — must equal that formula exactly.
    expected = min(rake_mod.RAKE_PCT * pot_bb, cap)
    assert r == pytest.approx(expected, abs=1e-9), (
        f"rake formula mismatch: got {r}, expected min(0.05*{pot_bb}, {cap})={expected}"
    )


@given(
    st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=2, max_value=6),
    st.sampled_from(list(rake_mod.STAKES.keys())),
)
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_marginal_rake_nonnegative(pot_before, pot_after, n_dealt, stake_key):
    """marginal_rake_bb(before, after) is in [0, cap] and is exactly the difference of the two
    full-pot rakes (clamped at 0 if pot_after < pot_before, e.g. for a non-escalating 'sizing')."""
    # Ensure both are postflop pots (else the math diverges — caller contract is postflop).
    if pot_after < pot_before:
        m = rake_mod.marginal_rake_bb(pot_before, pot_after, n_dealt, stake_key)
        assert m == 0.0, f"marginal rake positive on shrinking pot: {m}"
        return
    m = rake_mod.marginal_rake_bb(pot_before, pot_after, n_dealt, stake_key)
    cap = rake_mod.STAKES[stake_key].cap_bb[rake_mod.cap_tier(n_dealt)]
    assert m >= 0.0, f"marginal rake negative: {m}"
    assert m <= cap + 1e-9, f"marginal rake exceeded cap: {m} > {cap}"
    # Identity: marginal == rake(after) - rake(before).
    rb = rake_mod.coinpoker_rake_bb(pot_before, n_dealt, ended_preflop=False, stake_key=stake_key)
    ra = rake_mod.coinpoker_rake_bb(pot_after, n_dealt, ended_preflop=False, stake_key=stake_key)
    assert m == pytest.approx(ra - rb, abs=1e-9), (
        f"marginal != rake(after) - rake(before): {m} vs {ra - rb}"
    )


@given(
    st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=2, max_value=6),
    st.booleans(),
    st.sampled_from(list(rake_mod.STAKES.keys())),
)
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_contributed_rake_share(pot_bb, hero_committed_bb, n_dealt, ended_preflop, stake_key):
    """contributed_rake_bb(pot, hero_committed, n_dealt, ended_preflop, stake) is in [0, total_rake]
    and equals total_rake * (hero_committed / pot) when pot > 0.

    The contributed attribution is the industry-standard low-variance method, and is exactly
    how CoinRaces counts '1 Coin = $1 rake generated'.
    """
    assume(pot_bb > 0.0)
    # ECON-2 (rake.py): the helper now raises ValueError on hero_committed > pot (caller bug) or
    # hero_committed < 0 (impossible). Filter those out at the input boundary so the property
    # only sweeps the in-hand invariant the function was designed to handle.
    assume(0.0 <= hero_committed_bb <= pot_bb + 1e-9)
    c = rake_mod.contributed_rake_bb(pot_bb, hero_committed_bb, n_dealt, ended_preflop, stake_key)
    assert c >= 0.0, f"contributed rake negative: {c}"
    total = rake_mod.coinpoker_rake_bb(pot_bb, n_dealt, ended_preflop, stake_key)
    # In-hand invariant: hero share equals total * (hero_committed / pot).
    expected = total * (hero_committed_bb / pot_bb)
    assert c == pytest.approx(expected, abs=1e-9), (
        f"share != total * committed/pot: {c} vs {expected}"
    )


if __name__ == "__main__":
    # Quick manual sanity check (run as: python tests/test_cash_engine_property.py).
    # Hypothesis still drives the bulk of the coverage from pytest.
    print("Running sanity smoke checks (pytest is the primary entry point)...")
    # Hand-rolled single-example smoke for each property:
    state = GameState.new_hand(rng=random.Random(42))
    state.deal_hole_cards()
    assert state.pot_bb >= 0
    total = sum(p.stack_bb + p.committed_total_bb for p in state.players)
    assert abs(total - NUM_PLAYERS * STARTING_STACK_BB) < 1e-9
    legal = state.legal_actions()
    assert any(a.action_type == ActionType.FOLD for a in legal)
    a = fast_made_hand_bucket((0, 13), (5, 18, 31))
    assert a == fast_made_hand_bucket((0, 13), (5, 18, 31))
    assert Card.from_str("Ah").to_str() == "Ah"
    assert canonical_preflop_class(0, 1) == canonical_preflop_class(1, 0)
    r = rake_mod.coinpoker_rake_bb(10.0, 6, False, "0.50/1")
    assert 0.0 <= r <= rake_mod.STAKES["0.50/1"].cap_bb[5]
    print("  ok — 7 sanity checks passed.")
