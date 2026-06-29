"""P1 of the deep-ICM evaluator (workflow wexmo5p7v): the rollout producer + mean-preserving equity leaf.

Verifies the per-action chip-outcome tables are conservation-clean (each leaf zero-sum, probs sum to 1) and
feed decide_deep end-to-end — i.e., we can get a usable ICM-gradeable outcome table from a RESUMED deep
multiway postflop node (the prerequisite for the P2 held-out scoring).
"""

from __future__ import annotations

import random

from pokerbot.engine.game import ActionType, GameState
from pokerbot.tournament.action_outcomes import build_action_outcomes
from pokerbot.tournament.decide import decide_deep


def _deep_flop_node() -> GameState:
    """Deal a hand and play preflop passively to a multiway FLOP node with the hero to act (deep stacks)."""
    rng = random.Random(2)
    state = GameState.new_hand(rng=rng)
    state.deal_hole_cards()
    for _ in range(24):
        if state.is_terminal() or len(state.board) >= 3 or not state.actors_remaining():
            break
        seat = state.action_seat
        if state.players[seat].folded or state.players[seat].all_in:
            state.advance_to_next_actor()
            continue
        legal = state.legal_actions()
        act = next((a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)),
                   legal[0] if legal else None)
        if act is None:
            break
        state.apply_action(act)
        if not state.is_terminal():
            state.advance_to_next_actor()
    return state


def test_outcome_tables_conserve_and_grade_via_decide_deep():
    state = _deep_flop_node()
    assert len(state.board) >= 3 and not state.is_terminal(), "did not reach a live flop node"
    hero = state.action_seat
    candidates = state.legal_actions()[:2]                  # hero's top-2 candidate actions
    out = build_action_outcomes(state, candidates, rollouts=16, n_runouts=24, rng=random.Random(5))

    assert len(out) == len(candidates) and out
    for label, dist in out.items():
        assert abs(sum(p for p, _ in dist) - 1.0) < 1e-6   # outcome probs sum to 1
        assert dist, f"empty outcome dist for {label}"
        for _p, vec in dist:
            assert abs(sum(vec)) < 0.05                      # chip_delta_vector ~zero-sum (conservation)

    # End-to-end: decide_deep ICM-grades the produced tables (this is what the evaluator measures).
    stacks = [float(p.stack_bb) for p in state.players]
    payouts = [50.0, 30.0, 20.0]
    d = decide_deep(stacks, payouts, hero, out)
    assert d.action in out and d.source == "deep-icm-graded"


def test_chip_ev_toggle_picks_chip_ev_argmax():
    # decide_deep(deal_struck=True) must reduce to the payout-agnostic chip-EV choice over the SAME tables.
    state = _deep_flop_node()
    hero = state.action_seat
    out = build_action_outcomes(state, state.legal_actions()[:2], rollouts=12, n_runouts=20,
                                rng=random.Random(9))
    stacks = [float(p.stack_bb) for p in state.players]
    from pokerbot.tournament.icm_overlay import chip_ev_action_value
    deal = decide_deep(stacks, [50.0, 30.0, 20.0], hero, out, deal_struck=True)
    chip_vals = {a: chip_ev_action_value(stacks, o, hero) for a, o in out.items()}
    assert deal.action == max(chip_vals, key=chip_vals.get) and deal.source == "deal-chip-ev"
