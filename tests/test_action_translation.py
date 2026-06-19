"""Pseudo-harmonic action translation (pokerbot/engine/actions.py).

The serve path maps an opponent's off-tree bet size to a discrete bucket. Nearest-by-absolute-distance
is exploitable (bet just inside a boundary to swing our response); the pseudo-harmonic mapping is the
lowest-exploitability deterministic no-compute alternative. These tests pin its behavior AND guard the
train/serve contract: the raw `action_to_index` (used by the training encoders) must be untouched, and
the serve swap must be opt-in.
"""

from __future__ import annotations

from pokerbot.engine.actions import (
    BET_SIZE_BUCKETS,
    NUM_ACTIONS,
    Action,
    ActionType,
    action_to_index,
    action_to_index_pseudo_harmonic,
    pseudo_harmonic_bucket,
)


def test_exact_bucket_sizes_map_to_themselves():
    for i, frac in enumerate(BET_SIZE_BUCKETS):
        assert pseudo_harmonic_bucket(frac) == i


def test_out_of_range_clamps_to_ends():
    assert pseudo_harmonic_bucket(0.01) == 0                       # below smallest -> smallest
    assert pseudo_harmonic_bucket(99.0) == len(BET_SIZE_BUCKETS) - 1  # above largest -> largest


def test_diverges_from_linear_nearest_in_pot_odds_band():
    # The pseudo-harmonic crossover is pot-odds-weighted, NOT the arithmetic midpoint: for the (1.0, 1.5)
    # bracket it sits at ~1.222 (below the linear midpoint 1.25). So in the band (1.222, 1.25) the two
    # mappings DISAGREE — pseudo-harmonic rounds to 1.5 where nearest-by-distance rounds to 1.0. This
    # divergence is the whole point: an attacker can't read our bucket boundary off the linear midpoint.
    x = 1.24
    lin = min(range(len(BET_SIZE_BUCKETS)), key=lambda i: abs(BET_SIZE_BUCKETS[i] - x))
    assert BET_SIZE_BUCKETS[lin] == 1.0          # nearest-by-distance picks the smaller
    assert pseudo_harmonic_bucket(x) == 5        # pseudo-harmonic picks 1.5 -> they diverge


def test_pseudo_harmonic_threshold_formula():
    # The crossover x* solves f_a = 0.5: (b-x)(1+a) = 0.5 (b-a)(1+x). For a=1.0,b=1.5 -> x* ~ 1.30.
    a, b = 1.0, 1.5
    xstar = ((b * (1 + a)) - 0.5 * (b - a)) / ((1 + a) + 0.5 * (b - a))
    assert pseudo_harmonic_bucket(xstar - 0.02) == 4   # below crossover -> smaller (1.0)
    assert pseudo_harmonic_bucket(xstar + 0.02) == 5   # above crossover -> larger (1.5)


def test_non_bet_actions_match_raw():
    for at in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL, ActionType.ALL_IN):
        a = Action(at)
        assert action_to_index_pseudo_harmonic(a) == action_to_index(a)


def test_all_in_is_last_index():
    assert action_to_index_pseudo_harmonic(Action(ActionType.ALL_IN)) == NUM_ACTIONS - 1


def test_raw_action_to_index_unchanged_byte_identity():
    # The training-path function must keep its nearest-by-distance behavior exactly (fingerprint guard).
    assert action_to_index(Action(ActionType.RAISE, 1.27)) == 3 + min(
        range(len(BET_SIZE_BUCKETS)), key=lambda i: abs(BET_SIZE_BUCKETS[i] - 1.27))
    # bet/raise indices live in [3, NUM_ACTIONS-2]
    idx = action_to_index_pseudo_harmonic(Action(ActionType.BET, 0.6))
    assert 3 <= idx <= NUM_ACTIONS - 2
