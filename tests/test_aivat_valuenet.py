"""R1: value-net AIVAT — the leaf-net per-action baseline + the chance-node control variate.

These validate the RIGOROUS, deterministic claims: both control variates are exactly zero-mean (so the
estimator stays unbiased for ANY baseline), a correlated leaf-net-style baseline cuts more variance than
the crude proxy, and the ExploitHero `value_fn` seam is honored.
"""

from __future__ import annotations

import numpy as np

from pokerbot.eval.aivat import (
    AIVATAccumulator,
    baseline_from_leaf_net,
    chance_correction,
    decision_correction,
)


def test_leaf_net_baseline_masks_illegal_and_stays_zero_mean():
    # bv(a) = Q(h,a) with two illegal actions zeroed; the decision correction averaged over the policy
    # must still cancel exactly (illegal actions carry zero policy mass).
    q_row = np.array([0.4, -0.2, 0.9, 1.3, -0.5, 0.0, 0.7, 0.1, -0.3, 0.2, 1.1])
    legal = np.zeros(11, dtype=bool)
    legal[[1, 2, 3, 5, 10]] = True
    bv = baseline_from_leaf_net(q_row, legal)
    assert bv.shape == (11,)
    assert np.all(bv[~legal] == 0.0)          # illegal zeroed
    assert np.all(bv[legal] == q_row[legal])  # legal untouched

    probs = np.zeros(11)
    probs[[1, 2, 3, 5, 10]] = [0.1, 0.3, 0.25, 0.15, 0.2]
    e = sum(probs[i] * decision_correction(probs, bv, i) for i in range(11))
    assert abs(e) < 1e-12


def test_chance_correction_is_zero_mean_over_uniform_reveals():
    # ANY value function over the realized reveal set is zero-mean as long as the expectation averages
    # over the exact set the realized card is drawn from (unbiasedness independent of value quality).
    rng = np.random.default_rng(0)
    reveal_values = rng.normal(size=46)          # value(after) for each possible next card
    expected = float(reveal_values.mean())       # E[value | before]
    corrections = [chance_correction(expected, v) for v in reveal_values]
    assert abs(float(np.mean(corrections))) < 1e-12


def test_chance_corrections_flow_through_accumulator_unbiased():
    rng = np.random.default_rng(1)
    acc = AIVATAccumulator()
    for _ in range(3000):
        raw = float(rng.normal())
        # a zero-mean decision correction + a zero-mean chance correction
        dec = [float(rng.normal()) * 0.0]        # exactly zero here
        reveals = rng.normal(size=10)
        chance = [chance_correction(float(reveals.mean()), float(reveals[rng.integers(10)]))]
        acc.add_hand(raw, dec, chance)
    s = acc.summary()
    assert abs(s["adj_mean_bb"] - s["raw_mean_bb"]) < 0.1   # chance term doesn't bias the mean


def test_leaf_net_baseline_beats_crude_proxy_on_variance():
    # Outcome correlates with the TAKEN action's leaf value. A baseline that IS that leaf value cancels
    # far more variance than a flat/crude baseline → higher variance_reduction.
    rng = np.random.default_rng(7)
    q_table = np.array([0.0, 0.0, 0.3, 1.0, -0.8, 0.5, 0.9, 0.2, -0.4, 0.6, 1.4]) * 4.0
    crude = np.linspace(-1.0, 1.0, 11)           # a poorly-correlated baseline

    acc_leaf, acc_crude = AIVATAccumulator(), AIVATAccumulator()
    for _ in range(4000):
        probs = np.abs(rng.normal(size=11)) + 0.05
        probs /= probs.sum()
        idx = int(rng.choice(11, p=probs))
        outcome = q_table[idx] + rng.normal() * 0.5    # outcome tracks the taken action's leaf value
        acc_leaf.add_hand(outcome, [decision_correction(probs, q_table, idx)])
        acc_crude.add_hand(outcome, [decision_correction(probs, crude, idx)])

    sl, sc = acc_leaf.summary(), acc_crude.summary()
    assert sl["variance_reduction"] > sc["variance_reduction"]
    assert sl["variance_reduction"] > 0.3        # the matched leaf baseline cuts substantial variance


def test_exploit_hero_value_fn_seam_is_used():
    import random

    from pokerbot.engine.game import GameState
    from pokerbot.eval.exploitation_battery import ExploitHero

    calls = {"n": 0}

    def value_fn(state, seat, legal):
        calls["n"] += 1
        return np.arange(11, dtype=np.float64)   # a sentinel baseline

    hero = ExploitHero(random.Random(0), exploit=False, value_fn=value_fn)
    st = GameState.new_hand(button_seat=3, rng=random.Random(0))
    hero(st)
    assert calls["n"] >= 1
    _, bv, _ = hero.pop_decisions()[0]
    assert np.array_equal(np.asarray(bv), np.arange(11, dtype=np.float64))
