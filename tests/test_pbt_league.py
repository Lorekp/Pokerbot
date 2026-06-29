"""Tests for the PBT self-play league pure logic (scripts/train_pbt_league.py).

The heavy eval (run_archetype subprocess) is integration-smoked separately; here we lock the deterministic
core: the composite fitness (reward station value, penalize hurting regs), the bounded config perturbation,
and the weighted reg sampler.
"""

from __future__ import annotations

import importlib
import random

pbt = importlib.import_module("scripts.train_pbt_league")


def test_fitness_rewards_station_and_penalizes_reg_harm():
    # pure station value, no reg harm -> fitness == station
    assert pbt._fitness({pbt.TARGET: 100.0, "lag": 0.0}) == 100.0
    # hurting a reg subtracts the harm (downside guard)
    assert pbt._fitness({pbt.TARGET: 100.0, "lag": -30.0}) == 100.0 - pbt.SAFETY_PENALTY * 30.0
    # a reg GAIN is not rewarded (only harm is penalized) -> still just the station value
    assert pbt._fitness({pbt.TARGET: 100.0, "lag": 25.0}) == 100.0
    # harm across multiple regs accumulates
    assert pbt._fitness({pbt.TARGET: 50.0, "lag": -10.0, "tag": -5.0}) == 50.0 - 15.0


def test_perturb_stays_within_clip_bounds():
    rng = random.Random(0)
    base = {k: d for k, (d, *_rest) in pbt.KNOBS.items()}
    for _ in range(300):
        c = pbt._perturb(base, rng)
        for k, (_d, _sigma, lo, hi) in pbt.KNOBS.items():
            assert lo <= c[k] <= hi


def test_sample_reg_respects_weights():
    rng = random.Random(1)
    w = {"lag": 0.0, "nit": 1.0, "tag": 0.0}              # all mass on nit
    assert all(pbt._sample_reg(w, rng) == "nit" for _ in range(30))


def test_gen0_seed_is_the_deployed_config():
    # The deployed value-sizing gains seed generation 0 (so the league can only improve on / confirm them).
    base = {k: d for k, (d, *_rest) in pbt.KNOBS.items()}
    assert base["POKERBOT_VS_K_BIG"] == 1.20 and base["POKERBOT_VS_K_MID"] == 0.35
