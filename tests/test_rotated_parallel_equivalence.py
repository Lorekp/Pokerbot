"""Equivalence gate for the env-gated process-parallel OFF arm of run_field_absolute_rotated.

POKERBOT_ROTATED_WORKERS=N fans the OFF (pure-base headline) arm of the seat-rotated duplicate
estimator across N processes. Because each OFF deck value is a pure function of its seed (the OFF
hero carries no online-learning state; villains are reseeded per seat/deck), the parallel result must
be BYTE-IDENTICAL to the sequential default — this is an exact-equivalence optimization (concurrency
only, no algorithm change), so we require == not a tolerance.

The ON arm stays sequential (its bot_stats profile accrues across decks); these tests pin that the ON
seat-mean is ALSO byte-unchanged when the OFF arm is parallelized (the two arms share no RNG stream —
every random.Random is freshly seeded from the deck seed).

Uses the lightweight ExploitHero (no checkpoint). The pluribus/search hero is byte-verified manually
in the session report (its factory closes over a 40-60MB snapshot; cloudpickle ships it once per
worker, which is too slow to want in unit CI but is exercised by the harness).
"""

from __future__ import annotations

import os
import random

import pytest

from pokerbot.eval.exploitation_battery import (
    ExploitHero,
    field_spec,
    run_field_absolute_rotated,
)


def _hero_factory(*, exploit: bool):
    return ExploitHero(random.Random(0), exploit=exploit)


def _run(workers: int, *, decks: int, seed: int):
    prev = os.environ.get("POKERBOT_ROTATED_WORKERS")
    os.environ["POKERBOT_ROTATED_WORKERS"] = str(workers)
    try:
        return run_field_absolute_rotated(
            field_spec("homog:calling_station"), decks=decks, seed=seed,
            hero_factory=_hero_factory, allin_ev=True, progress_every=0)
    finally:
        if prev is None:
            os.environ.pop("POKERBOT_ROTATED_WORKERS", None)
        else:
            os.environ["POKERBOT_ROTATED_WORKERS"] = prev


def test_parallel_off_arm_byte_identical_to_sequential():
    seq = _run(0, decks=8, seed=3)
    par = _run(3, decks=8, seed=3)
    # Exact-equivalence optimization: require bit-identity, not a tolerance.
    assert seq["off_base"]["mean_bb"] == par["off_base"]["mean_bb"]
    assert seq["off_base"]["std"] == par["off_base"]["std"]
    assert seq["off_base"]["se"] == par["off_base"]["se"]
    # ON arm (sequential in both) must also be byte-unchanged — the OFF parallelization must not
    # perturb the ON RNG stream or the bot_stats learning order.
    assert seq["on_exploit"]["mean_bb"] == par["on_exploit"]["mean_bb"]
    assert seq["on_exploit"]["std"] == par["on_exploit"]["std"]
    assert seq["uplift_bb_per_hand"] == par["uplift_bb_per_hand"]
    assert seq["uplift_se"] == par["uplift_se"]
    assert seq["profile_contexts"] == par["profile_contexts"]


def test_parallel_off_disabled_for_single_deck_or_no_workers():
    # decks<=1 or workers<=1 -> the sequential path (the parallel branch is skipped); still correct.
    a = _run(1, decks=4, seed=5)
    b = _run(0, decks=4, seed=5)
    assert a["off_base"]["mean_bb"] == b["off_base"]["mean_bb"]
    assert a["on_exploit"]["mean_bb"] == b["on_exploit"]["mean_bb"]


def test_parallel_multiple_seeds_byte_identical():
    for seed in (0, 11):
        seq = _run(0, decks=6, seed=seed)
        par = _run(2, decks=6, seed=seed)
        assert seq["off_base"]["mean_bb"] == par["off_base"]["mean_bb"], f"seed {seed} OFF"
        assert seq["on_exploit"]["mean_bb"] == par["on_exploit"]["mean_bb"], f"seed {seed} ON"
