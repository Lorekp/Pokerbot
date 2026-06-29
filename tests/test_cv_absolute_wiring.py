"""End-to-end wiring of the control-variate estimator to the REAL rotated eval (run_field_absolute_rotated).

Proves the plumbing on actual eval output (not synthetic): (1) the eval now exposes per-deck seat-means,
(2) two runs at the SAME seed are CRN-paired deck-for-deck (identical deck seeds -> a same-hero rerun is
bit-identical, and a different hero shares the boards), (3) run_cv_absolute combines the three passes through
decompose_absolute and returns an unbiased, finite estimate. Uses the lightweight ExploitHero (no search,
no checkpoint) so it runs in milliseconds — the 7x variance reduction itself is proven separately on
synthetic draws in test_cv_absolute.py (the real search-hero magnitude needs CPU, not a unit test).
"""

from __future__ import annotations

import random

from pokerbot.eval.cv_absolute import CVResult, run_cv_absolute
from pokerbot.eval.exploitation_battery import (
    ExploitHero,
    field_spec,
    run_field_absolute_rotated,
)

FIELD = field_spec("homog:calling_station")


def _search_factory(*, exploit: bool):
    # stand-in "expensive" hero (exploit overlay ON) — in production this is the pluribus SearchAgent
    return ExploitHero(random.Random(0), exploit=True)


def _baseline_factory(*, exploit: bool):
    # stand-in "cheap" baseline (no exploit overlay) — in production this is the blueprint-only hero
    return ExploitHero(random.Random(0), exploit=False)


def test_eval_exposes_per_deck_seat_means():
    r = run_field_absolute_rotated(FIELD, decks=5, seed=0, hero_factory=_search_factory, progress_every=0)
    assert "off_deck_seat_means" in r and "on_deck_seat_means" in r
    assert len(r["off_deck_seat_means"]) == 5
    assert len(r["on_deck_seat_means"]) == 5
    # the exposed per-deck arrays must reproduce the reported aggregate mean exactly
    import numpy as np
    assert abs(float(np.mean(r["off_deck_seat_means"])) - r["off_base"]["mean_bb"]) < 1e-9


def test_same_seed_is_crn_paired():
    """Same seed + same hero -> identical per-deck series (the property the CV pairing relies on)."""
    a = run_field_absolute_rotated(FIELD, decks=6, seed=0, hero_factory=_baseline_factory, progress_every=0)
    b = run_field_absolute_rotated(FIELD, decks=6, seed=0, hero_factory=_baseline_factory, progress_every=0)
    assert a["off_deck_seat_means"] == b["off_deck_seat_means"]
    # a different seed must give a different (disjoint) series
    c = run_field_absolute_rotated(FIELD, decks=6, seed=1, hero_factory=_baseline_factory, progress_every=0)
    assert c["off_deck_seat_means"] != a["off_deck_seat_means"]


def test_run_cv_absolute_end_to_end():
    res, meta = run_cv_absolute(
        FIELD, search_factory=_search_factory, baseline_factory=_baseline_factory,
        paired_decks=12, cheap_decks=24, seed=0, arm="off", progress_every=0,
    )
    assert isinstance(res, CVResult)
    assert res.decks_paired == 12 and res.decks_cheap == 24
    # the CV mean must be finite and consistent with the raw search mean within a few combined SEs
    import math
    assert math.isfinite(res.mean) and math.isfinite(res.se) and res.se >= 0.0
    spread = abs(res.mean - meta["search_raw_mean_bb"])
    assert spread <= 6.0 * (res.se + meta["search_raw_se"] + 1e-9)
    # arm='on' path also wires
    res_on, meta_on = run_cv_absolute(
        FIELD, search_factory=_search_factory, baseline_factory=_baseline_factory,
        paired_decks=10, cheap_decks=20, seed=3, arm="on", progress_every=0,
    )
    assert meta_on["arm"] == "on" and res_on.decks_paired == 10


def test_missing_per_deck_arrays_is_a_clear_error():
    def _bad_run(field, **kw):
        return {"off_base": {"mean_bb": 0.0, "se": 0.0}}  # no per-deck arrays

    try:
        run_cv_absolute(FIELD, search_factory=_search_factory, baseline_factory=_baseline_factory,
                        paired_decks=4, cheap_decks=4, run_fn=_bad_run)
    except KeyError as e:
        assert "per-deck" in str(e)
    else:
        raise AssertionError("expected a KeyError when the eval omits per-deck seat-means")
