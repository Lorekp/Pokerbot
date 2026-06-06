"""Adaptive opponent model: cold-start population prior, recency-weighted adaptation to a strategy
switch (the key 'always adapts' property), change detection, confidence, persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from aof_cfr.adaptive import AdaptiveOpponentModel, PopulationPrior

_HAS = Path("data/aof/cfr_base/cfr_base_3p.npz").is_file()
SEAT, HIST, STACK, N = "BB", "A", 8.0, 3      # BB facing a jam (vs-jam node)


def _model(flat=0.4):
    return AdaptiveOpponentModel(prior=PopulationPrior(flat_rate=flat))


def _feed(m, ident, push_prob_seq):
    """Feed a sequence of booleans (deterministic)."""
    for pushed in push_prob_seq:
        m.observe(ident, N, SEAT, HIST, STACK, pushed)


def test_cold_start_uses_population_prior_not_gto() -> None:
    m = _model(flat=0.4)
    est = m.estimate("newbie", N, SEAT, HIST, STACK)
    assert est["rate"] == pytest.approx(0.4)        # exploit the FIELD from hand one
    assert est["confidence"] == 0.0 and not est["adapting"]


def test_tracks_strategy_switch_while_lifetime_stays_stale() -> None:
    m = _model(flat=0.4)
    ident = "villain"
    _feed(m, ident, [False] * 300)                  # long history as a nit (folds vs jams)
    nit = m.estimate(ident, N, SEAT, HIST, STACK)
    assert nit["rate"] < 0.15 and not nit["adapting"]

    _feed(m, ident, [True] * 20)                    # villain switches to a maniac (calls everything)
    sw = m.estimate(ident, N, SEAT, HIST, STACK)
    # adaptive read jumps toward the NEW behaviour; a lifetime-pooled read would still read ~nit
    lifetime = (0 * 300 + 20) / 320                  # = 0.0625, what plain additive counts would say
    assert sw["rate"] > 0.4, sw                      # jumped ~8x off the nit read in 20 hands
    assert sw["rate"] - lifetime > 0.35              # adaptive vs lifetime: huge gap
    assert sw["adapting"] is True                    # change detected


def test_adapt_flag_clears_once_change_is_established() -> None:
    m = _model(flat=0.4)
    ident = "shifter"
    _feed(m, ident, [False] * 200)
    _feed(m, ident, [True] * 15)
    assert m.is_adapting(ident, N, SEAT, HIST, STACK)        # mid-shift: fast != slow
    _feed(m, ident, [True] * 400)                            # sustained -> slow catches up
    est = m.estimate(ident, N, SEAT, HIST, STACK)
    assert not est["adapting"] and est["rate"] > 0.8         # settled on the new strategy


def test_confidence_rises_with_effective_sample() -> None:
    m = _model()
    ident = "v"
    c0 = m.estimate(ident, N, SEAT, HIST, STACK)["confidence"]
    _feed(m, ident, [True, False] * 30)
    c1 = m.estimate(ident, N, SEAT, HIST, STACK)["confidence"]
    assert c0 == 0.0 < c1 < 1.0


def test_persistence_roundtrip() -> None:
    m = _model()
    _feed(m, "p", [True, False, True, True, False])
    snap = m.to_dict()
    m2 = _model().load_dict(snap)
    a = m.estimate("p", N, SEAT, HIST, STACK)
    b = m2.estimate("p", N, SEAT, HIST, STACK)
    assert a["rate"] == pytest.approx(b["rate"]) and a["eff_n"] == pytest.approx(b["eff_n"])


@pytest.mark.skipif(not _HAS, reason="artifacts not built")
def test_estimator_rows_are_valid_policy() -> None:
    from aof_cfr.adaptive import estimate_policy_rows_adaptive
    from aof_cfr.query import load_cfr_base
    load_cfr_base()
    m = AdaptiveOpponentModel(prior=PopulationPrior())   # real population prior (needs base)
    _feed(m, "v", [True] * 30)
    rows = estimate_policy_rows_adaptive(m, "v", N, STACK)
    assert rows
    import numpy as np
    for arr in rows.values():
        assert arr.shape == (2,) and abs(float(arr.sum()) - 1.0) < 1e-9 and (arr >= 0).all()


@pytest.mark.skipif(not _HAS, reason="artifacts not built")
def test_population_prior_is_exploitable_vs_jam() -> None:
    # the default population over-folds vs jams -> prior push rate below the GTO base on vs-jam nodes
    from aof_cfr.profiler import _COMBO_W, _COMBO_W_SUM, _base_arrays_seathist, _query_base_push
    from aof_cfr.query import load_cfr_base
    load_cfr_base()
    pp = PopulationPrior(overfold_vs_jam=0.15)
    arrs = _base_arrays_seathist(N, STACK, _query_base_push)
    base = arrs.get((SEAT, HIST))
    if base is not None:
        base_agg = float((_COMBO_W * base).sum() / _COMBO_W_SUM)
        assert pp.push_rate(N, SEAT, HIST, STACK) < base_agg
