r"""Variance-reduced ABSOLUTE net bb/100 of an EXPENSIVE hero vs a fixed HETEROGENEOUS field.

Why this exists (M4 — and why AIVAT is BANNED on this path)
-----------------------------------------------------------
``run_field_absolute_rotated`` (pokerbot/eval/exploitation_battery.py) gives the gold-standard
seat-rotated duplicate win-rate — but ONLY for a HOMOGENEOUS field (it asserts all five villain specs
are identical). The realistic CoinPoker table is a MIX (a couple of fish + a weak reg + a solid reg), and
``run_field_absolute`` (the heterogeneous path) reduces variance with AIVAT — which on the pluribus/SEARCH
serve path gives *negative* variance reduction (the gadget-Q baseline is miscalibrated; the chance term is
absent). So measuring a few-bb/100 edge vs a realistic mix to a TIGHT CI with the AIVAT path is exactly the
"variance-dominated number" failure. This module is the VR-correct estimator for the heterogeneous case.

The estimand
------------
For a search+overlay hero ``S`` against a fixed heterogeneous ring of villains, the absolute per-hand EV in
**seat-mean** units (bb/hand), measured by a CO-ROTATED seat-rotated duplicate:

  * The hero visits EACH of the ``n_seats`` physical seats once on a deck. As the hero moves seat-to-seat,
    the FIXED heterogeneous ring co-rotates WITH it — the villain at relative offset ``k`` clockwise from the
    hero is ALWAYS ``field_specs[k]``. So every rotation faces the SAME villain-mix permutation; only the
    hero's hole cards (and thus its EV) change. This is the natural generalization of the homogeneity
    requirement: a homogeneous field is co-rotation-invariant for free; a heterogeneous ring must be carried
    along so the duplicate cancellation (the hero holds each dealt hand once vs the identical opponent set)
    survives. The per-deck SEAT-MEAN cancels the dominant hole-card-allocation luck exactly as in the
    homogeneous rotated path.

The reducer (PRIMARY: control variate)
--------------------------------------
Even after co-rotation, the per-deck seat-mean still carries deck-set + board + villain variance. We attack
it with the SAME control variate as ``cv_absolute``: pair the expensive search hero ``S`` with the cheap
blueprint-only hero ``B`` on the SAME co-rotated decks under common random numbers (CRN: identical deck
seed -> identical boards + villain RNG), and pin ``E[B]`` with a DISJOINT large cheap-``B`` sample:

    mu_hat = mean(S_paired) - beta*(mean(B_paired) - mean(B_cheap))          [cv_absolute.decompose_absolute]
    Var    = Var(S - beta*B)/n_paired  +  beta^2 * Var(B_cheap)/n_cheap

This is unbiased (``E[B_paired] = E[B_cheap]`` by construction — same field, same co-rotation, same deck-seed
generator, disjoint seeds) and CANNOT bias the headline — it only sharpens it. The realized variance
reduction is asserted ``> 1`` at runtime (a CV that fails to reduce variance must NEVER silently return — it
would mean the cheap baseline is anti-correlated / mis-wired, the same failure class as AIVAT-on-search).

The reducer (FALLBACK: pure co-rotated duplicate)
-------------------------------------------------
When no cheap baseline is wanted (``baseline_factory=None``), the estimator is the plain co-rotated duplicate
seat-mean over the expensive decks — still the dominant seat-luck cancellation, just without the CV layer.

RAKE (G6, additive)
-------------------
When ``net_of_rake_stake`` is given, EACH per-deck seat result additionally subtracts the hero's contributed
net cash cost via ``rake.net_cost_bb`` (5%-flat capped rake with the per-tier BB-cap hump, CoinRewards
rakeback, splash fee), captured per hand through a ``_play_capture`` shim that mirrors the rotated path. The
chip-EV column is byte-unchanged (same ``play_hand`` call); the net column is strictly additive.

Serve-skew (the #1 cash correctness risk)
-----------------------------------------
If this serves the 6-max blueprint (a ``search_factory`` whose hero reads the abstraction flags at import),
the caller MUST pass ``snap_abstraction`` so this module asserts the serving fingerprint matches the snapshot
stamp via ``pokerbot.eval.lbr.check_blueprint_serve_env(strict=True, expected=...)``. A mismatch fails closed
(that was the -559 bb/100 "config artifact"). ``tests/test_het_cv_absolute.py`` proves a deliberate mismatch
is DETECTED.

See ``tests/test_het_cv_absolute.py`` for unbiasedness, a measured ``var_reduction > 1`` on a correlated
synthetic het field, the documentation test that AIVAT-on-search yields ``var_reduction < 1``, and the
serve-skew fail-closed test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from pokerbot.eval.cv_absolute import CVResult, decompose_absolute
from pokerbot.engine.game import GameState, play_hand
from pokerbot.eval.bots import build_bot


# --------------------------------------------------------------------------------------------------
# Co-rotated heterogeneous deck play (NO AIVAT on this path — by construction and asserted)
# --------------------------------------------------------------------------------------------------
def _play_capture_het(hero_seat, hero_fn, bot_fn, state, n_seats, allin_settle=None):
    """Play one hand and return ``(hero_delta_bb, pot_bb, hero_committed_bb, ended_preflop)``.

    Mirrors ``exploitation_battery._play_capture`` (the rotated path's rake shim) but is duplicated here so
    this module owns DISJOINT files and never imports a private helper that could change underneath it. The
    chip-EV ``hero_delta`` is exactly what a plain play would return (same ``play_hand`` call); the extra
    three fields are the contributed-attribution rake inputs (``rake.net_cost_bb``)."""
    policies = {s: (hero_fn if s == hero_seat else bot_fn) for s in range(n_seats)}
    final, deltas = play_hand(policies, state=state, allin_settle=allin_settle)
    pot_bb = float(getattr(final, "pot_bb", 0.0) or 0.0)
    hero_committed = float(final.players[hero_seat].committed_total_bb)
    ended_preflop = len(final.board) == 0
    return float(deltas.get(hero_seat, 0.0)), pot_bb, hero_committed, ended_preflop


def _reseed_hero(hero, deck_seed: int, hero_seat: int) -> None:
    """Reseed a hero for one rotation, EXACTLY like the rotated path's ``_reseed``: set ``hero_seat`` BEFORE
    ``on_new_hand()`` (a search hero builds its subgame from ``self.hero_seat`` — without this it plays
    seat-0's cards from every physical seat and 5/6 rotations are garbage)."""
    if hasattr(hero, "hero_seat"):
        hero.hero_seat = hero_seat
    if hasattr(hero, "reseed"):
        hero.reseed(deck_seed)
    else:
        hero.rng = random.Random(deck_seed)
    if hasattr(hero, "on_new_hand"):
        hero.on_new_hand()


def _corotated_seat_means(hero, field_specs, mk, deck_seed, *, n_seats, button_seat, allin_ev,
                          rake_cost=None):
    """One deck's co-rotated seat-mean(s) for ``hero`` vs the FIXED heterogeneous ``field_specs`` ring.

    The hero occupies each physical seat ``hs`` in turn. The fixed ring co-rotates WITH the hero: the villain
    at clockwise offset ``k`` (1..n_seats-1) from the hero is ALWAYS ``field_specs[k-1]`` — so every rotation
    faces the SAME villain-mix permutation, the generalization of the homogeneity requirement. Villain RNG is
    a pure fn of ``(deck_seed, physical_seat)`` (the rotated path's ``5000 + deck_seed + 17*s`` stream), so the
    SAME deck_seed gives byte-identical boards + villain draws across heroes (CRN) and across the search/blueprint
    arms. Returns ``(seat_mean_chip,)`` or ``(seat_mean_chip, seat_mean_net)`` when ``rake_cost`` is given."""
    settle = (lambda st, nb, _s=deck_seed: _allin_settle(st, nb, 7777 + _s)) if allin_ev else None
    seat_chip, seat_net = [], []
    for hs in range(n_seats):
        # Co-rotation: physical seat (hs + offset) mod n_seats gets field_specs[offset-1]. The villain RNG
        # stream is keyed by PHYSICAL seat so it is identical across heroes/arms on the same deck (CRN), while
        # WHICH spec sits at a physical seat travels with the hero (so the mix permutation the hero faces is
        # constant across rotations).
        bots = {}
        for offset in range(1, n_seats):
            phys = (hs + offset) % n_seats
            name, params = field_specs[offset - 1]
            bots[phys] = mk(name, random.Random(5000 + deck_seed + 17 * phys), params=params)

        def bot_fn(state, _b=bots):
            return _b[state.action_seat](state)

        _reseed_hero(hero, deck_seed, hs)
        # Identical deck for every hero/arm/rotation: dealing is a pure fn of (button_seat, rng).
        st = GameState.new_hand(button_seat=button_seat, rng=random.Random(deck_seed))
        if rake_cost is not None:
            raw, pot, comm, pre = _play_capture_het(hs, hero, bot_fn, st, n_seats, settle)
            seat_net.append(raw - rake_cost(pot, comm, pre))
        else:
            raw, *_ = _play_capture_het(hs, hero, bot_fn, st, n_seats, settle)
        seat_chip.append(raw)
    chip_mean = float(np.mean(seat_chip))
    if rake_cost is not None:
        return chip_mean, float(np.mean(seat_net))
    return (chip_mean,)


def _allin_settle(state, num_board, seed):
    """Mean-preserving all-in equity settle, delegated to the battery's certified helper. Imported lazily so
    the module's top-level import graph stays light (and so a future battery refactor of the settle internals
    is picked up automatically — the seed protocol ``7777 + deck_seed`` matches the rotated path)."""
    from pokerbot.eval.exploitation_battery import _allin_ev_settle
    return _allin_ev_settle(state, num_board, seed)


def _run_corotated(hero, field_specs, mk, *, decks, seed, n_seats, button_seat, allin_ev,
                   rake_cost, progress_cb, tag):
    """Run ``decks`` co-rotated decks for ONE hero -> ``(chip_seat_means, net_seat_means_or_None)``.

    The deck seed is the rotated path's ``1000 + seed*1_000_000 + d`` — so two heroes (search & blueprint) run
    with the SAME ``seed`` are CRN-paired deck-for-deck, and a DISJOINT cheap sample at ``seed+offset`` strides
    into a non-overlapping seed band (the cv_absolute disjointness contract)."""
    chip, net = [], []
    want_rake = rake_cost is not None
    for d in range(decks):
        if progress_cb is not None:
            progress_cb(tag, d, decks)
        deck_seed = 1000 + seed * 1_000_000 + d
        out = _corotated_seat_means(hero, field_specs, mk, deck_seed, n_seats=n_seats,
                                    button_seat=button_seat, allin_ev=allin_ev,
                                    rake_cost=rake_cost)
        chip.append(out[0])
        if want_rake:
            net.append(out[1])
    if progress_cb is not None:
        progress_cb(tag, decks, decks)
    return chip, (net if want_rake else None)


# --------------------------------------------------------------------------------------------------
# Public result + entry point
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HetCVResult:
    """The heterogeneous-field VR measurement (M4). ``chip`` is the chip-EV control-variate result (always
    present). ``net`` is the net-of-rake control-variate result (present only when ``net_of_rake_stake`` was
    given). ``reducer`` is ``'control_variate'`` (PRIMARY) or ``'corotated_duplicate'`` (FALLBACK, no cheap
    baseline). ``meta`` carries raw aggregates + provenance (field, seats, button, rake stake/mode, seeds)."""

    chip: CVResult
    net: CVResult | None
    reducer: str
    meta: dict

    @property
    def chip_bb_per_100(self) -> float:
        return self.chip.mean * 100.0

    @property
    def net_bb_per_100(self) -> float | None:
        return None if self.net is None else self.net.mean * 100.0


class VarReductionError(RuntimeError):
    """The control variate FAILED to reduce variance (``var_reduction <= 1``). Raised (not silently
    returned) because a non-reducing CV means the cheap baseline is mis-wired / anti-correlated — exactly the
    silent-mis-measurement failure class (AIVAT-on-search) this estimator exists to replace."""


def _corotated_duplicate_result(chip_paired, net_paired):
    """FALLBACK reducer: pure co-rotated duplicate (no cheap baseline). Wrap the per-deck seat-means in a
    CVResult shell where the CV degenerates to the naive mean (beta=0, var_reduction=1, corr=0) so callers
    get a uniform return type. Var = Var(seat_mean)/n_decks."""
    def _shell(vals):
        a = np.asarray(vals, dtype=float).ravel()
        n = a.size
        if n < 2:
            raise ValueError("need >=2 decks for a variance estimate")
        m = float(a.mean())
        se = float(np.sqrt(np.var(a, ddof=1) / n))
        return CVResult(mean=m, se=se, naive_mean=m, naive_se=se, var_reduction=1.0,
                        corr=0.0, beta=0.0, decks_paired=n, decks_cheap=0)
    chip = _shell(chip_paired)
    net = _shell(net_paired) if net_paired is not None else None
    return chip, net


def het_cv_absolute(field_specs, *, search_factory, baseline_factory=None, paired_decks=None,
                    cheap_decks=None, seed=0, cheap_seed_offset=777, n_seats=6, button_seat=3,
                    net_of_rake_stake=None, rakeback_mode="modeled", chance_cv=False,
                    bot_builder=None, snap_abstraction=None, progress_cb=None,
                    _run_corotated_fn=None) -> tuple[HetCVResult, dict]:
    """Measure the EXPENSIVE hero's ABSOLUTE (and net-of-rake) bb/100 vs a FIXED HETEROGENEOUS field, with
    POSITIVE variance reduction, via a CO-ROTATED seat-rotated duplicate + a CHEAP-baseline control variate.

    Parameters
    ----------
    field_specs : list[(bot_name, params|None)]
        The FIXED heterogeneous ring for the ``n_seats - 1`` villain seats (the realistic MIX). The ring
        co-rotates WITH the hero, so the hero faces this same villain-mix permutation from every seat.
    search_factory : callable ``(exploit: bool) -> hero``
        Builds the EXPENSIVE hero (search + value/sizing overlays). Called with ``exploit=True`` (the
        production policy — the M1/M2 overlay arm the realistic mix is meant to capture).
    baseline_factory : callable ``(exploit: bool) -> hero`` | None
        Builds the CHEAP, highly-correlated blueprint-only hero. ``None`` -> the FALLBACK pure co-rotated
        duplicate (no CV). Called with ``exploit=False``.
    paired_decks, cheap_decks : int
        Expensive (paired) and cheap-baseline deck counts. ``cheap_decks`` is ignored when no baseline.
    seed, cheap_seed_offset : int
        CRN seed for the paired arms; the cheap arm runs at ``seed + cheap_seed_offset`` (disjoint band).
    net_of_rake_stake : str | None
        A ``pokerbot.economics.rake.STAKES`` key -> additionally report a net-of-rake CV column (G6).
    snap_abstraction : dict | None
        The abstraction dict STAMPED into the served blueprint snapshot. When given, the serving fingerprint
        is asserted to match via ``check_blueprint_serve_env(strict=True, expected=...)`` BEFORE any play —
        a mismatch fails closed (the -559 bb/100 serve-skew guard).

    Returns
    -------
    (HetCVResult, dict)
        The VR result + a meta dict. ``HetCVResult.chip`` is the chip-EV control-variate estimate; ``.net`` the
        net-of-rake one (when requested). Raises :class:`VarReductionError` if a CV arm reports
        ``var_reduction <= 1`` (a CV must reduce variance or it is mis-wired)."""
    if search_factory is None:
        raise ValueError("het_cv_absolute requires a search_factory")
    if paired_decks is None or paired_decks < 2:
        raise ValueError("paired_decks must be >= 2 (need a variance estimate)")
    specs = list(field_specs)
    if len(specs) != n_seats - 1:
        raise ValueError(f"field_specs has {len(specs)} villain specs, need n_seats-1 = {n_seats - 1}")
    if chance_cv:
        # ChanceCV is a board-runout control variate for the OFF arm only; it is orthogonal to this CV path
        # and not wired here. Reject loudly rather than silently ignore (no silent mis-measurement).
        raise ValueError("chance_cv is not supported on het_cv_absolute (use the CV/co-rotated reducer)")

    # ---- Serve-skew fingerprint guard (the #1 cash correctness risk) ----
    # If a blueprint is being served, the encoder baked POKERBOT_* abstraction flags AT IMPORT. If they differ
    # from the snapshot stamp the blueprint fingerprint-MISSES -> ~0% coverage -> garbage. Assert a MATCH and
    # FAIL CLOSED on mismatch (this was the -559 bb/100 artifact). Caller opts in by passing snap_abstraction.
    if snap_abstraction is not None:
        from pokerbot.eval.lbr import check_blueprint_serve_env
        check_blueprint_serve_env(strict=True, expected=snap_abstraction)

    # ---- AIVAT BAN (asserted): no AIVATAccumulator may be constructed on this path ----
    # AIVAT gives NEGATIVE variance reduction on the search serve path. This estimator deliberately never
    # imports/constructs it; the test suite proves AIVAT-on-search would fail the var_reduction>1 gate. We
    # assert at runtime that nothing on the call stack instantiated one (defense-in-depth, near-zero cost).
    _assert_no_aivat()

    mk = bot_builder or build_bot
    run_fn = _run_corotated_fn or _run_corotated

    def _rake_cost(stake):
        if stake is None:
            return None
        from pokerbot.economics.rake import STAKES, net_cost_bb
        if stake not in STAKES:
            raise ValueError(f"net_of_rake_stake {stake!r} not in {list(STAKES)}")

        def _cost(pot_bb, hero_committed, ended_preflop, _stake=stake):
            return float(net_cost_bb(pot_bb, hero_committed, n_seats, ended_preflop, _stake,
                                     rakeback_mode=rakeback_mode))
        return _cost

    rake_cost = _rake_cost(net_of_rake_stake)

    # ---- EXPENSIVE search arm (paired) ----
    hero_s = search_factory(exploit=True)
    s_chip, s_net = run_fn(hero_s, specs, mk, decks=paired_decks, seed=seed, n_seats=n_seats,
                           button_seat=button_seat, allin_ev=True, rake_cost=rake_cost,
                           progress_cb=progress_cb, tag="search")

    if baseline_factory is None:
        # FALLBACK: pure co-rotated duplicate, no CV layer.
        chip, net = _corotated_duplicate_result(s_chip, s_net)
        meta = _build_meta(specs, n_seats, button_seat, net_of_rake_stake, rakeback_mode, seed,
                           cheap_seed_offset, paired_decks, 0, "corotated_duplicate",
                           s_chip, None, None)
        return HetCVResult(chip=chip, net=net, reducer="corotated_duplicate", meta=meta), meta

    # ---- PRIMARY: control variate with a CHEAP blueprint-only baseline ----
    if cheap_decks is None or cheap_decks < 2:
        raise ValueError("cheap_decks must be >= 2 when a baseline_factory is given")
    hero_b = baseline_factory(exploit=False)            # CRN-paired with the search arm (same seed)
    b_chip, b_net = run_fn(hero_b, specs, mk, decks=paired_decks, seed=seed, n_seats=n_seats,
                           button_seat=button_seat, allin_ev=True, rake_cost=rake_cost,
                           progress_cb=progress_cb, tag="baseline_paired")
    hero_bc = baseline_factory(exploit=False)           # DISJOINT cheap sample (seed + offset)
    bc_chip, bc_net = run_fn(hero_bc, specs, mk, decks=cheap_decks, seed=seed + cheap_seed_offset,
                             n_seats=n_seats, button_seat=button_seat, allin_ev=True,
                             rake_cost=rake_cost, progress_cb=progress_cb, tag="baseline_cheap")

    chip = decompose_absolute(s_chip, b_chip, bc_chip)
    if chip.var_reduction <= 1.0:
        raise VarReductionError(
            f"chip-EV control variate did NOT reduce variance (var_reduction={chip.var_reduction:.3f} <= 1, "
            f"corr={chip.corr:.3f}). The cheap blueprint baseline is mis-wired or anti-correlated with the "
            "search hero on these co-rotated decks — refusing to return a non-reducing CV (silent "
            "mis-measurement is the exact AIVAT-on-search failure this estimator replaces).")

    net = None
    if rake_cost is not None:
        net = decompose_absolute(s_net, b_net, bc_net)
        if net.var_reduction <= 1.0:
            raise VarReductionError(
                f"net-of-rake control variate did NOT reduce variance "
                f"(var_reduction={net.var_reduction:.3f} <= 1, corr={net.corr:.3f}).")

    meta = _build_meta(specs, n_seats, button_seat, net_of_rake_stake, rakeback_mode, seed,
                       cheap_seed_offset, paired_decks, cheap_decks, "control_variate",
                       s_chip, b_chip, bc_chip)
    return HetCVResult(chip=chip, net=net, reducer="control_variate", meta=meta), meta


def _build_meta(specs, n_seats, button_seat, stake, rakeback_mode, seed, cheap_seed_offset,
                paired_decks, cheap_decks, reducer, s_chip, b_chip, bc_chip):
    meta = {
        "method": "het_corotated_duplicate",
        "reducer": reducer,
        "field": [s[0] for s in specs],
        "n_seats": n_seats,
        "button_seat": button_seat,
        "net_of_rake_stake": stake,
        "rakeback_mode": rakeback_mode if stake is not None else None,
        "seed": seed,
        "cheap_seed_offset": cheap_seed_offset,
        "paired_decks": paired_decks,
        "cheap_decks": cheap_decks,
        "search_raw_mean_bb": float(np.mean(s_chip)),
        "uses_aivat": False,            # asserted — this path BANS AIVAT (negative VR on search)
    }
    if b_chip is not None:
        meta["baseline_paired_mean_bb"] = float(np.mean(b_chip))
    if bc_chip is not None:
        meta["baseline_cheap_mean_bb"] = float(np.mean(bc_chip))
    return meta


def _assert_no_aivat() -> None:
    """Tripwire: assert no live ``AIVATAccumulator`` exists when this estimator runs. AIVAT is the broken
    reducer (NEGATIVE variance reduction on the search serve path) this module exists to replace; this path
    must never route through it. The estimator itself never imports/constructs one — this scans the GC once
    per call (a small, bounded set: AIVAT accumulators are eval-scoped, not per-decision) so an accidental
    construction in a caller-supplied factory that leaked into scope trips loudly rather than silently
    mis-measuring. Raises ``AssertionError`` on a live instance. The BINDING enforcement is behavioral (the
    ``var_reduction > 1`` gate, which AIVAT-on-search fails — see the test suite); this is defense-in-depth."""
    import gc
    import warnings
    try:
        from pokerbot.eval.aivat import AIVATAccumulator
    except Exception:  # noqa: BLE001 — if AIVAT can't even import, it certainly isn't constructed
        return
    # Silence any lazy-attribute deprecation warnings third-party objects (e.g. torch) emit when the gc walk
    # touches them during isinstance — the tripwire must be quiet, not spew unrelated warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        live = [o for o in gc.get_objects() if isinstance(o, AIVATAccumulator)]
    assert not live, (
        f"AIVAT BAN violated: {len(live)} live AIVATAccumulator instance(s) on the het_cv_absolute path — "
        "AIVAT gives NEGATIVE variance reduction on the search serve path and is forbidden here; use the "
        "control-variate / co-rotated-duplicate reducer instead.")
