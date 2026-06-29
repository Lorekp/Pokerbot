"""Tests for the full-lifecycle MTT/SNG $-ROI simulator (pokerbot/tournament/mtt_lifecycle.py).

The correctness anchors (all net-free / CPU / small-N so the suite stays fast):

  * E4 UNBIASEDNESS — a HOMOGENEOUS field (hero plays the same ``decide_fn`` as everyone) must have
    ``mean_finish ~ (N+1)/2`` and ``roi ~ 0`` THROUGH THE WHOLE LIFECYCLE (blind escalation + antes +
    table balancing + breaks). This is the decisive no-bias check: it exercises the schedule, the
    identity-tracked pid balancing through breaks, and the bust accounting. Run for ``chip_ev_nash``
    AND both net-free anchors (``anchor_fold`` / ``anchor_jam``) — the anchors are the harshest cases
    (all-fold / all-jam) and a seat-0 (hero) privilege would show up there first.

  * PER-HAND CHIP CONSERVATION — every confrontation's ``chip_delta`` vector sums to ~0 (antes removed
    from each seat + the all-in swing + the dead-money award all net out).

  * LIFECYCLE WIRING — the (previously unwired) ``structure`` ladder drives blinds UP as hands elapse
    and the regime crosses deep -> pushfold; the per-player ante is posted.

  * E3 ROI + 95% CI / CRN DUPLICATE — ``measure_lifecycle_roi`` reports gross+net ROI with a CI; the
    CRN-paired ``duplicate_roi`` runs two arms on the same rng and returns a paired-difference CI.

  * NET SERVE FAIL-SAFE + ENCODER FIDELITY — ``net_decide`` never raises (a missing/absent policy
    falls back to the net-free threshold); and the load-bearing ROUND-TRIP ``encode_info_state``
    byte-identity test (mirroring ``tests/test_seven_neural_serve.py``) proves the serve layer the net
    adapter consumes reproduces the trained 7-seat layout bit-for-bit (no serve-time encoder skew).

The simulator + adapters are torch-free (numpy + the dependency-light tournament primitives + a lazy
jax/engine import only on the net path), so the net-free body runs in any env; the net + encoder tests
``importorskip('jax')`` and skip cleanly when the checkpoint is absent.
"""

from __future__ import annotations

import random

import pytest

from pokerbot.tournament.mtt_lifecycle import (
    LifecycleSchedule,
    REGIME_DEEP,
    REGIME_PUSHFOLD,
    anchor_fold,
    anchor_jam,
    chip_ev_nash,
    duplicate_roi,
    icm_aware,
    measure_lifecycle_roi,
    net_decide,
    simulate_tournament_lifecycle,
    sng_schedule,
)
from pokerbot.tournament.structure import STARTING_CHIPS, ante_for_bb, blind_ladder


# --------------------------------------------------------------------------- LifecycleSchedule wiring
def test_schedule_walks_blinds_up_with_hands():
    """The schedule wires structure.blind_ladder: blinds rise as hands elapse; the ante is round(BB/8)."""
    sched = LifecycleSchedule(hands_per_level=5, start_level=1)
    bb0 = sched.bb_chips
    assert sched.ante_chips == ante_for_bb(bb0)
    # advance one full level worth of hands -> the level (and big blind) must step up.
    for _ in range(5):
        sched.advance(players_left=50)
    assert sched.level == 2
    assert sched.bb_chips > bb0
    assert sched.ante_chips == ante_for_bb(sched.bb_chips)


def test_schedule_final_table_rollback():
    """When the field reaches the final-table size the blinds ROLL BACK one level (structure rule)."""
    sched = LifecycleSchedule(hands_per_level=2, start_level=1, final_table_size=9)
    for _ in range(20):                       # climb several levels
        sched.advance(players_left=50)
    high = sched.level
    sched.advance(players_left=9)             # field hits the final table -> rollback
    assert sched.level == max(1, high - sched.ft_rollback_levels)


def test_effective_bb_shrinks_as_blinds_rise():
    sched = LifecycleSchedule(hands_per_level=3)
    deep = sched.effective_bb(STARTING_CHIPS)
    for _ in range(30):
        sched.advance(players_left=50)
    shallow = sched.effective_bb(STARTING_CHIPS)
    assert shallow < deep                     # same chips, more blinds -> fewer bb


# --------------------------------------------------------------------------- single tournament completes
def test_completes_to_valid_finish_and_payout():
    payouts = [50.0, 30.0, 20.0]
    r = simulate_tournament_lifecycle(12, payouts, chip_ev_nash(), rng=random.Random(1),
                                      realistic=False)
    assert 1 <= r.hero_finish <= 12
    assert r.players == 12
    pay = payouts + [0.0] * (12 - len(payouts))
    assert r.hero_payout == pay[r.hero_finish - 1]
    assert r.levels_played >= 1


# --------------------------------------------------------------------------- per-hand chip conservation
@pytest.mark.parametrize("realistic", [False, True])
@pytest.mark.parametrize("seed", [2, 3, 5])
def test_per_hand_chip_conservation(realistic, seed):
    """Every confrontation moves chips conservatively: the per-seat chip_delta (antes + all-in swing +
    dead-money award) sums to ~0. A non-zero sum would mean chips were created/destroyed in a hand."""
    r = simulate_tournament_lifecycle(9, [50.0, 30.0, 20.0], chip_ev_nash(), rng=random.Random(seed),
                                      realistic=realistic)
    assert r.trace, "a 9-player tournament must produce at least one confrontation"
    for sp in r.trace:
        assert abs(sum(sp.chip_delta)) < 1e-6, (sp.regime, sum(sp.chip_delta))
        assert len(sp.chip_delta) == len(sp.pre_stacks)


# --------------------------------------------------------------------------- E4 homogeneous unbiasedness
@pytest.mark.parametrize("decide_fn,label", [
    (chip_ev_nash(), "chip_ev"),
    (anchor_fold, "anchor_fold"),
    (anchor_jam, "anchor_jam"),
])
def test_homogeneous_field_is_unbiased(decide_fn, label):
    """Hero plays the SAME decide_fn as the field -> by symmetry mean_finish ~ (N+1)/2 and roi ~ 0,
    THROUGH the full lifecycle (blind escalation + antes + balancing + breaks). The anchors are the
    harshest no-bias cases (all-fold / all-jam): a seat-0 hero privilege surfaces here first."""
    N = 18                                    # 2 tables of 9 -> a table break occurs
    payouts = [40.0, 25.0, 18.0, 10.0, 7.0]   # 5 of 18 paid
    res = measure_lifecycle_roi(N, payouts, decide_fn, n_sims=160, rng=random.Random(4),
                                realistic=False,
                                schedule_factory=lambda: LifecycleSchedule(hands_per_level=20))
    expected = (N + 1) / 2                     # 9.5
    assert abs(res["mean_finish"] - expected) < 2.0, (label, res["mean_finish"])
    assert abs(res["roi"]) < 0.45, (label, res["roi"])


def test_itm_rate_tracks_places_paid():
    """A homogeneous field's ITM rate ~ places_paid / entrants within MC noise."""
    N = 16
    payouts = [60.0, 40.0]                     # 2 of 16 paid -> ~0.125
    res = measure_lifecycle_roi(N, payouts, chip_ev_nash(), n_sims=120, rng=random.Random(2),
                                realistic=False,
                                schedule_factory=lambda: LifecycleSchedule(hands_per_level=20))
    assert abs(res["buy_in_share"] - sum(payouts) / N) < 1e-9
    assert 0.02 <= res["itm_rate"] <= 0.30, res


# --------------------------------------------------------------------------- E3 ROI + 95% CI shape
def test_measure_lifecycle_roi_reports_gross_net_and_ci():
    """E3: gross ROI uses the rake-free pool share; net ROI uses the actual (rake-laden) buy-in, so
    net < gross. The 95% CI fields are present and well-ordered (lo <= mean <= hi)."""
    N = 12
    payouts = [50.0, 30.0, 20.0]               # pool 100; fair share 100/12 ~ 8.33
    buy_in = sum(payouts) / N / 0.92           # entry incl. ~8% rake -> net share < gross share
    res = measure_lifecycle_roi(N, payouts, chip_ev_nash(), n_sims=120, rng=random.Random(9),
                                realistic=False, buy_in=buy_in,
                                schedule_factory=lambda: LifecycleSchedule(hands_per_level=20))
    lo, hi = res["mean_payout_ci95"]
    assert lo <= res["mean_payout"] <= hi
    rlo, rhi = res["roi_ci95"]
    assert rlo <= res["roi"] <= rhi
    # net ROI is below gross by exactly the rake wedge (same mean payout, larger denominator/cost).
    assert res["roi_net"] < res["roi"]
    clo, chi = res["cv_mean_ci95"]
    assert clo <= res["cv_mean"] <= chi
    assert sum(res["finish_hist"].values()) == res["n_sims"]


def test_cv_mean_recovers_mean_payout():
    """The variance-reduced score is realized_payout - icm_baseline(equal opening stacks). At equal
    stacks the baseline == the fair buy-in share, so cv_mean + buy_in_share ~ mean_payout."""
    N = 9
    payouts = [50.0, 30.0, 20.0]
    res = measure_lifecycle_roi(N, payouts, chip_ev_nash(), n_sims=150, rng=random.Random(6),
                                realistic=False,
                                schedule_factory=lambda: LifecycleSchedule(hands_per_level=20))
    recovered = res["cv_mean"] + res["buy_in_share"]
    assert abs(recovered - res["mean_payout"]) < 1e-6, (recovered, res["mean_payout"])


# --------------------------------------------------------------------------- CRN duplicate
def test_duplicate_roi_is_crn_paired():
    """Two arms on the SAME per-tournament rng are CRN-paired: identical arms -> EXACTLY zero paired
    difference (the shared variance cancels perfectly), and the result blocks are well-formed."""
    payouts = [50.0, 30.0, 20.0, 12.0, 8.0]
    arm = chip_ev_nash(jam=0.45, call=0.60)
    res = duplicate_roi(12, payouts, arm, arm, n_sims=60, rng=random.Random(11), realistic=False,
                        schedule_factory=lambda: LifecycleSchedule(hands_per_level=20))
    # identical arms vs a common villain field on identical rng -> the paired difference is exactly 0.
    assert abs(res["paired_diff"]["mean"]) < 1e-12, res["paired_diff"]
    assert res["paired_diff"]["se"] < 1e-12
    for arm_key in ("arm_a", "arm_b"):
        blk = res[arm_key]
        assert "roi" in blk and "mean_finish" in blk and "cv_mean_ci95" in blk


def test_duplicate_roi_distinct_arms_runs():
    """Distinct arms produce a finite paired-difference CI + a verdict (the E3 headline reducer)."""
    payouts = [50.0, 30.0, 20.0, 12.0, 8.0]
    res = duplicate_roi(12, payouts, chip_ev_nash(jam=0.5, call=0.66), chip_ev_nash(jam=0.45, call=0.6),
                        n_sims=80, rng=random.Random(7), realistic=False,
                        schedule_factory=lambda: LifecycleSchedule(hands_per_level=20))
    pd = res["paired_diff"]
    assert pd["ci95"][0] <= pd["mean"] <= pd["ci95"][1]
    assert pd["verdict"] in ("POSITIVE", "NEGATIVE", "INCONCLUSIVE (CI spans 0)")


# --------------------------------------------------------------------------- regime routing crosses
def test_regime_crosses_deep_to_pushfold():
    """As blinds escalate the effective depth falls through PUSHFOLD_MAX_BB, so the regime crosses
    deep -> pushfold over the lifecycle (a frozen-blind sim would be one regime forever)."""
    r = simulate_tournament_lifecycle(18, [60.0, 40.0, 30.0, 20.0, 15.0, 10.0], chip_ev_nash(),
                                      rng=random.Random(4), realistic=False,
                                      schedule=LifecycleSchedule(hands_per_level=6))
    regimes = {sp.regime for sp in r.trace}
    assert REGIME_DEEP in regimes, "early hands (200bb start) must be deep"
    assert REGIME_PUSHFOLD in regimes, "late hands (escalated blinds) must reach push/fold"


# --------------------------------------------------------------------------- icm_aware runs end-to-end
def test_icm_aware_runs_and_ladders():
    """The ICM-aware decide_fn (field-bubble-factor tightening) completes and produces a sane ITM rate
    -> the field bubble-factor path is exercised end-to-end."""
    res = measure_lifecycle_roi(12, [50.0, 30.0, 20.0], icm_aware(), chip_ev_nash(), n_sims=30,
                                rng=random.Random(7), realistic=False,
                                schedule_factory=lambda: LifecycleSchedule(hands_per_level=20))
    assert 1.0 <= res["mean_finish"] <= 12.0
    assert 0.0 <= res["itm_rate"] <= 1.0


# --------------------------------------------------------------------------- net_decide fail-safe
def test_net_decide_with_no_policy_falls_back_and_runs():
    """net_decide with policy=None ALWAYS falls back to the net-free threshold (never raises): the
    tournament completes and the trace tags the fallback source."""
    r = simulate_tournament_lifecycle(9, [50.0, 30.0, 20.0], net_decide(policy=None),
                                      rng=random.Random(3), realistic=False,
                                      schedule=LifecycleSchedule(hands_per_level=10))
    assert 1 <= r.hero_finish <= 9
    assert any("net-fallback" in sp.source for sp in r.trace)


def test_net_decide_with_stub_policy_serves():
    """A stub policy exposing serve_aggression drives the net-served threshold path (no jax needed)."""

    class StubPolicy:
        def serve_aggression(self, ctx):
            return 0.7 if ctx.regime == REGIME_DEEP else 0.55

    r = simulate_tournament_lifecycle(9, [50.0, 30.0, 20.0], net_decide(policy=StubPolicy()),
                                      rng=random.Random(2), realistic=False,
                                      schedule=LifecycleSchedule(hands_per_level=10))
    assert any("net-served" in sp.source for sp in r.trace)


def test_net_decide_policy_that_raises_is_caught():
    """A policy whose serve_aggression RAISES must NOT break the sim — it falls back safely."""

    class BadPolicy:
        def serve_aggression(self, ctx):
            raise RuntimeError("boom")

    r = simulate_tournament_lifecycle(9, [50.0, 30.0, 20.0], net_decide(policy=BadPolicy()),
                                      rng=random.Random(2), realistic=False,
                                      schedule=LifecycleSchedule(hands_per_level=10))
    assert 1 <= r.hero_finish <= 9
    assert any("net-fallback" in sp.source for sp in r.trace)


# --------------------------------------------------------------------------- SNG schedule helper
def test_sng_schedule_completes_single_table():
    res = measure_lifecycle_roi(9, [50.0, 30.0, 20.0], chip_ev_nash(), n_sims=80,
                                rng=random.Random(5), realistic=False,
                                schedule_factory=lambda: sng_schedule(hands_per_level=10))
    assert abs(res["mean_finish"] - 5.0) < 2.0   # (9+1)/2
    assert abs(res["roi"]) < 0.45


# --------------------------------------------------------------------------- ANTI-SKEW encoder fidelity
def test_net_serve_round_trip_encode_info_state_byte_identity():
    """SERVE-SKEW GUARD (the #1 correctness risk). The ``net_decide`` adapter serves the 7-max net
    through ``neural_serve.SevenMaxNeuralPolicy`` / ``seat_remap.place_live_seats``. This proves that
    placement layer reproduces the native 7-seat training node BIT-FOR-BIT, so the ABSOLUTE seat/button
    one-hots in ``encode_info_state`` match what the net saw during training (the class of bug that
    silently turns a trained net into garbage). Mirrors
    ``tests/test_seven_neural_serve.py::test_seven_remap_reproduces_native_encoding`` for 7 seats: a
    genuinely-7-live root needs the IDENTITY remap (no permutation), unlike HU's {3,5} pin."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import numpy as np

    from pokerbot.engine.jax_engine import CHIP_UNIT, encode_info_state, init_hand
    from pokerbot.tournament.seat_remap import SEVEN, place_live_seats

    N7 = SEVEN
    occ = jnp.array([True] * N7, dtype=bool)
    button = 4                                # a rotated button (not seat 0) to catch button-onehot skew
    ante = jnp.int32(round(0.125 * CHIP_UNIT))   # the CoinPoker MTT per-player ante
    native = init_hand(jax.random.PRNGKey(7), button=jnp.int32(button), occupancy=occ, ante=ante)
    assert native.seat_status.shape[0] == N7
    hero = int(native.action_seat)
    assert 0 <= hero < N7

    native_np = jax.tree.map(lambda x: np.asarray(x), native)

    # Production placement: full table -> identity remap, must round-trip back to the native node.
    back, mapping = place_live_seats(native_np, list(range(N7)))
    assert mapping == {i: i for i in range(N7)}, "full table placement must be the identity map"
    hero_back = mapping[hero]

    for f in native_np._fields:
        a = np.asarray(getattr(native_np, f))
        b = np.asarray(getattr(back, f))
        assert np.array_equal(a, b), f"field {f!r} differs after round-trip: {a} != {b}"

    enc_native = np.asarray(encode_info_state(native, jnp.int32(hero)))
    back_j = type(back)(**{f: jnp.asarray(getattr(back, f)) for f in back._fields})
    enc_back = np.asarray(encode_info_state(back_j, jnp.int32(hero_back)))
    assert np.array_equal(enc_native, enc_back), "encode_info_state skew after bridge placement"


def test_net_decide_serves_real_checkpoint_through_skew_proof_bridge():
    """If seven_v1.pkl is present (and POKERBOT_INFO_SUITS matches the process), net_decide serves the
    REAL net through the bridge and the tournament completes. Skipped when the checkpoint is absent or
    the info-suits flag mismatches (the load fails SAFE -> net-free fallback, exercised elsewhere)."""
    pytest.importorskip("jax")
    import pathlib

    from pokerbot.tournament.neural_serve import SevenMaxNeuralPolicy

    ckpt = pathlib.Path(__file__).resolve().parents[1] / "checkpoints" / "sd_cfr_7max_neural" / "seven_v1.pkl"
    if not ckpt.exists():
        pytest.skip("seven_v1.pkl checkpoint not present")
    pol = SevenMaxNeuralPolicy(checkpoint=str(ckpt))
    if not pol._ensure_loaded():
        pytest.skip("7-max net failed to load (likely POKERBOT_INFO_SUITS mismatch); fail-safe path")

    r = simulate_tournament_lifecycle(9, [50.0, 30.0, 20.0], net_decide(policy=pol),
                                      rng=random.Random(2), realistic=False,
                                      schedule=LifecycleSchedule(hands_per_level=8))
    assert 1 <= r.hero_finish <= 9
    # at least some spots should be genuinely net-served (the real net puts mass on legal actions).
    assert any("net-served" in sp.source for sp in r.trace), \
        "the real net should serve at least one spot through the bridge"
