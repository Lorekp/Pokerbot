"""Tests for the ICM-aware 7-seat Local Best Response (pokerbot/tournament/lbr_icm.py).

What this proves
----------------
* E1 -- ``icm_lbr_exploitability`` measures a $-exploitability surface: a positive headline means an
  adversary CAN extract $ from a table of the target; it carries a CI, a per-seat breakdown, and BOTH
  chip-EV and $-EV valuations of the same run.
* Anchor recovery -- the always-fold target recovers ``sevenmax_tournament_anchor()`` in CHIPS (within the
  documented harness-overhead band) and its $ analogue (within ICM-curvature tolerance), because the
  always-steal best response scoops the dead money every hand.
* E2 -- where ICM binds (a bubble cell), the chip-LBR exploitability >= the $-LBR exploitability: the ICM
  risk premium suppresses the $-value of chips a spewing adversary wins.
* C1 -- at a SATELLITE seat-locked cell a chip-spewing (always-jam) adversary's $-EV at the locked seat is
  <= 0 (within CI): you cannot profit in $ by spewing chips when every paid finish is one identical seat.
* C0 -- the $-leaf is pool-conserving (max per-hand |sum over all seats| ~ 0) for NORMAL/SATELLITE.

THE LOAD-BEARING TEST is ``test_icm_lbr_root_reproduces_native_encoding``: the ICM-LBR serves the trained
7-max net on its OWN 7-seat root (occupancy [True]*7, button rotating, the MTT-ante deal). It MUST be
byte-identical to the native ``cfr_env.nlhe_7max_env_spec`` training node on every EnvState field AND
produce a byte-identical ``encode_info_state`` -- otherwise a silent seat/button one-hot skew turns the
served net into garbage. This mirrors ``tests/test_seven_neural_serve.py`` /
``tests/test_hu_neural_endgame.py``.

Smoke (WSL Debian):
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu python -m pytest tests/test_lbr_icm.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.eval.lbr6 import LBR6Config
from pokerbot.tournament import lbr_icm as L
from pokerbot.tournament.anchor import sevenmax_tournament_anchor
from pokerbot.tournament.payouts import TournamentFormat

# Small CPU-friendly configs (the 7-max GPU training holds the GPU; these are net-free anchors on CPU jax).
_NORMAL_PAY = [50.0, 30.0, 20.0]

# The $-scoring core (``eval_score``) transitively imports a numba dependency (rollout_leaf -> sim ->
# aof_equity_grids). ``import lbr_icm`` itself is numba-free (deferred), and the load-bearing ENCODER
# FIDELITY test needs only jax -- so it always runs. The tests that actually EXECUTE a $-LBR run need the
# numba-equipped env; skip them gracefully where numba is absent (e.g. the lean WSL jax venv) rather than
# erroring, so the suite stays green in both environments.
try:
    from pokerbot.tournament import eval_score as _eval_score_probe  # noqa: F401
    _HAS_SCORING = True
except Exception as _e:  # noqa: BLE001
    _HAS_SCORING = False
    _SCORING_ERR = repr(_e)

_needs_scoring = pytest.mark.skipif(
    not _HAS_SCORING,
    reason="eval_score/$-scoring core unavailable (likely missing numba in this env); "
           "encoder-fidelity test still runs",
)


def _fast_cfg(hands: int, seed: int = 0, **kw) -> LBR6Config:
    base = dict(hands=hands, seed=seed, settle_samples=6, eq_samples=10, eq_runouts=3, range_cap=24)
    base.update(kw)
    return LBR6Config(**base)


# --------------------------------------------------------------------------- the anti-skew fidelity test
def test_icm_lbr_root_reproduces_native_encoding():
    """The ICM-LBR's OWN 7-seat root (occupancy [True]*7, rotated button, the MTT-ante deal) must be
    byte-identical to the native training node AND produce a byte-identical ``encode_info_state`` -- the
    serve layout the trained 7-max net saw. A skew here silently turns the served net into garbage.

    The ICM-LBR deals exactly as ``cfr_env.nlhe_7max_env_spec`` does (init_hand with occupancy=[True]*7,
    a rotated button, ante=round(ante_bb*CHIP_UNIT), per-seat chip stacks), so a genuinely-7-live root is
    the IDENTITY layout -- no permutation. We rebuild that same root here and assert the seat/button
    one-hots land on the natural absolute indices.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import CHIP_UNIT, encode_info_state, init_hand
    from pokerbot.engine.np_engine import to_numpy_state

    N7 = L.SEVEN
    occ = jnp.array([True] * N7, dtype=bool)
    button = 4  # a rotated button (not seat 0) to catch any button-onehot skew
    ante = jnp.int32(round(0.125 * CHIP_UNIT))
    chip_stacks = jnp.full((N7,), int(round(25.0 * CHIP_UNIT)), dtype=jnp.int32)

    native = init_hand(jax.random.PRNGKey(7), button=jnp.int32(button), stacks=chip_stacks,
                       occupancy=occ, ante=ante)
    assert native.seat_status.shape[0] == N7, "the ICM-LBR root is a genuinely 7-wide table"
    hero = int(native.action_seat)
    assert 0 <= hero < N7

    # The ICM-LBR converts to a numpy EnvState (the np_engine path lbr6 + this module run on) and serves the
    # net on THAT, with the hero's ABSOLUTE seat -- no remap (the full-table identity case).
    env_np = to_numpy_state(native)
    for f in native._fields:
        a = np.asarray(getattr(native, f))
        b = np.asarray(getattr(env_np, f))
        assert np.array_equal(a, b), f"to_numpy_state changed field {f!r}: {a} != {b}"

    # encode_info_state on the np-rebuilt jax state must equal the native node's encoding (byte-identical).
    rebuilt = type(native)(**{f: jnp.asarray(getattr(env_np, f)) for f in env_np._fields})
    enc_native = np.asarray(encode_info_state(native, jnp.int32(hero)))
    enc_rebuilt = np.asarray(encode_info_state(rebuilt, jnp.int32(hero)))
    assert np.array_equal(enc_native, enc_rebuilt), "encode_info_state skew between native and served root"

    # The seat/button one-hots are on the natural absolute indices (button=4, hero is one of the 7 seats).
    assert int(rebuilt.button) == button


def test_root_deals_all_seven_live_with_ante():
    """The ICM-LBR root deals ALL 7 seats live (occupancy [True]*7) with the MTT ante -- the training
    layout. (A regression guard: if the deal silently dropped a seat or the ante, the anchor would
    undershoot.)"""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand, SEAT_EMPTY
    from pokerbot.engine.np_engine import to_numpy_state

    N7 = L.SEVEN
    occ = jnp.array([True] * N7, dtype=bool)
    env = to_numpy_state(init_hand(jax.random.PRNGKey(1), button=jnp.int32(0),
                                   stacks=jnp.full((N7,), 25 * CHIP_UNIT, dtype=jnp.int32),
                                   occupancy=occ, ante=jnp.int32(round(0.125 * CHIP_UNIT))))
    status = np.asarray(env.seat_status)
    assert status.shape[0] == N7
    assert (status != int(SEAT_EMPTY)).all(), "all 7 seats must be live (training layout)"
    # antes are in: every seat committed at least the ante (0.125 bb = round(0.125*100)=12 chips).
    assert (np.asarray(env.committed_total) >= round(0.125 * CHIP_UNIT)).all()


# --------------------------------------------------------------------------- net-free anchor policies
def test_always_fold_policy_mass_on_fold_or_check():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pokerbot.engine.actions import NUM_ACTIONS
    from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand
    from pokerbot.engine.np_engine import FOLD, legal_action_mask_np, to_numpy_state

    occ = jnp.array([True] * 7, dtype=bool)
    env = to_numpy_state(init_hand(jax.random.PRNGKey(0), button=jnp.int32(0),
                                   stacks=jnp.full((7,), 25 * CHIP_UNIT, dtype=jnp.int32), occupancy=occ))
    seat = int(env.action_seat)
    p = L.always_fold_policy(env, seat)
    assert p.shape[0] == NUM_ACTIONS
    assert abs(float(p.sum()) - 1.0) < 1e-9
    mask = np.asarray(legal_action_mask_np(env, seat), dtype=bool)
    assert (p[~mask] == 0.0).all(), "no mass on illegal actions"
    # facing the blinds preflop FOLD is legal -> a folder folds.
    if mask[FOLD]:
        assert p[FOLD] == 1.0


def test_always_jam_policy_jams_when_legal():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand
    from pokerbot.engine.np_engine import ALL_IN, legal_action_mask_np, to_numpy_state

    occ = jnp.array([True] * 7, dtype=bool)
    env = to_numpy_state(init_hand(jax.random.PRNGKey(0), button=jnp.int32(0),
                                   stacks=jnp.full((7,), 25 * CHIP_UNIT, dtype=jnp.int32), occupancy=occ))
    seat = int(env.action_seat)
    p = L.always_jam_policy(env, seat)
    mask = np.asarray(legal_action_mask_np(env, seat), dtype=bool)
    assert (p[~mask] == 0.0).all()
    if mask[ALL_IN]:
        assert p[ALL_IN] == 1.0


# --------------------------------------------------------------------------- E1 + anchor recovery (in $)
@_needs_scoring
def test_always_fold_recovers_chip_anchor_within_band():
    """The always-fold target's chip-LBR exploitability recovers ``sevenmax_tournament_anchor()`` (+203.6
    bb/100) within the harness-overhead band: the always-steal BR scoops the dead money every hand; the
    only shortfall is the SB/BB seats that already posted (so cannot steal their own blind)."""
    cfg = _fast_cfg(hands=147, seed=0, stack_bb=100.0)  # deal+price at 100bb deep (ICM ~ chip-linear there)
    res = L.icm_lbr_exploitability(L.always_fold_policy, cfg, _NORMAL_PAY, [100.0] * 7,
                                   fmt=TournamentFormat.NORMAL)
    anchor = sevenmax_tournament_anchor()  # +203.57 bb/100
    # The 7-max harness LBR recovers most of the formula vs a pure folder (the blinds-posted seats are the
    # documented residual). A generous band: within ~20 bb/100 below the formula, and clearly positive.
    assert res.chip_exploit_bb_per_100 > 0.0
    assert anchor - 30.0 <= res.chip_exploit_bb_per_100 <= anchor + 5.0, (
        f"chip anchor recovery {res.chip_exploit_bb_per_100:.1f} outside band around {anchor:.1f}")


@_needs_scoring
def test_always_fold_dollar_recovers_anchor_dollar_target():
    """The $-LBR exploitability of always-fold matches the always-fold anchor expressed in $ (the chip
    anchor mapped through ICM at the cell's pre-stacks) within ICM-curvature tolerance, and is POSITIVE
    with a CI excluding 0."""
    pre = [100.0] * 7
    cfg = _fast_cfg(hands=196, seed=0, stack_bb=100.0)
    res = L.icm_lbr_exploitability(L.always_fold_policy, cfg, _NORMAL_PAY, pre,
                                   fmt=TournamentFormat.NORMAL, buy_in=100.0)
    target = L.anchor_dollar_target(_NORMAL_PAY, pre)
    assert res.dollar_exploit_per_game > 0.0
    assert res.ci95[0] > 0.0, "always-fold is exploitable in $ (CI excludes 0)"
    assert res.verdict == "POSITIVE"
    # The linear $ target ignores ICM curvature near the steal; allow a 40% band around it.
    rel = abs(res.dollar_exploit_per_game - target) / max(target, 1e-9)
    assert rel < 0.4, f"$ recovery {res.dollar_exploit_per_game:.4f} far from target {target:.4f} (rel {rel:.2f})"
    assert res.pct_of_buyin is not None and res.pct_of_buyin > 0.0


@_needs_scoring
def test_result_carries_both_valuations_and_per_seat():
    cfg = _fast_cfg(hands=42, seed=1, stack_bb=100.0)
    res = L.icm_lbr_exploitability(L.always_fold_policy, cfg, _NORMAL_PAY, [100.0] * 7,
                                   fmt=TournamentFormat.NORMAL)
    # E1 headline scalings are internally consistent.
    assert abs(res.mbb_of_dollar_per_game - res.dollar_exploit_per_game * 1000.0) < 1e-9
    # both chip + $ valuations present (E2), and the per-seat breakdown covers all 7 seats.
    assert res.chip_exploit_bb_per_100 != 0.0
    assert set(res.per_seat.keys()) == set(range(7))
    for s, d in res.per_seat.items():
        assert d["n"] > 0
        assert "dollar_per_game" in d and "chip_bb_per_100" in d
    assert res.detail["n_seats"] == 7


# --------------------------------------------------------------------------- C0 leaf conservation
@_needs_scoring
def test_leaf_conservation_zero_for_normal():
    """C0: the $-leaf is pool-conserving for NORMAL (max per-hand |sum over all 7 seats| ~ 0)."""
    cfg = _fast_cfg(hands=63, seed=0, stack_bb=40.0)
    res = L.icm_lbr_exploitability(L.always_fold_policy, cfg, _NORMAL_PAY, [40.0] * 7,
                                   fmt=TournamentFormat.NORMAL)
    assert res.leaf_conservation < 1e-6, f"non-conserving $-leaf: {res.leaf_conservation}"


@_needs_scoring
def test_leaf_conservation_zero_for_satellite():
    seats, seat_val = 4, 10.0
    sat_pay = [seat_val] * seats
    pre = [50.0, 50.0, 50.0, 50.0, 5.0, 5.0, 5.0]
    cell = L.ICMCell(fmt=TournamentFormat.SATELLITE, payouts=tuple(sat_pay), pre_stacks=tuple(pre),
                     stack_bb=20.0)
    cfg = _fast_cfg(hands=63, seed=0)
    res = L.icm_lbr_exploitability(L.always_jam_policy, cfg, sat_pay, pre, cell=cell)
    assert res.leaf_conservation < 1e-6, f"non-conserving SATELLITE $-leaf: {res.leaf_conservation}"


# --------------------------------------------------------------------------- E2 chip >= $ where ICM binds
@_needs_scoring
def test_chip_lbr_ge_dollar_lbr_where_icm_binds():
    """E2: at a BUBBLE cell (a short stack one elimination from the money) a chip-spewing always-jam
    adversary wins CHIPS on average, but the ICM risk premium suppresses the $-value of those chips, so the
    chip-LBR exploitability >= the $-LBR exploitability (both on the same scale x100)."""
    pre = [40.0, 40.0, 40.0, 5.0, 1e-4, 1e-4, 1e-4]  # 3 big + 1 short = on the bubble (3 paid)
    cell = L.ICMCell(fmt=TournamentFormat.NORMAL, payouts=tuple(_NORMAL_PAY), pre_stacks=tuple(pre),
                     stack_bb=15.0)
    cfg = _fast_cfg(hands=98, seed=0)
    res = L.icm_lbr_exploitability(L.always_jam_policy, cfg, _NORMAL_PAY, pre, cell=cell)
    chip_scaled = res.chip_exploit_bb_per_100
    dollar_scaled = res.dollar_exploit_per_game * 100.0
    assert chip_scaled >= dollar_scaled - 1e-6, (
        f"ICM must suppress $ below chips at the bubble: chip {chip_scaled:.2f} < $ {dollar_scaled:.2f}")
    # And the chip-LBR is meaningfully larger (the spewer wins chips it cannot bank as $-equity).
    assert chip_scaled > dollar_scaled


# --------------------------------------------------------------------------- C1 satellite seat-lock
@_needs_scoring
def test_satellite_seat_lock_jam_adversary_cannot_profit():
    """C1: at a SATELLITE where the hero seat already holds a seat-equivalent (locked) stack, a
    chip-spewing always-jam adversary at that seat has $-EV <= 0 (within CI) -- every paid finish is one
    identical seat, so risking chips is pure downside. The locked seat's per-seat $/game must be <= 0."""
    seats, seat_val = 4, 10.0
    sat_pay = [seat_val] * seats
    # 4 locked giants (each already holds a seat) + 3 dust seats. The LBR at a locked giant cannot gain.
    pre = [200.0, 200.0, 200.0, 200.0, 1.0, 1.0, 1.0]
    cell = L.ICMCell(fmt=TournamentFormat.SATELLITE, payouts=tuple(sat_pay), pre_stacks=tuple(pre),
                     stack_bb=20.0)
    cfg = _fast_cfg(hands=98, seed=0)
    res = L.icm_lbr_exploitability(L.always_jam_policy, cfg, sat_pay, pre, cell=cell, seat_locked=0)
    # overall $-exploitability cannot be significantly positive (CI lower bound <= 0).
    assert res.ci95[0] <= 1e-6, f"a locked-seat jam adversary should not profit in $: ci {res.ci95}"
    # the specifically-locked seat's mean $/game is <= 0 (it can only lose $-equity by spewing).
    assert res.per_seat[0]["seat_locked"] is True
    assert res.per_seat[0]["dollar_per_game"] <= 1e-6, (
        f"locked seat $/game {res.per_seat[0]['dollar_per_game']:.4f} should be <= 0 (C1 violated)")


# --------------------------------------------------------------------------- mean-preserving leaf
def test_allin_showdown_equity_settle_is_zero_sum_and_averaged():
    """The terminal all-in showdown re-settle is mean-preserving (equity-averaged over board completions,
    NOT one engine runout) and zero-sum. A 7-way preflop jam settles to FRACTIONAL per-seat shares (an
    averaged equity vector), not a single winner's clean +(n-1)*stack / -(stack) split, and the per-seat
    deltas sum to ~0 (the chip game is zero-sum)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand
    from pokerbot.engine.np_engine import to_numpy_state

    occ = jnp.array([True] * 7, dtype=bool)
    stacks = jnp.full((7,), 20 * CHIP_UNIT, dtype=jnp.int32)
    ante = jnp.int32(round(0.125 * CHIP_UNIT))
    env0 = to_numpy_state(init_hand(jax.random.PRNGKey(5), button=jnp.int32(0), stacks=stacks,
                                    occupancy=occ, ante=ante))
    cfg = LBR6Config(hands=1, settle_samples=200)
    cfg.n_seats = 7
    leaf, lbr_bb, nd, info = L._play_hand_capturing_leaf(
        env0, 0, L.always_jam_policy, cfg, rng=np.random.default_rng(1),
        model_fn=L.always_jam_policy, settle_seed=11)
    assert abs(float(leaf.sum())) < 1e-6, "all-in showdown settle must be zero-sum"
    # mean-preserving => NOT the single-runout integer split (every seat -20 except one +120); at least one
    # seat carries a fractional (non-+-20bb-multiple) equity share.
    assert not np.allclose(np.abs(leaf), 20.0), "settle should be equity-averaged, not a single runout"


def test_foldout_leaf_is_exact_engine_rewards():
    """A non-showdown fold-out reads the engine's exact terminal rewards (no resampling): the always-fold
    table folds to the button, which wins the exact dead money, and the leaf is zero-sum."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand
    from pokerbot.engine.np_engine import to_numpy_state

    occ = jnp.array([True] * 7, dtype=bool)
    stacks = jnp.full((7,), 25 * CHIP_UNIT, dtype=jnp.int32)
    ante = jnp.int32(round(0.125 * CHIP_UNIT))
    env0 = to_numpy_state(init_hand(jax.random.PRNGKey(6), button=jnp.int32(0), stacks=stacks,
                                    occupancy=occ, ante=ante))
    cfg = LBR6Config(hands=1, settle_samples=64)
    cfg.n_seats = 7
    leaf, lbr_bb, nd, info = L._play_hand_capturing_leaf(
        env0, 0, L.always_fold_policy, cfg, rng=np.random.default_rng(1),
        model_fn=L.always_fold_policy, settle_seed=7)
    assert abs(float(leaf.sum())) < 1e-9
    assert info["folded"] is False and lbr_bb > 0.0  # button steals the dead money


# --------------------------------------------------------------------------- config / cell plumbing
@_needs_scoring
def test_pko_scores_prize_side_equal_to_normal():
    """PKO is scored on the PRIZE side only (the progressive-bounty uplift is not yet credited in the LBR
    leaf -- a documented PARTIAL). So a PKO cell's $-exploitability equals the same NORMAL cell's, and the
    result flags ``pko_bounty_credited=False``."""
    pre = [30.0, 30.0, 30.0, 30.0, 5.0, 5.0, 5.0]
    pay = (40.0, 24.0, 16.0)
    cfg = _fast_cfg(hands=49, seed=0)
    pko = L.icm_lbr_exploitability(
        L.always_jam_policy, cfg, list(pay), pre,
        cell=L.ICMCell(fmt=TournamentFormat.PKO, payouts=pay, pre_stacks=tuple(pre), stack_bb=15.0))
    nrm = L.icm_lbr_exploitability(
        L.always_jam_policy, cfg, list(pay), pre,
        cell=L.ICMCell(fmt=TournamentFormat.NORMAL, payouts=pay, pre_stacks=tuple(pre), stack_bb=15.0))
    assert abs(pko.dollar_exploit_per_game - nrm.dollar_exploit_per_game) < 1e-9
    assert pko.detail["pko_bounty_credited"] is False


def test_pre_stacks_must_be_seven_wide():
    cfg = _fast_cfg(hands=4)
    with pytest.raises(ValueError):
        L.icm_lbr_exploitability(L.always_fold_policy, cfg, _NORMAL_PAY, [100.0] * 6,
                                 fmt=TournamentFormat.NORMAL)


@_needs_scoring
def test_cell_overrides_explicit_args():
    """An explicit ``cell`` is the single source of the spot: its fmt/payouts/pre_stacks win over the
    positional args, and its deal depth/ante drive the root."""
    cell = L.ICMCell(fmt=TournamentFormat.NORMAL, payouts=(10.0, 6.0, 4.0),
                     pre_stacks=tuple([30.0] * 7), stack_bb=30.0, ante_bb=0.125)
    cfg = _fast_cfg(hands=21, seed=0)
    # pass DIFFERENT positional args; the cell must win.
    res = L.icm_lbr_exploitability(L.always_fold_policy, cfg, [999.0], [1.0] * 7,
                                   fmt=TournamentFormat.SATELLITE, cell=cell)
    assert res.detail["fmt"] == "normal"
    assert res.detail["payouts"] == [10.0, 6.0, 4.0]
    assert res.detail["pre_stacks"] == [30.0] * 7
    assert res.detail["stack_bb"] == 30.0


@_needs_scoring
def test_n_seats_forced_to_seven():
    """Even if the caller hands a 6-seat LBR6Config, the tournament driver forces a 7-seat table."""
    cfg = _fast_cfg(hands=14, seed=0, n_seats=6, stack_bb=50.0)
    res = L.icm_lbr_exploitability(L.always_fold_policy, cfg, _NORMAL_PAY, [50.0] * 7,
                                   fmt=TournamentFormat.NORMAL)
    assert res.detail["n_seats"] == 7
    assert cfg.n_seats == 7
    assert set(res.per_seat.keys()) == set(range(7))


def test_anchor_dollar_target_positive_and_scales_with_payouts():
    """The $-anchor target is positive and scales with the prize pool (bigger ladder -> bigger $ steal)."""
    pre = [100.0] * 7
    small = L.anchor_dollar_target([5.0, 3.0, 2.0], pre)
    big = L.anchor_dollar_target([500.0, 300.0, 200.0], pre)
    assert small > 0.0 and big > 0.0
    assert big > small * 50.0  # 100x larger pool -> ~100x larger $ steal (linear in pool)
