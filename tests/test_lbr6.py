"""6-max Local Best Response (pokerbot.eval.lbr6) -- harness validity tests.

LBR6 is the repo's first CERTIFIED 6-max adversary: a greedy best-responder seated at a table of N-1
copies of a target policy. The lower-bound property is only meaningful if the harness actually DETECTS
exploitability, so the load-bearing test is calibration: against a trivially exploitable target (always
fold, uniform) LBR6 must come back hugely positive. These run on the fast uniform-belief path so they
stay quick; the range-tracked path is exercised structurally.
"""

from __future__ import annotations

from pokerbot.eval import lbr6
from pokerbot.eval.lbr import always_fold_policy, make_nit_policy, uniform_policy  # noqa: F401


def test_lbr6_crushes_always_fold():
    """A table that always folds hands LBR the dead blinds nearly every hand -> a large positive lower
    bound. If this is not strongly positive the harness cannot detect exploitability at all."""
    cfg = lbr6.LBR6Config(hands=120, rollouts=8, seed=0, track_ranges=False, paired=False, settle_samples=8)
    res = lbr6.lbr6_exploitability(always_fold_policy, cfg)
    assert res.bb_per_100 > 50.0, res.bb_per_100
    lo = res.bb_per_100 - 1.96 * res.stderr_bb_per_hand * 100.0
    assert lo > 0.0, (res.bb_per_100, res.stderr_bb_per_hand)  # CI excludes zero -> certified beatable


def test_lbr6_crushes_always_fold_second_seed():
    """Calibration must be SEED-ROBUST: always-fold is low-variance (no stack-offs), so a different seed
    is also strongly positive. (Uniform/nit tables stack off and are too noisy for a single-seed assert --
    that high variance is a real property of the game, not a harness defect, so we calibrate on foldbot.)"""
    cfg = lbr6.LBR6Config(hands=120, rollouts=8, seed=5, track_ranges=False, paired=False, settle_samples=8)
    res = lbr6.lbr6_exploitability(always_fold_policy, cfg)
    assert res.bb_per_100 > 50.0, res.bb_per_100


def test_lbr6_result_shape_and_paired_mode():
    """Structural: both estimator modes run, rotate the LBR seat, and report the documented fields."""
    cfg = lbr6.LBR6Config(hands=24, rollouts=4, seed=2, track_ranges=True, paired=True, settle_samples=8)
    res = lbr6.lbr6_exploitability(make_nit_policy(), cfg)
    assert res.hands == 24
    assert res.detail["paired"] is True
    assert res.detail["track_ranges"] is True
    assert res.detail["lbr_decisions"] > 0
    assert "br_raw_bb_per_100" in res.detail
    # mbb/g and bb/100 are consistent rescalings of the per-hand EV.
    assert abs(res.mbb_per_g - res.ev_bb_per_hand * 1000.0) < 1e-6
    assert abs(res.bb_per_100 - res.ev_bb_per_hand * 100.0) < 1e-6


def test_lbr6_analytic_is_default_and_crushes_leaks_with_raises():
    """The analytic 1-ply LBR (Lisy & Bowling) is the default, and its raise valuation (calling-range
    equity) must still let it CRUSH a value-payoff leak with the full menu -- a regression guard for the
    fix that stopped it over-valuing bets and bleeding vs strong opponents."""
    from pokerbot.eval.lbr import always_fold_policy
    assert lbr6.LBR6Config().analytic is True
    cfg = lbr6.LBR6Config(hands=120, seed=0, paired=False, settle_samples=8)  # default full menu, analytic
    res = lbr6.lbr6_exploitability(always_fold_policy, cfg)
    assert res.detail["method"] == "analytic_1ply_lbr"
    assert res.bb_per_100 > 50.0, res.bb_per_100  # steals dead blinds via the (correct) raise value


def test_lbr6_two_policy_split_acts_via_policy_fn_not_model_fn():
    """The two-policy split must drive the OPPONENTS' real actions from ``policy_fn`` (ACT) and use only
    ``model_fn`` for the LBR's internal belief/fold reads. Proof: opponents ACT by always folding while the
    MODEL is uniform -- if the code wrongly used model_fn for opponent actions they would play uniform (and
    not fold), so a strongly-positive result confirms the seats acted via policy_fn (the dead blinds)."""
    cfg = lbr6.LBR6Config(hands=120, seed=0, track_ranges=True, paired=False, settle_samples=8)
    res = lbr6.lbr6_exploitability(always_fold_policy, cfg, model_fn=uniform_policy)
    # Strongly positive => opponents folded (acted via policy_fn); if the code wrongly drove opponent
    # ACTIONS from the uniform model_fn they would not fold and the edge would collapse toward 0. The
    # belief track now correctly uses model_fn (uniform), so this is lower than a policy_fn-belief run
    # would give -- the magnitude is the WHITE-BOX (uniform) belief's blind-steal, not a fold-tracked one.
    lo = res.bb_per_100 - 1.96 * res.stderr_bb_per_hand * 100.0
    assert lo > 0.0 and res.bb_per_100 > 25.0, (res.bb_per_100, res.stderr_bb_per_hand)


def test_lbr6_value_only_raises_still_extracts_value_from_a_station():
    """value_only_raises (the default vs un-modellable SEARCH) drops fold-equity credit (fp=0) so the LBR
    can never suicide-bluff -- but it must STILL crush a calling station by value-betting/calling, since the
    station is exactly the 'always called' world the value-only valuation assumes."""
    import numpy as np
    from pokerbot.engine.np_engine import CALL, CHECK, FOLD, NUM_ACTIONS, legal_action_mask_np

    def station(env, seat):
        mask = legal_action_mask_np(env, seat).astype(bool)
        p = np.zeros(NUM_ACTIONS, dtype=np.float64)
        p[CALL if mask[CALL] else (CHECK if mask[CHECK] else FOLD)] = 1.0
        return p

    cfg = lbr6.LBR6Config(hands=120, seed=3, track_ranges=True, paired=False, settle_samples=16,
                          value_only_raises=True)
    res = lbr6.lbr6_exploitability(station, cfg)
    assert res.bb_per_100 > 50.0, res.bb_per_100  # value extracted without any fold-equity gambling


def test_lbr6_exploit_table_observes_and_hooks_fire():
    """The deployed-stack harness: ExploitTable.observe must fold the LBR's decisions into its tracked
    profile via the real pipeline (so a read builds), and the lbr6 observe_fn/on_hand_start hooks must be
    invoked. Uses an empty snapshot (solver falls back to blueprint) so it stays fast -- this is a wiring
    test, not a search/EV test."""
    import jax
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import CALL, CHECK, NUM_PLAYERS, legal_action_mask_np, to_numpy_state
    from pokerbot.eval.lbr6_vs_exploit import ExploitTable

    table = ExploitTable({}, iterations=1, seed=0)   # empty snapshot -> blueprint fallback (fast)
    occ = jnp.asarray([s < 6 for s in range(NUM_PLAYERS)], dtype=bool)
    env = to_numpy_state(init_hand(jax.random.PRNGKey(0), occupancy=occ))
    seat = int(env.action_seat)
    table.on_hand_start(seat)                         # this hand's LBR seat == the actor we observe
    mask = legal_action_mask_np(env, seat).astype(bool)
    table.observe(env, seat, CALL if mask[CALL] else CHECK)
    rs = table.read_summary()
    assert rs["hands_seen"] == 1
    assert rs["n_ctx_keys"] >= 1, rs          # a context key was recorded from the observed decision
    # observing a DIFFERENT seat must NOT record (only the LBR-seat player is tracked)
    other = next(s for s in range(6) if s != seat)
    before = rs["n_ctx_keys"]
    table.observe(env, other, CALL)
    assert table.read_summary()["n_ctx_keys"] == before


def test_lbr6_all_in_ev_settle_reduces_variance_vs_uniform_table():
    """All-in EV settlement (averaging runouts) must not INCREASE variance on a spew-heavy table; it should
    reduce or match the single-runout stderr. Same seed/hands so the comparison is apples-to-apples."""
    base = dict(hands=60, rollouts=4, seed=7, track_ranges=False, paired=False)
    hi = lbr6.lbr6_exploitability(uniform_policy, lbr6.LBR6Config(settle_samples=1, **base))   # no settle
    lo = lbr6.lbr6_exploitability(uniform_policy, lbr6.LBR6Config(settle_samples=48, **base))  # settle
    assert lo.stderr_bb_per_hand <= hi.stderr_bb_per_hand + 1e-9, (lo.stderr_bb_per_hand,
                                                                   hi.stderr_bb_per_hand)
