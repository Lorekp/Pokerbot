"""Push/fold ICM ROOT prior (pokerbot/search/pushfold_prior.py) + its search wiring.

Pins: (1) spot detection from a real EnvState; (2) the prior reproduces the certified open_shove_icm /
HU push/fold ranges (range-sanity: position monotone, ante widens, HU reduction exact); (3) the prior
abstains (None) outside the certified push/fold regime; (4) the PluribusReBeLConfig default
(pushfold_prior_band_bb=None) is byte-identical (the prior never fires) — additive, no 6-max change.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from pokerbot.engine import jax_engine as je  # noqa: E402
from pokerbot.engine.jax_engine import ALL_IN, CALL, FOLD  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    to_numpy_state, legal_action_mask_np, apply_action_np,
)
from pokerbot.search.pushfold_prior import (  # noqa: E402
    detect_pushfold_spot, pushfold_root_prior,
)
from pokerbot.tournament.pushfold_icm import _hu_grid  # noqa: E402

_, _LABELS, _W = _hu_grid()


def _root(n, depth_bb=10.0, ante_bb=0.0, key=5, button=0):
    st = je.init_hand(
        jax.random.PRNGKey(key), button=jnp.int32(button),
        stacks=jnp.full(n, int(depth_bb * je.CHIP_UNIT), dtype=jnp.int32),
        occupancy=jnp.ones(n, dtype=bool),
        ante=jnp.int32(int(ante_bb * je.CHIP_UNIT)), n_players=n,
    )
    return to_numpy_state(st)


def _range_pct(env, hero, ante=0.0, iters=800):
    from pokerbot.search.pushfold_prior import detect_pushfold_spot as _d
    from pokerbot.tournament.open_shove_icm import solve_open_shove_icm
    spot = _d(env, hero)
    stacks_bb = ((np.asarray(env.stacks, float) + np.asarray(env.committed_total, float))
                 / je.CHIP_UNIT).tolist()
    sol = solve_open_shove_icm(stacks_bb, [1.0, 0.0], hero, list(spot["behind"]),
                               blinds=spot["blinds"], ante=ante, iters=iters)
    return float((sol.hero_jam * _W).sum() / _W.sum())


def test_detect_open_shove_7handed():
    env = _root(7)
    hero = int(np.asarray(env.action_seat))  # UTG
    spot = detect_pushfold_spot(env, hero)
    assert spot is not None and spot["kind"] == "open_shove"
    assert hero not in spot["behind"] and len(spot["behind"]) == 6


def test_prior_is_two_action_and_in_band():
    env = _root(7, depth_bb=10.0)
    hero = int(np.asarray(env.action_seat))
    legal = legal_action_mask_np(env, hero).astype(bool)
    prior = pushfold_root_prior(env, hero, legal, band_bb=(0.0, 20.0), iters=400)
    assert prior is not None
    # Mass only on ALL_IN (jam) and FOLD; sums to 1.
    assert abs(prior.sum() - 1.0) < 1e-9
    assert prior[ALL_IN] + prior[FOLD] > 0.999
    assert prior[CALL] == 0.0


def test_position_monotone_utg_tighter_than_button():
    env = _root(7, depth_bb=10.0)
    utg = int(np.asarray(env.action_seat))
    utg_pct = _range_pct(env, utg)
    # fold around to the button (only SB+BB behind => widest first-in)
    e = env
    for _ in range(4):
        e = apply_action_np(e, FOLD)
    btn = int(np.asarray(e.action_seat))
    btn_pct = _range_pct(e, btn)
    assert btn_pct > utg_pct + 0.05, f"button {btn_pct:.3f} should be wider than UTG {utg_pct:.3f}"


def test_ante_widens_range():
    no_ante = _range_pct(_root(7, depth_bb=10.0, ante_bb=0.0),
                         int(np.asarray(_root(7).action_seat)))
    env_a = _root(7, depth_bb=10.0, ante_bb=0.25)
    with_ante = _range_pct(env_a, int(np.asarray(env_a.action_seat)), ante=0.25)
    assert with_ante > no_ante + 0.02, f"ante {with_ante:.3f} should widen vs {no_ante:.3f}"


def test_hu_reduction_matches_certified_solver():
    from pokerbot.tournament.pushfold_icm import solve_hu_icm_pushfold
    env = _root(2, depth_bb=10.0, key=9)
    hero = int(np.asarray(env.action_seat))  # SB / button acts first HU
    spot = detect_pushfold_spot(env, hero)
    other = spot["behind"][0]
    from pokerbot.tournament.open_shove_icm import solve_open_shove_icm
    os_pct = float((solve_open_shove_icm([10.0, 10.0], [1.0, 0.0], hero, [other],
                                         blinds=spot["blinds"], ante=0.0, iters=2000).hero_jam * _W
                    ).sum() / _W.sum())
    hu = solve_hu_icm_pushfold([10.0, 10.0], [1.0, 0.0], hero, other, iters=4000)
    assert abs(os_pct - hu.sb_jam_pct) < 0.02, f"open-shove {os_pct:.3f} != certified HU {hu.sb_jam_pct:.3f}"


def test_abstains_deep_stack():
    env = _root(7, depth_bb=60.0)  # well above the push/fold band
    hero = int(np.asarray(env.action_seat))
    legal = legal_action_mask_np(env, hero).astype(bool)
    assert pushfold_root_prior(env, hero, legal, band_bb=(0.0, 20.0)) is None


def test_abstains_postflop():
    env = _root(7, depth_bb=10.0)
    # Move off preflop by faking the street (postflop is never a push/fold spot).
    env = env._replace(street=np.int32(1))
    hero = int(np.asarray(env.action_seat))
    assert detect_pushfold_spot(env, hero) is None


def test_config_default_is_disabled_byte_identical():
    """Default PluribusReBeLConfig has the prior OFF (band None) => _root_blueprint never calls it,
    so the 6-max / non-tournament path is byte-identical (additive)."""
    from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
    cfg = PluribusReBeLConfig()
    assert cfg.pushfold_prior_band_bb is None
    solver = PluribusReBeLSolver({}, config=cfg)
    env = _root(7, depth_bb=10.0)
    hero = int(np.asarray(env.action_seat))
    legal = legal_action_mask_np(env, hero).astype(bool)
    # With the band disabled the prior helper returns None (the uniform/blueprint fallback stands).
    assert solver._pushfold_prior(env, hero, legal) is None


def test_solver_serves_pushfold_prior_when_enabled():
    """With the band enabled, a shallow 7-handed preflop solve serves the certified push/fold prior
    (metadata 'pushfold_icm_prior') instead of uniform-over-legal search."""
    from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
    from pokerbot.search.state_bridge import SubgameRoot
    from pokerbot.search.pbs_tracker import PBSTracker
    cfg = PluribusReBeLConfig(pushfold_prior_band_bb=(0.0, 20.0))
    solver = PluribusReBeLSolver({}, config=cfg)
    # 6-handed shallow root (PBSTracker beliefs are NUM_PLAYERS=6-wide); the push/fold prior serve
    # returns BEFORE any PBS sampling, so the tracker is only a placeholder here.
    env = _root(6, depth_bb=10.0)
    hero = int(np.asarray(env.action_seat))
    legal = legal_action_mask_np(env, hero).astype(bool)
    pbs = PBSTracker(hero_seat=hero)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=pbs.get_public_belief_state(),
                       legal_mask=legal)
    res = solver.solve(root, pbs)
    assert res.metadata.get("served") == "pushfold_icm_prior"
    probs = np.asarray(res.action_probs)
    assert probs[ALL_IN] + probs[FOLD] > 0.999  # jam/fold only
