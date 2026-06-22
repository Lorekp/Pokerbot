"""Off-tree re-solve trigger (POKERBOT_OFFTREE_RESOLVE) — Pluribus-style real-time response to an
opponent bet size the static keep-list abstraction drops.

Covers the recommended-first-slice gate:
  (1) ON + an off-tree (over-bet) villain size makes the hero root policy/EV DIFFER from the rounded-leaf
      policy — proving the true-size leaf entered the solve;
  (2) on a constructed spot, hero EV against the off-tree response is no worse (<=) than the rounded one;
  (3) fold/call legality + the "no legal action -> CHECK" fallback hold under the augmented mask;
  (4) DEFAULT path (env unset) is byte-identical to today.

The slice touches ONLY pluribus_search.py: it un-masks ONE extra in-abstraction bet bucket (the bucket the
size belongs in, which the postflop keep-list drops) at the ROOT hero infoset for that solve. NUM_ACTIONS /
BET_SIZE_BUCKETS / the trained head / the engine triad / the abstraction encoder are all untouched.
"""

from __future__ import annotations

import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

import numpy as np
import jax

from pokerbot.engine.jax_engine import NUM_ACTIONS, NUM_PLAYERS, init_hand, BET_BUCKET_START
from pokerbot.engine.actions import BET_SIZE_BUCKETS
from pokerbot.engine.np_engine import (
    CALL,
    CHECK,
    FOLD,
    apply_action_np,
    is_chance_node_np,
    legal_action_mask_np,
    sample_chance_np,
    to_numpy_state,
)
from pokerbot.search.state_bridge import SubgameRoot
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
from pokerbot.search.pluribus_search import (
    PluribusReBeLSolver,
    PluribusReBeLConfig,
    set_offtree_resolve,
    offtree_resolve_enabled,
)

_OCC = np.array([True] * 6, dtype=bool)


class _StubPBS:
    """Returns the env's own dealt opponent holes (matches tests/test_pluribus_search)."""

    def __init__(self, env, hero):
        self._h = np.asarray(env.holes)
        self.hero = hero

    def sample_opponent_holes(self, rng):
        return {s: (int(self._h[s, 0]), int(self._h[s, 1])) for s in range(NUM_PLAYERS) if s != self.hero}


def _postflop_check_node(seed: int = 7):
    """A postflop node where everyone has checked to the actor (no facing bet yet)."""
    env = to_numpy_state(init_hand(jax.random.PRNGKey(seed), occupancy=_OCC))
    for _ in range(60):
        if bool(env.done) or is_chance_node_np(env):
            break
        seat = int(env.action_seat)
        if seat < 0:
            break
        m = legal_action_mask_np(env, seat).astype(bool)
        env = apply_action_np(env, CALL if m[CALL] else CHECK)
    if is_chance_node_np(env):
        env = sample_chance_np(env)
    return env


def _facing_overbet_env(seed: int = 7, hero: int = 0, villain: int = 5, frac: float = 2.0):
    """Construct a postflop spot where ``villain`` over-bet ``frac``x pot (an OFF-TREE size the postflop
    keep-list drops: it serves up to 1.5x, never 2.0x) and ``hero`` is now first to act facing it."""
    env = _postflop_check_node(seed)
    pot_chips = int(env.pot_dead) + int(np.asarray(env.committed_street).sum())
    bet_to = int(round(pot_chips * frac))  # bet-TO (street) = frac * pot-before (no prior street bet)
    cs = np.asarray(env.committed_street).copy()
    cs[villain] = bet_to
    st = np.asarray(env.stacks).copy()
    st[villain] = max(0, int(st[villain]) - bet_to)
    acted = np.zeros(NUM_PLAYERS, dtype=bool)
    acted[villain] = True
    env = env._replace(
        committed_street=cs.astype(np.int32),
        stacks=st.astype(np.int32),
        current_bet=np.int32(bet_to),
        action_seat=np.int32(hero),
        acted=acted,
        last_aggressor=np.int32(villain),
        min_raise=np.int32(bet_to),
    )
    return env


def _solve(env, hero, *, offtree: bool, max_iterations: int = 24, seed: int = 0):
    set_offtree_resolve(offtree)
    try:
        cfg = PluribusReBeLConfig(
            max_iterations=max_iterations, time_budget_ms=10 ** 9, client_bet_sizing=True,
            search_preflop=False,
            leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2),
        )
        root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                           legal_mask=legal_action_mask_np(env, hero).astype(bool))
        sol = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(seed))
        res = sol.solve(root, _StubPBS(env, hero))
        return sol, res
    finally:
        set_offtree_resolve(False)


# --------------------------------------------------------------------------- (1) true-size leaf enters
def test_offtree_inserts_true_size_child_and_changes_policy():
    """ON + an off-tree over-bet un-masks the dropped 2.0x bucket at the hero root, so the hero's policy
    has support the rounded-leaf solve cannot have (and the served distribution differs)."""
    hero = 0
    env = _facing_overbet_env(frac=2.0, hero=hero)
    overbet_action = BET_BUCKET_START + 6  # bucket idx 6 == 2.0x pot (dropped by the static keep-list)

    sol_off, res_off = _solve(env, hero, offtree=False)
    sol_on, res_on = _solve(env, hero, offtree=True)

    # The trigger fired and the extra child is the true-size (2.0x) bucket.
    assert res_on.metadata.get("offtree_inserted") is True
    assert res_off.metadata.get("offtree_inserted") in (False, None)

    # The dropped 2.0x bucket is legal at the root ONLY in the ON solve.
    assert not bool(sol_off._legal(env, hero)[overbet_action])
    assert bool(sol_on._legal(env, hero)[overbet_action])

    # The served policy differs (the true-size leaf entered the solve), and the ON policy puts the new
    # support on a bucket the OFF policy could never use.
    assert not np.allclose(res_on.action_probs, res_off.action_probs)
    assert res_off.action_probs[overbet_action] == 0.0


def test_offtree_trigger_does_not_fire_on_in_tree_size():
    """A clean, in-abstraction villain bet (1.0x pot) must NOT insert an extra child even with the env on
    — the served menu already represents it, so the path stays the static one."""
    hero = 0
    env = _facing_overbet_env(frac=1.0, hero=hero)
    sol_on, res_on = _solve(env, hero, offtree=True)
    assert res_on.metadata.get("offtree_inserted") in (False, None)
    # And the legal set at the root equals the static keep-list legal set.
    set_offtree_resolve(False)
    static_legal = PluribusReBeLSolver({}, PluribusReBeLConfig(client_bet_sizing=True))._legal(env, hero)
    assert np.array_equal(sol_on._legal(env, hero), static_legal)


# --------------------------------------------------------------------------- (2) EV no worse
def test_offtree_response_ev_no_worse_than_rounded():
    """Hero EV under the off-tree-augmented solve is >= the rounded-only solve: adding an action can only
    expand the hero's best-response set, never shrink it (a CFR best response over a superset of actions
    has value >= the same response over the subset). Measured as the hero's realized average value of the
    served policy against the same opponent rollout (CRN: identical public seed, identical leaf stream)."""
    hero = 0
    env = _facing_overbet_env(frac=2.0, hero=hero)

    def served_ev(offtree: bool) -> float:
        sol, res = _solve(env, hero, offtree=offtree, max_iterations=48)
        probs = np.asarray(res.action_probs, dtype=np.float64)
        legal = sol._legal(env, hero)
        # Per-action continuation-gadget value of the hero's response (the same leaf the solve uses),
        # weighted by the served policy: the realized value of playing that policy here.
        q = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in np.flatnonzero(legal):
            q[a] = sol._leaf_value(apply_action_np(env, int(a)), hero)
        return float(np.dot(probs, q))

    ev_rounded = served_ev(False)
    ev_offtree = served_ev(True)
    assert ev_offtree >= ev_rounded - 1e-6


# --------------------------------------------------------------------------- (3) legality invariants
def test_fold_call_legality_preserved_under_augmented_mask():
    """The augmented mask only ADDS a bet bucket — fold/call legality facing a bet is unchanged."""
    hero = 0
    env = _facing_overbet_env(frac=2.0, hero=hero)
    sol_on, _ = _solve(env, hero, offtree=True)
    legal = sol_on._legal(env, hero)
    assert bool(legal[FOLD])   # facing a bet => fold legal
    assert bool(legal[CALL])   # facing a bet => call legal
    assert not bool(legal[CHECK])  # facing a bet => check illegal


def test_no_legal_action_falls_back_to_check_under_augmentation():
    """The engine's "no legal action -> CHECK" fallback survives the augmented mask: a folded hero seat
    (no real actions) still yields a single CHECK-legal mask, ON or OFF."""
    hero = 0
    env = _facing_overbet_env(frac=2.0, hero=hero)
    # Fold the hero out so legal_action_mask_np hits its no-action fallback (mask[CHECK]=True).
    status = np.asarray(env.seat_status).copy()
    from pokerbot.engine.np_engine import SEAT_FOLDED
    status[hero] = SEAT_FOLDED
    env = env._replace(seat_status=status)
    sol_on, _ = _solve(env, hero, offtree=True)
    legal = sol_on._legal(env, hero)
    # The abstraction ANDs with the keep-list; CHECK is always in the keep-list, so the lone CHECK
    # survives and nothing illegal is added.
    assert bool(legal[CHECK])
    assert legal.sum() == 1


# --------------------------------------------------------------------------- (4) default byte-identity
def test_default_env_unset_is_byte_identical():
    """With POKERBOT_OFFTREE_RESOLVE unset (default OFF), a solve facing the off-tree over-bet is
    byte-identical to the same solve with the feature explicitly forced off — the augmentation never
    runs and the legal set / served policy are exactly the static-abstraction ones."""
    assert offtree_resolve_enabled() is False  # default OFF at import
    hero = 0
    env = _facing_overbet_env(frac=2.0, hero=hero)

    _, res_default = _solve(env, hero, offtree=False, seed=3)
    # Re-run with the env truly never touched (simulate "unset"): set_offtree_resolve(False) above already
    # restores it; assert the metadata flag is off and the served policy matches a fresh static solve.
    cfg = PluribusReBeLConfig(max_iterations=24, time_budget_ms=10 ** 9, client_bet_sizing=True,
                              search_preflop=False,
                              leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2))
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    res_static = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(3)).solve(root, _StubPBS(env, hero))

    assert res_default.metadata.get("offtree_inserted") in (False, None)
    assert np.array_equal(res_default.action_probs, res_static.action_probs)
    # And the dropped over-bet bucket is NOT legal in the default path.
    overbet_action = BET_BUCKET_START + 6
    assert res_default.action_probs[overbet_action] == 0.0
