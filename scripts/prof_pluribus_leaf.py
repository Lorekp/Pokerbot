"""Precise profile of the Pluribus depth-limited solve: leaf-vs-traversal split, playouts/leaf,
FFI call count per leaf, and leaf-cache hit rate. Run with the serve env (see lbr6_audit docstring).

  SNAP=...pkl SEARCH_ITERS=10 LEAF=uniform python scripts/prof_pluribus_leaf.py
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import time

import numpy as np
import jax
import jax.numpy as jnp

from pokerbot.engine.jax_engine import init_hand
from pokerbot.engine.np_engine import (
    NUM_PLAYERS,
    apply_action_np,
    is_chance_node_np,
    sample_chance_np,
    to_numpy_state,
)
from pokerbot.eval.solver_exploitability import make_search_policy_fn
from pokerbot.search import native_solver as ns
from pokerbot.search import pluribus_rollout as roll

SNAP_PATH = os.environ.get(
    "SNAP", r"checkpoints/bp6max_dcfrvr_iter200.pkl"
)
ITERS = int(os.environ.get("SEARCH_ITERS", "10"))
LEAF = os.environ.get("LEAF", "uniform")
HANDS = int(os.environ.get("PROF_HANDS", "12"))
WARMUP = int(os.environ.get("PROF_WARMUP", "2"))

import pickle  # noqa: E402

raw = pickle.load(open(SNAP_PATH, "rb"))
snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
print(f"native enabled: {ns.is_enabled()}  leaf={LEAF}  iters={ITERS}  snap_infosets={len(snap)}")

act = make_search_policy_fn(snap, leaf_select=LEAF, iterations=ITERS, seed=0)

# ---- instrument: FFI count via wrapping native primitives, playout count, leaf timing ----
counters = {"ffi": 0, "playout": 0, "leaf_calls": 0, "leaf_time": 0.0}

# wrap the four primitives used inside the leaf playouts (count FFI/native dispatches)
_orig = {
    "apply": roll.apply_action_np,
    "chance": roll.sample_chance_np,
    "mask": roll.legal_action_mask_np,
    "is_chance": roll.is_chance_node_np,
}


def _wrap(fn):
    def w(*a, **k):
        counters["ffi"] += 1
        return fn(*a, **k)
    return w


roll.apply_action_np = _wrap(_orig["apply"])
roll.sample_chance_np = _wrap(_orig["chance"])
roll.legal_action_mask_np = _wrap(_orig["mask"])
roll.is_chance_node_np = _wrap(_orig["is_chance"])

_orig_playout = roll._playout


def _count_playout(*a, **k):
    counters["playout"] += 1
    return _orig_playout(*a, **k)


roll._playout = _count_playout

_orig_leaf = roll.continuation_leaf_value


def _time_leaf(*a, **k):
    counters["leaf_calls"] += 1
    t0 = time.perf_counter()
    r = _orig_leaf(*a, **k)
    counters["leaf_time"] += time.perf_counter() - t0
    return r


roll.continuation_leaf_value = _time_leaf
# re-point the solver module's imported symbol (it did `from ... import continuation_leaf_value`)
import pokerbot.search.pluribus_search as ps  # noqa: E402
ps.continuation_leaf_value = _time_leaf


def gen_states():
    key = jax.random.PRNGKey(0)
    occ = jnp.asarray([s < 6 for s in range(NUM_PLAYERS)], dtype=bool)
    rng = np.random.default_rng(0)
    states = []
    for h in range(HANDS):
        key, k = jax.random.split(key)
        env = to_numpy_state(init_hand(k, occupancy=occ))
        depth = 0
        while depth < 400 and not bool(env.done):
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            seat = int(env.action_seat)
            if seat < 0:
                break
            if int(np.asarray(env.street)) > 0 and h >= WARMUP:
                states.append((env, seat))  # postflop decisions only (preflop served from bp)
            # advance toward postflop: prefer check/call (avoid mass folding so lines reach flop+)
            from pokerbot.search.native_solver import legal_action_mask_np as lm
            from pokerbot.engine.np_engine import CHECK, CALL
            m = lm(env, seat).astype(bool)
            if m[CHECK]:
                a = CHECK
            elif m[CALL] and rng.random() < 0.85:
                a = CALL
            else:
                legal = np.flatnonzero(m)
                # avoid fold most of the time to keep multiway lines alive
                nonfold = legal[legal != 0]
                a = int(rng.choice(nonfold if nonfold.size else legal))
            env = apply_action_np(env, a)
            depth += 1
    return states


states = gen_states()
print(f"postflop decision states collected: {len(states)}")

# ---- timing + counters ----
counters.update({k: (0 if isinstance(v, int) else 0.0) for k, v in counters.items()})
total_t = 0.0
N = 0
for (env, seat) in states:
    t0 = time.perf_counter()
    act(env, seat)
    total_t += time.perf_counter() - t0
    N += 1

solve_ms = 1e3 * total_t / max(N, 1)
leaf_ms = 1e3 * counters["leaf_time"] / max(N, 1)
print("\n========== AGGREGATE OVER POSTFLOP SOLVES ==========")
print(f"decisions timed         : {N}")
print(f"per-decision solve ms   : {solve_ms:8.3f}")
print(f"per-decision leaf ms    : {leaf_ms:8.3f}  ({100*leaf_ms/max(solve_ms,1e-9):5.1f}% of solve)")
print(f"per-decision traversal  : {solve_ms - leaf_ms:8.3f} ms ({100*(solve_ms-leaf_ms)/max(solve_ms,1e-9):5.1f}%)")
print(f"leaf_value calls/decision (incl cache hits): {counters['leaf_calls']/max(N,1):8.2f}")
print(f"playouts total          : {counters['playout']}  -> /leaf-eval {counters['playout']/max(counters['leaf_calls'],1):.2f}")
print(f"FFI native dispatches IN-LEAF total : {counters['ffi']}")
print(f"FFI native dispatches /decision     : {counters['ffi']/max(N,1):8.1f}")
print(f"FFI native dispatches /playout      : {counters['ffi']/max(counters['playout'],1):8.2f}")

# leaf cache hit rate: read from the solver instance inside `act` closure (rebuild a solver to inspect)
# Instead: re-run a few solves on a fresh solver and read its counters.
from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver  # noqa: E402
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig  # noqa: E402
from pokerbot.search.pbs_tracker import PBSTracker  # noqa: E402
from pokerbot.search.state_bridge import SubgameRoot  # noqa: E402
from pokerbot.engine.cards import Card  # noqa: E402
from pokerbot.eval.solver_exploitability import _RANKS, _SUITS  # noqa: E402

cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select=LEAF), search_preflop=False)
cfg.max_iterations = ITERS
cfg.time_budget_ms = 10 ** 9
solver = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(0))


def _card(idx):
    i = int(idx)
    return Card.from_str(_RANKS[i // 4] + _SUITS[i % 4])


hits = evals = 0
for (env, seat) in states[:40]:
    hole = (env.holes[seat][0], env.holes[seat][1])
    if int(hole[0]) < 0 or int(hole[1]) < 0:
        continue
    board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
    pbs = PBSTracker.init_uniform(seat, (_card(int(hole[0])), _card(int(hole[1]))), board)
    from pokerbot.search.native_solver import legal_action_mask_np as lm
    root = SubgameRoot(env=env, hero_seat=seat,
                       pbs_vector=pbs.get_public_belief_state(),
                       legal_mask=lm(env, seat).astype(bool))
    solver.solve(root, pbs)
    hits += solver._leaf_cache_hits
    evals += solver._leaf_eval_count

print(f"\nleaf cache (40 solves): evals(unique leaves)={evals}  hits(reuse)={hits}  "
      f"hit-rate={100*hits/max(hits+evals,1):.1f}%  unique-leaves/solve={evals/40:.2f}")
