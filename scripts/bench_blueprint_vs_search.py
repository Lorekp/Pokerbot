"""Exact per-decision speed: blueprint-only lookup vs blueprint+search (Pluribus depth-limited).

Self-plays 6 blueprint copies; at EVERY acting decision times BOTH policies on the SAME state,
bucketed by street (preflop vs postflop). Warmup hands excluded so JIT compile is not counted.
Run with the serve env matching the snapshot stamp, e.g.:
  SNAP=...pkl POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_POSTFLOP_NUM_BUCKETS=200 \
      python scripts/bench_blueprint_vs_search.py
"""
from __future__ import annotations

import os
import pickle
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
from pokerbot.eval.lbr import make_blueprint_policy
from pokerbot.eval.solver_exploitability import make_search_policy_fn

SNAP_PATH = os.environ["SNAP"]
ITERS = int(os.environ.get("SEARCH_ITERS", "10"))
HANDS = int(os.environ.get("BENCH_HANDS", "40"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))

raw = pickle.load(open(SNAP_PATH, "rb"))
snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
model = make_blueprint_policy(snap)                                   # blueprint-only (dict lookup)
act = make_search_policy_fn(snap, leaf_select="uniform", iterations=ITERS, seed=0)  # search, preflop DEFERRED
act_pf = make_search_policy_fn(snap, leaf_select="uniform", iterations=ITERS, seed=0,
                               search_preflop=True)                   # search, preflop SOLVED (deployed stopgap)

rng = np.random.default_rng(0)
key = jax.random.PRNGKey(0)
occ = jnp.asarray([s < 6 for s in range(NUM_PLAYERS)], dtype=bool)

bp = {"pre": [], "post": []}   # blueprint-only per-decision seconds
se = {"pre": [], "post": []}   # search, preflop deferred
sepf = {"pre": []}             # search, preflop SOLVED (only preflop differs from se)

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
        bucket = "pre" if int(np.asarray(env.street)) == 0 else "post"
        t0 = time.perf_counter(); probs = np.asarray(model(env, seat), dtype=np.float64); t1 = time.perf_counter()
        act(env, seat); t2 = time.perf_counter()
        if bucket == "pre":
            t3 = time.perf_counter(); act_pf(env, seat); t4 = time.perf_counter()
        if h >= WARMUP:
            bp[bucket].append(t1 - t0)
            se[bucket].append(t2 - t1)
            if bucket == "pre":
                sepf["pre"].append(t4 - t3)
        s = probs.sum()
        a = int(rng.choice(len(probs), p=(probs / s) if s > 0 else None))
        env = apply_action_np(env, a)
        depth += 1


def stats(xs):
    a = np.asarray(xs) * 1e3   # ms
    return f"n={len(a):4d}  mean={a.mean():8.3f}ms  median={np.median(a):8.3f}ms  p90={np.percentile(a,90):8.3f}ms" if len(a) else "n=0"


print(f"\nsnapshot={os.path.basename(SNAP_PATH)}  search_iters={ITERS}  hands={HANDS} (warmup {WARMUP})\n")
for st in ("pre", "post"):
    print(f"[{st:4s}] blueprint-only      : {stats(bp[st])}")
    print(f"[{st:4s}] search (pf deferred): {stats(se[st])}")
    if st == "pre":
        print(f"[{st:4s}] search (PREFLOP ON) : {stats(sepf['pre'])}")
        if bp[st] and sepf["pre"]:
            r = np.mean(sepf["pre"]) / max(np.mean(bp[st]), 1e-12)
            print(f"[{st:4s}] PREFLOP-SEARCH cost : {r:8.1f}x blueprint  (adds {1e3*(np.mean(sepf['pre'])-np.mean(bp[st])):.2f} ms/preflop decision)")
    if bp[st] and se[st]:
        r = (np.mean(se[st]) / max(np.mean(bp[st]), 1e-12))
        print(f"[{st:4s}] SLOWDOWN x         : {r:8.1f}x  (search adds {1e3*(np.mean(se[st])-np.mean(bp[st])):.2f} ms/decision)")
    print()

# per-hand estimate: count decisions/hand by street from the measured stream
all_bp = bp["pre"] + bp["post"]
all_se = se["pre"] + se["post"]
dec_per_hand = (len(all_bp)) / max(HANDS - WARMUP, 1)
print(f"~decisions/hand={dec_per_hand:.1f}")
print(f"per-hand  blueprint-only : {1e3*sum(all_bp)/max(HANDS-WARMUP,1):8.1f} ms/hand")
print(f"per-hand  blueprint+srch : {1e3*sum(all_se)/max(HANDS-WARMUP,1):8.1f} ms/hand")
