"""SCRATCH latency harness (no production edits): per-preflop-decision wall time, search ON vs OFF.

Drives representative 6-max PREFLOP decision nodes through the SAME deployed search path
(make_search_policy_fn, search-iters=10, uniform leaf) the live bot uses, and times policy_fn:
  - search_preflop=OFF -> preflop served from blueprint dict (one lookup) = the lossless-blueprint path
  - search_preflop=ON  -> one depth-limited CFR+ solve per preflop node = the stopgap path
The delta is the preflop latency the lossless retrain SAVES by letting search_preflop be turned off.

Env must be set before pokerbot import (see lbr6_quality_probe.py header). Uses the OLD/deployed
blueprint (the latency delta is solve-vs-lookup, ~blueprint-independent).

  python scripts/neural_cfr/lbr6_preflop_latency.py --snapshot <pkl> --nodes 60 --search-iters 10
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import init_hand
from pokerbot.engine.np_engine import (NUM_PLAYERS, SEAT_LIVE, apply_action_np, is_chance_node_np,
                                        legal_action_mask_np, sample_chance_np, to_numpy_state)
from pokerbot.eval.lbr import check_blueprint_serve_env, make_blueprint_policy
from pokerbot.eval.solver_exploitability import make_search_policy_fn


def _collect_preflop_nodes(n_nodes: int, n_seats: int, bp_policy, seed: int) -> list:
    """Walk hands on np_engine, sampling distinct PREFLOP hero-to-act nodes (street==0). Opponents
    play the blueprint so the node distribution matches real preflop spots (opens/calls/3bets)."""
    occ = jnp.asarray([s < n_seats for s in range(NUM_PLAYERS)], dtype=bool)
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    nodes = []
    while len(nodes) < n_nodes:
        key, k = jax.random.split(key)
        env = to_numpy_state(init_hand(k, occupancy=occ))
        depth = 0
        while depth < 60 and not bool(env.done):
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            seat = int(env.action_seat)
            if seat < 0:
                break
            if int(np.asarray(env.street)) == 0:           # preflop hero-to-act node: capture it
                nodes.append((env, seat))
                if len(nodes) >= n_nodes:
                    break
            # advance with a blueprint-sampled action to reach varied preflop spots
            p = bp_policy(env, seat)
            mask = legal_action_mask_np(env, seat).astype(bool)
            p = p * mask
            if p.sum() <= 1e-12:
                a = int(np.argmax(mask))
            else:
                a = int(rng.choice(len(p), p=p / p.sum()))
            env = apply_action_np(env, a)
            depth += 1
    return nodes[:n_nodes]


def _time_policy(policy_fn, nodes, warmup: int = 3) -> list:
    times_ms = []
    for i, (env, seat) in enumerate(nodes):
        t0 = time.perf_counter()
        policy_fn(env, seat)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:           # drop JIT/first-call warmup
            times_ms.append(dt)
    return times_ms


def _summ(label, ts):
    mean = statistics.mean(ts)
    med = statistics.median(ts)
    p95 = sorted(ts)[int(0.95 * (len(ts) - 1))]
    print(f"  {label:28s} mean {mean:8.2f} ms   median {med:8.2f} ms   p95 {p95:8.2f} ms   (n={len(ts)})")
    return mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--n-seats", type=int, default=6)
    ap.add_argument("--search-iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.snapshot, "rb") as f:
        snap_full = pickle.load(f)
    stamp = snap_full.get("abstraction") if isinstance(snap_full, dict) else None
    check_blueprint_serve_env(strict=True, expected=stamp)
    snap = (snap_full["strat_sum"]
            if isinstance(snap_full, dict) and "strat_sum" in snap_full else snap_full)
    print(f"snapshot: {args.snapshot}  ({len(snap)} infosets, search_iters={args.search_iters})")

    bp = make_blueprint_policy(snap)
    nodes = _collect_preflop_nodes(args.nodes, args.n_seats, bp, args.seed)
    print(f"collected {len(nodes)} preflop hero-to-act nodes\n")

    off = make_search_policy_fn(snap, leaf_select="uniform", iterations=args.search_iters,
                                seed=args.seed, search_preflop=False)
    on = make_search_policy_fn(snap, leaf_select="uniform", iterations=args.search_iters,
                               seed=args.seed, search_preflop=True)

    print("per-preflop-decision wall time through the deployed search path:")
    t_off = _time_policy(off, nodes)
    t_on = _time_policy(on, nodes)
    m_off = _summ("search_preflop=OFF (lookup)", t_off)
    m_on = _summ("search_preflop=ON  (solve)", t_on)
    print(f"\n  DELTA (ON - OFF): {m_on - m_off:+.2f} ms/preflop-decision   ({m_on / max(m_off,1e-9):.1f}x)")
    print("  => latency the lossless blueprint SAVES by letting search_preflop be turned OFF.")


if __name__ == "__main__":
    main()
