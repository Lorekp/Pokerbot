#!/usr/bin/env python3
"""Gate 2: per-leaf statistical equivalence of the native uniform leaf value vs Python
`continuation_leaf_value(select="uniform")`. Both estimate the SAME expectation; we compare the
mean over MANY independent seeds and report |Delta| vs the combined standard error.

Usage: python -m tools.battery_native.crosscheck_playout <blueprint.pkl>
"""
from __future__ import annotations

import os
import pickle
import random
import sys

os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_SEARCH_PREFLOP", "1")
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

import numpy as np

from pokerbot.search import native_solver as ns
from pokerbot.search.native_solver import _to_flat
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig, continuation_leaf_value
from pokerbot.search.tabular_blueprint_adapter import (
    DEFAULT_BIAS_MULT,
    TabularBlueprintAdapter,
)
from pokerbot.engine.np_engine import (
    apply_action_np as _apply_py,
    is_chance_node_np as _chance_py,
    legal_action_mask_np as _legal_py,
    sample_chance_np as _sample_py,
    to_numpy_state,
)
from pokerbot.engine.jax_engine import init_hand
from pokerbot.training.sd_cfr_nlhe import build_allowed_action_mask
import jax

NUM_BUCKETS = int(os.environ["POKERBOT_POSTFLOP_NUM_BUCKETS"])


def load_blueprint(path: str) -> dict:
    with open(path, "rb") as f:
        full = pickle.load(f)
    snap = full["strat_sum"] if isinstance(full, dict) and "strat_sum" in full else full
    # Normalize rows to probabilities (the adapter's _policy does legal-renorm at lookup; the native
    # engine likewise renormalizes over the abstracted legal mask, so pass the RAW strat_sum rows).
    return snap


def build_native_engine(snap: dict):
    allowed_pre = build_allowed_action_mask((4, 5, 6))
    allowed_post = build_allowed_action_mask((0, 1, 3, 4, 5))
    # Native expects dict[bytes16 -> list[float] len 11]; pass raw rows (engine renormalizes).
    bp = {bytes(k): [float(x) for x in np.asarray(v, dtype=np.float64).reshape(-1)[:11]]
          for k, v in snap.items()}
    eng = ns._EXT.PlayoutEngine(
        bp,
        [bool(x) for x in allowed_pre],
        [bool(x) for x in allowed_post],
        float(DEFAULT_BIAS_MULT),
        0.0,
        64,
        NUM_BUCKETS,
    )
    return eng


def collect_leaves(n: int, depth_target: int = 6) -> list:
    """Real depth-limit-ish leaf states: play random hands, stop at a postflop acting node."""
    rng = random.Random(99)
    leaves = []
    while len(leaves) < n:
        key = jax.random.PRNGKey(rng.randint(0, 10**9))
        env = to_numpy_state(init_hand(key, button=rng.randint(0, 5)))
        steps = 0
        while not bool(env.done) and steps < 40:
            if _chance_py(env):
                env = _sample_py(env)
                continue
            seat = int(env.action_seat)
            if seat < 0:
                break
            # snapshot a leaf at a few different depths/streets
            if steps >= depth_target and not bool(env.done) and int(env.action_seat) >= 0:
                leaves.append(env)
                break
            mask = _legal_py(env, seat)
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            env = _apply_py(env, int(rng.choice(list(legal))))
            steps += 1
    return leaves[:n]


def main(path: str) -> int:
    snap = load_blueprint(path)
    print(f"blueprint: {len(snap)} infosets")
    adapter = TabularBlueprintAdapter(snapshot=snap, client_bet_sizing=True,
                                      bias_mult=DEFAULT_BIAS_MULT, bias_floor=0.0)
    eng = build_native_engine(snap)
    print(f"native engine keys: {eng.n_keys()}")

    leaves = collect_leaves(12)
    hero = 0
    n_seeds = 240
    rollout_samples = 4
    worst = 0.0
    for li, env in enumerate(leaves):
        py_vals = []
        nat_vals = []
        sb = _to_flat(env).tobytes()
        for sd in range(n_seeds):
            seed = 1_000 + 7919 * sd
            cfg = ContinuationLeafConfig(select="uniform", rollout_samples=rollout_samples,
                                         max_playout_depth=64, base_seed=seed, resample_board=True)
            py_vals.append(continuation_leaf_value(env, hero, adapter, cfg, cont_weights=None))
            nat_vals.append(eng.leaf_value(sb, hero, seed, rollout_samples, True))
        py = np.array(py_vals); nat = np.array(nat_vals)
        py_m, nat_m = py.mean(), nat.mean()
        # SE of the difference of two independent means.
        se = float(np.sqrt(py.var(ddof=1) / len(py) + nat.var(ddof=1) / len(nat)))
        d = float(py_m - nat_m)
        z = d / se if se > 1e-12 else 0.0
        worst = max(worst, abs(z))
        print(f"leaf {li:2d} street={int(env.street)} seat={int(env.action_seat)} "
              f"py={py_m:+8.3f} nat={nat_m:+8.3f} dEV={d:+7.3f} se={se:6.3f} z={z:+5.2f}")
    print(f"max |z| over leaves = {worst:.2f}  (<~3 => statistically equivalent)")
    ok = worst < 4.0
    print("GATE2", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else r"checkpoints/bp6max_lossless2_iter200.pkl"
    sys.exit(main(p))
