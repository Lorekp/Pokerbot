#!/usr/bin/env python3
"""PARITY GATE: native settled showdown-leaf value == Python settled value, EXACT.

The SETTLE_SHOWDOWN_LEAF fix routes both engines through the trainer's terminal_return semantics
(n_show = LIVE|ALLIN contenders -> compute_returns), so parity is inherited from the certified
bit-identical compute_returns port. This script proves it end-to-end on a driven corpus:

  1. RANDOM-HANDS corpus: N random 6-max hands driven with a check/call-heavy stochastic policy
     (with occasional river jams) on the PURE-PYTHON np_engine until they land in the
     unsettled-showdown class (street=STREET_TERMINAL, done=False, stale rewards=0).
  2. For every such state: python settle (settle_showdown_value_np, per seat) vs native settle
     (battery_native.compute_returns with the SAME n_show rule) — EXACT equality required, plus
     zero-sum and the stale-zero repro assertions.
  3. ENGINE-LOCKSTEP: the same action sequences replayed through the NATIVE engine wrappers
     (native_solver dispatch) must land in a bit-identical terminal state (holes/board/status/
     committed/button), so the two engines settle from identical inputs by construction.
  4. LIVE-CORPUS states (read-only from the MAIN tree, mirrors track2_s1_ksweep's capture):
     artifacts/search_roots.pkl states driven check/call to the unsettled class, settled both ways.

Exit 0 = all exact. Any mismatch prints the state and exits 1.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
MAIN = Path(".")
sys.path.insert(0, str(WORKTREE))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pokerbot.engine.jax_engine import init_hand  # noqa: E402
from pokerbot.engine import np_engine as npe  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    ALL_IN, CALL, CHECK, SEAT_ALLIN, SEAT_LIVE, STREET_RIVER,
    is_unsettled_showdown_np, settle_showdown_value_np, to_numpy_state,
)
from pokerbot.search import native_solver as ns  # noqa: E402

N_RANDOM = int(os.environ.get("PARITY_N", "400"))
SEED0 = 20260710


def drive(env, rng, jam_p=0.15, max_steps=250):
    """Drive with pure-python np_engine: mostly check/call, occasional river jam. Returns
    (final_env, actions) where actions is the exact applied sequence (for native replay)."""
    actions = []
    for _ in range(max_steps):
        if bool(env.done):
            return env, actions
        if npe.is_chance_node_np(env):
            env = npe.sample_chance_np(env)
            continue
        seat = int(env.action_seat)
        if seat < 0:
            return env, actions
        m = npe.legal_action_mask_np(env, seat)
        a = None
        if int(env.street) == int(STREET_RIVER) and m[ALL_IN] and rng.random() < jam_p:
            a = ALL_IN
        elif m[CALL] and rng.random() < 0.9:
            a = CALL
        elif m[CHECK]:
            a = CHECK
        elif m[CALL]:
            a = CALL
        else:
            a = int(np.flatnonzero(m)[0])
        actions.append(a)
        env = npe.apply_action_np(env, a)
    return env, actions


def native_replay(seed_key, actions):
    """Replay the exact action sequence through the NATIVE engine wrappers (ns dispatch)."""
    occ = jnp.array([True] * 6)
    env = to_numpy_state(init_hand(jax.random.PRNGKey(seed_key), occupancy=occ))
    it = iter(actions)
    for _ in range(400):
        if bool(env.done):
            return env
        if ns.is_chance_node_np(env):
            env = ns.sample_chance_np(env)
            continue
        if int(env.action_seat) < 0:
            return env
        try:
            a = next(it)
        except StopIteration:
            return env
        env = ns.apply_action_np(env, a)
    return env


def settle_native(ext, env):
    st = np.asarray(env.seat_status)
    n_show = int(((st == SEAT_LIVE) | (st == SEAT_ALLIN)).sum())
    return [float(x) for x in ext.compute_returns(
        [int(x) for x in np.asarray(env.holes).reshape(-1)],
        [int(x) for x in np.asarray(env.board)],
        [int(x) for x in st],
        [int(x) for x in np.asarray(env.committed_total)],
        int(env.button), n_show,
    )]


def main():
    assert ns.available(), "battery_native unavailable (build the worktree pyd first)"
    ext = ns._EXT
    caps = set(ext.native_caps())
    assert "settle_showdown_leaf" in caps, f"stale pyd (caps={sorted(caps)})"
    assert ext.settle_showdown_leaf_enabled() is False, "native gate must default OFF"
    assert npe.settle_showdown_leaf_enabled() is False, "python gate must default OFF"

    occ = jnp.array([True] * 6)
    n_unsettled = n_settled = n_checked = 0
    max_abs = 0.0
    for i in range(N_RANDOM):
        seed = SEED0 + i
        rng = np.random.default_rng(seed)
        env0 = to_numpy_state(init_hand(jax.random.PRNGKey(seed), occupancy=occ))
        ns.disable()  # pure-python drive
        env, actions = drive(env0, rng)
        if bool(env.done):
            n_settled += 1
            continue
        if not is_unsettled_showdown_np(env):
            continue
        n_unsettled += 1
        # stale-zero repro
        assert all(int(r) == 0 for r in np.asarray(env.rewards)), f"seed {seed}: rewards not stale-0"
        # python settle
        py = [settle_showdown_value_np(env, s) for s in range(6)]
        assert abs(sum(py)) < 1e-9, f"seed {seed}: python settle not zero-sum: {py}"
        # native settle — EXACT
        nat = settle_native(ext, env)
        if nat != py:
            print(f"MISMATCH seed {seed}: native={nat} python={py}")
            return 1
        max_abs = max(max_abs, max(abs(v) for v in py))
        # engine lockstep on a subsample (native replay is marshaling-heavy)
        if n_unsettled % 10 == 1:
            ns.enable()
            env_n = native_replay(seed, actions)
            ns.disable()
            for f in ("street", "done", "button"):
                assert np.asarray(getattr(env, f)).tolist() == np.asarray(getattr(env_n, f)).tolist(), \
                    f"seed {seed}: lockstep field {f} diverged"
            for f in ("holes", "board", "seat_status", "committed_total", "rewards"):
                assert np.array_equal(np.asarray(getattr(env, f)), np.asarray(getattr(env_n, f))), \
                    f"seed {seed}: lockstep field {f} diverged"
            nat2 = settle_native(ext, env_n)
            assert nat2 == py, f"seed {seed}: native-driven settle diverged"
            n_checked += 1

    print(f"random corpus: {N_RANDOM} hands -> {n_unsettled} unsettled-showdown states, "
          f"{n_settled} engine-settled; {n_checked} native-lockstep replays; "
          f"max |settle| = {max_abs:.0f} chips — ALL EXACT")

    # live-corpus states (read-only from MAIN): drive check/call to the unsettled class.
    try:
        roots = pickle.load(open(MAIN / "artifacts" / "search_roots.pkl", "rb"))
    except Exception as e:  # noqa: BLE001
        print(f"live corpus skipped: {e!r}")
        roots = []
    n_live = 0
    for k, rec in enumerate(roots[:40]):
        env = rec.get("env") if isinstance(rec, dict) else getattr(rec, "env", None)
        if env is None:
            continue
        rng = np.random.default_rng(999_000 + k)
        ns.disable()
        env, _ = drive(env, rng, jam_p=0.0)
        if bool(env.done) or not is_unsettled_showdown_np(env):
            continue
        py = [settle_showdown_value_np(env, s) for s in range(6)]
        nat = settle_native(ext, env)
        if nat != py:
            print(f"MISMATCH live corpus state {k}: native={nat} python={py}")
            return 1
        assert abs(sum(py)) < 1e-9
        n_live += 1
    print(f"live corpus: {n_live} unsettled-showdown states settled — ALL EXACT")
    print("PARITY: PASS (native == python, exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
