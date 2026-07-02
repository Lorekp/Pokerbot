#!/usr/bin/env python3
"""PREFLOP-SEARCH ON vs OFF head-to-head (duplicate dealing + ALL-IN EV settle).

The deployment question: given the deck-repair + two-sided FIXES (which cured the one-sided preflop
over-jam), is SEARCHING preflop now a stronger full-game engine than serving the blueprint preflop
(Pluribus-style)? Both arms run the IDENTICAL native two-sided subgame solver (repair+decouple+range ON,
same K, same streams); the ONLY difference is:

  * ARM "ON"  (A):  search_preflop = True   -- search every street incl. preflop
  * ARM "OFF" (B):  search_preflop = False  -- blueprint preflop, search postflop only (the DEPLOYED default)

3 ON + 3 OFF seats alternate around a 6-max table; each deck played TWICE with arms swapped (card luck
cancels). ALL-IN EV settlement integrates the equity split over runouts. If ON wins (CI excludes 0), the
fixes made preflop search worth its extra compute -> flip search_preflop back ON. If OFF wins or ties,
keep blueprint preflop (cheaper and at-least-as-good).

Run from the native worktree (its patched .pyd loads):
  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 \
  POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS=200 \
  H2H_DECKS=400 H2H_ITERS=2000 ENS_STREAMS=1 \
  python scripts/preflop_onoff_h2h.py <worktree> <main>
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax


def main():
    WT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    MAIN = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    sys.path.insert(0, WT)

    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.search.pluribus_search import set_native_subgame
    from pokerbot.search import native_solver as _ns
    from pokerbot.eval.solver_exploitability import make_search_policy_fn
    import pickle

    if not _ns.native_subgame_available():
        _ns.enable()
    print(f"native available={_ns.available()} subgame_available={_ns.native_subgame_available()} "
          f"rayon={_ns._EXT.rayon_num_threads()}", flush=True)

    settle_samples = int(os.environ.get("H2H_SETTLE", "200"))

    def _nboard(env):
        return int((np.asarray(env.board) >= 0).sum())

    def ev_settle_via_action(prev_env, action, rng):
        ptr = int(prev_env.deck_ptr)
        base_deck = np.asarray(prev_env.deck).copy()
        tail = base_deck[ptr:].copy()
        if tail.shape[0] == 0:
            return np.asarray(apply_action_np(prev_env, action).rewards, dtype=np.float64)
        acc = np.zeros(6, dtype=np.float64)
        for _ in range(settle_samples):
            rng.shuffle(tail)
            d = base_deck.copy(); d[ptr:] = tail
            e = apply_action_np(prev_env._replace(deck=d), action)
            guard = 0
            while not is_terminal_np(e) and guard < 8:
                if is_chance_node_np(e):
                    e = sample_chance_np(e)
                else:
                    break
                guard += 1
            acc += np.asarray(e.rewards, dtype=np.float64)
        return acc / settle_samples

    n_decks = int(os.environ.get("H2H_DECKS", "400"))
    K = int(os.environ.get("H2H_ITERS", "2000"))
    streams = int(os.environ.get("ENS_STREAMS", "1"))
    repair = os.environ.get("H2H_REPAIR", "1") != "0"
    two = os.environ.get("H2H_TWOSIDED", "1") != "0"
    floor = float(os.environ.get("H2H_FLOOR", "0.25"))
    base = int(os.environ.get("H2H_SEED", "4242"))
    bp_path = os.environ.get("H2H_BLUEPRINT", f"{MAIN}/checkpoints/serve_blueprint_6max.pkl")
    out = os.environ.get("H2H_OUT", f"{MAIN}/artifacts/preflop_onoff_h2h.json")

    snap = (lambda o: o["strat_sum"] if isinstance(o, dict) and "strat_sum" in o else o)(
        pickle.load(open(bp_path, "rb")))
    occ = jax.numpy.array([True] * 6)

    def make_pf(preflop_on: bool):
        pf = make_search_policy_fn(snap, leaf_select="uniform", seed=base, iterations=K,
                                   search_preflop=preflop_on)
        sv = [c.cell_contents for c in pf.__closure__ if hasattr(getattr(c, "cell_contents", None), "config")][0]
        sv.config.native = True
        sv.config.repair_deck = repair
        sv.config.two_sided = two
        sv.config.blueprint_mix = 1.0
        sv.config.decouple_sampler = True
        sv.config.range_aware = True
        sv.config.range_uniform_floor = float(floor)
        sv.config.ensemble_streams = int(streams)
        return pf

    pf_on, pf_off = make_pf(True), make_pf(False)
    print(f"PREFLOP ON vs OFF  decks={n_decks} K={K} streams={streams} repair={repair} two_sided={two} "
          f"floor={floor} (3 ON vs 3 OFF, {len(snap)} bp)", flush=True)

    def act(is_on, env, seat, salt):
        set_native_subgame(True)
        p = np.asarray((pf_on if is_on else pf_off)(env, seat), dtype=np.float64)
        pc = np.clip(p, 0, None)
        s = pc.sum()
        if s <= 0:
            m = legal_action_mask_np(env, seat).astype(np.float64)
            pc = m / m.sum()
        else:
            pc = pc / s
        return int(np.random.default_rng(salt).choice(len(pc), p=pc))

    on_total = 0.0
    off_total = 0.0
    per_deck = []
    t0 = time.perf_counter()
    for d in range(n_decks):
        deck_on = 0.0
        deck_off = 0.0
        for rot in (0, 1):
            is_on = {s: ((s % 2 == 0) == (rot == 0)) for s in range(6)}
            env = to_numpy_state(init_hand(jax.random.PRNGKey(base * 100003 + d), occupancy=occ))
            settle_rng = np.random.default_rng(base * 991 + d * 61 + rot)
            prev_env, prev_action, prev_nboard = None, None, 0
            g = 0
            while not is_terminal_np(env) and g < 400:
                g += 1
                if is_chance_node_np(env):
                    env = sample_chance_np(env)
                    continue
                seat = current_player_np(env)
                if seat < 0:
                    break
                mask = legal_action_mask_np(env, seat).astype(bool)
                if int(mask.sum()) <= 1:
                    a = int(np.argmax(mask))
                else:
                    a = act(is_on[int(seat)], env, int(seat), base * 7 + d * 131 + g)
                prev_env, prev_action, prev_nboard = env, a, _nboard(env)
                env = apply_action_np(env, a)
            if settle_samples > 0 and prev_env is not None and _nboard(env) > prev_nboard:
                rew = ev_settle_via_action(prev_env, prev_action, settle_rng)
            else:
                rew = np.asarray(env.rewards, dtype=np.float64)
            for s in range(6):
                if is_on[s]:
                    deck_on += float(rew[s])
                else:
                    deck_off += float(rew[s])
        on_total += deck_on
        off_total += deck_off
        per_deck.append(deck_on - deck_off)
        el = time.perf_counter() - t0
        pd_arr = np.array(per_deck)
        se = float(pd_arr.std(ddof=1) / math.sqrt(len(pd_arr))) if len(pd_arr) > 1 else 0.0
        print(f"  deck {d+1}/{n_decks}: on-off net {deck_on-deck_off:+.1f} | cum on {on_total:+.0f}/off "
              f"{off_total:+.0f} | mean/deck {pd_arr.mean():+.2f}+-{1.96*se:.2f} | {el/60:.1f}min "
              f"ETA {el/(d+1)*(n_decks-d-1)/60:.1f}min", flush=True)
        json.dump({"decks_done": d + 1, "K": K, "streams": streams, "repair": repair, "two_sided": two,
                   "floor": floor, "on_total": on_total, "off_total": off_total, "per_deck_net": per_deck},
                  open(out, "w"), indent=1)

    pd = np.array(per_deck)
    mean = float(pd.mean())
    se = float(pd.std(ddof=1) / math.sqrt(len(pd))) if len(pd) > 1 else 0.0
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    print(f"\n=== PREFLOP ON vs OFF (paired by deck, {len(pd)} decks x 2 rotations) ===", flush=True)
    print(f"  K={K} streams={streams} two_sided={two} repair={repair} floor={floor}")
    print(f"  ON total:   {on_total:+.1f} chips")
    print(f"  OFF total:  {off_total:+.1f} chips")
    print(f"  ON - OFF per deck: {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (per deck = 2 rotations)")
    verdict = ("PREFLOP-SEARCH-ON STRONGER (CI excludes 0)" if lo > 0 else
               ("BLUEPRINT-PREFLOP(OFF) STRONGER (CI excludes 0)" if hi < 0 else "INCONCLUSIVE (CI includes 0)"))
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
