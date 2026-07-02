#!/usr/bin/env python3
"""DEPLOYED-REGIME ensemble A/B head-to-head (blueprint preflop + POSTFLOP-only ensemble search).

Same duplicate-dealing + ALL-IN-EV-settle full-game H2H as ensemble_ab_search_h2h.py, but run in the
regime we actually SHIP: search_preflop=OFF, so preflop is served from the blueprint (identical for both
arms, no ensemble effect) and the ONLY place the ensemble bites is POSTFLOP solves -- exactly where
subgames are big/slow and single-stream is starved. This answers the shippable question: does packing more
POSTFLOP search into ~1x wall-clock (via N independent parallel CFR streams) win chips in the full game?

  * ARM "NEW" (A):  ensemble_streams = ENS_STREAMS (default 8)
  * ARM "OLD" (B):  ensemble_streams = 1

Both arms: native two-sided subgame solver, repair+decouple+range ON @ floor 0.25 (the intended shipping
correctness config). 3 NEW + 3 OLD seats alternate; each deck played twice with arms swapped (card luck
cancels). Because most 6-max pots are decided PREFLOP (blueprint, no search) many decks net ~0 -- the
signal only lives in flop+ hands, so run MORE decks than the search-everywhere variant.

Run from the native worktree (its patched .pyd loads):
  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 \
  POKERBOT_POSTFLOP_MODE=emd POKERBOT_POSTFLOP_NUM_BUCKETS=160 \
  POKERBOT_POSTFLOP_EMD_FROZEN=<abs>/data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn \
  POKERBOT_EMD_FROZEN_MIRROR_FALLBACK=1 \
  H2H_DECKS=250 H2H_ITERS=2000 ENS_STREAMS=8 SEARCH_PREFLOP=0 \
  python scripts/ensemble_deployed_h2h.py <worktree> <main>
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
    print(f"native available={_ns.available()} enabled={_ns.is_enabled()} "
          f"subgame_available={_ns.native_subgame_available()} rayon={_ns._EXT.rayon_num_threads()}", flush=True)

    settle_samples = int(os.environ.get("H2H_SETTLE", "300"))
    search_preflop = os.environ.get("SEARCH_PREFLOP", "0") != "0"

    def _nboard(env):
        return int((np.asarray(env.board) >= 0).sum())

    def _street(env):
        return int(env.street)

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

    n_decks = int(os.environ.get("H2H_DECKS", "250"))
    K = int(os.environ.get("H2H_ITERS", "2000"))
    streams = int(os.environ.get("ENS_STREAMS", "8"))
    repair = os.environ.get("H2H_REPAIR", "1") != "0"
    two = os.environ.get("H2H_TWOSIDED", "1") != "0"
    floor = float(os.environ.get("H2H_FLOOR", "0.25"))
    base = int(os.environ.get("H2H_SEED", "4242"))
    bp_path = os.environ.get("H2H_BLUEPRINT", f"{MAIN}/checkpoints/serve_blueprint_6max.pkl")
    out = os.environ.get("H2H_OUT", f"{MAIN}/artifacts/ensemble_deployed_h2h.json")

    snap = (lambda o: o["strat_sum"] if isinstance(o, dict) and "strat_sum" in o else o)(
        pickle.load(open(bp_path, "rb")))
    occ = jax.numpy.array([True] * 6)

    def make_pf(new_arm: bool):
        pf = make_search_policy_fn(snap, leaf_select="uniform", seed=base, iterations=K,
                                   search_preflop=search_preflop)
        sv = [c.cell_contents for c in pf.__closure__ if hasattr(getattr(c, "cell_contents", None), "config")][0]
        sv.config.native = True
        sv.config.repair_deck = repair
        sv.config.two_sided = two
        sv.config.blueprint_mix = 1.0
        sv.config.decouple_sampler = True
        sv.config.range_aware = True
        sv.config.range_uniform_floor = float(floor)
        sv.config.ensemble_streams = int(streams) if new_arm else 1
        return pf

    pf_new, pf_old = make_pf(True), make_pf(False)
    print(f"DEPLOYED-REGIME ENSEMBLE A/B  decks={n_decks} K={K}/stream NEW_streams={streams} OLD_streams=1 "
          f"search_preflop={search_preflop} repair={repair} two_sided={two} floor={floor} "
          f"(3 NEW vs 3 OLD, {len(snap)} bp)", flush=True)

    postflop_decisions = 0  # count decisions that actually hit a solve (street>0 or search_preflop) for signal audit

    def act(is_new, env, seat, salt):
        set_native_subgame(True)
        p = np.asarray((pf_new if is_new else pf_old)(env, seat), dtype=np.float64)
        pc = np.clip(p, 0, None)
        s = pc.sum()
        if s <= 0:
            m = legal_action_mask_np(env, seat).astype(np.float64)
            pc = m / m.sum()
        else:
            pc = pc / s
        return int(np.random.default_rng(salt).choice(len(pc), p=pc))

    new_total = 0.0
    old_total = 0.0
    per_deck = []
    reached_flop_decks = 0
    n_games = 0
    t0 = time.perf_counter()
    for d in range(n_decks):
        deck_new = 0.0
        deck_old = 0.0
        deck_saw_flop = False
        for rot in (0, 1):
            is_new = {s: ((s % 2 == 0) == (rot == 0)) for s in range(6)}
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
                if _street(env) > 0:
                    deck_saw_flop = True
                    postflop_decisions += 1
                mask = legal_action_mask_np(env, seat).astype(bool)
                if int(mask.sum()) <= 1:
                    a = int(np.argmax(mask))
                else:
                    a = act(is_new[int(seat)], env, int(seat), base * 7 + d * 131 + g)
                prev_env, prev_action, prev_nboard = env, a, _nboard(env)
                env = apply_action_np(env, a)
            if settle_samples > 0 and prev_env is not None and _nboard(env) > prev_nboard:
                rew = ev_settle_via_action(prev_env, prev_action, settle_rng)
            else:
                rew = np.asarray(env.rewards, dtype=np.float64)
            for s in range(6):
                if is_new[s]:
                    deck_new += float(rew[s])
                else:
                    deck_old += float(rew[s])
            n_games += 1
        if deck_saw_flop:
            reached_flop_decks += 1
        new_total += deck_new
        old_total += deck_old
        per_deck.append(deck_new - deck_old)
        el = time.perf_counter() - t0
        pd_arr = np.array(per_deck)
        se = float(pd_arr.std(ddof=1) / math.sqrt(len(pd_arr))) if len(pd_arr) > 1 else 0.0
        print(f"  deck {d+1}/{n_decks}: new-old net {deck_new-deck_old:+.1f} | cum new {new_total:+.0f}/old "
              f"{old_total:+.0f} | mean/deck {pd_arr.mean():+.2f}+-{1.96*se:.2f} | flopdecks {reached_flop_decks}"
              f" pfdec {postflop_decisions} | {el/60:.1f}min "
              f"ETA {el/(d+1)*(n_decks-d-1)/60:.1f}min", flush=True)
        json.dump({"decks_done": d + 1, "K": K, "streams": streams, "repair": repair, "two_sided": two,
                   "floor": floor, "search_preflop": search_preflop, "reached_flop_decks": reached_flop_decks,
                   "postflop_decisions": postflop_decisions,
                   "new_total": new_total, "old_total": old_total, "per_deck_net": per_deck},
                  open(out, "w"), indent=1)

    pd = np.array(per_deck)
    mean = float(pd.mean())
    se = float(pd.std(ddof=1) / math.sqrt(len(pd))) if len(pd) > 1 else 0.0
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    print(f"\n=== DEPLOYED-REGIME ENSEMBLE A/B (paired by deck, {len(pd)} decks x 2 rotations) ===", flush=True)
    print(f"  search_preflop={search_preflop}  K={K}/stream  NEW_streams={streams} vs OLD_streams=1  "
          f"two_sided={two} repair={repair} floor={floor}")
    print(f"  decks reaching flop: {reached_flop_decks}/{n_decks}  postflop decisions: {postflop_decisions}")
    print(f"  NEW total:   {new_total:+.1f} chips")
    print(f"  OLD total:   {old_total:+.1f} chips")
    print(f"  NEW - OLD per deck: {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (per deck = 2 rotations)")
    verdict = ("NEW(ensemble) STRONGER (CI excludes 0)" if lo > 0 else
               ("OLD(single) STRONGER (CI excludes 0)" if hi < 0 else "INCONCLUSIVE (CI includes 0)"))
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
