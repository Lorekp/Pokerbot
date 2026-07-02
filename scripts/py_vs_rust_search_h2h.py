#!/usr/bin/env python3
"""DUPLICATE-DEALING head-to-head: 3 Python-search seats vs 3 Rust(native)-search seats, same 160k blueprint.

Does the native port converge to a BETTER strategy than interpreted Python (or worse -- i.e. is its measured
divergence a bug or a saner convergence)? We seat 3 Python searchers + 3 Rust searchers alternating around a
6-max table and play ~50 STATIC decks. Each deck is played TWICE with the two engines SWAPPED (rotation 0:
Rust on even seats / Python on odd; rotation 1: swapped), so across the pair each engine plays every card
position once -> card luck cancels and the net chip difference isolates the STRATEGY. Both engines use the
IDENTICAL search config (repair_deck ON, search_preflop ON, same iters) -- the ONLY difference is
native-vs-interpreted. Deterministic per (deck, decision) action sampling.

Run from the native worktree so its patched .pyd (repair_deck/two_sided) loads:
  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 \
  POKERBOT_POSTFLOP_NUM_BUCKETS=200 python scripts/py_vs_rust_search_h2h.py <worktree> <main>
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
    sys.path.insert(0, WT)  # worktree's pokerbot + patched native ext win

    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        SEAT_LIVE, apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.search.pluribus_search import set_native_subgame
    from pokerbot.eval.solver_exploitability import make_search_policy_fn
    import pickle

    settle_samples = int(os.environ.get("H2H_SETTLE", "300"))  # 0 => single-runout (high variance)

    def _nboard(env):
        return int((np.asarray(env.board) >= 0).sum())

    def ev_settle_via_action(prev_env, action, rng):
        """ALL-IN EV settlement. `apply_action_np` runs out the all-in board ATOMICALLY (run_out) -> the
        loop never sees the runout chance nodes. So we detect the runout at terminal (board grew) and
        re-apply the SAME closing `action` to `prev_env` with the UNDEALT deck tail reshuffled each time;
        averaging the resulting terminal rewards integrates the equity split over all runouts (mirrors
        lbr6._board_ev_settle) and removes the dominant stack-off variance while preserving the mean."""
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

    n_decks = int(os.environ.get("H2H_DECKS", "50"))
    K = int(os.environ.get("H2H_ITERS", "250"))
    repair = os.environ.get("H2H_REPAIR", "1") != "0"
    two = os.environ.get("H2H_TWOSIDED", "0") != "0"
    base = int(os.environ.get("H2H_SEED", "4242"))
    out = os.environ.get("H2H_OUT", f"{MAIN}/artifacts/py_vs_rust_h2h.json")

    snap = (lambda o: o["strat_sum"] if isinstance(o, dict) and "strat_sum" in o else o)(
        pickle.load(open(f"{MAIN}/checkpoints/serve_blueprint_6max.pkl", "rb")))
    occ = jax.numpy.array([True] * 6)

    def make_pf(native):
        pf = make_search_policy_fn(snap, leaf_select="uniform", seed=base, iterations=K, search_preflop=True)
        sv = [c.cell_contents for c in pf.__closure__ if hasattr(getattr(c, "cell_contents", None), "config")][0]
        sv.config.native = native
        sv.config.repair_deck = repair
        sv.config.two_sided = two
        sv.config.blueprint_mix = 1.0
        return pf

    pf_rust, pf_py = make_pf(True), make_pf(False)
    print(f"H2H  decks={n_decks} iters={K} repair={repair} two_sided={two}  (3 Rust vs 3 Python, 160k bp)",
          flush=True)

    def act(is_rust, env, seat, salt):
        set_native_subgame(bool(is_rust))
        p = np.asarray((pf_rust if is_rust else pf_py)(env, seat), dtype=np.float64)
        pc = np.clip(p, 0, None)
        s = pc.sum()
        if s <= 0:
            m = legal_action_mask_np(env, seat).astype(np.float64)
            pc = m / m.sum()
        else:
            pc = pc / s
        return int(np.random.default_rng(salt).choice(len(pc), p=pc))

    rust_total = 0.0
    py_total = 0.0
    per_deck = []
    n_games = 0
    t0 = time.perf_counter()
    for d in range(n_decks):
        deck_rust = 0.0
        deck_py = 0.0
        for rot in (0, 1):
            # even seats = rust when rot==0, python when rot==1 (engines swapped -> card luck cancels)
            is_rust = {s: ((s % 2 == 0) == (rot == 0)) for s in range(6)}
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
                    a = act(is_rust[int(seat)], env, int(seat), base * 7 + d * 131 + g)
                prev_env, prev_action, prev_nboard = env, a, _nboard(env)
                env = apply_action_np(env, a)
            # ALL-IN EV settle: if the final action ran the board OUT (board grew from the acting state to
            # the terminal), integrate the equity split over runouts instead of the single dealt board.
            if settle_samples > 0 and prev_env is not None and _nboard(env) > prev_nboard:
                rew = ev_settle_via_action(prev_env, prev_action, settle_rng)
            else:
                rew = np.asarray(env.rewards, dtype=np.float64)
            for s in range(6):
                if is_rust[s]:
                    deck_rust += float(rew[s])
                else:
                    deck_py += float(rew[s])
            n_games += 1
        rust_total += deck_rust
        py_total += deck_py
        per_deck.append(deck_rust - deck_py)  # net rust-minus-python for THIS deck (both rotations)
        el = time.perf_counter() - t0
        print(f"  deck {d+1}/{n_decks}: rust-python net {deck_rust-deck_py:+.1f} chips | "
              f"cum rust {rust_total:+.0f} / py {py_total:+.0f} | {el/60:.1f}min "
              f"ETA {el/(d+1)*(n_decks-d-1)/60:.1f}min", flush=True)
        json.dump({"decks_done": d + 1, "iters": K, "repair": repair, "two_sided": two,
                   "rust_total": rust_total, "py_total": py_total, "per_deck_net": per_deck},
                  open(out, "w"), indent=1)

    pd = np.array(per_deck)  # per-deck (rust-python), one paired sample per deck (both rotations)
    mean = float(pd.mean())
    se = float(pd.std(ddof=1) / math.sqrt(len(pd))) if len(pd) > 1 else 0.0
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    bb = 100.0  # chips are in 100bb-stack units? report raw chips + per-deck
    print(f"\n=== HEAD-TO-HEAD (paired by deck, {len(pd)} decks x 2 rotations) ===", flush=True)
    print(f"  Rust total:   {rust_total:+.1f} chips")
    print(f"  Python total: {py_total:+.1f} chips")
    print(f"  Rust - Python per deck: {mean:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  (per deck = 2 rotations)")
    verdict = ("RUST STRONGER (CI excludes 0)" if lo > 0 else
               ("PYTHON STRONGER (CI excludes 0)" if hi < 0 else "INCONCLUSIVE (CI includes 0)"))
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
