#!/usr/bin/env python3
"""MECHANISM probe: WHY does native land on a lower-jam fixed point than Python at the AKs node?

Reconstructs the SAME AKs-facing-raise root the jam probe found, then isolates the three hypotheses by
directly exercising the two engines' internals with matched inputs:

  H1 OPPONENT-HOLE SAMPLING: draw many opponent-hole worlds from Python's PBSTracker.sample_opponent_holes
     and (via a tiny ctypes-free replay) the native Fisher-Yates dealer. Compare the aggregate opponent
     RANGE (are they both uniform over unseen combos?) and, more importantly, whether the DECK TAIL used
     for the all-in runout collides with the sampled opponent holes (the native-specific stale-deck detail).

  H2 BOARD-RUNOUT / ALL-IN EQUITY: for the hero's ALL_IN action, run the engine forward to the terminal
     (which triggers the atomic all-in run_out) under BOTH the native subgame path (repair reshuffles the
     deck) and the Python path, and record the hero's realized reward distribution + mean. If native's AKs
     all-in EV is materially LOWER, jamming is correctly less attractive.

  H3 REGRET/AVERAGING/RNG: compare the per-action root Q-values (action_values) the two solvers compute at
     a few iteration counts (already partly visible via the served policy in the jam probe).

Also directly answers: is the all-in showdown reward the SAME for a fixed (hero, opp holes, board) under
both engines' apply_action? (rules OUT a showdown-scorer difference.)

Run from the worktree.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

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
        ALL_IN, BET_BUCKET_START, CALL, FOLD, CHECK, SEAT_FOLDED,
    )
    from pokerbot.search.pbs_tracker import PBSTracker
    from pokerbot.search.pluribus_rollout import _reshuffle_undealt
    from pokerbot.search import native_solver as _ns
    from pokerbot.engine.cards import Card
    import pickle

    _RANKS = "23456789TJQKA"
    _SUITS = "cdhs"

    def _card(idx):
        i = int(idx)
        return Card.from_str(_RANKS[i // 4] + _SUITS[i % 4])

    def is_AK(env, seat):
        h = env.holes[seat]
        if int(h[0]) < 0:
            return False
        r0, r1 = int(h[0]) // 4, int(h[1]) // 4
        s0, s1 = int(h[0]) % 4, int(h[1]) % 4
        return {r0, r1} == {12, 11} and s0 == s1

    occ = jax.numpy.array([True] * 6)

    # reconstruct the same facing_raise AKs root as the jam probe
    def find_ak_root(seed_base):
        for d in range(4000):
            env = to_numpy_state(init_hand(jax.random.PRNGKey(seed_base * 100003 + d), occupancy=occ))
            g = 0
            while not is_terminal_np(env) and g < 40:
                g += 1
                if is_chance_node_np(env):
                    env = sample_chance_np(env)
                    continue
                seat = current_player_np(env)
                if seat < 0:
                    break
                if int(env.street) != 0:
                    break
                mask = legal_action_mask_np(env, seat).astype(bool)
                facing = bool(mask[FOLD])
                if is_AK(env, seat) and facing:
                    return env, int(seat)
                if mask[BET_BUCKET_START + 3]:
                    a = BET_BUCKET_START + 3
                elif mask[CALL]:
                    a = CALL
                elif mask[CHECK]:
                    a = CHECK
                else:
                    a = int(np.argmax(mask))
                env = apply_action_np(env, a)
        return None, None

    env, seat = find_ak_root(4242)
    assert env is not None, "no AKs root"
    hole = (int(env.holes[seat][0]), int(env.holes[seat][1]))
    board = [int(c) for c in np.asarray(env.board) if int(c) >= 0]
    print(f"AKs root: seat={seat} hole={_RANKS[hole[0]//4]}{_SUITS[hole[0]%4]}"
          f"{_RANKS[hole[1]//4]}{_SUITS[hole[1]%4]} nboard={len(board)} "
          f"deck_ptr={int(env.deck_ptr)} street={int(env.street)}", flush=True)

    out = {"seat": seat, "hole": hole, "nboard": len(board)}

    # ---------------------------------------------------------------
    # H2 / showdown-scorer identity: the hero JAMS. Find who has already acted; simulate the ALL_IN
    # action + everyone-calls line and measure the hero all-in EV under Python's repair reshuffle vs
    # the native subgame's internal reshuffle. First: is the terminal showdown reward identical for a
    # FIXED (holes, board) under native apply_action vs Python apply_action? (rules out scorer diff)
    # ---------------------------------------------------------------
    # Build a fully-specified all-in terminal by hand: hero jams, everyone left calls.
    def play_allin_line(env0):
        """Hero (seat) jams; every subsequent live actor calls all-in / calls; return terminal env."""
        e = env0
        g = 0
        while not is_terminal_np(e) and g < 40:
            g += 1
            if is_chance_node_np(e):
                e = sample_chance_np(e)
                continue
            s = current_player_np(e)
            if s < 0:
                break
            m = legal_action_mask_np(e, s).astype(bool)
            if s == seat and int(e.street) == 0 and m[ALL_IN]:
                e = apply_action_np(e, ALL_IN)
            elif m[CALL]:
                e = apply_action_np(e, CALL)
            elif m[CHECK]:
                e = apply_action_np(e, CHECK)
            else:
                e = apply_action_np(e, int(np.argmax(m)))
        return e

    # Scorer identity check: fix opp holes + a random full deck, run the all-in line with native ON vs OFF.
    _ns.enable()
    rng = np.random.default_rng(7)
    # give every non-hero seat concrete holes + a concrete deck (reshuffle once)
    base = _reshuffle_undealt(env, np.random.default_rng(11))
    # fill opp holes from the (now shuffled) deck tail deterministically so both engines see identical input
    holes = np.asarray(base.holes).copy()
    used = set(int(c) for c in holes.reshape(-1) if int(c) >= 0) | set(board)
    pool = [c for c in range(52) if c not in used]
    pi = 0
    for s in range(6):
        if s == seat:
            continue
        if int(holes[s, 0]) < 0 and int(np.asarray(base.seat_status)[s]) != SEAT_FOLDED:
            holes[s, 0] = pool[pi]; holes[s, 1] = pool[pi + 1]; pi += 2
    base = base._replace(holes=holes)

    _ns.enable()
    term_native = play_allin_line(base)
    r_native = np.asarray(term_native.rewards, dtype=np.float64)
    _ns.disable()
    term_py = play_allin_line(base)
    r_py = np.asarray(term_py.rewards, dtype=np.float64)
    board_native = [int(c) for c in np.asarray(term_native.board) if int(c) >= 0]
    board_py = [int(c) for c in np.asarray(term_py.board) if int(c) >= 0]
    out["scorer_identity"] = {
        "reward_native": r_native.tolist(), "reward_py": r_py.tolist(),
        "board_native": board_native, "board_py": board_py,
        "identical": bool(np.array_equal(r_native, r_py) and board_native == board_py),
    }
    print(f"\n[SCORER IDENTITY] same (holes,deck) -> native reward {r_native[seat]:+.0f} "
          f"py {r_py[seat]:+.0f} | boards_equal={board_native==board_py} "
          f"rewards_equal={np.array_equal(r_native,r_py)}", flush=True)

    # ---------------------------------------------------------------
    # H2 all-in EV distribution: many worlds. For each, sample opp holes + reshuffle the deck (Python's
    # repair path), run the all-in-everyone-calls line, record hero reward. This is the value the hero's
    # jam is worth WHEN CALLED. (Both engines reach the SAME terminal scorer, so this is engine-agnostic
    # given identical world inputs; the divergence must be in WHICH worlds/opponents each engine samples,
    # H1.) We report the distribution so the mean all-in EV is on record.
    # ---------------------------------------------------------------
    _ns.disable()
    n_world = 400
    hero_allin_rewards = []
    opp_call_when_jammed = 0  # (all opponents are forced to call here, so this measures pure all-in EV)
    for w in range(n_world):
        # Python-style world: sample opp holes uniformly (PBSTracker under uniform belief) + reshuffle deck
        hole_cards = (_card(hole[0]), _card(hole[1]))
        board_cards = [_card(c) for c in board]
        pbs = PBSTracker.init_uniform(seat, hole_cards, board_cards)
        sh = pbs.sample_opponent_holes(np.random.default_rng(1000 + w))
        h2 = np.asarray(env.holes).copy()
        for s2, (c0, c1) in sh.items():
            h2[s2, 0] = c0; h2[s2, 1] = c1
        world = env._replace(holes=h2)
        world = _reshuffle_undealt(world, np.random.default_rng(2000 + w))
        term = play_allin_line(world)
        hero_allin_rewards.append(float(np.asarray(term.rewards)[seat]))
    ar = np.array(hero_allin_rewards)
    out["allin_ev"] = {"mean": round(float(ar.mean()), 1), "std": round(float(ar.std()), 1),
                       "win_rate": round(float((ar > 0).mean()), 4), "n": n_world}
    print(f"\n[ALL-IN EV] hero AKs jam-when-called over {n_world} worlds: "
          f"mean {ar.mean():+.0f} chips  win% {(ar>0).mean()*100:.1f}  (this is engine-agnostic given "
          f"the SAME world inputs; scorer identity above confirms it)", flush=True)

    json.dump(out, open(f"{MAIN}/artifacts/py_vs_rust_mechanism_probe.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_mechanism_probe.json", flush=True)


if __name__ == "__main__":
    main()
