#!/usr/bin/env python3
"""Generalized full-game duplicate-deal H2H: OUR deployed arm vs a registry external opponent.

Pattern mirrors scripts/preflop_onoff_h2h.py exactly (arm construction, EV settle, progress
logging, duplicate dealing). The external arm requires NO config introspection.

Usage:
    python -u scripts/external_ladder_h2h.py \
        --opponent fish \
        --decks 100 \
        --iters 2000 \
        --settle-samples 200 \
        --seed 4242 \
        --out artifacts/ladder_fish.json

    # Quick smoke (tiny K):
    python -u scripts/external_ladder_h2h.py \
        --opponent fish --decks 3 --iters 200 \
        --out artifacts/_ladder_smoke_fish.json

Environment overrides (same as preflop_onoff_h2h.py):
    POKERBOT_BETTING_ABSTRACTION=1
    POKERBOT_POSTFLOP_BUCKETS=1
    POKERBOT_PREFLOP_LOSSLESS=1
    POKERBOT_POSTFLOP_MODE=emd
    POKERBOT_POSTFLOP_NUM_BUCKETS=160
    POKERBOT_POSTFLOP_EMD_FROZEN=<abs>/data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn
    POKERBOT_EMD_FROZEN_MIRROR_FALLBACK=1
    H2H_SETTLE=200      (runout samples for all-in EV settle)
    H2H_REPAIR=1        (deck repair gate)
    H2H_TWOSIDED=1      (two-sided subgame CFR gate)
    H2H_FLOOR=0.25      (range-uniform floor)

Output JSON: {opponent, decks, ours_minus_them_per_deck, ci95, per_deck_nets, flopdecks, wall_s}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# ---------------------------------------------------------------------------
# POKERBOT abstraction env vars MUST be set before importing pokerbot (copy
# from preflop_onoff_h2h.py — these must arrive before any pokerbot import).
# ---------------------------------------------------------------------------
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "emd")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "160")
# EMD-K160 deployed regime (promoted 2026-07-02, certified +32.89 bb/100 vs made-hand iter62000):
# frozen flop/turn EMD lookup + train-mirrored made-hand fallback on the river (the river is not
# in the blob; without the mirror flag the river frozen-miss serves a UNIFORM policy).
os.environ.setdefault("POKERBOT_POSTFLOP_EMD_FROZEN", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")))
os.environ.setdefault("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax


def parse_args():
    p = argparse.ArgumentParser(description="External opponent ladder H2H")
    p.add_argument("--opponent", required=True,
                   help="Opponent name from registry (fold_man / random / fish / honest / ...)")
    p.add_argument("--deck-start", type=int, default=0,
                   help="First deck index (for CRN-exact sharding across processes)")
    p.add_argument("--deck-stride", type=int, default=1,
                   help="Deck index stride (round-robin sharding: start=i, stride=n_shards)")
    p.add_argument("--decks", type=int, default=100)
    p.add_argument("--iters", type=int, default=2000,
                   help="CFR iterations for OUR arm (K)")
    p.add_argument("--settle-samples", type=int, default=200,
                   help="Runout samples for all-in EV settlement (0 = chip settle only)")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path (default artifacts/ladder_<opponent>.json)")
    return p.parse_args()


def main():
    args = parse_args()
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    n_decks = args.decks
    K = args.iters
    settle_samples = args.settle_samples
    base = args.seed
    opponent_name = args.opponent
    out_path = args.out or os.path.join(REPO, "artifacts", f"ladder_{opponent_name}.json")
    bp_path = os.environ.get("H2H_BLUEPRINT",
                             os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl"))

    repair = os.environ.get("H2H_REPAIR", "1") != "0"
    two = os.environ.get("H2H_TWOSIDED", "1") != "0"
    floor = float(os.environ.get("H2H_FLOOR", "0.25"))

    # ---------------------------------------------------------------------------
    # Imports (after env vars are set)
    # ---------------------------------------------------------------------------
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.search.pluribus_search import set_native_subgame
    from pokerbot.search import native_solver as _ns
    from pokerbot.eval.solver_exploitability import make_search_policy_fn
    from pokerbot.eval.external_bots.registry import get_opponent_policy
    import pickle

    # Enable native subgame solver (matches preflop_onoff_h2h.py exactly)
    if not _ns.native_subgame_available():
        _ns.enable()
    print(f"native available={_ns.available()} subgame_available={_ns.native_subgame_available()} "
          f"rayon={_ns._EXT.rayon_num_threads()}", flush=True)

    # ---------------------------------------------------------------------------
    # Load blueprint + build OUR arm
    # ---------------------------------------------------------------------------
    snap = (lambda o: o["strat_sum"] if isinstance(o, dict) and "strat_sum" in o else o)(
        pickle.load(open(bp_path, "rb")))

    def make_our_arm():
        pf = make_search_policy_fn(snap, leaf_select="uniform", seed=base, iterations=K,
                                   search_preflop=False)
        sv = [c.cell_contents for c in pf.__closure__
              if hasattr(getattr(c, "cell_contents", None), "config")][0]
        sv.config.native = True
        sv.config.repair_deck = repair
        sv.config.two_sided = two
        sv.config.blueprint_mix = 1.0
        sv.config.decouple_sampler = True
        sv.config.range_aware = True
        sv.config.range_uniform_floor = float(floor)
        sv.config.ensemble_streams = 1
        return pf

    pf_ours = make_our_arm()

    # ---------------------------------------------------------------------------
    # Build external opponent arm
    # ---------------------------------------------------------------------------
    pf_ext = get_opponent_policy(opponent_name)
    print(f"Loaded opponent: {opponent_name}", flush=True)

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------
    occ = jax.numpy.array([True] * 6)

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

    def act_ours(env, seat, salt):
        set_native_subgame(True)
        p = np.asarray(pf_ours(env, seat), dtype=np.float64)
        pc = np.clip(p, 0, None)
        s = pc.sum()
        if s <= 0:
            m = legal_action_mask_np(env, seat).astype(np.float64)
            pc = m / m.sum()
        else:
            pc = pc / s
        return int(np.random.default_rng(salt).choice(len(pc), p=pc))

    def act_ext(env, seat, salt):
        p = np.asarray(pf_ext(env, seat), dtype=np.float64)
        pc = np.clip(p, 0, None)
        s = pc.sum()
        if s <= 0:
            m = legal_action_mask_np(env, seat).astype(np.float64)
            pc = m / m.sum()
        else:
            pc = pc / s
        return int(np.random.default_rng(salt).choice(len(pc), p=pc))

    # ---------------------------------------------------------------------------
    # Main loop: 3-vs-3, 2 rotations per deck
    # ---------------------------------------------------------------------------
    print(f"LADDER H2H  opponent={opponent_name}  decks={n_decks}  K={K}  "
          f"settle={settle_samples}  repair={repair}  two_sided={two}  floor={floor}  "
          f"seed={base}  bp={len(snap)} infosets", flush=True)
    print(f"  OUR arm: search_preflop=OFF  ensemble_streams=1  (deployed default)", flush=True)
    print(f"  Seats 0,2,4 = OUR arm on rotation 0; seats 1,3,5 = OUR arm on rotation 1", flush=True)

    ours_total = 0.0
    ext_total = 0.0
    per_deck = []
    flopdecks = 0
    t0 = time.perf_counter()

    for _idx in range(n_decks):
        d = args.deck_start + _idx * args.deck_stride
        deck_ours = 0.0
        deck_ext = 0.0
        deck_saw_flop = False

        for rot in (0, 1):
            # is_ours[seat] True = this seat uses OUR arm in this rotation
            is_ours = {s: ((s % 2 == 0) == (rot == 0)) for s in range(6)}
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
                if int(env.street) > 0:
                    deck_saw_flop = True
                mask = legal_action_mask_np(env, seat).astype(bool)
                if int(mask.sum()) <= 1:
                    a = int(np.argmax(mask))
                else:
                    salt = base * 7 + d * 131 + g
                    if is_ours[int(seat)]:
                        a = act_ours(env, int(seat), salt)
                    else:
                        a = act_ext(env, int(seat), salt)
                prev_env, prev_action, prev_nboard = env, a, _nboard(env)
                env = apply_action_np(env, a)

            if settle_samples > 0 and prev_env is not None and _nboard(env) > prev_nboard:
                rew = ev_settle_via_action(prev_env, prev_action, settle_rng)
            else:
                rew = np.asarray(env.rewards, dtype=np.float64)

            for s in range(6):
                if is_ours[s]:
                    deck_ours += float(rew[s])
                else:
                    deck_ext += float(rew[s])

        if deck_saw_flop:
            flopdecks += 1
        ours_total += deck_ours
        ext_total += deck_ext
        net = deck_ours - deck_ext
        per_deck.append(net)

        el = time.perf_counter() - t0
        pd_arr = np.array(per_deck)
        se = float(pd_arr.std(ddof=1) / math.sqrt(len(pd_arr))) if len(pd_arr) > 1 else 0.0
        eta = el / len(pd_arr) * (n_decks - len(pd_arr)) / 60
        print(
            f"  deck {len(pd_arr)}/{n_decks} (idx {d}): ours-ext net {net:+.1f} | cum ours {ours_total:+.0f}/ext "
            f"{ext_total:+.0f} | mean/deck {pd_arr.mean():+.2f}+-{1.96*se:.2f} | "
            f"flopdecks={flopdecks} | {el/60:.1f}min ETA {eta:.1f}min",
            flush=True,
        )

        # Incremental checkpoint
        ci95 = [float(pd_arr.mean() - 1.96 * se), float(pd_arr.mean() + 1.96 * se)]
        json.dump(
            {
                "opponent": opponent_name,
                "decks_done": len(per_deck),
                "K": K,
                "settle_samples": settle_samples,
                "repair": repair,
                "two_sided": two,
                "floor": floor,
                "seed": base,
                "ours_total": ours_total,
                "ext_total": ext_total,
                "ours_minus_them_per_deck": float(pd_arr.mean()),
                "ci95": ci95,
                "per_deck_nets": per_deck,
                "flopdecks": flopdecks,
                "wall_s": el,
            },
            open(out_path, "w"),
            indent=1,
        )

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    pd_arr = np.array(per_deck)
    mean = float(pd_arr.mean())
    se = float(pd_arr.std(ddof=1) / math.sqrt(len(pd_arr))) if len(pd_arr) > 1 else 0.0
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    wall_s = time.perf_counter() - t0

    print(f"\n=== LADDER H2H: OUR vs {opponent_name} "
          f"({len(pd_arr)} decks x 2 rotations) ===", flush=True)
    print(f"  K={K}  settle={settle_samples}  repair={repair}  two_sided={two}  floor={floor}")
    print(f"  OUR total:    {ours_total:+.1f} chips")
    print(f"  EXT total:    {ext_total:+.1f} chips")
    print(f"  OUR - EXT per deck: {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    verdict = (
        "OUR ARM STRONGER (CI excludes 0)" if lo > 0
        else ("OPPONENT STRONGER (CI excludes 0)" if hi < 0
              else "INCONCLUSIVE (CI includes 0)")
    )
    print(f"  VERDICT: {verdict}  flopdecks={flopdecks}/{n_decks}", flush=True)
    print(f"\nwrote {out_path}  (wall {wall_s:.1f}s)")

    # Final JSON write (complete)
    json.dump(
        {
            "opponent": opponent_name,
            "decks": n_decks,
            "K": K,
            "settle_samples": settle_samples,
            "repair": repair,
            "two_sided": two,
            "floor": floor,
            "seed": base,
            "ours_minus_them_per_deck": mean,
            "ci95": [lo, hi],
            "per_deck_nets": per_deck,
            "flopdecks": flopdecks,
            "wall_s": wall_s,
        },
        open(out_path, "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
