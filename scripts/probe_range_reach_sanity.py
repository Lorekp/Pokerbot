#!/usr/bin/env python3
"""R1 REACH-RANGE SANITY + COST PROBE (real blueprint, GAME-FLOW states only).

Validates the native reach-consistent opponent root ranges (`range_reach_full`) against the DEPLOYED
EMD-K160 blueprint on real game-flow states (standing rule: never hand-build postflop nodes):

  1. REPLAY SUCCESS RATE — `SubgameSolver.range_weights(state, hero, "reach_strict")` returns None
     whenever the native history replay fails verification; on engine-dealt game-flow states this
     must be ~100% (the fallback exists for live/bridge states, not for engine states).
  2. DIRECTIVE SANITY PROPERTY — for opponents who VOLUNTARILY CALLED PREFLOP (limp/call line), the
     reach weight of blueprint-always-fold trash (72o-type) must sit near the bottom of the range,
     premium always-raise hands (AA) must be down-weighted for a call line, and the seat's ACTUAL
     dealt hand should rank ABOVE the uniform 0.5 percentile on average (the log-loss-lite check:
     the reach posterior should dominate uniform at predicting real holdings).
  3. PER-SOLVE COST — wall-clock of the deployed-baseline native solve (K=2000) with the knob OFF vs
     ON over the same states + the standalone reach-table build time.

Usage:
  python scripts/probe_range_reach_sanity.py --states 24 --seed 7 \
      --out artifacts/r1_reach_sanity.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# DEPLOYED EMD-K160 serve env — MUST be set before any pokerbot import (flags read at import).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
_FROZEN = os.path.join(REPO, "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")
for _k, _v in {
    "POKERBOT_BETTING_ABSTRACTION": "1",
    "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "160",
    "POKERBOT_POSTFLOP_MODE": "emd",
    "POKERBOT_POSTFLOP_EMD_FROZEN": _FROZEN,
    "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402

FOLD, CHECK, CALL = 0, 1, 2
STRIDE = 52 * 52


def _pick_action(rng: random.Random, legal: "np.ndarray") -> int:
    """Passive-leaning game-flow policy (mirrors probe_native_emd_parity): mostly check/call so
    hands reach the flop with limp/call lines in the history, some aggression for variety."""
    legal_set = set(int(a) for a in np.flatnonzero(legal))
    u = rng.random()
    passive = [a for a in (CHECK, CALL) if a in legal_set]
    aggro = [a for a in legal_set if a >= 3]
    if u < 0.80 and passive:
        return rng.choice(passive)
    if u < 0.93 and aggro:
        return rng.choice(aggro)
    return int(rng.choice(sorted(legal_set)))


def _combo_ids(rank_hi: int, rank_lo: int, suited: bool) -> list[tuple[int, int]]:
    """All card-id combos of an unpaired rank pair (ranks 0..12; suited or offsuit)."""
    out = []
    for s0 in range(4):
        for s1 in range(4):
            if suited and s0 != s1:
                continue
            if not suited and s0 == s1:
                continue
            c0, c1 = rank_hi * 4 + s0, rank_lo * 4 + s1
            out.append((min(c0, c1), max(c0, c1)))
    return out


def _pair_ids(rank: int) -> list[tuple[int, int]]:
    cards = [rank * 4 + s for s in range(4)]
    return [(cards[i], cards[j]) for i in range(4) for j in range(i + 1, 4)]


def main() -> int:
    ap = argparse.ArgumentParser(description="R1 reach-range sanity + cost probe")
    ap.add_argument("--states", type=int, default=24, help="flop decision states to collect")
    ap.add_argument("--timing-states", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--max-hands", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--blueprint", default=os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl"))
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts", "r1_reach_sanity.json"))
    args = ap.parse_args()

    import pickle
    import jax
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS
    from pokerbot.search import native_solver as _ns
    from pokerbot.eval.solver_exploitability import make_search_policy_fn

    if not _ns.native_subgame_available():
        _ns.enable()
    if not _ns.native_subgame_available():
        print("FATAL: native SubgameSolver unavailable", flush=True)
        return 2
    if not hasattr(_ns._EXT.SubgameSolver, "range_weights"):
        print("FATAL: installed battery_native predates range_weights (rebuild + install)", flush=True)
        return 2

    raw = pickle.load(open(args.blueprint, "rb"))
    snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
    print(f"blueprint rows: {len(snap)}", flush=True)

    # Adapter via the same construction the H2H scaffold uses (fish the solver out of the closure).
    pf = make_search_policy_fn(snap, leaf_select="uniform", seed=args.seed, iterations=200,
                               time_budget_ms=2000, search_preflop=False)
    sv = [c.cell_contents for c in pf.__closure__
          if hasattr(getattr(c, "cell_contents", None), "config")][0]
    adapter = sv.adapter
    eng = _ns._build_subgame_engine(adapter, int(_POSTFLOP_NUM_BUCKETS),
                                    int(sv.config.leaf.max_playout_depth))
    if eng is None:
        print("FATAL: could not build native subgame engine", flush=True)
        return 2

    # ---- collect flop decision states where >=1 opponent voluntarily CALLED preflop ----
    rng = random.Random(args.seed)
    occ = jax.numpy.array([True] * 6)
    states = []  # (env, hero, [seats that called preflop])
    for d in range(args.max_hands):
        if len(states) >= args.states:
            break
        e = to_numpy_state(init_hand(jax.random.PRNGKey(args.seed * 100003 + d), occupancy=occ))
        guard = 0
        while not is_terminal_np(e) and guard < 60:
            guard += 1
            if is_chance_node_np(e):
                e = sample_chance_np(e)
                continue
            s = current_player_np(e)
            if s < 0:
                break
            street = int((np.asarray(e.board) >= 0).sum())
            if street == 3:  # flop decision node
                hs = np.asarray(e.hist_seat)
                ha = np.asarray(e.hist_action)
                n = int(e.hist_len) if hasattr(e, "hist_len") else int((hs >= 0).sum())
                # preflop voluntary calls: CALL actions before the first street close. The flat
                # history spans the whole hand; preflop entries are the prefix while street was 0 —
                # conservatively approximate with "CALL by a non-hero seat anywhere in history"
                # (at a first flop decision node the history IS the preflop history).
                callers = sorted({int(hs[i]) for i in range(n)
                                  if int(ha[i]) == CALL and int(hs[i]) != int(s)})
                if callers:
                    states.append((e, int(s), callers))
                break
            m = legal_action_mask_np(e, s).astype(bool)
            e = apply_action_np(e, _pick_action(rng, m))
    print(f"collected {len(states)} flop states with a preflop caller", flush=True)
    if not states:
        return 2

    trash = _combo_ids(5, 0, suited=False)   # 72o (rank 5 = seven, rank 0 = deuce)
    aa = _pair_ids(12)                        # AA
    mid = _combo_ids(6, 5, suited=True)       # 87s

    results = {"replay_ok": 0, "replay_total": 0, "per_state": [], "percentiles": []}
    t_build = []
    for (e, hero, callers) in states:
        sb = _ns._to_flat(e).tobytes()
        t0 = time.perf_counter()
        w_strict = eng.range_weights(sb, hero, "reach_strict")
        t_build.append(time.perf_counter() - t0)
        results["replay_total"] += 1
        if w_strict is None:
            results["per_state"].append({"replay": False})
            continue
        results["replay_ok"] += 1
        w = np.asarray(w_strict, dtype=np.float64)
        holes = np.asarray(e.holes)
        board = {int(c) for c in np.asarray(e.board) if int(c) >= 0}
        known = board | {int(holes[hero][0]), int(holes[hero][1])}

        def _w(seat, c0, c1):
            return float(w[seat * STRIDE + c0 * 52 + c1])

        def _mean_class(seat, combos):
            vals = [_w(seat, c0, c1) for (c0, c1) in combos
                    if c0 not in known and c1 not in known]
            return float(np.mean(vals)) if vals else None

        st = {"replay": True, "callers": callers, "seats": {}}
        for seat in callers:
            unseen = [c for c in range(52) if c not in known]
            allw = np.array([_w(seat, unseen[i], unseen[j])
                             for i in range(len(unseen)) for j in range(i + 1, len(unseen))])
            h0, h1 = sorted((int(holes[seat][0]), int(holes[seat][1])))
            w_act = _w(seat, h0, h1)
            pct = float((np.sum(allw < w_act) + 0.5 * np.sum(allw == w_act)) / len(allw))
            results["percentiles"].append(pct)
            st["seats"][seat] = {
                "w_72o_mean": _mean_class(seat, trash),
                "w_AA_mean": _mean_class(seat, aa),
                "w_87s_mean": _mean_class(seat, mid),
                "w_median": float(np.median(allw)),
                "w_actual_hand": w_act,
                "actual_hand_percentile": pct,
            }
        results["per_state"].append(st)

    # ---- per-solve cost: deployed baseline knobs, reach OFF vs ON ----
    deployed = dict(max_iterations=int(args.iterations), max_depth=48, blueprint_mix=1.0,
                    rollout_samples=4, resample_board=True, max_playout_depth=int(sv.config.leaf.max_playout_depth),
                    num_buckets=int(_POSTFLOP_NUM_BUCKETS), linear_averaging=True, repair_deck=True,
                    two_sided=True, decouple_sampler=True, range_aware=True, range_uniform_floor=0.25,
                    ensemble_streams=1, leaf_cache_worlds=4, avg_skip_frac=0.0,
                    leaf_biased_continuations=False, leaf_bias_mult=5.0, serve_final_iterate=False)
    t_off, t_on = [], []
    for (e, hero, _c) in states[: args.timing_states]:
        for reach, sink in ((False, t_off), (True, t_on)):
            t0 = time.perf_counter()
            _ns.native_subgame_solve(e, hero, adapter, public_seed=12345,
                                     range_reach_full=reach, **deployed)
            sink.append(time.perf_counter() - t0)

    summary = {
        "states": len(states),
        "replay_success": f"{results['replay_ok']}/{results['replay_total']}",
        "actual_hand_percentile_mean": (float(np.mean(results["percentiles"]))
                                        if results["percentiles"] else None),
        "reach_table_build_ms_mean": float(np.mean(t_build) * 1000.0),
        "solve_ms_reach_off_mean": float(np.mean(t_off) * 1000.0),
        "solve_ms_reach_on_mean": float(np.mean(t_on) * 1000.0),
        "per_solve_added_ms": float((np.mean(t_on) - np.mean(t_off)) * 1000.0),
        "iterations": int(args.iterations),
    }
    print(json.dumps(summary, indent=2), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
