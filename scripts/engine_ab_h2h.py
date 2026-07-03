#!/usr/bin/env python3
"""Generalized self-play A/B head-to-head: compare two configurations of our own engine.

Each arm is specified as a JSON dict of knobs overriding the DEPLOYED baseline.  The empty dict {}
gives the exact deployed config (blueprint preflop OFF, K=2000, streams=1, native flags ON at floor
0.25).  Unknown knobs hard-fail with the valid-knob list.

Game loop: duplicate dealing (each deck played TWICE with arms swapped 3v3), all-in-EV settlement,
forced-node autoplay, per-deck progress with running mean / 95% CI / ETA.

CRITICAL CORRECTNESS PROPERTY — exact-zero gate:
  When arm_a spec == arm_b spec, the per-deck diff is EXACTLY 0.0 on every deck.
  This holds because the action-sampling salt depends ONLY on (deck index, step counter) and the
  global seed — never on which arm / rotation is currently acting — so both arms sample identical
  action indices from identical probability vectors whenever their configs are the same.

Usage
-----
  python scripts/engine_ab_h2h.py \\
      --arm-a '{}' \\
      --arm-b '{"iterations": 150}' \\
      --decks 40 --seed 4242 --settle-samples 200 \\
      --label "K2000_vs_K150" \\
      --out artifacts/my_ab.json

  # Zero-gate smoke (both arms must be identical, e.g. both override iterations=150):
  python scripts/engine_ab_h2h.py \\
      --arm-a '{"iterations": 150}' \\
      --arm-b '{"iterations": 150}' \\
      --decks 6 --seed 9999 --label "zero_gate" \\
      --out artifacts/zero_gate.json

Environment preamble (set before running — the DEPLOYED EMD-K160 regime, promoted 2026-07-02):
  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 \\
  POKERBOT_PREFLOP_LOSSLESS=1 POKERBOT_POSTFLOP_MODE=emd \\
  POKERBOT_POSTFLOP_NUM_BUCKETS=160 \\
  POKERBOT_POSTFLOP_EMD_FROZEN=data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn \\
  POKERBOT_EMD_FROZEN_MIRROR_FALLBACK=1
(use the ABSOLUTE frozen path so it string-matches the snapshot stamp; the mirror flag keys the
river made-hand exactly as the trainer did — without it the river serves UNIFORM)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# CRITICAL: POKERBOT_* env vars must be set BEFORE any pokerbot import.
# Mirrors the preamble in adaptive_field_h2h.py / blueprint_ranking.py.
# These setdefaults are the deployed EMD-K160 regime (promoted 2026-07-02).
# A pinned external env always wins (setdefault never overwrites an existing value).
# ---------------------------------------------------------------------------
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "emd")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "160")
# EMD-K160 deployed regime (certified +32.89 bb/100 vs made-hand iter62000).
# Frozen flop/turn lookup; river uses train-mirrored made-hand fallback.
# POKERBOT_EMD_FROZEN_MIRROR_FALLBACK=1 is required — without it the river serves UNIFORM.
os.environ.setdefault("POKERBOT_POSTFLOP_EMD_FROZEN", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "abstraction",
    "emd_frozen_flop_turn_160_r400.emdfrzn")))
os.environ.setdefault("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax

# ---------------------------------------------------------------------------
# VALID KNOBS — every knob that can appear in an arm spec.
# Unknown keys hard-fail at parse time (no silent typo-ignoring).
# ---------------------------------------------------------------------------
_DEPLOYED_BASELINE = {
    # search iterations (fixed-count mode; 0 = use time_budget_ms instead)
    "iterations": 2000,
    # wall-clock budget per decision (ms); only active when iterations==0
    "time_budget_ms": 2000,
    # blueprint preflop: False = serve preflop from blueprint (DEPLOYED default)
    "search_preflop": False,
    # leaf continuation strategy: "gadget" (primary) | "uniform" (bootstrap)
    "leaf_select": "uniform",
    # number of independent CFR ensemble streams (1 = single stream, DEPLOYED default)
    "ensemble_streams": 1,
    # native two-sided subgame CFR (both arms must correct; deployed ON)
    "two_sided": True,
    # stale-deck repair (deployed ON)
    "repair_deck": True,
    # decouple opponent-hole sampler RNG (deployed ON)
    "decouple_sampler": True,
    # range-aware opponent-hole sampling (deployed ON)
    "range_aware": True,
    # uniform robustness floor for range-aware sampling (deployed 0.25)
    "range_uniform_floor": 0.25,
    # R1 reach-consistent opponent root ranges: per-combo weight = blueprint REACH product along the
    # hand's observed public history (natively replayed + verified; falls back to fold-now/uniform on
    # an unverifiable history) instead of the fold-now proxy. Floored by range_uniform_floor.
    # (False = fold-now proxy per range_aware, DEPLOYED default, byte-identical)
    "range_reach_full": False,
    # BUG-1 leaf-cache hole-world integration: running mean over the first M visiting worlds
    # (4 = ADOPTED 2026-07-02 default, +146/deck CI-certified; 1 = legacy first-touch freeze)
    "leaf_cache_worlds": 4,
    # BUG-5 delayed averaging: strat_sum accumulation starts at floor(frac * K) iterations
    # (0.0 = legacy accumulate-from-t=0, DEPLOYED default, byte-identical)
    "avg_skip_frac": 0.0,
    # R2 Pluribus biased continuation leaves: every depth-limit leaf becomes a 4-way continuation-
    # choice node per live seat, solved by the same subgame CFR; leaf playout biased by
    # leaf_bias_mult (False = legacy uniform-draw leaf, DEPLOYED default, byte-identical)
    "leaf_biased_continuations": False,
    # R2 bias multiplier for the biased continuations (Pluribus 5, Modicum 10; read only when
    # leaf_biased_continuations is on)
    "leaf_bias_mult": 5.0,
    # R5a final-iterate serving: serve the root policy from the final iterate's regret-matching
    # distribution (regret-match over the root's cumulative CFR+ regrets after the last iteration)
    # instead of the averaged strategy_sum — Pluribus plays the final iterate, not the average
    # (False = averaged serve, DEPLOYED default, byte-identical)
    "serve_final_iterate": False,
    # blueprint checkpoint path (None = auto-detect from cwd)
    "blueprint_path": None,
    # max CFR depth (safety cap)
    "max_depth": 48,
    # blueprint mix at opponent nodes: (1-mix)*regret_match + mix*blueprint
    "blueprint_mix": 1.0,
    # CFR+ linear averaging (True = CFR+; False = uniform average)
    "linear_averaging": True,
    # deterministic per-decision RNG seeding
    "per_decision_seed": True,
    # RNG salt (vary for independent replicas)
    "rng_salt": 0,
    # blueprint blend when budget-bound (None = follow env)
    "blueprint_blend": None,
    # frozen-read DBR response
    "frozen_read_response": False,
    # RNR restriction of solver-internal best-response
    "rnr_restrict": False,
    # leaf rollout samples per board runout
    "leaf_rollout_samples": 4,
    # leaf gadget iterations
    "leaf_gadget_iters": 256,
    # leaf max playout depth
    "leaf_max_playout_depth": 64,
    # single-table iteration floor (None = disabled)
    "single_table_iter_floor": None,
    # single-table budget ms (None = disabled)
    "single_table_budget_ms": None,
}

_VALID_KNOBS = set(_DEPLOYED_BASELINE.keys())


def _parse_spec(raw: str, label: str) -> dict:
    """Parse an arm spec JSON string; hard-fail on unknown knobs."""
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"[engine_ab_h2h] ERROR: {label} spec is not valid JSON: {e}\n  raw: {raw!r}")
    if not isinstance(spec, dict):
        sys.exit(f"[engine_ab_h2h] ERROR: {label} spec must be a JSON object (dict), got {type(spec).__name__}")
    unknown = set(spec.keys()) - _VALID_KNOBS
    if unknown:
        valid_list = "\n  ".join(sorted(_VALID_KNOBS))
        sys.exit(
            f"[engine_ab_h2h] ERROR: {label} spec contains unknown knob(s): {sorted(unknown)}\n"
            f"Valid knobs:\n  {valid_list}"
        )
    return spec


def _resolve(spec: dict, key: str):
    """Return spec[key] if present, else the deployed baseline default."""
    return spec.get(key, _DEPLOYED_BASELINE[key])


def _build_policy(spec: dict, snap, base_seed: int, repo_root: str):
    """Build a search policy function from a knob spec overriding the deployed baseline."""
    from pokerbot.eval.solver_exploitability import make_search_policy_fn

    iterations = int(_resolve(spec, "iterations"))
    time_budget_ms = int(_resolve(spec, "time_budget_ms"))
    search_preflop = bool(_resolve(spec, "search_preflop"))
    leaf_select = str(_resolve(spec, "leaf_select"))

    pf = make_search_policy_fn(
        snap,
        leaf_select=leaf_select,
        seed=base_seed,
        iterations=iterations,
        time_budget_ms=time_budget_ms,
        search_preflop=search_preflop,
    )

    # Fish the solver out of the closure (same pattern as the source scripts).
    sv = [
        c.cell_contents
        for c in pf.__closure__
        if hasattr(getattr(c, "cell_contents", None), "config")
    ][0]

    # Apply every config knob that maps directly to sv.config attributes.
    sv.config.native = True
    sv.config.repair_deck = bool(_resolve(spec, "repair_deck"))
    sv.config.two_sided = bool(_resolve(spec, "two_sided"))
    sv.config.blueprint_mix = float(_resolve(spec, "blueprint_mix"))
    sv.config.decouple_sampler = bool(_resolve(spec, "decouple_sampler"))
    sv.config.range_aware = bool(_resolve(spec, "range_aware"))
    sv.config.range_uniform_floor = float(_resolve(spec, "range_uniform_floor"))
    sv.config.range_reach_full = bool(_resolve(spec, "range_reach_full"))
    sv.config.ensemble_streams = int(_resolve(spec, "ensemble_streams"))
    sv.config.leaf_cache_worlds = int(_resolve(spec, "leaf_cache_worlds"))
    sv.config.avg_skip_frac = float(_resolve(spec, "avg_skip_frac"))
    sv.config.leaf_biased_continuations = bool(_resolve(spec, "leaf_biased_continuations"))
    sv.config.leaf_bias_mult = float(_resolve(spec, "leaf_bias_mult"))
    sv.config.serve_final_iterate = bool(_resolve(spec, "serve_final_iterate"))
    sv.config.max_depth = int(_resolve(spec, "max_depth"))
    sv.config.linear_averaging = bool(_resolve(spec, "linear_averaging"))
    sv.config.per_decision_seed = bool(_resolve(spec, "per_decision_seed"))
    sv.config.rng_salt = int(_resolve(spec, "rng_salt"))
    sv.config.frozen_read_response = bool(_resolve(spec, "frozen_read_response"))
    sv.config.rnr_restrict = bool(_resolve(spec, "rnr_restrict"))

    _bb = _resolve(spec, "blueprint_blend")
    if _bb is not None:
        sv.config.blueprint_blend = bool(_bb)

    _stif = _resolve(spec, "single_table_iter_floor")
    if _stif is not None:
        sv.config.single_table_iter_floor = int(_stif)

    _stbm = _resolve(spec, "single_table_budget_ms")
    if _stbm is not None:
        sv.config.single_table_budget_ms = int(_stbm)

    # Leaf config knobs
    sv.config.leaf.rollout_samples = int(_resolve(spec, "leaf_rollout_samples"))
    sv.config.leaf.gadget_iters = int(_resolve(spec, "leaf_gadget_iters"))
    sv.config.leaf.max_playout_depth = int(_resolve(spec, "leaf_max_playout_depth"))

    return pf


def _ev_settle(prev_env, action, settle_rng, settle_samples, apply_action_np,
               is_terminal_np, is_chance_node_np, sample_chance_np):
    """All-in EV settlement: integrate equity split over `settle_samples` runouts.

    The settle_rng is seeded from (base_seed, deck, step) — NOT from the rotation — so both
    rotations of the same deck produce the SAME EV estimate when both arms take identical
    actions (same prev_env + same action).  This is required for the exact-zero property:
    if spec_a == spec_b, per-deck diff must be exactly 0.0 on every deck.
    """
    ptr = int(prev_env.deck_ptr)
    base_deck = np.asarray(prev_env.deck).copy()
    tail = base_deck[ptr:].copy()
    if tail.shape[0] == 0:
        return np.asarray(apply_action_np(prev_env, action).rewards, dtype=np.float64)
    acc = np.zeros(6, dtype=np.float64)
    for _ in range(settle_samples):
        settle_rng.shuffle(tail)
        d = base_deck.copy()
        d[ptr:] = tail
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


def _nboard(env):
    return int((np.asarray(env.board) >= 0).sum())


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Generalized engine A/B H2H: compare two configs of our own engine."
    )
    parser.add_argument("--arm-a", required=True,
                        help="JSON dict of knobs for arm A (overrides deployed baseline; {} = exact deployed)")
    parser.add_argument("--arm-b", required=True,
                        help="JSON dict of knobs for arm B (overrides deployed baseline; {} = exact deployed)")
    parser.add_argument("--deck-start", type=int, default=0,
                        help="First deck index (for CRN-exact sharding across processes)")
    parser.add_argument("--deck-stride", type=int, default=1,
                        help="Deck index stride (round-robin sharding: start=i, stride=n_shards)")
    parser.add_argument("--decks", type=int, default=40,
                        help="Number of duplicate decks (each played twice, arms swapped)")
    parser.add_argument("--seed", type=int, default=4242, help="Base RNG seed")
    parser.add_argument("--settle-samples", type=int, default=200,
                        help="All-in EV runouts per all-in action (0 = disable settle)")
    parser.add_argument("--label", default="ab_h2h", help="Human label for this run")
    parser.add_argument("--out", default="artifacts/engine_ab_h2h.json",
                        help="Output JSON path")
    parser.add_argument("--repo-root", default=os.getcwd(),
                        help="Repo root (for blueprint auto-detect; default=cwd)")
    args = parser.parse_args()

    spec_a = _parse_spec(args.arm_a, "arm-a")
    spec_b = _parse_spec(args.arm_b, "arm-b")

    repo_root = args.repo_root
    base_seed = args.seed
    n_decks = args.decks
    settle_samples = args.settle_samples

    # Blueprint path resolution (arm-specific, defaulting to the deployed checkpoint).
    def _bp_path(spec):
        p = _resolve(spec, "blueprint_path")
        if p is None:
            p = f"{repo_root}/checkpoints/serve_blueprint_6max.pkl"
        return p

    bp_path_a = _bp_path(spec_a)
    bp_path_b = _bp_path(spec_b)

    # --- Imports (kept inside main so sys.path tricks work if needed) ---
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.search.pluribus_search import set_native_subgame
    from pokerbot.search import native_solver as _ns
    import pickle

    if not _ns.native_subgame_available():
        _ns.enable()
    print(
        f"native available={_ns.available()} subgame_available={_ns.native_subgame_available()} "
        f"rayon={_ns._EXT.rayon_num_threads()}",
        flush=True,
    )

    # Load blueprints (may be the same file; load separately so arm-b can use a different checkpoint).
    def _load_snap(path):
        raw = pickle.load(open(path, "rb"))
        return raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw

    print(f"Loading blueprint A: {bp_path_a}", flush=True)
    snap_a = _load_snap(bp_path_a)
    if bp_path_b == bp_path_a:
        snap_b = snap_a
    else:
        print(f"Loading blueprint B: {bp_path_b}", flush=True)
        snap_b = _load_snap(bp_path_b)

    occ = jax.numpy.array([True] * 6)

    # Build policies.
    print(f"\nBuilding arm A policy (spec={spec_a or '{}'})...", flush=True)
    pf_a = _build_policy(spec_a, snap_a, base_seed, repo_root)
    print(f"Building arm B policy (spec={spec_b or '{}'})...", flush=True)
    pf_b = _build_policy(spec_b, snap_b, base_seed, repo_root)

    # Resolved config summary.
    def _summary(spec):
        return {k: _resolve(spec, k) for k in sorted(_DEPLOYED_BASELINE.keys())}

    print(
        f"\n=== ENGINE A/B H2H: {args.label} ===\n"
        f"  decks={n_decks}  seed={base_seed}  settle_samples={settle_samples}\n"
        f"  ARM A: {spec_a or '(deployed baseline)'}\n"
        f"  ARM B: {spec_b or '(deployed baseline)'}\n"
        f"  3 A-seats vs 3 B-seats, each deck played twice (arms swapped)",
        flush=True,
    )

    # --- GAME LOOP ---
    # CRITICAL ZERO-GATE DESIGN: the action-sampling salt is:
    #   base_seed * 7 + deck_idx * 131 + step
    # This depends ONLY on (seed, deck, step) — NOT on which arm/rotation is acting.
    # When spec_a == spec_b, both arms produce identical probability vectors at every node,
    # and they sample the same action index from those vectors -> per-deck diff is EXACTLY 0.0.

    def act(pf, env, seat, salt):
        set_native_subgame(True)
        p = np.asarray(pf(env, seat), dtype=np.float64)
        pc = np.clip(p, 0, None)
        s = pc.sum()
        if s <= 0:
            m = legal_action_mask_np(env, seat).astype(np.float64)
            pc = m / m.sum()
        else:
            pc = pc / s
        # Salt depends ONLY on (deck_idx, step) — never on arm identity or rotation.
        return int(np.random.default_rng(salt).choice(len(pc), p=pc))

    a_total = 0.0
    b_total = 0.0
    per_deck = []
    flopdecks = 0
    t0 = time.perf_counter()

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    for _idx in range(n_decks):
        d = args.deck_start + _idx * args.deck_stride
        deck_a = 0.0
        deck_b = 0.0
        deck_saw_flop = False

        for rot in (0, 1):
            # Seat assignment: even seats = arm_a in rot=0; swapped in rot=1.
            is_a = {s: ((s % 2 == 0) == (rot == 0)) for s in range(6)}

            env = to_numpy_state(
                init_hand(jax.random.PRNGKey(base_seed * 100003 + d), occupancy=occ)
            )
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
                    # Salt is arm-INDEPENDENT: same for A and B at the same (deck, step).
                    salt = base_seed * 7 + d * 131 + g
                    pf = pf_a if is_a[int(seat)] else pf_b
                    a = act(pf, env, int(seat), salt)
                prev_env, prev_action, prev_nboard = env, a, _nboard(env)
                env = apply_action_np(env, a)

            # ZERO-GATE CRITICAL: settle_rng seeded from (base_seed, deck, step_count) ONLY —
            # never from the rotation.  When both arms are identical, `g` (total steps) is the
            # same in both rotations, so the settle draws the identical runouts and the EV
            # estimates cancel exactly, giving per-deck diff == 0.0.
            settle_rng = np.random.default_rng(base_seed * 991 + d * 61 + g)

            if settle_samples > 0 and prev_env is not None and _nboard(env) > prev_nboard:
                rew = _ev_settle(
                    prev_env, prev_action, settle_rng, settle_samples,
                    apply_action_np, is_terminal_np, is_chance_node_np, sample_chance_np,
                )
            else:
                rew = np.asarray(env.rewards, dtype=np.float64)

            # ZERO-GATE CRITICAL — rotation-symmetric accumulation.
            # Sum each rotation's arm rewards into per-rotation partials (fixed seat order),
            # then fold ONE partial per rotation into the deck totals.  When both arms are
            # identical, the per-seat rewards are bitwise identical across rotations, so
            # rot0's A-partial == rot1's B-partial bitwise (same values, same seat order)
            # and vice versa; deck_a = X + Y and deck_b = Y + X are then bitwise equal by
            # IEEE-754 addition commutativity -> per-deck diff EXACTLY 0.0 even when
            # _ev_settle returns fractional rewards.  (The previous interleaved
            # `deck_a += rew[s]` accumulated the same six floats in a rotation-dependent
            # ORDER — non-associativity produced deterministic ~1e-13 dust on settle decks.)
            rot_a = 0.0
            rot_b = 0.0
            for s in range(6):
                if is_a[s]:
                    rot_a += float(rew[s])
                else:
                    rot_b += float(rew[s])
            deck_a += rot_a
            deck_b += rot_b

        if deck_saw_flop:
            flopdecks += 1

        a_total += deck_a
        b_total += deck_b
        net = deck_a - deck_b
        per_deck.append(net)

        el = time.perf_counter() - t0
        pd_arr = np.array(per_deck)
        n = len(pd_arr)
        mean = float(pd_arr.mean())
        se = float(pd_arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        eta = el / n * (n_decks - n) / 60 if n < n_decks else 0.0
        print(
            f"  deck {d + 1}/{n_decks}: a-b net {net:+.1f} | "
            f"cum a {a_total:+.0f}/b {b_total:+.0f} | "
            f"mean/deck {mean:+.2f}+-{1.96 * se:.2f} | "
            f"flopdecks {flopdecks} | "
            f"{el / 60:.1f}min ETA {eta:.1f}min",
            flush=True,
        )

        # Incremental checkpoint.
        _checkpoint = {
            "label": args.label,
            "decks_done": len(per_deck),
            "arm_a": _summary(spec_a),
            "arm_b": _summary(spec_b),
            "a_total": a_total,
            "b_total": b_total,
            "per_deck_nets": per_deck,
            "flopdecks": flopdecks,
        }
        try:
            json.dump(_checkpoint, open(out_path, "w"), indent=1)
        except Exception:
            pass

    # --- Final summary ---
    pd_arr = np.array(per_deck)
    mean = float(pd_arr.mean())
    se = float(pd_arr.std(ddof=1) / math.sqrt(len(pd_arr))) if len(pd_arr) > 1 else 0.0
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    wall_s = time.perf_counter() - t0
    git_rev = _git_rev()

    verdict = (
        "A STRONGER (CI excludes 0)" if lo > 0 else
        ("B STRONGER (CI excludes 0)" if hi < 0 else "INCONCLUSIVE (CI includes 0)")
    )

    print(f"\n=== {args.label} (paired by deck, {len(pd_arr)} decks x 2 rotations) ===", flush=True)
    print(f"  ARM A spec: {spec_a or '(deployed baseline)'}")
    print(f"  ARM B spec: {spec_b or '(deployed baseline)'}")
    print(f"  A total:  {a_total:+.1f} chips")
    print(f"  B total:  {b_total:+.0f} chips")
    print(f"  A-B per deck: {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (per deck = 2 rotations)")
    print(f"  flopdecks: {flopdecks}/{n_decks}")
    print(f"  wall: {wall_s:.1f}s  git: {git_rev}")
    print(f"  VERDICT: {verdict}", flush=True)

    result = {
        "label": args.label,
        "arm_a": _summary(spec_a),
        "arm_b": _summary(spec_b),
        "decks": n_decks,
        "seed": base_seed,
        "settle_samples": settle_samples,
        "a_minus_b_per_deck": mean,
        "ci95": [lo, hi],
        "per_deck_nets": per_deck,
        "flopdecks": flopdecks,
        "a_total": a_total,
        "b_total": b_total,
        "wall_s": wall_s,
        "git_rev": git_rev,
        "verdict": verdict,
    }
    json.dump(result, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
