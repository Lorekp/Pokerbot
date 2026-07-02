#!/usr/bin/env python3
"""PARITY GATE for the native EMD frozen-lookup keying (battery_native fingerprint fast path).

Walks GAME-FLOW states ONLY (real hands dealt through the engine; standing rule: NEVER hand-build
or microbench postflop nodes — known nondeterministic native hangs). For every decision state the
acting seat's blueprint key is computed three ways:

  * py_key  — ``jax_weva_fast.fingerprint_fast`` under the DEPLOYED EMD-K160 serve env (the
              parity-verified Python reference the promoted blueprint is served with);
  * nat_key — ``battery_native.fingerprint_fast_emd`` (the new native EMD frozen-lookup keying that
              the native SubgameSolver / PlayoutEngine / Lbr6Harness now key with);
  * old_key — ``battery_native.fingerprint_fast`` (the PRE-PORT made-hand keying, kept only to
              reproduce the promotion's miskey measurement).

GATE: nat_key == py_key for EVERY state (and the blueprint row looked up under each key is
identical). Also reports the first-15-states-per-street summary, the promotion probe's shape:
the old made-hand keying measured 12/15 flop + 14/15 turn WRONG; the new keying must be 15/15/15.

Usage:
  python scripts/probe_native_emd_parity.py --min-postflop 500 --seed 7 \
      --out artifacts/native_emd_parity.json
  python scripts/probe_native_emd_parity.py --bench --bench-iters 2000   # + speedup measurement
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

# DEPLOYED EMD-K160 serve env — MUST be set before any pokerbot import (flags are read at import).
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

STREET_NAMES = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


def _street_of(env) -> int:
    return int((np.asarray(env.board) >= 0).sum())


def _pick_action(rng: random.Random, legal: "np.ndarray") -> int:
    """Passive-leaning GAME-FLOW action policy: mostly check/call so hands reach turn/river (a
    uniform-over-legal walk folds out preflop and never populates the later streets), with some
    bets/raises mixed in so postflop betting states vary. Real engine transitions only — no
    hand-built nodes."""
    legal_set = set(int(a) for a in legal)
    u = rng.random()
    passive = [a for a in (1, 2) if a in legal_set]           # CHECK / CALL
    aggro = [a for a in legal_set if a >= 3]                  # bet buckets / all-in
    if u < 0.75 and passive:
        return rng.choice(passive)
    if u < 0.92 and aggro:
        return rng.choice(aggro)
    return int(rng.choice(sorted(legal_set)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Native EMD keying parity gate (game-flow states only)")
    ap.add_argument("--min-postflop", type=int, default=500,
                    help="Minimum flop+turn+river decision states to collect (default 500)")
    ap.add_argument("--min-per-street", type=int, default=60,
                    help="Minimum decision states per postflop street (default 60)")
    ap.add_argument("--max-hands", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--blueprint", default=os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl"))
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts", "native_emd_parity.json"))
    ap.add_argument("--bench", action="store_true",
                    help="Also measure native-vs-python search decision time on a game-flow flop state")
    ap.add_argument("--bench-iters", type=int, default=2000)
    args = ap.parse_args()

    import jax
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, is_chance_node_np, legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.engine.jax_weva_fast import fingerprint_fast as py_fingerprint
    from pokerbot.search import native_solver as _ns

    if not _ns.available():
        print("FATAL: battery_native extension unavailable", flush=True)
        return 2
    ext = _ns._EXT
    if not hasattr(ext, "fingerprint_fast_emd") or not hasattr(ext, "native_caps"):
        print("FATAL: battery_native binary predates the EMD port (rebuild --features python)", flush=True)
        return 2
    print(f"native_caps={sorted(ext.native_caps())}", flush=True)
    ecfg = _ns._emd_native_key_config()
    if ecfg is None:
        print(f"FATAL: frozen blob missing: {_FROZEN}", flush=True)
        return 2
    blob_path, miss_made_hand = ecfg
    num_buckets = int(os.environ["POKERBOT_POSTFLOP_NUM_BUCKETS"])
    print(f"blob={blob_path} miss_made_hand={miss_made_hand} num_buckets={num_buckets}", flush=True)

    snap = None
    if os.path.exists(args.blueprint):
        import pickle
        raw = pickle.load(open(args.blueprint, "rb"))
        snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
        print(f"blueprint rows: {len(snap)} ({args.blueprint})", flush=True)
    else:
        print(f"WARNING: blueprint not found ({args.blueprint}) — key-parity only, no row check", flush=True)

    rng = random.Random(args.seed)
    counts = {s: 0 for s in ("preflop", "flop", "turn", "river")}
    match_new = {s: 0 for s in counts}
    match_old = {s: 0 for s in counts}
    first15_new = {s: [] for s in ("flop", "turn", "river")}
    first15_old = {s: [] for s in ("flop", "turn", "river")}
    row_mismatch = 0
    mismatches = []
    t0 = time.time()
    hands = 0
    bench_state = None

    def postflop_total() -> int:
        return counts["flop"] + counts["turn"] + counts["river"]

    def done_collecting() -> bool:
        return (postflop_total() >= args.min_postflop
                and all(counts[s] >= args.min_per_street for s in ("flop", "turn", "river")))

    while hands < args.max_hands and not done_collecting():
        key = jax.random.PRNGKey(rng.randrange(0, 2**31 - 1))
        env = to_numpy_state(init_hand(key, button=hands % 6))
        hands += 1
        depth = 0
        while not bool(env.done) and depth < 200:
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            seat = int(env.action_seat)
            if seat < 0:
                break
            mask = legal_action_mask_np(env, seat)
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            nb = _street_of(env)
            sname = STREET_NAMES.get(nb, "preflop")
            # --- the parity check on this GAME-FLOW decision state ---
            # (preflop is capped: it dominates the walk and the preflop key path is shared with the
            #  made-hand port that already has golden byte-identity gates)
            if sname == "preflop" and counts["preflop"] >= 2000:
                env = apply_action_np(env, _pick_action(rng, legal))
                depth += 1
                continue
            flat = _ns._to_flat(env).tobytes()
            py_key = bytes(py_fingerprint(env, seat))
            nat_key = bytes(ext.fingerprint_fast_emd(flat, seat, num_buckets, [], blob_path,
                                                     miss_made_hand))
            old_key = bytes(ext.fingerprint_fast(flat, seat, num_buckets))
            counts[sname] += 1
            ok_new = nat_key == py_key
            ok_old = old_key == py_key
            match_new[sname] += int(ok_new)
            match_old[sname] += int(ok_old)
            if sname in first15_new and len(first15_new[sname]) < 15:
                first15_new[sname].append(ok_new)
                first15_old[sname].append(ok_old)
            if snap is not None:
                py_row = snap.get(py_key)
                nat_row = snap.get(nat_key)
                rows_eq = ((py_row is None and nat_row is None)
                           or (py_row is not None and nat_row is not None
                               and np.array_equal(np.asarray(py_row), np.asarray(nat_row))))
                if not rows_eq:
                    row_mismatch += 1
            if not ok_new and len(mismatches) < 20:
                mismatches.append({
                    "street": sname, "seat": seat,
                    "board": [int(c) for c in np.asarray(env.board)],
                    "hole": [int(c) for c in np.asarray(env.holes)[seat]],
                    "py_key": py_key.hex(), "nat_key": nat_key.hex(),
                })
            if bench_state is None and sname == "flop":
                bench_state = env  # first game-flow flop decision state (for --bench)
            env = apply_action_np(env, _pick_action(rng, legal))
            depth += 1
        if hands % 100 == 0:
            el = time.time() - t0
            print(f"[progress] hands={hands} states={sum(counts.values())} "
                  f"(pre={counts['preflop']} f={counts['flop']} t={counts['turn']} r={counts['river']}) "
                  f"elapsed={el:.0f}s", flush=True)

    total = sum(counts.values())
    new_fail = {s: counts[s] - match_new[s] for s in counts}
    all_pass = all(v == 0 for v in new_fail.values()) and row_mismatch == 0 and done_collecting()

    def f15(d):
        return {s: f"{sum(v)}/{len(v)}" for s, v in d.items()}

    print("\n=== NATIVE EMD KEYING PARITY ===")
    print(f"hands={hands} states={total} per-street={dict(counts)}")
    print(f"NEW native EMD key == python key: "
          + ", ".join(f"{s}: {match_new[s]}/{counts[s]}" for s in counts))
    print(f"OLD native made-hand key == python key (miskey reference): "
          + ", ".join(f"{s}: {match_old[s]}/{counts[s]}" for s in counts))
    print(f"15-state probe (promotion measurement, NEW): {f15(first15_new)}  (must be 15/15 each)")
    print(f"15-state probe (promotion measurement, OLD): {f15(first15_old)}")
    print(f"blueprint row mismatches: {row_mismatch}" if snap is not None else "row check skipped")
    print(f"PARITY GATE: {'PASS' if all_pass else 'FAIL'}", flush=True)

    result = {
        "seed": args.seed, "hands": hands, "counts": counts,
        "match_new": match_new, "match_old": match_old,
        "first15_new": f15(first15_new), "first15_old": f15(first15_old),
        "row_mismatch": row_mismatch, "row_check": snap is not None,
        "blob": blob_path, "miss_made_hand": miss_made_hand,
        "pass": bool(all_pass), "mismatch_samples": mismatches,
    }

    # ------------------------------------------------------------------ bench
    if args.bench and all_pass:
        if bench_state is None:
            print("bench: no flop decision state captured — skipping", flush=True)
        elif snap is None:
            print("bench: blueprint missing — skipping", flush=True)
        else:
            from pokerbot.eval.solver_exploitability import make_search_policy_fn
            from pokerbot.search.pluribus_search import set_native_subgame

            seat = int(bench_state.action_seat)
            set_native_subgame(True)
            if not _ns.native_subgame_available():
                _ns.enable()
            print(f"\nbench: game-flow flop state, seat={seat}, K={args.bench_iters}", flush=True)

            def _time_policy(label):
                pf = make_search_policy_fn(snap, leaf_select="uniform", seed=args.seed,
                                           iterations=args.bench_iters, search_preflop=False)
                pf(bench_state, seat)  # warm-up (engine build + blob load excluded)
                ts = []
                for _ in range(3):
                    t1 = time.time()
                    pf(bench_state, seat)
                    ts.append(time.time() - t1)
                med = sorted(ts)[1]
                print(f"bench[{label}]: per-decision {[f'{x*1e3:.0f}ms' for x in ts]} median={med*1e3:.0f}ms",
                      flush=True)
                return med

            t_native = _time_policy("native")
            # Python arm: force the pre-port fail-closed gate (native subgame/rollout unavailable).
            saved = _ns._native_fingerprint_matches_serve_abstraction
            _ns._native_fingerprint_matches_serve_abstraction = lambda: False
            try:
                t_python = _time_policy("python")
            finally:
                _ns._native_fingerprint_matches_serve_abstraction = saved
            result["bench"] = {"iters": args.bench_iters, "native_ms": t_native * 1e3,
                               "python_ms": t_python * 1e3, "speedup": t_python / max(t_native, 1e-9)}
            print(f"bench: speedup = {t_python / max(t_native, 1e-9):.1f}x "
                  f"(python {t_python*1e3:.0f}ms -> native {t_native*1e3:.0f}ms)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
