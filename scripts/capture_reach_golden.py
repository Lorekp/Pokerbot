#!/usr/bin/env python3
"""Stage 2a gate 3 — reach-table BYTE-IDENTITY golden (engine-generated game-flow states).

Two modes:
  --capture : collect >=12 engine-generated flop/turn decision states (game-flow deals, NEVER
              hand-built nodes; same collection pattern as probe_live_reach_replay_perturb.py),
              probe `SubgameSolver.range_weights(state, hero, "reach_strict")` with the CURRENTLY
              INSTALLED pyd, and save {state bytes hex, hero, street, table sha256, table len}
              to artifacts/reach_golden_pre2a.json.  Run this BEFORE rebuilding the pyd.
  --compare : reload the saved state BYTES (exact same inputs, no regeneration), re-probe with the
              currently installed pyd, and require sha256(table.tobytes()) byte-identical per state.
              Run this AFTER rebuilding.  Any mismatch => exit 1 (gate FAIL).

Env preamble MUST precede all pokerbot imports (standing rule).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

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
GOLDEN_PATH = os.path.join(REPO, "artifacts", "reach_golden_pre2a.json")


def _pick_action(rng: random.Random, legal: "np.ndarray") -> int:
    legal_set = set(int(a) for a in np.flatnonzero(legal))
    u = rng.random()
    passive = [a for a in (CHECK, CALL) if a in legal_set]
    aggro = [a for a in legal_set if a >= 3]
    if u < 0.70 and passive:
        return rng.choice(passive)
    if u < 0.95 and aggro:
        return rng.choice(aggro)
    return int(rng.choice(sorted(legal_set)))


def _build_engine():
    import pickle
    from pokerbot.search import native_solver as _ns
    from pokerbot.eval.solver_exploitability import make_search_policy_fn
    from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS

    if not _ns.native_subgame_available():
        _ns.enable()
    if not _ns.native_subgame_available():
        raise SystemExit("FATAL: native SubgameSolver unavailable")
    raw = pickle.load(open(os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl"), "rb"))
    snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
    pf = make_search_policy_fn(snap, leaf_select="uniform", seed=7, iterations=200,
                               time_budget_ms=2000, search_preflop=False)
    sv = [c.cell_contents for c in pf.__closure__
          if hasattr(getattr(c, "cell_contents", None), "config")][0]
    eng = _ns._build_subgame_engine(sv.adapter, int(_POSTFLOP_NUM_BUCKETS),
                                    int(sv.config.leaf.max_playout_depth))
    if eng is None:
        raise SystemExit("FATAL: could not build native subgame engine")
    return eng


def _collect_states(n_flop: int = 8, n_turn: int = 6):
    """Engine-generated game-flow flop/turn decision states (>=1 bet/raise in history preferred)."""
    import jax
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
    )

    rng = random.Random(20260703)
    occ = jax.numpy.array([True] * 6)
    flop_cases, turn_cases = [], []
    for d in range(8000):
        if len(flop_cases) >= n_flop and len(turn_cases) >= n_turn:
            break
        e = to_numpy_state(init_hand(jax.random.PRNGKey(880001 + d), occupancy=occ))
        guard = 0
        while not is_terminal_np(e) and guard < 80:
            guard += 1
            if is_chance_node_np(e):
                e = sample_chance_np(e)
                continue
            s = current_player_np(e)
            if s < 0:
                break
            n_board = int((np.asarray(e.board) >= 0).sum())
            ha = np.asarray(e.hist_action)
            nh = int((np.asarray(e.hist_seat) >= 0).sum())
            has_bet = any(int(ha[i]) >= 3 for i in range(nh))
            if n_board == 3 and has_bet and len(flop_cases) < n_flop:
                flop_cases.append((e, int(s), "flop"))
                break
            if n_board == 4 and len(turn_cases) < n_turn:
                turn_cases.append((e, int(s), "turn"))
                break
            m = legal_action_mask_np(e, s).astype(bool)
            e = apply_action_np(e, _pick_action(rng, m))
    return flop_cases + turn_cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.capture == args.compare:
        raise SystemExit("pass exactly one of --capture / --compare")

    from pokerbot.search import native_solver as _ns
    eng = _build_engine()

    if args.capture:
        cases = _collect_states()
        print(f"collected {len(cases)} engine game-flow states", flush=True)
        entries = []
        for (e, hero, street) in cases:
            sb = _ns._to_flat(e).tobytes()
            w = eng.range_weights(sb, hero, "reach_strict")
            if w is None:
                print(f"WARN: pristine engine state returned None (street={street}) — skipped")
                continue
            arr = np.asarray(w, dtype=np.float64)
            entries.append({
                "street": street,
                "hero": hero,
                "state_hex": sb.hex(),
                "table_len": int(arr.size),
                "table_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "table_sum": float(arr.sum()),
            })
        if len(entries) < 12:
            raise SystemExit(f"FAIL: only {len(entries)} usable golden states (<12)")
        with open(GOLDEN_PATH, "w") as f:
            json.dump({"n": len(entries), "entries": entries}, f, indent=1)
        print(f"wrote {GOLDEN_PATH} with {len(entries)} goldens", flush=True)
        return 0

    # --compare
    with open(GOLDEN_PATH) as f:
        gold = json.load(f)
    n_ok = 0
    n_bad = 0
    for i, ent in enumerate(gold["entries"]):
        sb = bytes.fromhex(ent["state_hex"])
        w = eng.range_weights(sb, ent["hero"], "reach_strict")
        if w is None:
            print(f"[{i}] FAIL: post-change probe returned None (was OK pre-change)")
            n_bad += 1
            continue
        arr = np.asarray(w, dtype=np.float64)
        h = hashlib.sha256(arr.tobytes()).hexdigest()
        if h == ent["table_sha256"] and int(arr.size) == ent["table_len"]:
            n_ok += 1
        else:
            print(f"[{i}] FAIL: table hash mismatch street={ent['street']} "
                  f"pre={ent['table_sha256'][:16]} post={h[:16]}")
            n_bad += 1
    print(f"byte-identity: {n_ok}/{len(gold['entries'])} identical, {n_bad} mismatched", flush=True)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
