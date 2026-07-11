#!/usr/bin/env python3
"""LEG A (decision-quality audit): extract the SERVED preflop policy per position per 169-hand class.

The deployed engine serves preflop DIRECTLY from the blueprint (search_preflop=False), through
make_search_policy_fn -> PluribusReBeLSolver.solve -> _root_blueprint (adapter.blueprint * abstracted
legal). This script drives that EXACT serve path (the same `pf` stack engine_ab_h2h._build_policy
builds) on game-flow preflop nodes:

  RFI nodes: init_hand deal -> fold every seat that acts before the target position -> query hero.
  vs-RFI nodes: fold to the opener -> opener takes its own modal raise action -> fold intermediates
                -> query hero.

Per (position, hand-class): mean served action distribution, RFI/3bet/call/fold frequencies, and a
within-class consistency check (preflop abstraction is LOSSLESS-169, so all combos of a class must
serve the same row; any spread is a bug flag, not sampling noise).

Output: artifacts/audit_preflop_blueprint.json
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# POKERBOT_* env preamble BEFORE any pokerbot import (deployed EMD-K160 regime).
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
BET_BUCKET_START, ALL_IN = 3, 10
_RANKS = "23456789TJQKA"

POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")


def hand_class(c0: int, c1: int) -> str:
    r0, r1 = c0 // 4, c1 // 4
    s0, s1 = c0 % 4, c1 % 4
    hi, lo = max(r0, r1), min(r0, r1)
    if hi == lo:
        return _RANKS[hi] * 2
    return _RANKS[hi] + _RANKS[lo] + ("s" if s0 == s1 else "o")


def seat_of(button: int, pos: str) -> int:
    off = {"BTN": 0, "SB": 1, "BB": 2, "UTG": 3, "HJ": 4, "CO": 5}[pos]
    return (button + off) % 6


def preflop_order(button: int) -> list[int]:
    return [(button + 3 + i) % 6 for i in range(6)]  # UTG, HJ, CO, BTN, SB, BB


def main() -> int:
    import pickle
    import jax
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_terminal_np, legal_action_mask_np,
        to_numpy_state, _action_amount,
    )
    from pokerbot.eval.solver_exploitability import make_search_policy_fn

    n_deals = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    t0 = time.time()

    # Optional overrides (blueprint-v2 gate reuse): audit an arbitrary checkpoint without
    # touching the deployed serve pkl or the committed baseline artifact.
    #   AUDIT_PF_CHECKPOINT: path to a .pkl snapshot OR a raw .nlcfr native checkpoint
    #   AUDIT_PF_OUT:        output json path (default artifacts/audit_preflop_blueprint.json)
    ckpt = os.environ.get("AUDIT_PF_CHECKPOINT",
                          os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl"))
    if ckpt.endswith(".nlcfr"):
        # Pure-python NLCFR001 -> snapshot conversion; avg_strategy mirrors
        # trainer.rs::snapshot (row-normalized strat_sum, uniform fallback) exactly.
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        from f32_regret_strategy_delta import load_checkpoint, avg_strategy
        _ck = load_checkpoint(ckpt)
        snap = {k: avg_strategy(row) for k, row in _ck["strat"].items()}
    else:
        # Format-aware loader (legacy dict pkl OR nlcfr_stream_filter arrays_v1).
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        from engine_ab_h2h import load_snapshot
        snap = load_snapshot(ckpt)
    # The deployed serve stack (engine_ab_h2h `{}` baseline). At preflop nodes solve() returns
    # _root_blueprint without search, so iterations/native knobs are inert here — but we go through
    # the same wrapper to audit what is actually SERVED.
    pf = make_search_policy_fn(snap, leaf_select="uniform", seed=0, iterations=2000,
                               time_budget_ms=2000, search_preflop=False)

    occ = jax.numpy.array([True] * 6)

    # acc[node_kind][pos][class] -> list of 11-prob rows
    acc: dict[str, dict[str, dict[str, list]]] = {}
    open_sizes: dict[str, dict[int, list]] = {}   # pos -> action -> list of raise-to bb
    vs_meta: dict[str, dict] = {}

    def record(kind: str, pos: str, cls: str, probs: np.ndarray):
        acc.setdefault(kind, {}).setdefault(pos, {}).setdefault(cls, []).append(
            [round(float(x), 6) for x in probs])

    prog = open(os.path.join(REPO, "artifacts", "decision_quality_progress.log"), "a")

    def log(msg):
        prog.write(f"[{time.strftime('%F %T')}] [legA] {msg}\n")
        prog.flush()
        print(msg, flush=True)

    log(f"start extraction: {n_deals} deals")

    for d in range(n_deals):
        env0 = to_numpy_state(init_hand(jax.random.PRNGKey(910001 + d), occupancy=occ))
        btn = int(env0.button)
        order = preflop_order(btn)

        # ---- RFI nodes (fold to hero) ----
        for pi, pos in enumerate(("UTG", "HJ", "CO", "BTN", "SB")):
            e = env0
            ok = True
            for s in order[:pi]:
                if is_terminal_np(e) or current_player_np(e) != s:
                    ok = False
                    break
                e = apply_action_np(e, FOLD)
            if not ok or is_terminal_np(e):
                continue
            hero = seat_of(btn, pos)
            if current_player_np(e) != hero:
                continue
            probs = np.asarray(pf(e, hero), dtype=np.float64)
            cls = hand_class(int(e.holes[hero][0]), int(e.holes[hero][1]))
            record("rfi", pos, cls, probs)
            if d < 3:  # document the engine's open sizes once
                for a in range(BET_BUCKET_START, ALL_IN + 1):
                    if legal_action_mask_np(e, hero)[a]:
                        committed = int(e.committed_street[hero])
                        amt = _action_amount(e, hero, a)
                        open_sizes.setdefault(pos, {}).setdefault(a, []).append(
                            round((committed + amt) / 100.0, 2))

        # ---- vs-RFI nodes ----
        # (opener_pos, hero_pos, seats folded in between after the open)
        for opener_pos, hero_pos in (("BTN", "BB"), ("CO", "BTN"), ("BTN", "SB"), ("UTG", "BB")):
            oi = ("UTG", "HJ", "CO", "BTN", "SB", "BB").index(opener_pos)
            hi = ("UTG", "HJ", "CO", "BTN", "SB", "BB").index(hero_pos)
            e = env0
            ok = True
            for s in order[:oi]:
                if is_terminal_np(e) or current_player_np(e) != s:
                    ok = False
                    break
                e = apply_action_np(e, FOLD)
            if not ok or is_terminal_np(e):
                continue
            opener = seat_of(btn, opener_pos)
            if current_player_np(e) != opener:
                continue
            # Opener FORCED to the 2.5bb open (action 7 = 1.0x pot, the smallest raise in the
            # deployed preflop menu) — matches the published charts' 2.3-2.5x RFI assumption.
            # (The blueprint's own modal raise-given-raise is often the 4bb bucket or the jam,
            # which no published chart covers; forcing the chart-matched size keeps the vs-RFI
            # comparison apples-to-apples. Caveat: the blueprint rarely opens 2.5x itself, so the
            # hero response row may be off the blueprint's self-play distribution.)
            open_a = 7
            if not legal_action_mask_np(e, opener)[open_a]:
                continue
            key = f"{hero_pos}_vs_{opener_pos}"
            if key not in vs_meta:
                committed = int(e.committed_street[opener])
                op_probs = np.asarray(pf(e, opener), dtype=np.float64)
                vs_meta[key] = {
                    "opener_forced_action": int(open_a),
                    "open_to_bb": round((committed + _action_amount(e, opener, open_a)) / 100.0, 2),
                }
            e = apply_action_np(e, open_a)
            for s in order[oi + 1:hi]:
                if is_terminal_np(e) or current_player_np(e) != s:
                    ok = False
                    break
                e = apply_action_np(e, FOLD)
            if not ok or is_terminal_np(e):
                continue
            hero = seat_of(btn, hero_pos)
            if current_player_np(e) != hero:
                continue
            probs = np.asarray(pf(e, hero), dtype=np.float64)
            cls = hand_class(int(e.holes[hero][0]), int(e.holes[hero][1]))
            record(f"vs_{opener_pos}", hero_pos, cls, probs)

        if (d + 1) % 500 == 0:
            n_rfi = sum(len(v) for v in acc.get("rfi", {}).values())
            log(f"deal {d + 1}/{n_deals} ({100.0 * (d + 1) / n_deals:.0f}%) "
                f"rfi-classes={n_rfi} elapsed={time.time() - t0:.0f}s")

    # ---- Aggregate ----
    out = {"meta": {
        "n_deals": n_deals,
        "game": "6-max 100bb cash, no ante, SB 0.5bb/BB 1bb (engine defaults)",
        "serve_path": "make_search_policy_fn(search_preflop=False) -> _root_blueprint "
                      "(deployed engine_ab_h2h {} baseline preflop)",
        "action_layout": "0 fold, 1 check, 2 call, 3..9 pot-frac bets (1/3,.5,.66,.75,1,1.5,2), 10 all-in",
        "preflop_allowed_buckets": "raise menu = {1.0, 1.5, 2.0} x pot + all-in (adapter keep (4,5,6))",
        "open_sizes_bb_observed": open_sizes,
        "vs_rfi_meta": vs_meta,
    }, "nodes": {}}

    for kind, poss in acc.items():
        out["nodes"][kind] = {}
        for pos, classes in poss.items():
            rows = {}
            max_spread = 0.0
            for cls, plist in classes.items():
                arr = np.asarray(plist)
                mean = arr.mean(axis=0)
                spread = float(np.abs(arr - mean[None, :]).sum(axis=1).max()) if len(plist) > 1 else 0.0
                max_spread = max(max_spread, spread)
                raise_f = float(mean[BET_BUCKET_START:ALL_IN + 1].sum())
                rows[cls] = {
                    "n": len(plist),
                    "raise": round(raise_f, 4),
                    "call": round(float(mean[CALL]), 4),
                    "check": round(float(mean[CHECK]), 4),
                    "fold": round(float(mean[FOLD]), 4),
                    "allin": round(float(mean[ALL_IN]), 4),
                    "probs": [round(float(x), 5) for x in mean],
                    "spread_L1_max": round(spread, 5),
                }
            out["nodes"][kind][pos] = {
                "classes": rows,
                "n_classes": len(rows),
                "within_class_L1_spread_max": round(max_spread, 5),
            }

    path = os.environ.get("AUDIT_PF_OUT",
                          os.path.join(REPO, "artifacts", "audit_preflop_blueprint.json"))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    log(f"DONE wrote {path} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
