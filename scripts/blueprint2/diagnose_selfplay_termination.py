#!/usr/bin/env python3
"""Self-play street-termination diagnosis for a blueprint checkpoint.

Plays N hands of 6-max self-play (ALL seats = the same pure-tabular blueprint via
TabularBlueprintAgent) and records WHERE hands end and WHY — the empirical ground
for the "92.5% of self-play hands end preflop" stat and the postflop-coverage
starvation hypothesis.

Serve env (mode/buckets/blob) is set from argv BEFORE any pokerbot import (the
encoder bakes env at import), so run once per blueprint:

  python scripts/blueprint2/diagnose_selfplay_termination.py \
      --blueprint runs/native_cfr/bp_emd_k160_1.nlcfr --postflop-mode emd \
      --num-buckets 160 --emd-frozen data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn \
      --hands 4000 --out artifacts/bp2_selfplay_term_emd.json

  python scripts/blueprint2/diagnose_selfplay_termination.py \
      --blueprint checkpoints/serve_blueprints/bp_madehand_long1_iter160000.pkl \
      --postflop-mode fast --num-buckets 200 --hands 4000 \
      --out artifacts/bp2_selfplay_term_madehand.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--postflop-mode", choices=("fast", "emd"), required=True)
    ap.add_argument("--num-buckets", type=int, required=True)
    ap.add_argument("--emd-frozen", default=None)
    ap.add_argument("--hands", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def _configure_env(args) -> None:
    os.environ["POKERBOT_BETTING_ABSTRACTION"] = "1"
    os.environ["POKERBOT_POSTFLOP_BUCKETS"] = "1"
    os.environ["POKERBOT_PREFLOP_LOSSLESS"] = "1"
    os.environ["POKERBOT_POSTFLOP_NUM_BUCKETS"] = str(args.num_buckets)
    os.environ["POKERBOT_POSTFLOP_MODE"] = args.postflop_mode
    if args.postflop_mode == "emd":
        if not args.emd_frozen or not os.path.exists(args.emd_frozen):
            raise SystemExit("--emd-frozen path required (and must exist) for emd mode")
        os.environ["POKERBOT_POSTFLOP_EMD_FROZEN"] = args.emd_frozen
        os.environ["POKERBOT_EMD_FROZEN_MIRROR_FALLBACK"] = "1"
    os.environ["POKERBOT_BP_COVERAGE"] = "1"


def _load_snapshot(path: str):
    import numpy as np
    if path.endswith((".nlcfr", ".bin")):
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, ".."))
        from f32_regret_strategy_delta import avg_strategy, load_checkpoint
        ck = load_checkpoint(path)
        return {bytes(k): np.asarray(avg_strategy(row), dtype=np.float64)
                for k, row in ck["strat"].items()}
    import pickle
    with open(path, "rb") as f:
        obj = pickle.load(f)
    tab = obj.get("strat_sum", obj) if isinstance(obj, dict) else obj
    out = {}
    for k, v in tab.items():
        row = np.asarray(v, dtype=np.float64)
        s = float(row.sum())
        out[bytes(k)] = (row / s) if s > 0 else np.full(11, 1.0 / 11)
    return out


def main() -> int:
    args = _parse_args()
    _configure_env(args)

    import random

    from pokerbot.engine.game import Action, ActionType, GameState, Street, play_hand
    from pokerbot.eval import lbr
    from pokerbot.eval.blueprint_ranking import TabularBlueprintAgent

    snap = _load_snapshot(args.blueprint)
    print(f"[bp] {args.blueprint}: {len(snap)} infosets, mode={args.postflop_mode}/"
          f"{args.num_buckets}", flush=True)

    lbr.bp_coverage_reset()
    agents = {s: TabularBlueprintAgent(snap, random.Random(1000 + s)) for s in range(6)}

    S = {Street.PREFLOP: "preflop", Street.FLOP: "flop", Street.TURN: "turn",
         Street.RIVER: "river"}
    term_street = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
    fold_out = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}   # ended by folds (no showdown)
    allin_by = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}   # first all-in happened on street
    showdowns = 0
    walks = 0
    n_decisions = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
    act_hist_pre = [0] * 12  # 11 action kinds + raise split later; use type buckets below
    pre_kinds = {"fold": 0, "check": 0, "call": 0, "raise": 0, "allin": 0}
    pot_sum = 0.0

    t0 = time.time()
    rep_every = max(1, args.hands // 20)
    for h in range(args.hands):
        for s, ag in agents.items():
            ag.reseed(args.seed * 1_000_003 + h * 31 + s)
        state, deltas = play_hand(dict(agents), rng=random.Random(args.seed * 77 + h))

        hist = list(state.action_history_with_street())
        # Voluntary decisions only (skip blind posts recorded as history? play_hand history
        # includes only applied actions via apply_action -> all are decisions).
        last_street = None
        first_allin_street = None
        n_vol = 0
        for seat, action, street in hist:
            sname = S.get(street)
            if sname is None:
                continue
            n_vol += 1
            n_decisions[sname] += 1
            last_street = sname
            at = action.action_type
            if sname == "preflop":
                if at == ActionType.FOLD:
                    pre_kinds["fold"] += 1
                elif at == ActionType.CHECK:
                    pre_kinds["check"] += 1
                elif at == ActionType.CALL:
                    pre_kinds["call"] += 1
                elif at in (ActionType.RAISE, ActionType.BET):
                    # all-in raise = raise for the player's whole stack; detect via amount vs 100bb
                    if getattr(action, "amount", 0) and action.amount >= 9000:
                        pre_kinds["allin"] += 1
                    else:
                        pre_kinds["raise"] += 1
                else:
                    pre_kinds["raise"] += 1
            if first_allin_street is None and any(p.all_in for p in state.players):
                pass  # resolved post-hoc below (cheaper)
        if last_street is None:
            last_street = "preflop"
        term_street[last_street] += 1

        n_allin = sum(1 for p in state.players if p.all_in)
        went_showdown = state.active_player_count() >= 2 and len(state.board) == 5
        if went_showdown:
            showdowns += 1
        else:
            fold_out[last_street] += 1
        if n_allin >= 1:
            allin_by[last_street] += 1
        # walk = everyone folded to BB preflop with zero raises
        if last_street == "preflop" and not went_showdown and n_allin == 0:
            n_raises = sum(1 for _s, a, st in hist
                           if S.get(st) == "preflop" and a.action_type in (ActionType.RAISE, ActionType.BET))
            if n_raises == 0:
                walks += 1
        pot_sum += float(state.pot_bb)
        if (h + 1) % rep_every == 0:
            el = time.time() - t0
            print(f"[{h+1}/{args.hands}] {el:.0f}s  eta {el/(h+1)*(args.hands-h-1):.0f}s  "
                  f"pre-term so far {100.0*term_street['preflop']/(h+1):.1f}%", flush=True)

    n = args.hands
    total_dec = sum(n_decisions.values()) or 1
    cov = lbr.bp_coverage_report() if getattr(lbr, "_BP_COVERAGE_ON", False) else {}
    fb = sum(a.fallback_count for a in agents.values())
    dc = sum(a.decision_count for a in agents.values())

    out = {
        "blueprint": args.blueprint, "hands": n, "seed": args.seed,
        "mode": f"{args.postflop_mode}/{args.num_buckets}",
        "termination_street_pct": {k: 100.0 * v / n for k, v in term_street.items()},
        "fold_out_pct": {k: 100.0 * v / n for k, v in fold_out.items()},
        "hands_with_allin_by_last_street_pct": {k: 100.0 * v / n for k, v in allin_by.items()},
        "showdown_pct": 100.0 * showdowns / n,
        "walk_pct": 100.0 * walks / n,
        "decisions_per_street_pct": {k: 100.0 * v / total_dec for k, v in n_decisions.items()},
        "decisions_per_street_abs": n_decisions,
        "preflop_action_kinds": pre_kinds,
        "avg_final_pot_bb": pot_sum / n,
        "adapter_fallback_rate": fb / max(1, dc),
        "bp_coverage": {k: v for k, v in cov.items()} if cov else None,
    }
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
