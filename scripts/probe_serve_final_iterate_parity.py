#!/usr/bin/env python3
"""R5a parity probe: serve_final_iterate Python-loop vs native, ON vs OFF, at fixed equal K.

Bitwise Python-vs-native equality is impossible BY DESIGN (the native solve uses a fast xoshiro RNG
in place of numpy's PCG64 — the documented "statistically equivalent, not bit-identical" contract),
so this probe supplies the state-flip / L1 parity evidence instead:

  * EXTRACTION SEMANTICS are proven exactly elsewhere (Rust golden
    `n7_subgame_serve_final_iterate_golden_extraction` + Python golden
    `test_serve_final_iterate_golden_and_zero_gate`): both paths serve regret_match(root cumulative
    CFR+ regrets after the last iteration) when ON and the averaged strategy_sum when OFF.
  * This probe shows the knob ENGAGES on both paths at real deployed-config postflop states:
    per-state L1(ON, OFF) and argmax flips for the Python loop and for the native solve, plus the
    cross-path L1 at each knob state (bounded RNG-level divergence, no systematic semantic gap).

Config: eval base arm shape shared by BOTH paths (two_sided + repair_deck + decouple_sampler +
leaf_cache_worlds=4; range_aware OFF because it is native-only — the Python loop samples holes via
PBSTracker, so leaving it on would compare different sampling models, not the knob).

Writes artifacts/serve_final_iterate_parity.json.
"""
from __future__ import annotations

import json
import os

# POKERBOT_* env before ANY pokerbot import (deployed EMD-K160 regime, mirrors engine_ab_h2h.py).
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "emd")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "160")
os.environ.setdefault("POKERBOT_POSTFLOP_EMD_FROZEN", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "abstraction",
    "emd_frozen_flop_turn_160_r400.emdfrzn")))
os.environ.setdefault("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "1")
os.environ.setdefault("POKERBOT_SEARCH_BLUEPRINT_BLEND", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax

K = 300          # fixed iteration count, identical across all four (path, knob) cells
N_STATES = 8     # postflop game-flow states (dealt hands driven to the flop)
SEEDS = (7, 11, 23, 31, 47, 59, 71, 83)


def main():
    import pickle

    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        CALL, CHECK, apply_action_np, is_chance_node_np, legal_action_mask_np,
        sample_chance_np, to_numpy_state,
    )
    from pokerbot.eval.solver_exploitability import make_search_policy_fn
    from pokerbot.search.pluribus_search import set_native_subgame
    from pokerbot.search import native_solver as _ns

    if not _ns.native_subgame_available():
        _ns.enable()
    assert _ns.native_subgame_available(), "native SubgameSolver unavailable — rebuild the .pyd"

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw = pickle.load(open(os.path.join(repo, "checkpoints", "serve_blueprint_6max.pkl"), "rb"))
    snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw

    pf = make_search_policy_fn(
        snap, leaf_select="uniform", seed=0, iterations=K,
        time_budget_ms=10 ** 9, search_preflop=False,
    )
    sv = [c.cell_contents for c in pf.__closure__
          if hasattr(getattr(c, "cell_contents", None), "config")][0]
    sv.config.native = True
    sv.config.two_sided = True
    sv.config.repair_deck = True
    sv.config.decouple_sampler = True
    sv.config.range_aware = False   # native-only lever: OFF so both paths share the sampling model
    sv.config.leaf_cache_worlds = 4
    sv.config.blueprint_blend = False

    occ = jax.numpy.array([True] * 6)

    def postflop_state(seed):
        env = to_numpy_state(init_hand(jax.random.PRNGKey(seed), occupancy=occ))
        for _ in range(60):
            if bool(env.done) or is_chance_node_np(env):
                break
            seat = int(env.action_seat)
            if seat < 0:
                break
            m = legal_action_mask_np(env, seat).astype(bool)
            env = apply_action_np(env, CALL if m[CALL] else CHECK)
        while (not bool(env.done)) and is_chance_node_np(env):
            env = sample_chance_np(env)
        return env

    rows = []
    for seed in SEEDS[:N_STATES]:
        env = postflop_state(seed)
        if bool(env.done) or int(env.street) == 0:
            continue
        seat = int(env.action_seat)
        cell = {}
        for native in (False, True):
            set_native_subgame(native)
            for flag in (False, True):
                sv.config.serve_final_iterate = flag
                p = np.asarray(pf(env, seat), dtype=np.float64)
                cell[("nat" if native else "py") + ("_on" if flag else "_off")] = p
        r = {
            "seed": int(seed),
            "street": int(env.street),
            "py_l1_on_off": float(np.abs(cell["py_on"] - cell["py_off"]).sum()),
            "nat_l1_on_off": float(np.abs(cell["nat_on"] - cell["nat_off"]).sum()),
            "py_flip": bool(int(cell["py_on"].argmax()) != int(cell["py_off"].argmax())),
            "nat_flip": bool(int(cell["nat_on"].argmax()) != int(cell["nat_off"].argmax())),
            "cross_l1_off": float(np.abs(cell["py_off"] - cell["nat_off"]).sum()),
            "cross_l1_on": float(np.abs(cell["py_on"] - cell["nat_on"]).sum()),
        }
        rows.append(r)
        print(f"seed={seed} street={r['street']} "
              f"py L1(on,off)={r['py_l1_on_off']:.4f} flip={r['py_flip']} | "
              f"nat L1(on,off)={r['nat_l1_on_off']:.4f} flip={r['nat_flip']} | "
              f"cross L1 off={r['cross_l1_off']:.4f} on={r['cross_l1_on']:.4f}", flush=True)

    n = len(rows)
    summary = {
        "K": K,
        "n_states": n,
        "py_knob_nonzero": int(sum(r["py_l1_on_off"] > 1e-9 for r in rows)),
        "nat_knob_nonzero": int(sum(r["nat_l1_on_off"] > 1e-9 for r in rows)),
        "py_mean_l1_on_off": float(np.mean([r["py_l1_on_off"] for r in rows])) if n else None,
        "nat_mean_l1_on_off": float(np.mean([r["nat_l1_on_off"] for r in rows])) if n else None,
        "cross_mean_l1_off": float(np.mean([r["cross_l1_off"] for r in rows])) if n else None,
        "cross_mean_l1_on": float(np.mean([r["cross_l1_on"] for r in rows])) if n else None,
        "rows": rows,
    }
    print("\n=== R5a serve_final_iterate parity summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    out = os.path.join(repo, "artifacts", "serve_final_iterate_parity.json")
    json.dump(summary, open(out, "w"), indent=1)
    print(f"wrote {out}")
    # Gate: the knob must engage on BOTH paths (nonzero L1 on at least one state each).
    assert summary["py_knob_nonzero"] > 0, "knob inert on the Python loop"
    assert summary["nat_knob_nonzero"] > 0, "knob inert on the native path"
    print("PARITY/ENGAGEMENT GATE: PASS")


if __name__ == "__main__":
    main()
