#!/usr/bin/env python3
"""Inspect SD-CFR Leduc suspects: avg-strategy vs tabular, reach trace, derived CF bound."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pokerbot.engine.leduc_bridge import (
    PyLeducState,
    apply_action_py,
    collect_reachable_infosets,
    jax_from_py,
    py_info_key,
    py_legal,
    py_state_from_deal,
)
from pokerbot.engine.leduc_jax import NUM_ACTIONS
from pokerbot.training.leduc_tabular_cfr import (
    _policy_for_key,
    _terminal_reward_bounds,
    all_leduc_deals,
    tabular_cfr,
)
from pokerbot.training.sd_cfr import SDCFRConfig, SDCFRTrainer, load_checkpoint


def derive_instant_cf_bound() -> dict:
    """Max |opp_reach * (util[a]-node_util)| with opp_reach<=1 and exact continuation values."""
    deals = all_leduc_deals()
    roots = [py_state_from_deal(c, p) for c, p in deals]
    max_terminal = 0.0
    max_spread = 0.0

    def action_value(state: PyLeducState, traverser: int, action: int) -> float:
        child = apply_action_py(state, action)
        if child.done:
            return float(child.rewards[traverser])
        mask = py_legal(child)
        vals = [action_value(child, traverser, a) for a in range(NUM_ACTIONS) if mask[a]]
        return float(np.mean(vals)) if vals else 0.0

    def walk(state: PyLeducState, traverser: int) -> None:
        nonlocal max_spread
        if state.done:
            return
        mask = py_legal(state)
        utils = [action_value(state, traverser, a) for a in range(NUM_ACTIONS) if mask[a]]
        if utils:
            max_spread = max(max_spread, max(utils) - min(utils))
        for a in range(NUM_ACTIONS):
            if mask[a]:
                walk(apply_action_py(state, a), traverser)

    for root in roots:
        for trav in (0, 1):
            walk(root, trav)
            lo, hi = _terminal_reward_bounds(root, trav)
            max_terminal = max(max_terminal, abs(lo), abs(hi))

    # Tight bound: opp_reach in [0,1], |util[a]-node_util| <= max spread at node
    # node_util is convex combo of util[a], so spread <= max_a util[a] - min_a util[a]
    bound = float(max_spread)  # opp_reach max = 1.0
    return {
        "max_abs_terminal_reward": max_terminal,
        "max_action_value_spread": max_spread,
        "derived_instant_cf_bound": bound,
        "formula": "max opp_reach (<=1) * max_a,b |V(a)-V(b)| over traverser nodes",
    }


def hand_trace_reach() -> dict:
    """Deal (0,1)/pub=2: private CC, then trace one external-sampling path for traverser=0."""
    root = py_state_from_deal((0, 1), 2)
    s = apply_action_py(apply_action_py(root, 1), 1)  # CALL=1 both streets private
    assert s.public == 2 and s.round == 1

    trainer = SDCFRTrainer(SDCFRConfig(iterations=1, traversals_per_iter=1, seed=99))
    path_log: list[dict] = []

    def traced_traverse(
        state: PyLeducState,
        traverser: int,
        key,
        p0: float = 1.0,
        p1: float = 1.0,
    ) -> float:
        if state.done:
            return float(state.rewards[traverser])
        seat = state.acting
        mask = py_legal(state)
        opp_reach = p1 if traverser == 0 else p0
        if seat == traverser:
            util = np.zeros(NUM_ACTIONS, dtype=np.float64)
            for a in range(NUM_ACTIONS):
                if mask[a]:
                    util[a] = traced_traverse(apply_action_py(state, a), traverser, key, p0, p1)
            sigma = trainer._regret_matching_sigma(state, seat)
            node_util = float(sum(sigma[a] * util[a] for a in range(NUM_ACTIONS) if mask[a]))
            inst = {a: opp_reach * (util[a] - node_util) for a in range(NUM_ACTIONS) if mask[a]}
            path_log.append(
                {
                    "seat": seat,
                    "traverser": traverser,
                    "p0": p0,
                    "p1": p1,
                    "opp_reach": opp_reach,
                    "util": {a: float(util[a]) for a in inst},
                    "node_util": node_util,
                    "inst": inst,
                    "info_key": py_info_key(state, seat),
                }
            )
            return node_util
        sigma = trainer._regret_matching_sigma(state, seat)
        legal_idx = np.flatnonzero(mask)
        # deterministic: CALL when legal so path reaches a traverser decision
        call_idx = np.where(legal_idx == 1)[0]
        pick = int(call_idx[0]) if len(call_idx) else 0
        action = int(legal_idx[pick])
        np0 = p0 * sigma[action] if seat == 0 else p0
        np1 = p1 * sigma[action] if seat == 1 else p1
        path_log.append(
            {
                "seat": seat,
                "traverser": traverser,
                "action": action,
                "sigma_action": float(sigma[action]),
                "p0_before": p0,
                "p1_before": p1,
                "p0_after": np0,
                "p1_after": np1,
            }
        )
        return traced_traverse(apply_action_py(state, action), traverser, key, np0, np1)

    traced_traverse(s, traverser=0, key=None, p0=1.0, p1=1.0)

    trav_node = next((e for e in path_log if "opp_reach" in e), None)
    opp_steps = [e for e in path_log if "p1_after" in e]
    manual_p1 = 1.0
    for e in opp_steps:
        if e["seat"] == 1:
            manual_p1 = e["p1_after"]

    match = trav_node is not None and abs(trav_node["opp_reach"] - manual_p1) < 1e-9

    return {
        "path_steps": path_log,
        "traverser_node": trav_node,
        "manual_p1_product": manual_p1,
        "opp_reach_matches_manual": match,
    }


def compare_avg_vs_tabular(
    avg: dict,
    trainer: SDCFRTrainer,
    *,
    final_iterate_only: bool = False,
) -> dict:
    deals = all_leduc_deals()
    roots = [py_state_from_deal(c, p) for c, p in deals]
    infosets: set[tuple] = set()
    for r in roots:
        infosets |= collect_reachable_infosets([r])

    policy_fn = (
        trainer.regret_matching_policy
        if final_iterate_only
        else trainer.average_strategy_policy
    )

    l1_list: list[float] = []
    max_l1 = 0.0
    worst_key = None
    n_compared = 0
    for key in sorted(infosets):
        seat = key[0]
        # reconstruct a representative state from key is hard; walk tree instead
        pass

    # Walk tree and compare at each acting node
    stack = list(roots)
    seen: set[tuple] = set()
    samples: list[dict] = []
    while stack:
        state = stack.pop()
        fp = (state.deal_public, state.cards, state.public, state.round, state.hist, state.pot)
        if fp in seen:
            continue
        seen.add(fp)
        if state.done:
            continue
        seat = state.acting
        mask = py_legal(state)
        key = py_info_key(state, seat)
        tab = _policy_for_key(avg, key, mask)
        jax_s = jax_from_py(state)
        sd = policy_fn(jax_s, seat)
        l1 = float(np.sum(np.abs(tab - sd)))
        l1_list.append(l1)
        n_compared += 1
        if l1 > max_l1:
            max_l1 = l1
            worst_key = key
            worst = {"key": key, "tabular": tab.tolist(), "sd_cfr": sd.tolist(), "l1": l1}
        for a in range(NUM_ACTIONS):
            if mask[a]:
                stack.append(apply_action_py(state, a))
    l1_arr = np.asarray(l1_list)
    return {
        "n_infosets": n_compared,
        "mean_l1": float(l1_arr.mean()),
        "median_l1": float(np.median(l1_arr)),
        "max_l1": max_l1,
        "worst": worst if n_compared else None,
        "frac_l1_gt_0.25": float((l1_arr > 0.25).mean()),
        "frac_l1_gt_0.5": float((l1_arr > 0.5).mean()),
    }


def policy_from_cumulative_regrets(regret: dict) -> callable:
    """Oracle: regret-match tabular cumulative regrets (what tabular CFR uses per-iter)."""

    def policy(state, seat: int) -> np.ndarray:
        from pokerbot.engine.leduc_jax import legal_actions

        mask = np.asarray(legal_actions(state), dtype=bool)
        key = py_info_key(PyLeducState.from_jax(state), seat)
        reg = np.asarray(regret.get(key, np.zeros(NUM_ACTIONS)), dtype=np.float64)
        positive = np.maximum(reg, 0.0) * mask
        s = positive.sum()
        if s > 0:
            return positive / s
        return mask.astype(np.float64) / max(mask.sum(), 1)

    return policy


def compare_snapshot_weighting(trainer: SDCFRTrainer, avg: dict) -> dict:
    """Check linear-CFR weights vs uniform snapshot weights on same net snapshots."""
    deals = all_leduc_deals()
    roots = [py_state_from_deal(c, p) for c, p in deals[:5]]
    state = roots[0]
    while not state.done:
        mask = py_legal(state)
        if mask.any():
            break
        state = apply_action_py(state, 0)
    seat = state.acting
    jax_s = jax_from_py(state)
    key = py_info_key(state, seat)
    tab = _policy_for_key(avg, key, mask)

    linear = trainer.average_strategy_policy(jax_s, seat)
    if not trainer._strategy_snapshots:
        return {"error": "no snapshots"}

    agg_u = np.zeros(NUM_ACTIONS)
    for _, params in trainer._strategy_snapshots:
        agg_u += trainer._sigma_from_params(params, jax_s, seat)
    uniform = agg_u / len(trainer._strategy_snapshots)
    uniform = uniform * mask
    uniform = uniform / uniform.sum()

    return {
        "info_key": key,
        "tabular_avg": tab.tolist(),
        "sd_linear_cfr_avg": linear.tolist(),
        "sd_uniform_snapshot_avg": uniform.tolist(),
        "l1_linear_vs_tabular": float(np.sum(np.abs(tab - linear))),
        "l1_uniform_vs_tabular": float(np.sum(np.abs(tab - uniform))),
        "n_snapshots": len(trainer._strategy_snapshots),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None, help="SD-CFR .pkl (optional)")
    p.add_argument("--tabular-iters", type=int, default=800)
    args = p.parse_args()

    print("=== Derived instantaneous CF bound (from game tree) ===")
    bound_info = derive_instant_cf_bound()
    for k, v in bound_info.items():
        print(f"  {k}: {v}")

    print("\n=== Hand-traced reach / advantage (deterministic path) ===")
    trace = hand_trace_reach()
    for step in trace["path_steps"][:6]:
        print(f"  {step}")
    if trace.get("traverser_node"):
        print(f"  traverser_node: {trace['traverser_node']}")

    print(f"\n=== Tabular CFR ({args.tabular_iters} iters) for avg-strategy reference ===")
    avg, regret = tabular_cfr(iterations=args.tabular_iters, seed=0)

    trainer = SDCFRTrainer(SDCFRConfig())
    if args.checkpoint and args.checkpoint.exists():
        n = load_checkpoint(trainer, args.checkpoint)
        print(f"Loaded checkpoint {args.checkpoint} (iter={n}, snapshots={len(trainer._strategy_snapshots)})")
    else:
        print("No checkpoint — comparing weighting logic only (untrained net).")

    trainer.clear_policy_cache()
    print("\n=== Suspect 1: recovered avg strategy vs tabular (full tree) ===")
    comp_avg = compare_avg_vs_tabular(avg, trainer, final_iterate_only=False)
    comp_final = compare_avg_vs_tabular(avg, trainer, final_iterate_only=True)
    for label, comp in [("average_strategy_policy", comp_avg), ("regret_matching_policy (final iter)", comp_final)]:
        print(f"  [{label}]")
        print(f"    n_infosets={comp['n_infosets']} mean_l1={comp['mean_l1']:.4f} "
              f"median_l1={comp['median_l1']:.4f} max_l1={comp['max_l1']:.4f}")
        print(f"    frac_l1>0.25={comp['frac_l1_gt_0.25']:.3f} frac_l1>0.5={comp['frac_l1_gt_0.5']:.3f}")
        if comp.get("worst"):
            print(f"    worst: {comp['worst']}")

    if trainer._strategy_snapshots:
        print("\n=== Snapshot weighting spot-check (linear vs uniform) ===")
        w = compare_snapshot_weighting(trainer, avg)
        for k, v in w.items():
            print(f"  {k}: {v}")

    print("\n=== Suspect 2 architecture: policy uses regret-match(net), net trained on instantaneous ===")
    print("  Tabular sigma = regret_match(cumulative_regret)")
    print("  SD-CFR sigma  = regret_match(network_output) where network targets opp_reach*(util[a]-node_util)")
    print("  These are different objects unless the net accumulates regrets externally (it does not).")


if __name__ == "__main__":
    main()
