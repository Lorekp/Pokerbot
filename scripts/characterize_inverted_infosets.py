#!/usr/bin/env python3
"""Characterize inverted Leduc infosets and hand-trace worst example."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pokerbot.engine.leduc_bridge import (
    apply_action_py,
    jax_from_py,
    py_info_key,
    py_legal,
    py_state_from_deal,
)
from pokerbot.engine.leduc_jax import CALL, FOLD, RAISE, ROUND_PRIVATE, ROUND_PUBLIC
from pokerbot.training.leduc_tabular_cfr import (
    _policy_for_key,
    tabular_cfr,
    all_leduc_deals,
)
from pokerbot.training.sd_cfr import (
    SDCFRConfig,
    SDCFRTrainer,
    encode_leduc_py,
    load_checkpoint,
)

ACTION_NAMES = ("FOLD", "CALL", "RAISE")


def dominant_action(sigma: np.ndarray, mask: np.ndarray) -> int:
    legal = [a for a in range(3) if mask[a]]
    return max(legal, key=lambda a: sigma[a])


def is_inverted(tab: np.ndarray, sd: np.ndarray, mask: np.ndarray, l1: float) -> bool:
    if l1 < 1.0:
        return False
    dt = dominant_action(tab, mask)
    ds = dominant_action(sd, mask)
    if dt == ds:
        return False
    # confident disagreement
    if tab[dt] < 0.55 or sd[ds] < 0.55:
        return False
    return True


def structural_features(ik: tuple) -> dict:
    seat, card, pub, rnd, to_call, raises, hist = ik
    facing_bet = to_call > 0
    return {
        "seat": seat,
        "round": "public" if rnd == ROUND_PUBLIC else "private",
        "round_id": int(rnd),
        "to_call": int(to_call),
        "facing_bet": facing_bet,
        "raises": int(raises),
        "hist_len": len(hist),
        "public_card": int(pub),
        "hole_rank": int(card),
        "node_type": "facing_bet" if facing_bet else "first_to_act",
    }


def cf_util_full(
    state,
    traverser: int,
    action: int,
    regret_fn,
) -> float:
    """Full-tree expected value for action (enumerate opponent responses below)."""
    child = apply_action_py(state, action)
    if child.done:
        return float(child.rewards[traverser])

    def expected(s) -> float:
        if s.done:
            return float(s.rewards[traverser])
        seat = s.acting
        mask = py_legal(s)
        if seat == traverser:
            sigma = regret_fn(s, seat)
            return float(
                sum(
                    sigma[a] * cf_util_full(s, traverser, a, regret_fn)
                    for a in range(3)
                    if mask[a]
                )
            )
        vals = []
        for a in range(3):
            if mask[a]:
                vals.append(cf_util_full(s, traverser, a, regret_fn))
        return float(np.mean(vals)) if vals else 0.0

    return expected(child)


def _policy_sigma(tabular_avg: dict, state, seat: int) -> np.ndarray:
    mask = py_legal(state)
    return _policy_for_key(tabular_avg, py_info_key(state, seat), mask)


def _cf_value(state, traverser: int, tabular_avg: dict) -> float:
    if state.done:
        return float(state.rewards[traverser])
    seat = state.acting
    mask = py_legal(state)
    if seat == traverser:
        sigma = _policy_sigma(tabular_avg, state, seat)
        return float(
            sum(
                sigma[a] * _cf_action_value(state, traverser, a, tabular_avg)
                for a in range(3)
                if mask[a]
            )
        )
    sigma = _policy_sigma(tabular_avg, state, seat)
    return float(
        sum(
            sigma[a] * _cf_value(apply_action_py(state, a), traverser, tabular_avg)
            for a in range(3)
            if mask[a]
        )
    )


def _cf_action_value(state, traverser: int, action: int, tabular_avg: dict) -> float:
    return _cf_value(apply_action_py(state, action), traverser, tabular_avg)


def hand_trace_state(state, traverser: int, tabular_avg: dict, trainer: SDCFRTrainer) -> dict:
    seat = state.acting
    mask = py_legal(state)
    ik = py_info_key(state, seat)
    tab = _policy_for_key(tabular_avg, ik, mask)
    sd = trainer.average_strategy_policy(jax_from_py(state), seat)
    reg = trainer._cum_regret.get(
        tuple(np.asarray(encode_leduc_py(state, seat)).tolist()),
        np.zeros(3),
    )

    util = np.zeros(3)
    for a in range(3):
        if mask[a]:
            util[a] = _cf_action_value(state, traverser, a, tabular_avg)

    sigma = trainer._regret_matching_sigma(state, seat)
    node_util = float(sum(sigma[a] * util[a] for a in range(3) if mask[a]))
    inst = {a: util[a] - node_util for a in range(3) if mask[a]}

    return {
        "ik": ik,
        "traverser": traverser,
        "acting_seat": seat,
        "mask": mask.tolist(),
        "util_traverser": {ACTION_NAMES[a]: util[a] for a in range(3) if mask[a]},
        "sigma_rm": {ACTION_NAMES[a]: float(sigma[a]) for a in range(3) if mask[a]},
        "node_util": node_util,
        "inst_no_reach": inst,
        "cum_regret": {ACTION_NAMES[a]: float(reg[a]) for a in range(3)},
        "tabular": {ACTION_NAMES[a]: float(tab[a]) for a in range(3)},
        "sd_avg": {ACTION_NAMES[a]: float(sd[a]) for a in range(3)},
        "pot": state.pot,
        "public": state.public,
        "round": "public" if state.round == ROUND_PUBLIC else "private",
    }


def find_state_for_ik(target_ik: tuple):
    for cards, pub in all_leduc_deals():
        stack = [py_state_from_deal(cards, pub)]
        seen = set()
        while stack:
            s = stack.pop()
            fp = (s.deal_public, s.cards, s.public, s.round, s.hist, s.pot)
            if fp in seen:
                continue
            seen.add(fp)
            if s.done:
                continue
            seat = s.acting
            if py_info_key(s, seat) == target_ik:
                return s, cards, pub
            m = py_legal(s)
            for a in range(3):
                if m[a]:
                    stack.append(apply_action_py(s, a))
    return None, None, None


def main() -> None:
    avg, _ = tabular_cfr(800, seed=0)
    trainer = SDCFRTrainer(SDCFRConfig())
    ckpt = Path("checkpoints/sd_cfr_leduc_gate/sd_cfr_leduc_latest.pkl")
    load_checkpoint(trainer, ckpt)

    inverted: list[dict] = []
    by_ik: dict[tuple, dict] = {}

    for cards, pub in all_leduc_deals():
        stack = [py_state_from_deal(cards, pub)]
        seen = set()
        while stack:
            s = stack.pop()
            fp = (s.deal_public, s.cards, s.public, s.round, s.hist, s.pot)
            if fp in seen:
                continue
            seen.add(fp)
            if s.done:
                continue
            seat = s.acting
            m = py_legal(s)
            ik = py_info_key(s, seat)
            if ik in by_ik:
                m = py_legal(s)
                for a in range(3):
                    if m[a]:
                        stack.append(apply_action_py(s, a))
                continue
            tab = _policy_for_key(avg, ik, m)
            sd = trainer.average_strategy_policy(jax_from_py(s), seat)
            l1 = float(np.sum(np.abs(tab - sd)))
            feat = structural_features(ik)
            feat["l1"] = l1
            feat["ik"] = ik
            feat["tabular_dom"] = ACTION_NAMES[dominant_action(tab, m)]
            feat["sd_dom"] = ACTION_NAMES[dominant_action(sd, m)]
            by_ik[ik] = feat
            if is_inverted(tab, sd, m, l1):
                inverted.append(feat)
            m = py_legal(s)
            for a in range(3):
                if m[a]:
                    stack.append(apply_action_py(s, a))

    print(f"=== (1) Inverted infosets: {len(inverted)} / 288 ===\n")
    if not inverted:
        return

    counters = {k: Counter() for k in ("seat", "round", "node_type", "facing_bet", "raises", "to_call")}
    for f in inverted:
        for k in counters:
            counters[k][f[k] if k != "facing_bet" else f["facing_bet"]] += 1

    for name, c in counters.items():
        print(f"  {name}: {dict(c)}")

    print("\n=== (2) Public round cluster? ===")
    pub_n = counters["round"].get("public", 0)
    priv_n = counters["round"].get("private", 0)
    print(f"  inverted in public round: {pub_n}/{len(inverted)} ({100*pub_n/len(inverted):.0f}%)")
    print(f"  inverted in private round: {priv_n}/{len(inverted)}")

    worst_ik = max(inverted, key=lambda f: f["l1"])["ik"]
    print(f"\n=== (3) Hand trace worst infoset ===")
    print(f"  ik={worst_ik}")
    state, cards, deal_pub = find_state_for_ik(worst_ik)
    if state is None:
        print("  state not found")
        return
    print(f"  deal cards={cards} fixed_public={deal_pub}")
    for trav in (0, 1):
        tr = hand_trace_state(state, trav, avg, trainer)
        print(f"\n  --- traverser P{trav} (acting seat {tr['acting_seat']}) ---")
        print(f"  round={tr['round']} pot={tr['pot']} public={tr['public']}")
        print(f"  util[traverser]: {tr['util_traverser']}")
        print(f"  node_util (RM-weighted): {tr['node_util']:.4f}")
        print(f"  inst (util-node, no reach): {tr['inst_no_reach']}")
        print(f"  cum_regret: {tr['cum_regret']}")
        print(f"  tabular: {tr['tabular']}  sd_avg: {tr['sd_avg']}  sigma_rm: {tr['sigma_rm']}")


if __name__ == "__main__":
    main()
