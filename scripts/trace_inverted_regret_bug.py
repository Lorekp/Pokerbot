#!/usr/bin/env python3
"""Compare hand CF values, SD traverse utils, and cum_regret at inverted infoset."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokerbot.engine.leduc_bridge import apply_action_py, py_info_key, py_legal, py_state_from_deal
from pokerbot.engine.leduc_jax import CALL, FOLD
from pokerbot.training.leduc_tabular_cfr import _policy_for_key, tabular_cfr
from pokerbot.training.sd_cfr import SDCFRConfig, SDCFRTrainer, encode_leduc_py, load_checkpoint

TARGET_IK = (0, 2, 1, 1, 2, 2, (1, 1, 1, 2, 2))


def tabular_regret_at(avg, regret, state, seat):
    ik = py_info_key(state, seat)
    mask = py_legal(state)
    reg = regret.get(ik, np.zeros(3))
    pol = _policy_for_key(avg, ik, mask)
    return ik, reg, pol


def main() -> None:
    state = py_state_from_deal((2, 0), 1)
    # Reach public infoset: private CC then public betting to match hist (1,1,1,2,2)
    for a in (CALL, CALL):
        state = apply_action_py(state, a)
    # Build path: need hist (1,1,1,2,2) on public round seat 0 facing to_call=2
    # Walk from deal to find matching state
    from pokerbot.training.leduc_tabular_cfr import all_leduc_deals
    from pokerbot.engine.leduc_bridge import apply_action_py as ap

    found = None
    for cards, pub in all_leduc_deals():
        if cards != (2, 0) or pub != 1:
            continue
        stack = [py_state_from_deal(cards, pub)]
        seen = set()
        while stack:
            s = stack.pop()
            fp = (s.deal_public, s.cards, s.public, s.round, s.hist, s.pot, s.acting)
            if fp in seen:
                continue
            seen.add(fp)
            if s.done:
                continue
            if py_info_key(s, s.acting) == TARGET_IK:
                found = s
                break
            m = py_legal(s)
            for i in range(3):
                if m[i]:
                    stack.append(ap(s, i))
        if found:
            break

    if found is None:
        print("target state not found")
        return
    state = found
    print("state:", "round", state.round, "pot", state.pot, "acting", state.acting, "to_call",
          max(0, state.current_bet - state.committed[state.acting]))
    print("hist", state.hist)

    avg, regret = tabular_cfr(800, seed=0)
    ik, treg, tpol = tabular_regret_at(avg, regret, state, 0)
    print("\ntabular at ik:", ik)
    print("  regret", treg)
    print("  policy", tpol)

    trainer = SDCFRTrainer(SDCFRConfig())
    load_checkpoint(trainer, Path("checkpoints/sd_cfr_leduc_gate/sd_cfr_leduc_latest.pkl"))
    ek = tuple(np.asarray(encode_leduc_py(state, 0)).tolist())
    print("\nSD cum_regret", trainer._cum_regret[ek])
    print("SD sigma_rm", trainer._regret_matching_sigma(state, 0))

    mask = py_legal(state)
    trav = 0
    util = {}
    for a in range(3):
        if mask[a]:
            u = trainer.traverse(apply_action_py(state, a), trav, trainer.key, 1.0, 1.0)
            util[a] = u
    sigma = trainer._regret_matching_sigma(state, 0)
    nu = sum(sigma[a] * util[a] for a in range(3) if mask[a])
    print("\nSD traverse util (trav=0):", util)
    print("  node_util", nu)
    for a in range(3):
        if mask[a]:
            print(f"  inst[{a}] would be", util[a] - nu)

    # Terminal checks for immediate children
    for name, a in [("FOLD", FOLD), ("CALL", CALL)]:
        if mask[a]:
            c = apply_action_py(state, a)
            print(f"\nchild {name}: done={c.done} rewards={c.rewards} round={c.round}")


if __name__ == "__main__":
    main()
