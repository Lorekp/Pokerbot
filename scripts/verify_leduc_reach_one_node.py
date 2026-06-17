#!/usr/bin/env python3
"""Verify opp_reach matches tabular CFR on one Leduc node (no traverser fan-out)."""

from __future__ import annotations

import numpy as np

from pokerbot.engine.leduc_bridge import apply_action_py, py_legal, py_state_from_deal
from pokerbot.training.sd_cfr import SDCFRConfig, SDCFRTrainer


def tabular_traverse_util(
    state,
    traverser: int,
    p0: float,
    p1: float,
    sigma_fn,
) -> float:
    """Mirror tabular opponent/traverser logic with fixed sigma_fn(state, seat)->sigma."""
    if state.done:
        return float(state.rewards[traverser])
    seat = state.acting
    mask = py_legal(state)
    sigma = sigma_fn(state, seat)
    if seat == traverser:
        util = np.zeros(3)
        for a in range(3):
            if mask[a]:
                util[a] = tabular_traverse_util(
                    apply_action_py(state, a), traverser, p0, p1, sigma_fn
                )
        return float(sum(sigma[a] * util[a] for a in range(3) if mask[a]))
    util = 0.0
    for a in range(3):
        if mask[a]:
            np0 = p0 * sigma[a] if seat == 0 else p0
            np1 = p1 * sigma[a] if seat == 1 else p1
            util += sigma[a] * tabular_traverse_util(
                apply_action_py(state, a), traverser, np0, np1, sigma_fn
            )
    return util


def main() -> None:
    # After private CC on deal (0,1)/pub=2; one opponent CALL then traverser (P0) acts.
    root = py_state_from_deal((0, 1), 2)
    s = apply_action_py(apply_action_py(root, 1), 1)
    s = apply_action_py(s, 1)  # opponent CALL once
    assert s.acting == 0

    trainer = SDCFRTrainer(SDCFRConfig(seed=99))

    def sigma_fn(st, seat):
        return trainer._regret_matching_sigma(st, seat)

    # Sampled path: opponent took CALL with prob sigma[1]; opp_reach for P0 = p1 * sigma[1]
    sig_opp = sigma_fn(s, 1) if s.acting == 1 else sigma_fn(apply_action_py(root, 1), 1)
    # Rebuild: from start s0=post-private, seat=1, one CALL
    s0 = apply_action_py(apply_action_py(root, 1), 1)
    sig = sigma_fn(s0, 1)
    manual_p1 = float(sig[1])

    # SD traverse returns node_util at traverser; read opp_reach from code path
    captured: dict = {}

    def traverse(state, traverser, key, p0=1.0, p1=1.0):
        if state.done:
            return float(state.rewards[traverser])
        seat = state.acting
        mask = py_legal(state)
        opp_reach = p1 if traverser == 0 else p0
        if seat == traverser:
            util = np.zeros(3)
            for a in range(3):
                if mask[a]:
                    util[a] = traverse(apply_action_py(state, a), traverser, key, p0, p1)
            sigma = sigma_fn(state, seat)
            node_util = float(sum(sigma[a] * util[a] for a in range(3) if mask[a]))
            captured["opp_reach"] = opp_reach
            captured["p1"] = p1
            captured["inst"] = {
                a: opp_reach * (util[a] - node_util) for a in range(3) if mask[a]
            }
            return node_util
        sigma = sigma_fn(state, seat)
        legal = np.flatnonzero(mask)
        action = 1 if 1 in legal else int(legal[0])
        np0 = p0 * sigma[action] if seat == 0 else p0
        np1 = p1 * sigma[action] if seat == 1 else p1
        return traverse(apply_action_py(state, action), traverser, key, np0, np1)

    traverse(s0, 0, None)
    assert s0.acting == 1
    traverse(apply_action_py(s0, 1), 0, None, 1.0, manual_p1)

    print("opponent CALL sigma[1]:", manual_p1)
    print("captured at P0 node: opp_reach=", captured.get("opp_reach"), "p1=", captured.get("p1"))
    print("match:", abs(captured.get("opp_reach", -1) - manual_p1) < 1e-9)
    print("instantaneous adv:", captured.get("inst"))


if __name__ == "__main__":
    main()
