#!/usr/bin/env python3
"""strat_sum coverage vs 288 infosets; L1 on visited-only vs all."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokerbot.engine.leduc_bridge import (
    apply_action_py,
    jax_from_py,
    py_info_key,
    py_legal,
    py_state_from_deal,
)
from pokerbot.engine.leduc_jax import NUM_ACTIONS
from pokerbot.training.leduc_tabular_cfr import (
    _policy_for_key,
    mean_l1_recovered_avg_vs_tabular,
    tabular_cfr,
)
from pokerbot.training.leduc_tabular_cfr import all_leduc_deals
from pokerbot.training.sd_cfr import SDCFRConfig, SDCFRTrainer, load_checkpoint, encode_leduc_py


def main() -> None:
    ckpt = Path("checkpoints/sd_cfr_leduc_gate/sd_cfr_leduc_latest.pkl")
    if not ckpt.exists():
        ckpt = Path("checkpoints/sd_cfr_leduc_gate/final.pkl")
    tabular_avg, _ = tabular_cfr(800, seed=0)
    trainer = SDCFRTrainer(SDCFRConfig())
    load_checkpoint(trainer, ckpt)
    trainer._tabular_avg_ref = tabular_avg

    all_ik = set()
    visited_encode = set(trainer._strat_sum.keys())
    zero_mass = 0
    l1_all = []
    l1_visited = []

    for cards, pub in all_leduc_deals():
        stack = [py_state_from_deal(cards, pub)]
        seen = set()
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
            ik = py_info_key(state, seat)
            all_ik.add(ik)
            ek = tuple(np.asarray(encode_leduc_py(state, seat)).tolist())
            tab = _policy_for_key(tabular_avg, ik, mask)
            sd = trainer.average_strategy_policy(jax_from_py(state), seat)
            l1 = float(np.sum(np.abs(tab - sd)))
            l1_all.append(l1)
            mass = float(trainer._strat_sum.get(ek, np.zeros(3)).sum())
            if mass <= 0:
                zero_mass += 1
            if ek in visited_encode and mass > 0:
                l1_visited.append(l1)

            for a in range(NUM_ACTIONS):
                if mask[a]:
                    stack.append(apply_action_py(state, a))

    mean_all, _ = mean_l1_recovered_avg_vs_tabular(
        trainer.average_strategy_policy, tabular_avg
    )
    print(f"Checkpoint: {ckpt}")
    print(f"strat_sum keys: {len(visited_encode)} / 288 infosets")
    print(f"acting nodes with zero strat_sum mass: {zero_mass}")
    print(f"mean L1 (all nodes): {mean_all:.4f}")
    print(f"mean L1 (visited encode keys with mass>0): {np.mean(l1_visited):.4f} n={len(l1_visited)}")
    print(f"mean L1 (nodes, local walk): {np.mean(l1_all):.4f}")


if __name__ == "__main__":
    main()
