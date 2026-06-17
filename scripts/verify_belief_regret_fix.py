#!/usr/bin/env python3
"""Verify reach-weighted regret flush at worst inverted infoset (tabular favors CALL)."""

from __future__ import annotations

import numpy as np

from pokerbot.engine.leduc_bridge import apply_action_py, py_info_key, py_legal, py_state_from_deal
from pokerbot.engine.leduc_jax import CALL, FOLD
from pokerbot.training.leduc_tabular_cfr import all_leduc_deals, tabular_cfr
from pokerbot.training.sd_cfr import SDCFRConfig, SDCFRTrainer, encode_leduc_py

TARGET_IK = (0, 2, 1, 1, 2, 2, (1, 1, 1, 2, 2))


def find_target_state():
    for cards, pub in all_leduc_deals():
        if cards[0] != 2:
            continue
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
            if py_info_key(s, s.acting) == TARGET_IK:
                return s
            m = py_legal(s)
            for a in range(3):
                if m[a]:
                    stack.append(apply_action_py(s, a))
    return None


def main() -> None:
    state = find_target_state()
    if state is None:
        raise SystemExit("TARGET_IK state not found")

    avg, _ = tabular_cfr(400, seed=0)
    tab = avg[TARGET_IK]
    print("tabular policy", tab, "dominant", "CALL" if tab[CALL] >= tab[FOLD] else "FOLD")

    cfg = SDCFRConfig(
        iterations=40,
        seed=2,
        enumerate_all_deals=True,
        deal_sweeps=6,
    )
    trainer = SDCFRTrainer(cfg)
    trainer.train(log_every=10, checkpoint_every=0)

    ek = tuple(np.asarray(encode_leduc_py(state, 0)).tolist())
    reg = trainer._cum_regret.get(ek, np.zeros(3))
    sigma = trainer._regret_matching_sigma(state, 0)
    print("SD cum_regret", reg)
    print("SD sigma_rm", sigma)
    print("CALL regret > FOLD:", reg[CALL] > reg[FOLD])
    print("sigma favors CALL:", sigma[CALL] >= sigma[FOLD])

    # One-iteration reach audit at TARGET_IK
    trainer2 = SDCFRTrainer(cfg)
    trainer2._log_regret_aggregation = True
    trainer2._begin_iteration_regret()
    for cards, pub in all_leduc_deals():
        if cards[0] != 2:
            continue
        root = py_state_from_deal(cards, pub)
        trainer2.key, k2 = __import__("jax").random.split(trainer2.key)
        trainer2.traverse(root, 0, k2)
    visits = trainer2._pending_regret.get(ek, [])
    reach_by_card = {}
    for v in visits:
        reach_by_card[v.opponent_card] = reach_by_card.get(v.opponent_card, 0.0) + v.opp_reach
    print("pending visits at TARGET", len(visits), "reach_by_opponent_card", reach_by_card)
    trainer2._flush_pending_regret()


if __name__ == "__main__":
    main()
