#!/usr/bin/env python3
"""Structural features for infosets with L1 > 0.25 (broader bad set)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from pokerbot.engine.leduc_bridge import apply_action_py, jax_from_py, py_info_key, py_legal, py_state_from_deal
from pokerbot.engine.leduc_jax import ROUND_PUBLIC
from pokerbot.training.leduc_tabular_cfr import _policy_for_key, tabular_cfr, all_leduc_deals
from pokerbot.training.sd_cfr import SDCFRConfig, SDCFRTrainer, load_checkpoint


def main() -> None:
    avg, _ = tabular_cfr(800, seed=0)
    t = SDCFRTrainer(SDCFRConfig())
    load_checkpoint(t, Path("checkpoints/sd_cfr_leduc_gate/sd_cfr_leduc_latest.pkl"))

    by_ik = {}
    for cards, pub in all_leduc_deals():
        stack = [py_state_from_deal(cards, pub)]
        seen = set()
        while stack:
            s = stack.pop()
            fp = (s.deal_public, s.cards, s.public, s.round, s.hist)
            if fp in seen:
                continue
            seen.add(fp)
            if s.done:
                continue
            seat = s.acting
            m = py_legal(s)
            ik = py_info_key(s, seat)
            if ik not in by_ik:
                tab = _policy_for_key(avg, ik, m)
                sd = t.average_strategy_policy(jax_from_py(s), seat)
                l1 = float(np.sum(np.abs(tab - sd)))
                by_ik[ik] = (l1, s.round == ROUND_PUBLIC, seat, max(0, s.current_bet - s.committed[seat]) > 0)
            m = py_legal(s)
            for a in range(3):
                if m[a]:
                    stack.append(apply_action_py(s, a))

    bad = [v for v in by_ik.values() if v[0] > 0.25]
    print(f"infosets L1>0.25: {len(bad)} / 288")
    c_round = Counter("public" if b[1] else "private" for b in bad)
    c_seat = Counter(b[2] for b in bad)
    c_face = Counter("facing" if b[3] else "first" for b in bad)
    print("  round:", dict(c_round))
    print("  seat:", dict(c_seat))
    print("  node:", dict(c_face))


if __name__ == "__main__":
    main()
