"""CFR+ regret matching for HU toy game."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from aof_solver.game import Action, HuGameConfig, HuState, terminal_utility_sb


class RegretTable:
    def __init__(self) -> None:
        self.regret: dict[str, np.ndarray] = {}
        self.strategy_sum: dict[str, np.ndarray] = {}

    def _ensure(self, key: str, n_actions: int = 2) -> None:
        if key not in self.regret:
            self.regret[key] = np.zeros(n_actions, dtype=np.float64)
            self.strategy_sum[key] = np.zeros(n_actions, dtype=np.float64)

    def get_strategy(self, key: str, n_actions: int = 2) -> np.ndarray:
        self._ensure(key, n_actions)
        r = self.regret[key]
        pos = np.maximum(r, 0.0)
        s = pos.sum()
        if s > 0:
            return pos / s
        return np.full(n_actions, 1.0 / n_actions)

    def accumulate_strategy(self, key: str, strategy: np.ndarray) -> None:
        self._ensure(key, len(strategy))
        self.strategy_sum[key] += strategy

    def cfr_plus_update(
        self, key: str, action_utilities: np.ndarray, reach: float
    ) -> None:
        self._ensure(key, len(action_utilities))
        strat = self.get_strategy(key, len(action_utilities))
        ev = float(np.dot(strat, action_utilities))
        for i, u in enumerate(action_utilities):
            self.regret[key][i] = max(
                0.0, self.regret[key][i] + reach * (u - ev)
            )
        self.accumulate_strategy(key, strat * reach)


def run_one_cfr_iteration(
    table: RegretTable,
    buckets: list[str],
    bucket_eq_sb: dict[str, float],
    cfg: HuGameConfig,
) -> None:
    """Single traversal for SB root; BB responses sampled per bucket pair."""

    def traverse(state: HuState, player: str, bucket: str, reach: float) -> float:
        if state.is_terminal():
            if player == "SB":
                return terminal_utility_sb(
                    state, sb_equity_vs_bb_call=bucket_eq_sb.get(bucket, 0.5), cfg=cfg
                )
            return -terminal_utility_sb(
                state, sb_equity_vs_bb_call=bucket_eq_sb.get(bucket, 0.5), cfg=cfg
            )

        legal = state.legal_actions()
        if not legal:
            return 0.0

        key = f"{player}|{bucket}|{''.join(state.actions) or '-'}"
        n = len(legal)
        utils = np.zeros(n)
        for i, act in enumerate(legal):
            child = state.apply(act)
            utils[i] = traverse(child, player, bucket, reach)
        table.cfr_plus_update(key, utils, reach)
        strat = table.get_strategy(key, n)
        return float(np.dot(strat, utils))

    for b in buckets:
        traverse(HuState(), "SB", b, 1.0)
