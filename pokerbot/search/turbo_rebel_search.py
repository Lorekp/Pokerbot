"""Depth-limited TurboReBeL subgame solver with SSMIG and batched leaf evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import (
    NUM_ACTIONS,
    NUM_PLAYERS,
    encode_history_state,
    legal_action_mask,
    step_state,
)
from pokerbot.search.isomorphism import cached_canonicalize_key, canonicalize_history_encoding
from pokerbot.search.leaf_evaluator import HistoryValueEvaluator
from pokerbot.search.pbs_tracker import PBSTracker
from pokerbot.search.state_bridge import SubgameRoot, env_to_numpy


@dataclass
class TurboReBeLConfig:
    max_depth: int = 4
    max_iterations: int = 200
    ssmig_inner_iters: int = 10
    time_budget_ms: int = 2000
    leaf_batch_size: int = 64


@dataclass
class SolveResult:
    action_probs: np.ndarray
    ev: float
    iterations: int
    metadata: dict[str, float] = field(default_factory=dict)


def _regret_match(regrets: np.ndarray, legal: np.ndarray) -> np.ndarray:
    pos = np.maximum(regrets, 0.0) * legal
    s = pos.sum()
    if s > 0:
        return pos / s
    return legal.astype(np.float64) / max(legal.sum(), 1)


class TurboReBeLSolver:
    """Depth-limited CFR+ with single-sample multi-iteration generation."""

    def __init__(
        self,
        leaf_eval: HistoryValueEvaluator,
        config: TurboReBeLConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.leaf_eval = leaf_eval
        self.config = config or TurboReBeLConfig()
        self.rng = rng or np.random.default_rng(0)
        self._regrets: dict[str, np.ndarray] = {}
        self._strategy_sum: dict[str, np.ndarray] = {}
        self._leaf_cache: dict[str, np.ndarray] = {}

    def _infoset_key(self, env, seat: int) -> str:
        hist_len = int(np.asarray(env.hist_len))
        seats = tuple(int(x) for x in np.asarray(env.hist_seat[:hist_len]))
        actions = tuple(int(x) for x in np.asarray(env.hist_action[:hist_len]))
        board = tuple(int(x) for x in np.asarray(env.board) if int(x) >= 0)
        return f"{seat}|{int(np.asarray(env.street))}|{board}|{seats}|{actions}"

    def _get_regrets(self, key: str, legal: np.ndarray) -> np.ndarray:
        if key not in self._regrets:
            self._regrets[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
            self._strategy_sum[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        return self._regrets[key]

    def _leaf_q(self, env) -> np.ndarray:
        hist = np.asarray(encode_history_state(env), dtype=np.float32)
        hist = canonicalize_history_encoding(hist)
        hist_len = int(np.asarray(env.hist_len))
        action_seq = tuple(int(x) for x in np.asarray(env.hist_action[:hist_len]))
        board = tuple(int(x) for x in np.asarray(env.board) if int(x) >= 0)
        street = int(np.asarray(env.street))
        cache_key = cached_canonicalize_key(board, street, action_seq)
        if cache_key in self._leaf_cache:
            return self._leaf_cache[cache_key]
        q = self.leaf_eval.evaluate_batch(hist)[0]
        self._leaf_cache[cache_key] = q
        return q

    def _cfv(
        self,
        env,
        seat: int,
        hero_seat: int,
        depth: int,
        reach: float,
        sampled_holes: dict[int, tuple[int, int]],
    ) -> float:
        if bool(np.asarray(env.done)) or depth >= self.config.max_depth:
            legal = np.asarray(legal_action_mask(env, jnp.int32(seat)), dtype=bool)
            q = self._leaf_q(env)
            return float(np.max(q[legal])) if legal.any() else 0.0

        actor = int(np.asarray(env.action_seat))
        legal = np.asarray(legal_action_mask(env, jnp.int32(actor)), dtype=bool)
        if not legal.any():
            return 0.0

        key = self._infoset_key(env, actor)
        regrets = self._get_regrets(key, legal)
        strategy = _regret_match(regrets, legal)

        if actor != hero_seat:
            a_idx = int(self.rng.choice(np.where(legal)[0], p=strategy[legal] / strategy[legal].sum()))
            holes = np.asarray(env.holes).copy()
            if actor in sampled_holes:
                c0, c1 = sampled_holes[actor]
                holes[actor, 0] = c0
                holes[actor, 1] = c1
                env = env._replace(holes=jnp.asarray(holes))
            next_env = step_state(env, jnp.int32(a_idx))
            return self._cfv(next_env, seat, hero_seat, depth + 1, reach, sampled_holes)

        action_values = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in np.where(legal)[0]:
            next_env = step_state(env, jnp.int32(int(a)))
            action_values[a] = self._cfv(next_env, seat, hero_seat, depth + 1, reach, sampled_holes)

        ev = float(np.dot(strategy, action_values))
        for a in np.where(legal)[0]:
            regrets[a] += action_values[a] - ev
            regrets[a] = max(regrets[a], 0.0)
        self._strategy_sum[key] += reach * strategy
        return ev

    def solve(self, root: SubgameRoot, pbs: PBSTracker) -> SolveResult:
        """Run time-budgeted SSMIG CFR+ from root."""
        t0 = time.perf_counter()
        cfg = self.config
        hero = root.hero_seat
        iterations = 0
        leaf_ms = 0.0
        cache_hits_start = len(self._leaf_cache)

        sampled_holes = pbs.sample_opponent_holes(self.rng)

        while iterations < cfg.max_iterations:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms >= cfg.time_budget_ms:
                break

            for _ in range(cfg.ssmig_inner_iters):
                if (time.perf_counter() - t0) * 1000.0 >= cfg.time_budget_ms:
                    break
                env = env_to_numpy(root.env)
                holes = np.asarray(env.holes)
                if hero < NUM_PLAYERS and holes[hero, 0] >= 0:
                    pass
                for seat, (c0, c1) in sampled_holes.items():
                    holes = np.array(holes)
                    holes[seat, 0] = c0
                    holes[seat, 1] = c1
                env = env._replace(holes=jnp.asarray(holes))
                leaf_before = time.perf_counter()
                self._cfv(env, hero, hero, 0, 1.0, sampled_holes)
                leaf_ms += (time.perf_counter() - leaf_before) * 1000.0
                iterations += 1
                if iterations >= cfg.max_iterations:
                    break

            if (time.perf_counter() - t0) * 1000.0 >= cfg.time_budget_ms * 0.8:
                break

        root_key = self._infoset_key(root.env, hero)
        legal = root.legal_mask.astype(bool)
        if root_key in self._strategy_sum and self._strategy_sum[root_key].sum() > 0:
            avg = self._strategy_sum[root_key].copy()
            avg = avg * legal
            probs = avg / max(avg.sum(), 1e-12)
        else:
            regrets = self._regrets.get(root_key, np.zeros(NUM_ACTIONS))
            probs = _regret_match(regrets, legal)

        probs = probs * legal
        probs = probs / max(probs.sum(), 1e-12)

        ev = float(np.sum(probs * self._leaf_q(root.env))) if legal.any() else 0.0
        budget_ms = (time.perf_counter() - t0) * 1000.0

        return SolveResult(
            action_probs=probs.astype(np.float32),
            ev=ev,
            iterations=iterations,
            metadata={
                "leaf_eval_ms": leaf_ms,
                "cache_hits": float(len(self._leaf_cache) - cache_hits_start),
                "budget_used_ms": budget_ms,
            },
        )
