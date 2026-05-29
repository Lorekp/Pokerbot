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
    encode_info_state,
    legal_action_mask,
    step_state,
)
from pokerbot.search.isomorphism import cached_canonicalize_key, canonicalize_history_encoding
from pokerbot.search.leaf_evaluator import HistoryValueEvaluator
from pokerbot.search.pbs_tracker import PBSTracker
from pokerbot.search.state_bridge import SubgameRoot, env_to_numpy
from pokerbot.training.neural_cfr import AverageStrategyNet


@dataclass
class TurboReBeLConfig:
    max_depth: int = 4
    max_iterations: int = 200
    ssmig_inner_iters: int = 10
    time_budget_ms: int = 2000
    leaf_batch_size: int = 64
    ssmig_resample_every: int = 1
    blueprint_mix: float = 0.5


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
        self._pending_leaf_keys: list[str] = []
        self._pending_leaf_hists: list[np.ndarray] = []
        self._leaf_eval_count = 0
        self._leaf_cache_hits = 0
        self._frozen_opp_avg: dict[str, np.ndarray] = {}

        self._blueprint_module = None
        self._blueprint_params = None
        if getattr(leaf_eval, "blueprint_params", None) is not None:
            hidden = int(getattr(leaf_eval, "hidden_dim", 256))
            self._blueprint_module = AverageStrategyNet(hidden)
            self._blueprint_params = leaf_eval.blueprint_params

    def _infoset_key(self, env, seat: int) -> str:
        hist_len = int(np.asarray(env.hist_len))
        seats = tuple(int(x) for x in np.asarray(env.hist_seat[:hist_len]))
        actions = tuple(int(x) for x in np.asarray(env.hist_action[:hist_len]))
        board = tuple(int(x) for x in np.asarray(env.board) if int(x) >= 0)
        pot = float(np.asarray(env.pot))
        stacks = np.asarray(env.stacks, dtype=np.float64)
        folded = np.asarray(env.folded, dtype=bool)
        active_mean = float(stacks[~folded].mean()) if (~folded).any() else float(stacks.mean())
        spr_bucket = int(pot / max(active_mean, 1.0) * 4)
        return f"{seat}|{int(np.asarray(env.street))}|{board}|{seats}|{actions}|spr{spr_bucket}"

    def _get_regrets(self, key: str, legal: np.ndarray) -> np.ndarray:
        if key not in self._regrets:
            self._regrets[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
            self._strategy_sum[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        return self._regrets[key]

    def _leaf_cache_key(self, env) -> str:
        hist_len = int(np.asarray(env.hist_len))
        action_seq = tuple(int(x) for x in np.asarray(env.hist_action[:hist_len]))
        board = tuple(int(x) for x in np.asarray(env.board) if int(x) >= 0)
        street = int(np.asarray(env.street))
        pot = float(np.asarray(env.pot))
        stacks = np.asarray(env.stacks, dtype=np.float64)
        folded = np.asarray(env.folded, dtype=bool)
        active_mean = float(stacks[~folded].mean()) if (~folded).any() else float(stacks.mean())
        pot_bucket = int(pot * 2)
        spr_bucket = int(pot / max(active_mean, 1.0) * 4)
        return cached_canonicalize_key(board, street, action_seq + (pot_bucket, spr_bucket))

    def _queue_leaf(self, env) -> str:
        cache_key = self._leaf_cache_key(env)
        if cache_key in self._leaf_cache:
            self._leaf_cache_hits += 1
            return cache_key
        hist = np.asarray(encode_history_state(env), dtype=np.float32)
        pot = float(np.asarray(env.pot))
        stacks = np.asarray(env.stacks, dtype=np.float64)
        folded = np.asarray(env.folded, dtype=bool)
        active_mean = float(stacks[~folded].mean()) if (~folded).any() else float(stacks.mean())
        spr_bucket = int(pot / max(active_mean, 1.0) * 4)
        hist = canonicalize_history_encoding(hist, pot_bb=pot, spr_bucket=spr_bucket)
        self._pending_leaf_keys.append(cache_key)
        self._pending_leaf_hists.append(hist)
        if len(self._pending_leaf_hists) >= self.config.leaf_batch_size:
            self._flush_leaf_batch()
        return cache_key

    def _flush_leaf_batch(self) -> None:
        if not self._pending_leaf_hists:
            return
        batch = np.stack(self._pending_leaf_hists, axis=0)
        qs = self.leaf_eval.evaluate_batch(batch)
        for key, q in zip(self._pending_leaf_keys, qs, strict=True):
            self._leaf_cache[key] = q
            self._leaf_eval_count += 1
        self._pending_leaf_keys.clear()
        self._pending_leaf_hists.clear()

    def _leaf_q(self, env) -> np.ndarray:
        cache_key = self._queue_leaf(env)
        if cache_key in self._leaf_cache:
            return self._leaf_cache[cache_key]
        self._flush_leaf_batch()
        return self._leaf_cache[cache_key]

    def _opponent_strategy(
        self,
        env,
        actor: int,
        legal: np.ndarray,
        regrets: np.ndarray,
        *,
        use_frozen: bool = False,
    ) -> np.ndarray:
        key = self._infoset_key(env, actor)
        if use_frozen and key in self._frozen_opp_avg:
            strategy = self._frozen_opp_avg[key].copy() * legal
            s = strategy.sum()
            if s > 0:
                return strategy / s

        strategy = _regret_match(regrets, legal)
        if self._blueprint_module is not None and self._blueprint_params is not None:
            info = encode_info_state(env, jnp.int32(actor))
            mask = legal_action_mask(env, jnp.int32(actor))
            bp = np.asarray(self._blueprint_module.apply(self._blueprint_params, info, mask))
            mix = float(self.config.blueprint_mix)
            strategy = (1.0 - mix) * strategy + mix * bp * legal
            s = strategy.sum()
            if s > 0:
                strategy = strategy / s
            else:
                strategy = legal.astype(np.float64) / max(legal.sum(), 1)
        return strategy

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

        if actor != hero_seat:
            strategy = self._opponent_strategy(env, actor, legal, regrets, use_frozen=getattr(self, "_ssmig_use_frozen", False))
            a_idx = int(self.rng.choice(np.where(legal)[0], p=strategy[legal] / strategy[legal].sum()))
            holes = np.asarray(env.holes).copy()
            if actor in sampled_holes:
                c0, c1 = sampled_holes[actor]
                holes[actor, 0] = c0
                holes[actor, 1] = c1
                env = env._replace(holes=jnp.asarray(holes))
            next_env = step_state(env, jnp.int32(a_idx))
            return self._cfv(next_env, seat, hero_seat, depth + 1, reach, sampled_holes)

        strategy = _regret_match(regrets, legal)
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
        cache_hits_start = self._leaf_cache_hits
        self._leaf_eval_count = 0

        self._traversal_ms = 0.0
        sampled_holes = pbs.sample_opponent_holes(self.rng)

        while iterations < cfg.max_iterations:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms >= cfg.time_budget_ms:
                break

            self._frozen_opp_avg.clear()
            for key, sums in self._strategy_sum.items():
                if sums.sum() > 1e-12:
                    self._frozen_opp_avg[key] = (sums / sums.sum()).astype(np.float64)

            for inner in range(cfg.ssmig_inner_iters):
                if (time.perf_counter() - t0) * 1000.0 >= cfg.time_budget_ms:
                    break
                if inner % max(cfg.ssmig_resample_every, 1) == 0:
                    sampled_holes = pbs.sample_opponent_holes(self.rng)

                self._ssmig_use_frozen = inner > 0
                env = env_to_numpy(root.env)
                holes = np.asarray(env.holes)
                for seat, (c0, c1) in sampled_holes.items():
                    holes = np.array(holes)
                    holes[seat, 0] = c0
                    holes[seat, 1] = c1
                env = env._replace(holes=jnp.asarray(holes))
                iter_before = time.perf_counter()
                self._cfv(env, hero, hero, 0, 1.0, sampled_holes)
                self._traversal_ms += (time.perf_counter() - iter_before) * 1000.0
                iterations += 1
                if iterations >= cfg.max_iterations:
                    break

            if (time.perf_counter() - t0) * 1000.0 >= cfg.time_budget_ms * 0.8:
                break

        self._flush_leaf_batch()

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
        self._flush_leaf_batch()
        traversal_ms = self._traversal_ms
        leaf_ms_est = max(budget_ms - traversal_ms, 0.0)

        return SolveResult(
            action_probs=probs.astype(np.float32),
            ev=ev,
            iterations=iterations,
            metadata={
                "leaf_eval_ms": leaf_ms_est,
                "leaf_eval_count": float(self._leaf_eval_count),
                "cache_hits": float(self._leaf_cache_hits - cache_hits_start),
                "budget_used_ms": budget_ms,
                "traversal_ms": traversal_ms,
                "traversal_fraction": float(traversal_ms / max(budget_ms, 1e-6)),
                "native_cfr_recommended": traversal_ms / max(budget_ms, 1e-6) > 0.4,
                "leaf_ms_per_iter": float(leaf_ms_est / max(iterations, 1)),
                "decision_latency_ms_p50": float(budget_ms),
                "decision_latency_ms_p95": float(budget_ms),
            },
        )
