"""Single Deep CFR (SD-CFR) — advantage network + external sampling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax import Array

from pokerbot.engine.leduc_bridge import (
    PyLeducState,
    apply_action_py,
    init_py_from_key,
    jax_from_py,
    py_legal,
)
from pokerbot.engine.leduc_jax import (
    ANTE,
    MAX_HISTORY,
    MAX_RAISES,
    NUM_ACTIONS,
    NUM_PLAYERS,
    RAISE_SIZE,
    LeducState,
    legal_actions,
)

LEDUC_ENCODE_DIM = 7 + MAX_HISTORY  # card, pub, pot, seat, round, to_call, raises + history


class LeducAdvantageNet(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, x: Array, legal_mask: Array) -> Array:
        h = nn.relu(nn.Dense(self.hidden)(x))
        h = nn.relu(nn.Dense(self.hidden)(h))
        logits = nn.Dense(NUM_ACTIONS)(h)
        return jnp.where(legal_mask, logits, -1e9)


def encode_leduc(state: LeducState, seat: int) -> Array:
    """History-aware Leduc infoset encoding — betting history is part of the information set."""
    card = state.cards[seat].astype(jnp.float32) / 3.0
    pub = jnp.maximum(state.public, 0).astype(jnp.float32) / 3.0
    pot = state.pot.astype(jnp.float32) / 100.0
    seat_f = jnp.float32(seat)
    rnd = state.round.astype(jnp.float32)
    to_call = jnp.maximum(0, state.current_bet - state.committed[seat]).astype(jnp.float32)
    to_call = to_call / float(RAISE_SIZE * MAX_RAISES + ANTE)
    raises_n = state.raises.astype(jnp.float32) / float(MAX_RAISES)
    base = jnp.array([card, pub, pot, seat_f, rnd, to_call, raises_n], dtype=jnp.float32)
    slots = []
    for i in range(MAX_HISTORY):
        a = jnp.where(i < state.hist_len, state.hist_action[i], jnp.int32(-1))
        slots.append(jnp.where(a >= 0, a.astype(jnp.float32) / 2.0, jnp.float32(0.0)))
    return jnp.concatenate([base, jnp.stack(slots)])


def encode_leduc_py(state: PyLeducState, seat: int) -> Array:
    card = float(state.cards[seat]) / 3.0
    pub = float(max(state.public, 0)) / 3.0
    pot = float(state.pot) / 100.0
    seat_f = float(seat)
    rnd = float(state.round)
    to_call = float(max(0, state.current_bet - state.committed[seat]))
    to_call /= float(RAISE_SIZE * MAX_RAISES + ANTE)
    raises_n = float(state.raises) / float(MAX_RAISES)
    hist = list(state.hist) + [0.0] * (MAX_HISTORY - len(state.hist))
    for i, a in enumerate(state.hist):
        hist[i] = float(a) / 2.0
    return jnp.array(
        [card, pub, pot, seat_f, rnd, to_call, raises_n, *hist[:MAX_HISTORY]],
        dtype=jnp.float32,
    )


@dataclass
class SDCFRConfig:
    iterations: int = 200
    traversals_per_iter: int = 32
    lr: float = 1e-3
    seed: int = 0


class SDCFRTrainer:
    """External-sampling SD-CFR on Leduc (validation ruler)."""

    def __init__(self, config: SDCFRConfig | None = None) -> None:
        self.config = config or SDCFRConfig()
        self.key = jax.random.PRNGKey(self.config.seed)
        net = LeducAdvantageNet()
        self.key, k = jax.random.split(self.key)
        dummy_x = jnp.zeros(LEDUC_ENCODE_DIM)
        dummy_m = jnp.ones(NUM_ACTIONS, dtype=bool)
        params = net.init(k, dummy_x, dummy_m)
        self.state = train_state.TrainState.create(
            apply_fn=net.apply,
            params=params,
            tx=optax.adam(self.config.lr),
        )
        self.adv_buffer: list[tuple[Array, int, Array, Array]] = []
        self._strategy_snapshots: list[tuple[int, dict]] = []
        self._avg_policy_cache: dict[tuple, np.ndarray] = {}

    def clear_policy_cache(self) -> None:
        self._avg_policy_cache.clear()

    def _regret_matching_sigma(self, state: PyLeducState, seat: int) -> np.ndarray:
        return self.regret_matching_policy(jax_from_py(state), seat)

    def traverse(self, state: PyLeducState, traverser: int, key: Array) -> float:
        """External-sampling traversal on pure-Python Leduc (no per-node JAX sync)."""
        if state.done:
            return float(state.rewards[traverser])

        seat = state.acting
        mask = py_legal(state)
        if seat == traverser:
            util = np.zeros(NUM_ACTIONS, dtype=np.float64)
            for a in range(NUM_ACTIONS):
                if mask[a]:
                    util[a] = self.traverse(apply_action_py(state, a), traverser, key)
            sigma = self._regret_matching_sigma(state, seat)
            node_util = float(sum(sigma[a] * util[a] for a in range(NUM_ACTIONS) if mask[a]))
            for a in range(NUM_ACTIONS):
                if mask[a]:
                    self.adv_buffer.append(
                        (
                            encode_leduc_py(state, seat),
                            a,
                            jnp.float32(util[a] - node_util),
                            jnp.asarray(mask),
                        )
                    )
            return node_util

        sigma = self._regret_matching_sigma(state, seat)
        legal_idx = np.flatnonzero(mask)
        probs = sigma[legal_idx]
        probs = probs / probs.sum()
        key, sub = jax.random.split(key)
        pick = int(jax.random.choice(sub, legal_idx.shape[0], p=jnp.asarray(probs, dtype=jnp.float32)))
        action = int(legal_idx[pick])
        return self.traverse(apply_action_py(state, action), traverser, key)

    def train_iteration(self) -> float:
        self.adv_buffer.clear()
        for _ in range(self.config.traversals_per_iter):
            self.key, k1, k2 = jax.random.split(self.key, 3)
            state = init_py_from_key(k1)
            for seat in range(NUM_PLAYERS):
                self.traverse(state, seat, k2)

        if not self.adv_buffer:
            return 0.0

        xs = jnp.stack([b[0] for b in self.adv_buffer])
        actions = jnp.array([b[1] for b in self.adv_buffer], dtype=jnp.int32)
        targets = jnp.stack([b[2] for b in self.adv_buffer])
        masks = jnp.stack([b[3] for b in self.adv_buffer])

        def loss_fn(params):
            pred = jax.vmap(lambda x, m: self.state.apply_fn(params, x, m))(xs, masks)
            picked = pred[jnp.arange(pred.shape[0]), actions]
            return jnp.mean(jnp.square(picked - targets))

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(self.state.params)
        self.state = self.state.apply_gradients(grads=grads)
        return float(loss)

    def train(
        self,
        log_every: int = 1,
        checkpoint_dir: Path | None = None,
        checkpoint_every: int = 0,
        start_iter: int = 0,
    ) -> list[float]:
        losses: list[float] = []
        t0 = time.perf_counter()
        for i in range(start_iter, self.config.iterations):
            losses.append(self.train_iteration())
            iter_num = i + 1
            if log_every > 0 and iter_num % log_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"SD-CFR iter {iter_num}/{self.config.iterations} "
                    f"loss={losses[-1]:.6f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            self._record_strategy_snapshot(iter_num)
            if (
                checkpoint_dir is not None
                and checkpoint_every > 0
                and iter_num % checkpoint_every == 0
            ):
                self._completed_iterations = iter_num
                ckpt_dir = Path(checkpoint_dir)
                save_checkpoint(self, ckpt_dir / f"sd_cfr_leduc_iter_{iter_num:04d}.pkl")
                save_checkpoint(self, ckpt_dir / "sd_cfr_leduc_latest.pkl")
                print(f"Checkpoint saved at iter {iter_num}", flush=True)
        return losses

    def _record_strategy_snapshot(self, iteration: int) -> None:
        """Linear-CFR weighted snapshot for average-strategy recovery."""
        params_copy = jax.tree.map(lambda x: np.asarray(x), self.state.params)
        self._strategy_snapshots.append((iteration, params_copy))

    def _sigma_from_params(
        self, params: dict, state: LeducState, seat: int
    ) -> np.ndarray:
        x = encode_leduc(state, seat)
        mask = np.asarray(legal_actions(state), dtype=bool)
        adv = np.asarray(self.state.apply_fn(params, x, mask), dtype=np.float64)
        positive = np.maximum(adv, 0.0) * mask
        total = positive.sum()
        if total > 0:
            return positive / total
        return mask.astype(np.float64) / max(mask.sum(), 1)

    def regret_matching_policy(self, state: LeducState, seat: int) -> np.ndarray:
        """Regret-matching strategy from the current (final-iterate) advantage net."""
        return self._sigma_from_params(self.state.params, state, seat)

    def average_strategy_policy(self, state: LeducState, seat: int) -> np.ndarray:
        """Linear-CFR average of regret-matched policies across iteration snapshots."""
        cache_key = (tuple(np.asarray(encode_leduc(state, seat)).tolist()), int(seat))
        cached = self._avg_policy_cache.get(cache_key)
        if cached is not None:
            return cached
        mask = np.asarray(legal_actions(state), dtype=bool)
        if not self._strategy_snapshots:
            sigma = self.regret_matching_policy(state, seat)
            self._avg_policy_cache[cache_key] = sigma
            return sigma
        agg = np.zeros(NUM_ACTIONS, dtype=np.float64)
        total_w = 0.0
        for weight, params in self._strategy_snapshots:
            agg += float(weight) * self._sigma_from_params(params, state, seat)
            total_w += float(weight)
        sigma = agg / total_w
        sigma = sigma * mask
        s = sigma.sum()
        if s > 0:
            sigma = sigma / s
        else:
            sigma = mask.astype(np.float64) / max(mask.sum(), 1)
        self._avg_policy_cache[cache_key] = sigma
        return sigma


def save_checkpoint(trainer: SDCFRTrainer, path: Path) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "params": trainer.state.params,
                "config": trainer.config,
                "completed_iterations": getattr(trainer, "_completed_iterations", 0),
                "strategy_snapshots": getattr(trainer, "_strategy_snapshots", []),
            },
            f,
        )


def load_checkpoint(trainer: SDCFRTrainer, path: Path) -> int:
    """Restore params from checkpoint; returns completed iteration count."""
    import pickle

    with Path(path).open("rb") as f:
        data = pickle.load(f)
    trainer.state = trainer.state.replace(params=data["params"])
    if "config" in data:
        trainer.config = data["config"]
    trainer._strategy_snapshots = list(data.get("strategy_snapshots", []))
    return int(data.get("completed_iterations", 0))


# ---------------------------------------------------------------------------
# HU NLHE SD-CFR (Phase 2b)
# ---------------------------------------------------------------------------

from pokerbot.engine.jax_engine import (
    CHIP_UNIT,
    INFO_FEATURE_DIM,
    NUM_ACTIONS as NLHE_NUM_ACTIONS,
    EnvState,
    encode_info_state,
    is_chance_node,
    legal_action_mask,
    reset_state,
    sample_chance,
    step_state,
)


class NLHEAdvantageNet(nn.Module):
    hidden: int = 256

    @nn.compact
    def __call__(self, x: Array, legal_mask: Array) -> Array:
        h = nn.relu(nn.Dense(self.hidden)(x))
        h = nn.relu(nn.Dense(self.hidden)(h))
        logits = nn.Dense(NLHE_NUM_ACTIONS)(h)
        return jnp.where(legal_mask, logits, -1e9)


@dataclass
class SDCFRNLHEConfig:
    iterations: int = 10
    traversals_per_iter: int = 8
    lr: float = 1e-4
    seed: int = 0
    occupancy: tuple[bool, ...] = (True, True, False, False, False, False)


class SDCFRNLHETrainer:
    """External-sampling SD-CFR on validated JAX NLHE env (HU by default)."""

    def __init__(self, config: SDCFRNLHEConfig | None = None) -> None:
        self.config = config or SDCFRNLHEConfig()
        self.key = jax.random.PRNGKey(self.config.seed)
        self.occupancy = jnp.array(self.config.occupancy, dtype=bool)
        net = NLHEAdvantageNet()
        self.key, k = jax.random.split(self.key)
        dummy_x = jnp.zeros(INFO_FEATURE_DIM)
        dummy_m = jnp.ones(NLHE_NUM_ACTIONS, dtype=bool)
        params = net.init(k, dummy_x, dummy_m)
        self.state = train_state.TrainState.create(
            apply_fn=net.apply,
            params=params,
            tx=optax.adam(self.config.lr),
        )
        self.adv_buffer: list[tuple[Array, Array, Array]] = []

    def _init_state(self, key: Array) -> EnvState:
        return reset_state(key, occupancy=self.occupancy)

    def traverse(self, state: EnvState, traverser: int, key: Array, depth: int = 0) -> Array:
        if depth > 200 or bool(state.done):
            return state.rewards[traverser].astype(jnp.float32) / float(CHIP_UNIT)
        if bool(is_chance_node(state)):
            state = sample_chance(state)
        seat = int(state.action_seat)
        mask = legal_action_mask(state)
        if seat == traverser:
            util = jnp.zeros(NLHE_NUM_ACTIONS, dtype=jnp.float32)
            for a in range(NLHE_NUM_ACTIONS):
                if bool(mask[a]):
                    util = util.at[a].set(self.traverse(step_state(state, jnp.int32(a)), traverser, key, depth + 1))
            sigma = jax.nn.softmax(jnp.where(mask, util, -1e9))
            node_util = jnp.sum(sigma * util)
            for a in range(NLHE_NUM_ACTIONS):
                if bool(mask[a]):
                    self.adv_buffer.append(
                        (
                            encode_info_state(state, jnp.int32(seat)),
                            jnp.float32(util[a] - node_util),
                            mask,
                        )
                    )
            return node_util
        key, sub = jax.random.split(key)
        legal = jnp.where(mask, 1.0, 0.0)
        probs = legal / jnp.maximum(jnp.sum(legal), 1.0)
        action = jax.random.choice(sub, NLHE_NUM_ACTIONS, p=probs)
        return self.traverse(step_state(state, action), traverser, key, depth + 1)

    def train_iteration(self) -> float:
        self.adv_buffer.clear()
        for _ in range(self.config.traversals_per_iter):
            self.key, k1, k2 = jax.random.split(self.key, 3)
            state = self._init_state(k1)
            for seat in range(int(jnp.sum(self.occupancy))):
                self.traverse(state, seat, k2)
        if not self.adv_buffer:
            return 0.0
        xs = jnp.stack([b[0] for b in self.adv_buffer])
        targets = jnp.stack([b[1] for b in self.adv_buffer])
        masks = jnp.stack([b[2] for b in self.adv_buffer])

        def loss_fn(params):
            pred = jax.vmap(lambda x, m: self.state.apply_fn(params, x, m))(xs, masks)
            return jnp.mean(jnp.square(pred[:, 0] - targets))

        loss, grads = jax.value_and_grad(loss_fn)(self.state.params)
        self.state = self.state.apply_gradients(grads=grads)
        return float(loss)

    def train(self) -> list[float]:
        return [self.train_iteration() for _ in range(self.config.iterations)]

