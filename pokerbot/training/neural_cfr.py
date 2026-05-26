"""Variance-Reduced Deep Predictive Discounted CFR+ (VR-DeepPDCFR+) trainer.

Implements the AAAI 2026 algorithm with three Flax networks per player:
cumulative advantage, instantaneous advantage, and a shared history value
network for baseline variance reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import Array

from pokerbot.engine.jax_engine import (
    HISTORY_FEATURE_DIM,
    INFO_FEATURE_DIM,
    NUM_ACTIONS,
    NUM_PLAYERS,
    EnvState,
    encode_history_state,
    encode_info_state,
    legal_action_mask,
    reset_state,
    step_state,
)
from pokerbot.training.replay_buffers import (
    AdvantageBuffers,
    HistoryValueBuffer,
    StrategyBuffer,
    clear_all_advantage_buffers,
    init_advantage_buffers,
    init_history_value_buffer,
    init_strategy_buffer,
    insert_advantage_sample,
    insert_history_value_sample,
    insert_strategy_sample,
    sample_advantage_batch,
    sample_history_value_batch,
    sample_strategy_batch,
)

# Paper defaults: α=2.3, γ=2 for VR-DeepPDCFR+
DEFAULT_ALPHA: float = 2.3
DEFAULT_GAMMA: float = 2.0
DEFAULT_EPSILON: float = 0.1
MAX_TRAVERSAL_STEPS: int = 128


class NetworkParams(NamedTuple):
    """All trainable networks for VR-DeepPDCFR+."""

    cumulative: Any  # per-player TrainState
    instantaneous: Any
    history_value: Any
    average_strategy: Any


@dataclass(frozen=True)
class VRDeepPDCFRConfig:
    total_iterations: int = 100
    traversals_per_iter: int = 64
    batch_size: int = 512
    learning_rate: float = 1e-4
    hidden_dim: int = 256
    alpha: float = DEFAULT_ALPHA
    gamma: float = DEFAULT_GAMMA
    epsilon: float = DEFAULT_EPSILON
    buffer_capacity: int = 100_000
    seed: int = 0


# ---------------------------------------------------------------------------
# Flax network definitions
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.out_dim)(x)


class CumulativeAdvantageNet(nn.Module):
    """R(I, a | θ) — cumulative advantage head for one player."""

    hidden_dim: int = 256

    @nn.compact
    def __call__(self, info_state: Array, legal_mask: Array) -> Array:
        logits = MLP(self.hidden_dim, NUM_ACTIONS)(info_state)
        return jnp.where(legal_mask, logits, 0.0)


class InstantaneousAdvantageNet(nn.Module):
    """r(I, a | φ) — instantaneous advantage head for one player."""

    hidden_dim: int = 256

    @nn.compact
    def __call__(self, info_state: Array, legal_mask: Array) -> Array:
        logits = MLP(self.hidden_dim, NUM_ACTIONS)(info_state)
        return jnp.where(legal_mask, logits, 0.0)


class HistoryValueNet(nn.Module):
    """Q(h, a | ω) — history value baseline for variance reduction."""

    hidden_dim: int = 256

    @nn.compact
    def __call__(self, history: Array) -> Array:
        return MLP(self.hidden_dim, NUM_ACTIONS)(history)


class AverageStrategyNet(nn.Module):
    """Π(I, a | ψ) — average strategy approximation."""

    hidden_dim: int = 256

    @nn.compact
    def __call__(self, info_state: Array, legal_mask: Array) -> Array:
        logits = MLP(self.hidden_dim, NUM_ACTIONS)(info_state)
        masked = jnp.where(legal_mask, logits, -1e9)
        return jax.nn.softmax(masked)


# ---------------------------------------------------------------------------
# Regret matching & PDCFR+ strategy
# ---------------------------------------------------------------------------


@jax.jit
def regret_matching(regrets: Array, legal_mask: Array) -> Array:
    """ReLU regret matching with legal action masking."""

    positive = jnp.maximum(regrets, 0.0) * legal_mask
    denom = jnp.sum(positive)
    uniform = legal_mask.astype(jnp.float32) / jnp.maximum(jnp.sum(legal_mask), 1.0)
    return jnp.where(denom > 0, positive / denom, uniform)


@jax.jit
def pdcfr_predicted_regrets(
    cum_prev: Array,
    inst_prev: Array,
    iteration: Array,
    alpha: float,
) -> Array:
    """Predicted cumulative advantages for VR-DeepPDCFR+ strategy (Alg. 2)."""

    t = jnp.maximum(iteration.astype(jnp.float32), 1.0)
    discount = (t - 1.0) ** alpha / ((t - 1.0) ** alpha + 1.0)
    return jnp.maximum(cum_prev, 0.0) * discount + inst_prev


@jax.jit
def dcfr_discount_factor(iteration: Array, alpha: float) -> Array:
    t = jnp.maximum(iteration.astype(jnp.float32), 1.0)
    return (t - 1.0) ** alpha / ((t - 1.0) ** alpha + 1.0)


@jax.jit
def sampling_strategy(policy: Array, legal_mask: Array, epsilon: float) -> Array:
    """ξ(I,a) = ε/|A| + (1-ε)σ(I,a) for the updating player."""

    n_legal = jnp.sum(legal_mask)
    uniform = legal_mask.astype(jnp.float32) / jnp.maximum(n_legal, 1.0)
    return epsilon * uniform + (1.0 - epsilon) * policy


# ---------------------------------------------------------------------------
# Loss functions (Algorithm 1)
# ---------------------------------------------------------------------------


def cumulative_advantage_loss(
    params: Any,
    apply_fn: Any,
    batch: dict[str, Array],
    prev_params: Any,
    prev_apply_fn: Any,
    iteration: int,
    alpha: float,
) -> Array:
    """PDCFR+ bootstrapped cumulative advantage loss with discount + clip target."""

    legal = batch["legal_mask"]
    pred = apply_fn(params, batch["info_state"], legal)
    with_prev = prev_apply_fn(prev_params, batch["info_state"], legal)
    discount = dcfr_discount_factor(jnp.int32(iteration), alpha)
    target = jnp.maximum(with_prev, 0.0) * discount + batch["advantages"]
    err = (target - pred) * legal
    return jnp.mean(jnp.sum(err**2, axis=-1))


def instantaneous_advantage_loss(
    params: Any,
    apply_fn: Any,
    batch: dict[str, Array],
) -> Array:
    """Variance-reduced instantaneous advantage regression loss."""

    legal = batch["legal_mask"]
    pred = apply_fn(params, batch["info_state"], legal)
    err = (batch["advantages"] - pred) * legal
    return jnp.mean(jnp.sum(err**2, axis=-1))


def history_value_loss(
    params: Any,
    apply_fn: Any,
    batch: dict[str, Array],
    next_policy: Array,
) -> Array:
    """Bellman-style history value loss with variance-reduced target (Alg. 1)."""

    q = apply_fn(params, batch["history"])
    q_next = apply_fn(params, batch["next_history"])
    bootstrap = jnp.sum(next_policy * q_next, axis=-1)
    taken = q[jnp.arange(q.shape[0]), batch["action"]]
    target = batch["utility"] + bootstrap - batch["utility"]  # simplified terminal backup
    target = batch["utility"] + bootstrap * 0.0 + bootstrap  # v(h') aggregate
    # Alg 1: (û + Σ σ(I',a') Q(h',a') - Q(h,a))
    target = batch["utility"] + jnp.sum(next_policy * q_next, axis=-1)
    return jnp.mean((target - taken) ** 2)


def average_strategy_loss(
    params: Any,
    apply_fn: Any,
    batch: dict[str, Array],
    total_iterations: int,
    gamma: float,
) -> Array:
    """Weighted MSE toward sampled strategies: (t/T)^γ (σ^t - Π)^2."""

    legal = batch["legal_mask"]
    pred = apply_fn(params, batch["info_state"], legal)
    weight = (batch["iteration"].astype(jnp.float32) / float(total_iterations)) ** gamma
    err = (batch["strategy"] - pred) * legal
    return jnp.mean(weight * jnp.sum(err**2, axis=-1))


# ---------------------------------------------------------------------------
# Traversal with outcome sampling + variance reduction
# ---------------------------------------------------------------------------


@jax.jit
def _variance_reduced_advantages(
    q_values: Array,
    sampled_action: Array,
    child_value: Array,
    xi_prob: Array,
) -> Array:
    """Compute v̄(I,a) per Algorithm 2 lines 12-17."""

    def for_action(a: Array) -> Array:
        q_a = q_values[a]
        correction = (child_value - q_a) / jnp.maximum(xi_prob, 1e-8)
        return q_a + correction * (sampled_action == a).astype(jnp.float32)

    return jax.vmap(for_action)(jnp.arange(NUM_ACTIONS))


def _apply_player_net(module: nn.Module, all_params: tuple[Any, ...], seat: Array, info: Array, mask: Array) -> Array:
    """Select per-player parameters via ``lax.switch`` for JIT compatibility."""

    branches = tuple(lambda i=i, m=module: m.apply(all_params[i], info, mask) for i in range(NUM_PLAYERS))
    return jax.lax.switch(seat, branches)


@jax.jit
def traversal_step(
    state: EnvState,
    key: Array,
    traverser: Array,
    iteration: Array,
    cum_params: tuple[Any, ...],
    inst_params: tuple[Any, ...],
    hv_params: Any,
    alpha: float,
    epsilon: float,
) -> tuple[EnvState, Array, dict[str, Array], dict[str, Array], dict[str, Array], Array]:
    """One outcome-sampling step with variance-reduced advantage targets."""

    seat = state.action_seat
    mask = legal_action_mask(state)
    info = encode_info_state(state, seat)
    hist = encode_history_state(state)

    cum_mod = CumulativeAdvantageNet()
    inst_mod = InstantaneousAdvantageNet()
    hv_mod = HistoryValueNet()

    cum = _apply_player_net(cum_mod, cum_params, seat, info, mask)
    inst = _apply_player_net(inst_mod, inst_params, seat, info, mask)
    predicted = pdcfr_predicted_regrets(cum, inst, iteration, alpha)
    policy = regret_matching(predicted, mask)

    is_traverser = seat == traverser
    xi = jax.lax.cond(
        is_traverser,
        lambda p: sampling_strategy(p, mask, epsilon),
        lambda p: p,
        policy,
    )

    key, ka = jax.random.split(key)
    action = jax.random.choice(ka, NUM_ACTIONS, p=xi)

    q = hv_mod.apply(hv_params, hist)
    child_value = jnp.float32(0.0)
    xi_a = xi[action]
    v_bar = _variance_reduced_advantages(q, action, child_value, xi_a)
    v_node = jnp.sum(policy * v_bar)
    inst_adv = (v_bar - v_node) * mask

    next_state = step_state(state, action)
    next_info = encode_info_state(next_state, next_state.action_seat)
    next_hist = encode_history_state(next_state)

    adv_sample = {
        "info_state": info,
        "advantages": inst_adv,
        "legal_mask": mask,
        "iteration": iteration,
    }
    strat_sample = {
        "info_state": info,
        "strategy": policy,
        "legal_mask": mask,
        "iteration": iteration,
    }
    hist_sample = {
        "history": hist,
        "next_history": next_hist,
        "action": action,
        "utility": next_state.rewards[0],
        "player": seat,
        "iteration": iteration,
        "next_info_state": next_info,
    }
    return next_state, key, adv_sample, strat_sample, hist_sample, is_traverser


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _make_train_state(module: nn.Module, key: Array, lr: float) -> train_state.TrainState:
    tx = optax.adam(lr)
    if isinstance(module, HistoryValueNet):
        dummy = jnp.zeros((1, HISTORY_FEATURE_DIM), dtype=jnp.float32)
        params = module.init(key, dummy)
    else:
        dummy_info = jnp.zeros((1, INFO_FEATURE_DIM), dtype=jnp.float32)
        dummy_mask = jnp.ones((1, NUM_ACTIONS), dtype=bool)
        params = module.init(key, dummy_info, dummy_mask)
    return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)


class VRDeepPDCFRPlusTrainer:
    """Hardware-accelerated VR-DeepPDCFR+ training loop."""

    def __init__(self, config: VRDeepPDCFRConfig | None = None) -> None:
        self.config = config or VRDeepPDCFRConfig()
        self.key = jax.random.PRNGKey(self.config.seed)

        self.adv_buffers = init_advantage_buffers(self.config.buffer_capacity)
        self.strat_buffer = init_strategy_buffer(self.config.buffer_capacity)
        self.hist_buffer = init_history_value_buffer(self.config.buffer_capacity)

        hidden = self.config.hidden_dim
        lr = self.config.learning_rate

        self.key, *subkeys = jax.random.split(self.key, NUM_PLAYERS * 2 + 3)

        self.cum_states = tuple(
            _make_train_state(CumulativeAdvantageNet(hidden), subkeys[i], lr) for i in range(NUM_PLAYERS)
        )
        self.inst_states = tuple(
            _make_train_state(InstantaneousAdvantageNet(hidden), subkeys[NUM_PLAYERS + i], lr)
            for i in range(NUM_PLAYERS)
        )
        self.hv_state = _make_train_state(HistoryValueNet(hidden), subkeys[-2], lr)
        self.avg_state = _make_train_state(AverageStrategyNet(hidden), subkeys[-1], lr)

        self._prev_cum_params = tuple(s.params for s in self.cum_states)

    def _train_player_cumulative(self, player: int, iteration: int) -> None:
        cfg = self.config
        buf = self.adv_buffers.buffers[player]
        if int(buf.size) == 0:
            return
        self.key, k = jax.random.split(self.key)
        batch = sample_advantage_batch(buf, k, cfg.batch_size)

        state = self.cum_states[player]
        prev_params = self._prev_cum_params[player]

        def loss_fn(params: Any) -> Array:
            return cumulative_advantage_loss(
                params,
                state.apply_fn,
                batch,
                prev_params,
                state.apply_fn,
                iteration,
                cfg.alpha,
            )

        grads = jax.grad(loss_fn)(state.params)
        self.cum_states = self.cum_states[:player] + (state.apply_gradients(grads=grads),) + self.cum_states[player + 1 :]

    def _train_player_instantaneous(self, player: int) -> None:
        cfg = self.config
        buf = self.adv_buffers.buffers[player]
        if int(buf.size) == 0:
            return
        self.key, k = jax.random.split(self.key)
        batch = sample_advantage_batch(buf, k, cfg.batch_size)
        state = self.inst_states[player]

        def loss_fn(params: Any) -> Array:
            return instantaneous_advantage_loss(params, state.apply_fn, batch)

        grads = jax.grad(loss_fn)(state.params)
        self.inst_states = self.inst_states[:player] + (state.apply_gradients(grads=grads),) + self.inst_states[player + 1 :]

    def _train_history_value(self, iteration: int) -> None:
        cfg = self.config
        buf = self.hist_buffer.buffer
        if int(buf.size) == 0:
            return
        self.key, k = jax.random.split(self.key)
        batch = sample_history_value_batch(buf, k, cfg.batch_size)
        next_info = batch["next_info_state"]
        legal = jnp.ones((cfg.batch_size, NUM_ACTIONS), dtype=bool)
        next_policy = self.avg_state.apply_fn(self.avg_state.params, next_info, legal)
        hv_module = HistoryValueNet(cfg.hidden_dim)

        def loss_fn(params: Any) -> Array:
            q = hv_module.apply(params, batch["history"])
            q_next = hv_module.apply(params, batch["next_history"])
            taken = q[jnp.arange(q.shape[0]), batch["action"]]
            bootstrap = jnp.sum(next_policy * q_next, axis=-1)
            target = batch["utility"] + bootstrap
            return jnp.mean((target - taken) ** 2)

        grads = jax.grad(loss_fn)(self.hv_state.params)
        self.hv_state = self.hv_state.apply_gradients(grads=grads)

    def _train_average_strategy(self, iteration: int) -> None:
        cfg = self.config
        buf = self.strat_buffer.buffer
        if int(buf.size) == 0:
            return
        self.key, k = jax.random.split(self.key)
        batch = sample_strategy_batch(buf, k, cfg.batch_size)
        state = self.avg_state

        def loss_fn(params: Any) -> Array:
            return average_strategy_loss(params, state.apply_fn, batch, cfg.total_iterations, cfg.gamma)

        grads = jax.grad(loss_fn)(state.params)
        self.avg_state = state.apply_gradients(grads=grads)

    def run_traversals(self, iteration: int) -> None:
        """Collect K outcome-sampled traversals for each player."""

        cfg = self.config
        self.adv_buffers = clear_all_advantage_buffers(self.adv_buffers)
        cum_params = tuple(s.params for s in self.cum_states)
        inst_params = tuple(s.params for s in self.inst_states)
        hv_params = self.hv_state.params
        iter_arr = jnp.int32(iteration)

        for traverser in range(NUM_PLAYERS):
            trav_arr = jnp.int32(traverser)
            for _ in range(cfg.traversals_per_iter):
                self.key, k1, k2 = jax.random.split(self.key, 3)
                state = reset_state(k1)
                key = k2
                for _step in range(MAX_TRAVERSAL_STEPS):
                    if bool(state.done):
                        break
                    state, key, adv_s, strat_s, hist_s, is_trav = traversal_step(
                        state,
                        key,
                        trav_arr,
                        iter_arr,
                        cum_params,
                        inst_params,
                        hv_params,
                        cfg.alpha,
                        cfg.epsilon,
                    )
                    if bool(is_trav):
                        buf = self.adv_buffers.buffers[traverser]
                        self.adv_buffers = AdvantageBuffers(
                            buffers=self.adv_buffers.buffers[:traverser]
                            + (insert_advantage_sample(buf, **adv_s),)
                            + self.adv_buffers.buffers[traverser + 1 :]
                        )
                    else:
                        self.key, ks = jax.random.split(self.key)
                        self.strat_buffer = StrategyBuffer(
                            buffer=insert_strategy_sample(self.strat_buffer.buffer, ks, **strat_s)
                        )
                    self.key, kh = jax.random.split(self.key)
                    self.hist_buffer = HistoryValueBuffer(
                        buffer=insert_history_value_sample(self.hist_buffer.buffer, kh, **hist_s)
                    )

    def train_iteration(self, iteration: int) -> dict[str, float]:
        """Single CFR iteration: traverse, then train all networks."""

        self.run_traversals(iteration)

        for player in range(NUM_PLAYERS):
            self._train_player_cumulative(player, iteration)
            self._train_player_instantaneous(player)

        self._train_history_value(iteration)
        self._train_average_strategy(iteration)

        self._prev_cum_params = tuple(s.params for s in self.cum_states)
        return {"iteration": float(iteration)}

    def train(self) -> NetworkParams:
        """Run full VR-DeepPDCFR+ training."""

        for t in range(1, self.config.total_iterations + 1):
            self.train_iteration(t)
        return NetworkParams(
            cumulative=self.cum_states,
            instantaneous=self.inst_states,
            history_value=self.hv_state,
            average_strategy=self.avg_state,
        )

    def get_strategy(self, info_state: Array, legal_mask: Array, player: int, iteration: int) -> Array:
        """Return regret-matched strategy for inference."""

        cum = self.cum_states[player].apply_fn(self.cum_states[player].params, info_state, legal_mask)
        inst = self.inst_states[player].apply_fn(self.inst_states[player].params, info_state, legal_mask)
        predicted = pdcfr_predicted_regrets(cum, inst, jnp.int32(iteration), self.config.alpha)
        return regret_matching(predicted, legal_mask)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_network_params(
    params: NetworkParams,
    path: str | Path,
    config: VRDeepPDCFRConfig | None = None,
) -> Path:
    """Persist HistoryValueNet and blueprint params for search inference."""

    import pickle

    cfg = config or VRDeepPDCFRConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "history_value_params": jax.device_get(params.history_value.params),
        "average_strategy_params": jax.device_get(params.average_strategy.params),
        "hidden_dim": cfg.hidden_dim,
        "total_iterations": cfg.total_iterations,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path


def load_network_params(path: str | Path) -> dict[str, Any]:
    """Load serialized network params dict from checkpoint."""

    import pickle

    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def save_trainer_checkpoint(trainer: VRDeepPDCFRPlusTrainer, path: str | Path) -> Path:
    """Save all networks from a trained trainer instance."""

    params = NetworkParams(
        cumulative=trainer.cum_states,
        instantaneous=trainer.inst_states,
        history_value=trainer.hv_state,
        average_strategy=trainer.avg_state,
    )
    return save_network_params(params, path, trainer.config)
