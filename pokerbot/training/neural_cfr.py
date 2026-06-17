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
    is_chance_node,
    legal_action_mask,
    reset_state,
    sample_chance,
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


@dataclass
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
    avg_apply_fn: Any,
    avg_params: Any,
) -> Array:
    """Bellman-style history value loss: û + Σ σ(I',a') Q(h',a') − Q(h,a)."""

    q = apply_fn(params, batch["history"])
    q_next = apply_fn(params, batch["next_history"])
    next_policy = avg_apply_fn(avg_params, batch["next_info_state"], batch["next_legal_mask"])
    bootstrap = jnp.sum(next_policy * q_next, axis=-1)
    taken = q[jnp.arange(q.shape[0]), batch["action"]]
    target = batch["utility"] + bootstrap
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

    state = jax.lax.cond(is_chance_node(state), sample_chance, lambda s: s, state)
    next_state = step_state(state, action)
    next_seat = next_state.action_seat
    next_mask = legal_action_mask(next_state)
    next_info = encode_info_state(next_state, next_seat)
    next_hist = encode_history_state(next_state)

    next_cum = _apply_player_net(cum_mod, cum_params, next_seat, next_info, next_mask)
    next_inst = _apply_player_net(inst_mod, inst_params, next_seat, next_info, next_mask)
    next_predicted = pdcfr_predicted_regrets(next_cum, next_inst, iteration, alpha)
    next_policy = regret_matching(next_predicted, next_mask)
    q_next = hv_mod.apply(hv_params, next_hist)
    bootstrap = jnp.sum(next_policy * q_next)
    # Acting-player payoff at terminal; regret-matched bootstrap otherwise.
    terminal_payoff = next_state.rewards[seat].astype(jnp.float32)
    child_value = jax.lax.cond(next_state.done, lambda: terminal_payoff, lambda: bootstrap)

    q = hv_mod.apply(hv_params, hist)
    xi_a = xi[action]
    v_bar = _variance_reduced_advantages(q, action, child_value, xi_a)
    v_node = jnp.sum(policy * v_bar)
    inst_adv = (v_bar - v_node) * mask

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
        "utility": next_state.rewards[seat].astype(jnp.float32),
        "player": seat,
        "iteration": iteration,
        "next_info_state": next_info,
        "next_legal_mask": next_mask,
    }
    return next_state, key, adv_sample, strat_sample, hist_sample, is_traverser


def _empty_step_outputs() -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], Array, Array]:
    adv = {
        "info_state": jnp.zeros(INFO_FEATURE_DIM, dtype=jnp.float32),
        "advantages": jnp.zeros(NUM_ACTIONS, dtype=jnp.float32),
        "legal_mask": jnp.zeros(NUM_ACTIONS, dtype=bool),
        "iteration": jnp.int32(0),
    }
    strat = {
        "info_state": jnp.zeros(INFO_FEATURE_DIM, dtype=jnp.float32),
        "strategy": jnp.zeros(NUM_ACTIONS, dtype=jnp.float32),
        "legal_mask": jnp.zeros(NUM_ACTIONS, dtype=bool),
        "iteration": jnp.int32(0),
    }
    hist = {
        "history": jnp.zeros(HISTORY_FEATURE_DIM, dtype=jnp.float32),
        "next_history": jnp.zeros(HISTORY_FEATURE_DIM, dtype=jnp.float32),
        "action": jnp.int32(0),
        "utility": jnp.float32(0.0),
        "player": jnp.int32(0),
        "iteration": jnp.int32(0),
        "next_info_state": jnp.zeros(INFO_FEATURE_DIM, dtype=jnp.float32),
        "next_legal_mask": jnp.zeros(NUM_ACTIONS, dtype=bool),
    }
    return adv, strat, hist, jnp.bool_(False), jnp.bool_(False)


def _scan_one_traversal(
    reset_key: Array,
    trav_key: Array,
    traverser: Array,
    iteration: Array,
    cum_params: tuple[Any, ...],
    inst_params: tuple[Any, ...],
    hv_params: Any,
    alpha: float,
    epsilon: float,
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], Array, Array]:
    """Run one hand via ``lax.scan`` (fixed horizon, mask early termination)."""

    def body(carry: tuple[EnvState, Array, Array], _: None) -> tuple[tuple[EnvState, Array, Array], tuple]:
        state, key, active = carry

        def do_step() -> tuple[tuple[EnvState, Array, Array], tuple]:
            next_state, next_key, adv, strat, hist, is_trav = traversal_step(
                state,
                key,
                traverser,
                iteration,
                cum_params,
                inst_params,
                hv_params,
                alpha,
                epsilon,
            )
            still_active = jnp.logical_and(active, jnp.logical_not(state.done))
            return (next_state, next_key, still_active), (adv, strat, hist, is_trav, still_active)

        def skip_step() -> tuple[tuple[EnvState, Array, Array], tuple]:
            return (state, key, jnp.bool_(False)), _empty_step_outputs()

        return jax.lax.cond(active, do_step, skip_step)

    init_state = reset_state(reset_key)
    init = (init_state, trav_key, jnp.logical_not(init_state.done))
    _, steps = jax.lax.scan(body, init, None, length=MAX_TRAVERSAL_STEPS)
    adv_t, strat_t, hist_t, is_trav_t, active_t = steps
    adv = {k: adv_t[k] for k in ("info_state", "advantages", "legal_mask", "iteration")}
    strat = {k: strat_t[k] for k in ("info_state", "strategy", "legal_mask", "iteration")}
    hist = {k: hist_t[k] for k in hist_t}
    return adv, strat, hist, is_trav_t, active_t


def _batched_traversal_scans(
    reset_keys: Array,
    trav_keys: Array,
    traverser: Array,
    iteration: Array,
    cum_params: tuple[Any, ...],
    inst_params: tuple[Any, ...],
    hv_params: Any,
    alpha: float,
    epsilon: float,
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], Array, Array]:
    """``vmap`` over parallel outcome-sampled traversals for one updating player."""

    vmapped = jax.vmap(
        lambda rk, tk: _scan_one_traversal(
            rk, tk, traverser, iteration, cum_params, inst_params, hv_params, alpha, epsilon
        )
    )
    return vmapped(reset_keys, trav_keys)


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

    def __init__(
        self,
        config: VRDeepPDCFRConfig | None = None,
        *,
        start_iteration: int = 0,
        prng_key: Array | None = None,
        cum_states: tuple[Any, ...] | None = None,
        inst_states: tuple[Any, ...] | None = None,
        hv_state: Any | None = None,
        avg_state: Any | None = None,
    ) -> None:
        self.config = config or VRDeepPDCFRConfig()
        self._start_iteration = start_iteration
        self._iteration = start_iteration
        self.key = prng_key if prng_key is not None else jax.random.PRNGKey(self.config.seed)

        self.adv_buffers = init_advantage_buffers(self.config.buffer_capacity)
        self.strat_buffer = init_strategy_buffer(self.config.buffer_capacity)
        self.hist_buffer = init_history_value_buffer(self.config.buffer_capacity)

        hidden = self.config.hidden_dim
        lr = self.config.learning_rate

        if cum_states is not None and inst_states is not None and hv_state is not None and avg_state is not None:
            self.cum_states = cum_states
            self.inst_states = inst_states
            self.hv_state = hv_state
            self.avg_state = avg_state
        else:
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
        self._batched_scans = jax.jit(_batched_traversal_scans, static_argnames=("alpha", "epsilon"))

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
        hv_module = HistoryValueNet(cfg.hidden_dim)

        def loss_fn(params: Any) -> Array:
            return history_value_loss(
                params,
                hv_module.apply,
                batch,
                self.avg_state.apply_fn,
                self.avg_state.params,
            )

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

    def _insert_traversal_batch(
        self,
        traverser: int,
        adv: dict[str, Array],
        strat: dict[str, Array],
        hist: dict[str, Array],
        is_trav: Array,
        active: Array,
    ) -> None:
        """Transfer JIT rollout samples into host replay buffers."""

        adv_np = jax.device_get(adv)
        strat_np = jax.device_get(strat)
        hist_np = jax.device_get(hist)
        is_trav_np = jax.device_get(is_trav)
        active_np = jax.device_get(active)

        n_batch = int(adv_np["info_state"].shape[0])
        n_steps = int(adv_np["info_state"].shape[1])
        for b in range(n_batch):
            for t in range(n_steps):
                if not active_np[b, t]:
                    continue
                if is_trav_np[b, t]:
                    buf = self.adv_buffers.buffers[traverser]
                    self.adv_buffers = AdvantageBuffers(
                        buffers=self.adv_buffers.buffers[:traverser]
                        + (
                            insert_advantage_sample(
                                buf,
                                info_state=jnp.asarray(adv_np["info_state"][b, t]),
                                advantages=jnp.asarray(adv_np["advantages"][b, t]),
                                legal_mask=jnp.asarray(adv_np["legal_mask"][b, t]),
                                iteration=jnp.asarray(adv_np["iteration"][b, t]),
                            ),
                        )
                        + self.adv_buffers.buffers[traverser + 1 :]
                    )
                else:
                    self.key, ks = jax.random.split(self.key)
                    self.strat_buffer = StrategyBuffer(
                        buffer=insert_strategy_sample(
                            self.strat_buffer.buffer,
                            ks,
                            info_state=jnp.asarray(strat_np["info_state"][b, t]),
                            strategy=jnp.asarray(strat_np["strategy"][b, t]),
                            legal_mask=jnp.asarray(strat_np["legal_mask"][b, t]),
                            iteration=jnp.asarray(strat_np["iteration"][b, t]),
                        )
                    )
                self.key, kh = jax.random.split(self.key)
                self.hist_buffer = HistoryValueBuffer(
                    buffer=insert_history_value_sample(
                        self.hist_buffer.buffer,
                        kh,
                        history=jnp.asarray(hist_np["history"][b, t]),
                        next_history=jnp.asarray(hist_np["next_history"][b, t]),
                        action=jnp.asarray(hist_np["action"][b, t]),
                        utility=jnp.asarray(hist_np["utility"][b, t]),
                        player=jnp.asarray(hist_np["player"][b, t]),
                        iteration=jnp.asarray(hist_np["iteration"][b, t]),
                        next_info_state=jnp.asarray(hist_np["next_info_state"][b, t]),
                        next_legal_mask=jnp.asarray(hist_np["next_legal_mask"][b, t]),
                    )
                )

    def run_traversals(self, iteration: int) -> None:
        """Collect K outcome-sampled traversals for each player (scan + vmap)."""

        cfg = self.config
        self.adv_buffers = clear_all_advantage_buffers(self.adv_buffers)
        cum_params = tuple(s.params for s in self.cum_states)
        inst_params = tuple(s.params for s in self.inst_states)
        hv_params = self.hv_state.params
        iter_arr = jnp.int32(iteration)
        k = cfg.traversals_per_iter

        for traverser in range(NUM_PLAYERS):
            self.key, key_block = jax.random.split(self.key)
            keys = jax.random.split(key_block, k * 2)
            reset_keys = keys[:k]
            trav_keys = keys[k:]
            adv, strat, hist, is_trav, active = self._batched_scans(
                reset_keys,
                trav_keys,
                jnp.int32(traverser),
                iter_arr,
                cum_params,
                inst_params,
                hv_params,
                cfg.alpha,
                cfg.epsilon,
            )
            self._insert_traversal_batch(traverser, adv, strat, hist, is_trav, active)

    def train_iteration(self, iteration: int) -> dict[str, float]:
        """Single CFR iteration: traverse, then train all networks."""

        self._iteration = iteration
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

        start = self._start_iteration
        for t in range(start + 1, self.config.total_iterations + 1):
            self.train_iteration(t)
        return NetworkParams(
            cumulative=self.cum_states,
            instantaneous=self.inst_states,
            history_value=self.hv_state,
            average_strategy=self.avg_state,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> VRDeepPDCFRPlusTrainer:
        """Restore trainer from a full checkpoint written by ``save_trainer_checkpoint``."""

        return load_trainer_checkpoint(path)

    def get_strategy(self, info_state: Array, legal_mask: Array, player: int, iteration: int) -> Array:
        """Return regret-matched strategy for inference."""

        cum = self.cum_states[player].apply_fn(self.cum_states[player].params, info_state, legal_mask)
        inst = self.inst_states[player].apply_fn(self.inst_states[player].params, info_state, legal_mask)
        predicted = pdcfr_predicted_regrets(cum, inst, jnp.int32(iteration), self.config.alpha)
        return regret_matching(predicted, legal_mask)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def _config_to_dict(config: VRDeepPDCFRConfig) -> dict[str, Any]:
    return {
        "total_iterations": config.total_iterations,
        "traversals_per_iter": config.traversals_per_iter,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "hidden_dim": config.hidden_dim,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "epsilon": config.epsilon,
        "buffer_capacity": config.buffer_capacity,
        "seed": config.seed,
    }


def _train_state_from_params(module: nn.Module, params: Any, lr: float) -> train_state.TrainState:
    tx = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)


def save_network_params(
    params: NetworkParams,
    path: str | Path,
    config: VRDeepPDCFRConfig | None = None,
    *,
    iteration: int = 0,
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
        "iteration": iteration,
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
    """Save inference + full trainer checkpoint (all four networks, config, iteration)."""

    import pickle

    path = Path(path)
    params = NetworkParams(
        cumulative=trainer.cum_states,
        instantaneous=trainer.inst_states,
        history_value=trainer.hv_state,
        average_strategy=trainer.avg_state,
    )
    save_network_params(params, path, trainer.config, iteration=trainer._iteration)

    full_path = path.with_name(f"{path.stem}_full{path.suffix}")
    full_payload = {
        "history_value_params": jax.device_get(trainer.hv_state.params),
        "average_strategy_params": jax.device_get(trainer.avg_state.params),
        "cumulative_advantage_params": [jax.device_get(s.params) for s in trainer.cum_states],
        "instantaneous_advantage_params": [jax.device_get(s.params) for s in trainer.inst_states],
        "config": _config_to_dict(trainer.config),
        "iteration": trainer._iteration,
        "prng_key": jax.device_get(trainer.key),
    }
    with full_path.open("wb") as f:
        pickle.dump(full_payload, f)
    return full_path


def load_trainer_checkpoint(path: str | Path) -> VRDeepPDCFRPlusTrainer:
    """Load a full trainer checkpoint for resume."""

    import pickle

    path = Path(path)
    if path.stem.endswith("_full"):
        full_path = path
    else:
        full_path = path.with_name(f"{path.stem}_full{path.suffix}")
    if not full_path.exists():
        raise FileNotFoundError(f"Full trainer checkpoint not found: {full_path}")

    with full_path.open("rb") as f:
        payload = pickle.load(f)

    cfg = VRDeepPDCFRConfig(**payload["config"])
    hidden = cfg.hidden_dim
    lr = cfg.learning_rate
    cum_states = tuple(
        _train_state_from_params(CumulativeAdvantageNet(hidden), p, lr)
        for p in payload["cumulative_advantage_params"]
    )
    inst_states = tuple(
        _train_state_from_params(InstantaneousAdvantageNet(hidden), p, lr)
        for p in payload["instantaneous_advantage_params"]
    )
    hv_state = _train_state_from_params(HistoryValueNet(hidden), payload["history_value_params"], lr)
    avg_state = _train_state_from_params(AverageStrategyNet(hidden), payload["average_strategy_params"], lr)
    return VRDeepPDCFRPlusTrainer(
        cfg,
        start_iteration=int(payload["iteration"]),
        prng_key=jnp.asarray(payload["prng_key"]),
        cum_states=cum_states,
        inst_states=inst_states,
        hv_state=hv_state,
        avg_state=avg_state,
    )
