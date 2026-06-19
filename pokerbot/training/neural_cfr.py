"""Variance-Reduced Deep Predictive Discounted CFR+ (VR-DeepPDCFR+) trainer.

Implements the AAAI 2026 algorithm with three Flax networks per player:
cumulative advantage, instantaneous advantage, and a shared history value
network for baseline variance reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
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
    STARTING_STACK_CHIPS,
    EnvState,
    encode_history_state,
    encode_info_state,
    is_chance_node,
    legal_action_mask,
    reset_state,
    sample_chance,
    step_state,
)
from pokerbot.training.cfr_env import EnvSpec, nlhe_env_spec
from pokerbot.training.replay_buffers import (
    AdvantageBuffers,
    HistoryValueBuffer,
    StrategyBuffer,
    advantage_buffer_spec,
    clear_all_advantage_buffers,
    history_value_buffer_spec,
    init_advantage_buffers,
    init_history_value_buffer,
    init_strategy_buffer,
    insert_advantage_sample,
    insert_history_value_sample,
    insert_strategy_sample,
    linear_write_batch,
    reservoir_insert_batch,
    sample_advantage_batch,
    sample_history_value_batch,
    sample_strategy_batch,
    strategy_buffer_spec,
)

# Reference defaults (configs/VRDeepPDCFRPlus.yaml): alpha=2.3, gamma=2, epsilon=0.6.
DEFAULT_ALPHA: float = 2.3
DEFAULT_GAMMA: float = 2.0
DEFAULT_EPSILON: float = 0.6        # reference yaml; was 0.1 (too little exploration in the traverser)
# Legacy default scan horizon. The active bound is now per-game via EnvSpec.max_steps (NLHE=64,
# Leduc=24, both >2x the measured worst-case hand depth); this constant only documents the prior
# value / the EnvSpec.max_steps default for adapters that omit it.
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
    # Leaf returns are divided by this before entering advantage/Q targets (reference /max_utility ->
    # [-1,1] scale). Default = a full NLHE stack in chips; the Leduc/Kuhn gate overrides it.
    reward_scale: float = float(STARTING_STACK_CHIPS)
    # SGD steps per network per CFR iteration. The reference fits each net for hundreds of steps/iter;
    # 1 underfits the advantage/strategy targets. The Leduc gate calibrates this.
    train_steps_per_iter: int = 1
    # Per-net step OVERRIDES (None => fall back to train_steps_per_iter, preserving the uniform-budget
    # behavior exactly). The reference (rpSebastian/DeepPDCFR) fits the SERVED average-policy net for far
    # more steps (~5000) than the advantage nets (~750), since the average net is what's evaluated for
    # exploitability and a single uniform knob under-serves it. Set avg_strategy_steps_per_iter higher
    # than train_steps_per_iter to fit the served strategy harder without paying that cost on every net.
    avg_strategy_steps_per_iter: int | None = None
    history_value_steps_per_iter: int | None = None
    # MULTIPROCESS SATURATION (>0 enables it): number of independent worker PROCESSES that generate
    # traversal samples data-parallel. A single JAX-CPU process is ~1-2 cores on the scan-heavy NLHE
    # traversal regardless of thread count (measured flat), so process-parallelism is the ONLY way to
    # saturate a many-vCPU box. Set ~= vcpus (each worker pinned to one compute thread => W threads
    # total, zero oversubscription). 0 = in-process single-device path (default; correct, not saturated).
    traversal_workers: int = 0


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
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, info_state: Array, legal_mask: Array) -> Array:
        logits = MLP(self.hidden_dim, self.num_actions)(info_state)
        return jnp.where(legal_mask, logits, 0.0)


class InstantaneousAdvantageNet(nn.Module):
    """r(I, a | φ) — instantaneous advantage head for one player."""

    hidden_dim: int = 256
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, info_state: Array, legal_mask: Array) -> Array:
        logits = MLP(self.hidden_dim, self.num_actions)(info_state)
        return jnp.where(legal_mask, logits, 0.0)


class HistoryValueNet(nn.Module):
    """Q(h, a | ω) — history value baseline for variance reduction."""

    hidden_dim: int = 256
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, history: Array) -> Array:
        return MLP(self.hidden_dim, self.num_actions)(history)


class AverageStrategyNet(nn.Module):
    """Π(I, a | ψ) — average strategy approximation."""

    hidden_dim: int = 256
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, info_state: Array, legal_mask: Array) -> Array:
        logits = MLP(self.hidden_dim, self.num_actions)(info_state)
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
    # all-negative fallback: put mass 1 on the highest-regret LEGAL action -- the PDCFR+ optimistic
    # step (reference use_regret_matching_argmax=True), NOT uniform (which changes early dynamics and
    # the served strategy at rarely-visited nodes).
    masked = jnp.where(legal_mask, regrets, -jnp.inf)
    argmax_onehot = jax.nn.one_hot(jnp.argmax(masked), regrets.shape[-1], dtype=jnp.float32) * legal_mask
    return jnp.where(denom > 0, positive / denom, argmax_onehot)


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
    # outer max(.,0) matches reference predictive_regret_matching (variants.py:360-368): the
    # discounted-cumulative + instantaneous prediction is re-clamped to non-negative before RM.
    return jnp.maximum(jnp.maximum(cum_prev, 0.0) * discount + inst_prev, 0.0)


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
    # PERSPECTIVE FIX (F4): Q(h,a) is the value to the player acting at h (batch["player"]). The
    # bootstrap Sum sigma Q(h') is in the NEXT acting player's perspective; convert it to the current
    # player's perspective (2-player zero-sum: negate when the turn passes to the opponent) before
    # forming the Bellman target. Otherwise Q learns an incoherent mixed-perspective value and the VR
    # baseline injects sign-flipped continuation values into the advantage targets.
    if "next_seat" in batch and "player" in batch:
        persp_sign = jnp.where(batch["next_seat"] == batch["player"], 1.0, -1.0)
    else:
        persp_sign = 1.0  # legacy batches without seat info: no perspective conversion
    # Terminal transitions carry NO bootstrap (reference: target = r + (1-done)*sum sigma*Q), and the
    # target is fixed (stop_gradient) so the Q-net does not chase a target it can itself shift.
    target = jax.lax.stop_gradient(
        batch["utility"] + (1.0 - batch["done"]) * persp_sign * bootstrap)
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
# Fused multi-step SGD (device-resident, single dispatch per net per iteration)
# ---------------------------------------------------------------------------
#
# PERFORMANCE: the trainer fits each net for K SGD steps/iter. The naive loop ran K *separate* Python
# dispatches per net (split key -> sample minibatch -> jax.grad(closure) -> apply_gradients), each a
# fresh host->device round-trip AND a fresh trace of the un-jitted grad closure. For tiny MLPs the
# per-step XLA kernels are tens of microseconds but the dispatch+retrace overhead was ~40-80 ms/step
# (GPU ~35% utilized) -- the dominant iteration cost. The helpers below FUSE the K steps into ONE
# jitted ``lax.scan``: the buffer stays device-resident, K minibatches are drawn inside the scan, and
# the optimizer update is carried across steps. One dispatch + one trace per net per iter replaces K.
#
# Learning equivalence: each scanned step draws an INDEPENDENT minibatch (a fresh sub-key per step,
# same ``randint``-with-replacement sampler as the old ``sample_*`` helpers) and runs the IDENTICAL
# Adam grad+update on the IDENTICAL loss. The only change is the RNG bookkeeping (one key split into K
# sub-keys, vs K sequential host-side splits) -- statistically the same minibatch stream, not bit-for-
# bit. ``prev_params``/iteration/avg-params are captured ONCE for the whole scan exactly as the old
# per-iteration loop captured them (they did not change across the K steps before, and do not here).


def _sample_indices(key: Array, size: Array, batch_size: int) -> Array:
    """Uniform-with-replacement row indices into a buffer of logical length ``size`` (>=1)."""
    return jax.random.randint(key, (batch_size,), 0, jnp.maximum(size.astype(jnp.int32), 1))


def _gather_rows(data: dict[str, Array], idx: Array) -> dict[str, Array]:
    return {k: v[idx] for k, v in data.items()}


def _scan_sgd(state: Any, key: Array, step_fn: Any, num_steps: int) -> Any:
    """Run ``num_steps`` Adam steps via ``lax.scan``, carrying the TrainState. ``step_fn(state, step_key)
    -> new_state`` does one sample+grad+update. ``num_steps`` is static (scan length)."""

    def body(st: Any, k: Array) -> tuple[Any, None]:
        return step_fn(st, k), None

    keys = jax.random.split(key, num_steps)
    new_state, _ = jax.lax.scan(body, state, keys)
    return new_state


@partial(jax.jit, static_argnums=(5, 6, 7))
def _fused_cumulative_update(state, data, size, key, prev_params, apply_fn, batch_size, num_steps,
                             iteration, alpha):
    """K fused cumulative-advantage Adam steps. ``data``/``size`` = the player's advantage buffer
    (device-resident); ``prev_params``/``iteration``/``alpha`` fixed across the scan (== old loop)."""

    def step(st, k):
        idx = _sample_indices(k, size, batch_size)
        batch = _gather_rows(data, idx)
        batch["advantages"] = batch["advantages"] * batch["legal_mask"]  # == sample_advantage_batch

        def loss_fn(params):
            return cumulative_advantage_loss(
                params, apply_fn, batch, prev_params, apply_fn, iteration, alpha)

        grads = jax.grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads)

    return _scan_sgd(state, key, step, num_steps)


@partial(jax.jit, static_argnums=(4, 5, 6))
def _fused_instantaneous_update(state, data, size, key, apply_fn, batch_size, num_steps):
    """K fused instantaneous-advantage Adam steps over the player's advantage buffer."""

    def step(st, k):
        idx = _sample_indices(k, size, batch_size)
        batch = _gather_rows(data, idx)
        batch["advantages"] = batch["advantages"] * batch["legal_mask"]

        def loss_fn(params):
            return instantaneous_advantage_loss(params, apply_fn, batch)

        grads = jax.grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads)

    return _scan_sgd(state, key, step, num_steps)


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def _fused_history_value_update(state, data, size, key, avg_params, avg_apply_fn, apply_fn,
                                batch_size, num_steps):
    """K fused history-value (VR baseline) Adam steps. ``avg_params``/``avg_apply_fn`` are the served
    average-strategy net used in the Bellman bootstrap; they are FIXED across the scan (the avg net is
    trained after the HV net within the iteration, exactly as before)."""

    def step(st, k):
        idx = _sample_indices(k, size, batch_size)
        batch = _gather_rows(data, idx)

        def loss_fn(params):
            return history_value_loss(params, apply_fn, batch, avg_apply_fn, avg_params)

        grads = jax.grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads)

    return _scan_sgd(state, key, step, num_steps)


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def _fused_average_strategy_update(state, data, size, key, apply_fn, batch_size, num_steps,
                                   total_iterations, gamma):
    """K fused average-strategy Adam steps over the reservoir strategy buffer."""

    def step(st, k):
        idx = _sample_indices(k, size, batch_size)
        batch = _gather_rows(data, idx)

        def loss_fn(params):
            return average_strategy_loss(params, apply_fn, batch, total_iterations, gamma)

        grads = jax.grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads)

    return _scan_sgd(state, key, step, num_steps)


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

    return jax.vmap(for_action)(jnp.arange(q_values.shape[-1]))


def _stack_player_params(all_params: tuple[Any, ...]) -> Any:
    """Stack a tuple of per-player param pytrees into one pytree with a leading player axis.

    Done ONCE per traversal-batch (outside the scan/vmap), so the hot loop only gathers + applies."""

    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *all_params)


def _apply_player_net(module: nn.Module, stacked_params: Any, seat: Array, info: Array,
                      mask: Array) -> Array:
    """Apply ``module`` for the acting player by GATHERING that seat's params from a stacked pytree.

    Replaces the prior ``lax.switch(seat, branches)``: under ``vmap`` a switch is lowered to compute
    ALL ``num_players`` branches (one full MLP apply each) and select, so each call did N-1 wasted
    net evals. Gathering the seat's params (``params[seat]``) and applying once does exactly one MLP
    evaluation per call -- statistically/numerically identical output, N-fold less net compute."""

    seat_params = jax.tree_util.tree_map(lambda x: x[seat], stacked_params)
    return module.apply(seat_params, info, mask)


def _traversal_step_partial(
    state: Any,
    key: Array,
    traverser: Array,
    iteration: Array,
    cum_params: Any,
    inst_params: Any,
    hv_params: Any,
    alpha: float,
    epsilon: float,
    reward_scale: float,
    hidden_dim: int,
    env: "EnvSpec",
) -> tuple[Any, Array, dict[str, Array]]:
    """Like ``traversal_step`` but emits ONLY the current node's nets (cum/inst/hv -> policy, q).

    The next-node evaluation (``next_cum``/``next_inst``/``q_next``) that the old step did to form the
    VR ``bootstrap`` is the SAME computation as the NEXT step's current node (same state, seat, params
    and ``iteration``), so it is recomputed redundantly every step. We drop it here and reconstruct the
    bootstrap in a single post-scan pass by shifting these per-step ``policy``/``q`` arrays by one (see
    ``_finalize_advantages``). Halves the net applies/step (6 -> 3) with an EXACT algebraic identity --
    the emitted samples are numerically identical to ``traversal_step`` (verified by verify_scan_equiv).
    The carried ``policy``/``q`` and the record below hold everything the post-pass needs."""

    seat = env.action_seat(state)
    mask = env.legal_mask(state)
    info = env.encode_info(state, seat)
    hist = env.encode_history(state)

    cum_mod = CumulativeAdvantageNet(hidden_dim, env.num_actions)
    inst_mod = InstantaneousAdvantageNet(hidden_dim, env.num_actions)
    hv_mod = HistoryValueNet(hidden_dim, env.num_actions)

    cum = _apply_player_net(cum_mod, cum_params, seat, info, mask)
    inst = _apply_player_net(inst_mod, inst_params, seat, info, mask)
    predicted = pdcfr_predicted_regrets(cum, inst, iteration, alpha)
    policy = regret_matching(predicted, mask)
    q = hv_mod.apply(hv_params, hist)

    is_traverser = seat == traverser
    xi = jax.lax.cond(
        is_traverser,
        lambda p: sampling_strategy(p, mask, epsilon),
        lambda p: p,
        policy,
    )

    key, ka = jax.random.split(key)
    action = jax.random.choice(ka, env.num_actions, p=xi)

    state = jax.lax.cond(env.is_chance(state), env.sample_chance, lambda s: s, state)
    next_state = env.step(state, action)
    next_seat = env.action_seat(next_state)
    next_mask = env.legal_mask(next_state)
    next_info = env.encode_info(next_state, next_seat)
    next_hist = env.encode_history(next_state)
    utility = env.rewards(next_state)[seat] / reward_scale
    done = env.done(next_state)

    record = {
        "info": info,
        "hist": hist,
        "mask": mask,
        "seat": seat,
        "policy": policy,          # current-node policy: also serves as PREV step's next_policy
        "q": q,                    # current-node q:      also serves as PREV step's q_next
        "action": action,
        "xi_a": xi[action],
        "iteration": iteration,
        "is_trav": is_traverser,
        "next_done": done.astype(jnp.float32),
        "terminal_payoff": utility,        # acting-player payoff at next_state (== utility)
        "next_info": next_info,
        "next_mask": next_mask,
        "next_hist": next_hist,
        "next_seat": next_seat,
    }
    return next_state, key, record


def _finalize_advantages(record: dict[str, Array], active: Array) -> tuple[
        dict[str, Array], dict[str, Array], dict[str, Array]]:
    """Reconstruct the VR advantage/strategy/history samples from the per-step partial records.

    ``record`` fields are stacked along a leading STEP axis (length = horizon) for ONE outcome-sampled
    path (single sampled action per node -> a chain root..terminal, then padded). The reference's
    DeepCumuAdv.dfs returns ``value = dot(q_values, policy)`` -- the VR-CORRECTED node value -- and the
    PARENT uses THAT recursive return as its child's ``action_value``. So the correct child value at
    step ``t`` is the *recursively corrected* value of step ``t+1`` (``corrected_value_{t+1}``), NOT a
    one-step bootstrap off the Q-baseline net. Using ``sum(policy_{t+1} * Q(h_{t+1}))`` instead makes
    the VR correction collapse to ``(Q - Q) ~ 0`` once the Q-net fits, killing the learning signal.

    Along a single sampled path the recursion
        corrected_value_T   = terminal_payoff
        corrected_value_t   = dot(q_base_t, policy_t)
                              + policy_t[a_t] * (persp_sign_t*corrected_value_{t+1} - q_base_t[a_t])
                                                 / sample_prob_t
    is a BACKWARD scan over the step axis, carrying ``corrected_value``. We seed the carry at terminal
    (``next_done==1``) with the terminal payoff and reset it on padding rows so post-terminal junk does
    not leak into the (chronologically earlier, reverse-first) real terminal step's carry."""

    policy = record["policy"]
    q = record["q"]
    done = record["next_done"]
    mask = record["mask"]
    action = record["action"]
    xi_a = record["xi_a"]
    # PERSPECTIVE FIX (F4, BUG #1): the child's corrected value is expressed in the NEXT acting
    # player's perspective (terminal payoff / Q are per-acting-player). The acting `seat`'s
    # continuation value is that quantity in `seat`'s perspective; in a 2-player zero-sum game they
    # differ by a sign when the action passes to the opponent. Without this negation the VR advantage
    # of any action that cedes the turn (e.g. RAISE/bet) gets the WRONG SIGN.
    persp_sign = jnp.where(record["next_seat"] == record["seat"], 1.0, -1.0)

    # BUG #2 FIX: backward scan carrying the RECURSIVE corrected child value (corrected_value_{t+1}),
    # replacing the old one-step Q-bootstrap. Per step we need child_value_t (this node's actor
    # perspective) -> v_bar_t -> corrected_value_t = dot(policy_t, v_bar_t), and the carry passes
    # corrected_value_t up to step t-1.
    q_taken = jnp.take_along_axis(q, action[:, None], axis=1)[:, 0]  # q_base_t[a_t]
    sample_prob = jnp.maximum(xi_a, 1e-8)

    def back_body(carry_cv: Array, step: dict[str, Array]) -> tuple[Array, Array]:
        # child_value in THIS node's actor perspective: terminal -> terminal_payoff, else the
        # persp-converted recursively-corrected child value carried up from step t+1.
        child_value = jnp.where(
            step["done"] > 0.5,
            step["payoff"],
            step["persp"] * carry_cv,
        )
        # corrected_value_t = dot(q_base, policy) + policy[a]*(child_value - q_base[a])/sample_prob.
        node_value = (
            jnp.sum(step["q"] * step["policy"])
            + step["policy"][step["action"]]
            * (child_value - step["q_taken"]) / step["sample_prob"]
        )
        # padding rows (inactive) must not propagate a meaningful carry to the (reverse-earlier) real
        # terminal step; that step ignores the carry anyway (done==1), but reset to be safe + explicit.
        next_carry = jnp.where(step["active"], node_value, 0.0)
        return next_carry, child_value

    scan_in = {
        "done": done,
        "payoff": record["terminal_payoff"],
        "persp": persp_sign,
        "q": q,
        "policy": policy,
        "action": action,
        "q_taken": q_taken,
        "sample_prob": sample_prob,
        "active": active.astype(jnp.float32),
    }
    # reverse=True walks t = T-1 .. 0; outputs child_value_t stacked back in forward order.
    _, child_value = jax.lax.scan(back_body, jnp.float32(0.0), scan_in, reverse=True)

    qv = record["q"]
    v_bar = jax.vmap(_variance_reduced_advantages)(qv, action, child_value, xi_a)
    v_node = jnp.sum(policy * v_bar, axis=-1, keepdims=True)
    inst_adv = (v_bar - v_node) * mask

    adv = {
        "info_state": record["info"],
        "advantages": inst_adv,
        "legal_mask": mask,
        "iteration": record["iteration"],
    }
    strat = {
        "info_state": record["info"],
        "strategy": policy,
        "legal_mask": mask,
        "iteration": record["iteration"],
    }
    hist = {
        "history": record["hist"],
        "next_history": record["next_hist"],
        "action": action,
        "utility": record["terminal_payoff"],
        "player": record["seat"],
        "iteration": record["iteration"],
        "next_info_state": record["next_info"],
        "next_legal_mask": record["next_mask"],
        "done": record["next_done"],
        "next_seat": record["next_seat"],
    }
    return adv, strat, hist


def _empty_partial_record(env: "EnvSpec") -> dict[str, Array]:
    """All-zero partial record for skipped (post-terminal) scan steps. Matches
    ``_traversal_step_partial``'s record schema. Zero ``policy``/``q`` make any accidental bootstrap
    use harmless; these rows are masked out of ingestion by ``active`` regardless."""

    info_dim, hist_dim, n_act = env.info_dim, env.history_dim, env.num_actions
    return {
        "info": jnp.zeros(info_dim, dtype=jnp.float32),
        "hist": jnp.zeros(hist_dim, dtype=jnp.float32),
        "mask": jnp.zeros(n_act, dtype=bool),
        "seat": jnp.int32(0),
        "policy": jnp.zeros(n_act, dtype=jnp.float32),
        "q": jnp.zeros(n_act, dtype=jnp.float32),
        "action": jnp.int32(0),
        "xi_a": jnp.float32(1.0),       # nonzero to avoid 0/0 in VR correction on dead rows
        "iteration": jnp.int32(0),
        "is_trav": jnp.bool_(False),
        "next_done": jnp.float32(0.0),
        "terminal_payoff": jnp.float32(0.0),
        "next_info": jnp.zeros(info_dim, dtype=jnp.float32),
        "next_mask": jnp.zeros(n_act, dtype=bool),
        "next_hist": jnp.zeros(hist_dim, dtype=jnp.float32),
        "next_seat": jnp.int32(0),
    }


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
    reward_scale: float,
    hidden_dim: int,
    env: "EnvSpec",
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], Array, Array]:
    """Run one hand via ``lax.scan`` (fixed horizon, mask early termination).

    The scan body computes ONLY each node's own nets (``_traversal_step_partial``, 3 applies); the VR
    bootstrap that needs the child node's policy/q is reconstructed once after the scan by
    ``_finalize_advantages`` (a one-step shift of the per-step policy/q). This halves the per-step net
    applies vs the prior body that recomputed the next node inline -- numerically identical output."""

    def body(carry: tuple[Any, Array, Array], _: None) -> tuple[tuple[Any, Array, Array], tuple]:
        state, key, active = carry

        def do_step() -> tuple[tuple[Any, Array, Array], tuple]:
            next_state, next_key, record = _traversal_step_partial(
                state,
                key,
                traverser,
                iteration,
                cum_params,
                inst_params,
                hv_params,
                alpha,
                epsilon,
                reward_scale,
                hidden_dim,
                env,
            )
            still_active = jnp.logical_and(active, jnp.logical_not(env.done(state)))
            return (next_state, next_key, still_active), (record, still_active)

        def skip_step() -> tuple[tuple[Any, Array, Array], tuple]:
            return (state, key, jnp.bool_(False)), (_empty_partial_record(env), jnp.bool_(False))

        return jax.lax.cond(active, do_step, skip_step)

    init_state = env.reset(reset_key)
    init = (init_state, trav_key, jnp.logical_not(env.done(init_state)))
    # env.max_steps is the per-game fixed horizon (>= true worst-case hand depth; see EnvSpec).
    # Tightening it from the legacy 128 to the game's real ceiling cuts the wasted masked-skip
    # steps without changing results (no real hand is ever truncated).
    _, (record_t, active_t) = jax.lax.scan(body, init, None, length=env.max_steps)
    adv, strat, hist = _finalize_advantages(record_t, active_t)
    is_trav_t = record_t["is_trav"]
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
    reward_scale: float,
    hidden_dim: int,
    env: "EnvSpec",
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], Array, Array]:
    """``vmap`` over parallel outcome-sampled traversals for one updating player.

    Stacks the per-player cum/inst params ONCE here (leading player axis) so every scan step gathers
    the acting seat's slice and does a single MLP apply, instead of the old per-step ``lax.switch``
    that (under ``vmap``) evaluated all ``num_players`` branches."""

    cum_params = _stack_player_params(cum_params)
    inst_params = _stack_player_params(inst_params)

    vmapped = jax.vmap(
        lambda rk, tk: _scan_one_traversal(
            rk, tk, traverser, iteration, cum_params, inst_params, hv_params, alpha, epsilon,
            reward_scale, hidden_dim, env
        )
    )
    return vmapped(reset_keys, trav_keys)


def _pmapped_traversal_scans(
    reset_keys: Array,        # (n_devices, per_dev) PRNGKey
    trav_keys: Array,         # (n_devices, per_dev) PRNGKey
    traverser: Array,
    iteration: Array,
    cum_params: tuple[Any, ...],
    inst_params: tuple[Any, ...],
    hv_params: Any,
    alpha: float,
    epsilon: float,
    reward_scale: float,
    hidden_dim: int,
    env: "EnvSpec",
):
    """``pmap`` the traversal batch across host CPU devices, ``vmap`` within each device.

    The traversal phase is embarrassingly parallel (each outcome-sampled hand is independent), so we
    split the ``k`` traversals into ``n_devices`` shards of ``k//n_devices`` and run one ``vmap`` scan
    per device IN PARALLEL across devices. With ``--xla_force_host_platform_device_count=N`` (set before
    jax import) this maps the work onto N separate XLA executables that run concurrently on N disjoint
    thread pools -- the only way a single JAX-on-CPU process lights up more than the handful of cores a
    single device's vmap reaches. Output keeps the leading device axis; the trainer reshapes
    ``(n_devices, per_dev, steps, ...) -> (k, steps, ...)`` before ingestion (identical samples to the
    single-device path; only the parallelization of independent traversals differs)."""

    stacked_cum = _stack_player_params(cum_params)
    stacked_inst = _stack_player_params(inst_params)

    def per_device(rk_dev, tk_dev, cum_p, inst_p, hv_p, trav, it):
        vmapped = jax.vmap(
            lambda rk, tk: _scan_one_traversal(
                rk, tk, trav, it, cum_p, inst_p, hv_p, alpha, epsilon,
                reward_scale, hidden_dim, env
            )
        )
        return vmapped(rk_dev, tk_dev)

    # Keys are SHARDED (in_axes=0 -> one shard per device); params + traverser + iteration are
    # BROADCAST to every device (in_axes=None). static_broadcasted_argnums keeps the python-level
    # alpha/.../env as compile-time constants (same as the jit path).
    pmapped = jax.pmap(
        per_device,
        in_axes=(0, 0, None, None, None, None, None),
    )
    return pmapped(reset_keys, trav_keys, stacked_cum, stacked_inst, hv_params,
                   traverser, iteration)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _make_train_state(module: nn.Module, key: Array, lr: float, env: "EnvSpec") -> train_state.TrainState:
    tx = optax.adam(lr)
    if isinstance(module, HistoryValueNet):
        dummy = jnp.zeros((1, env.history_dim), dtype=jnp.float32)
        params = module.init(key, dummy)
    else:
        dummy_info = jnp.zeros((1, env.info_dim), dtype=jnp.float32)
        dummy_mask = jnp.ones((1, env.num_actions), dtype=bool)
        params = module.init(key, dummy_info, dummy_mask)
    return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)


class VRDeepPDCFRPlusTrainer:
    """Hardware-accelerated VR-DeepPDCFR+ training loop."""

    def __init__(
        self,
        config: VRDeepPDCFRConfig | None = None,
        *,
        env: EnvSpec | None = None,
        start_iteration: int = 0,
        prng_key: Array | None = None,
        cum_states: tuple[Any, ...] | None = None,
        inst_states: tuple[Any, ...] | None = None,
        hv_state: Any | None = None,
        avg_state: Any | None = None,
    ) -> None:
        self.config = config or VRDeepPDCFRConfig()
        # Pluggable game. Default = 6-max NLHE (production); the gate passes a Leduc EnvSpec.
        self.env = env if env is not None else nlhe_env_spec()
        np_ = self.env.num_players
        self._start_iteration = start_iteration
        self._iteration = start_iteration
        self.key = prng_key if prng_key is not None else jax.random.PRNGKey(self.config.seed)

        cap = self.config.buffer_capacity
        self.adv_buffers = init_advantage_buffers(
            spec=advantage_buffer_spec(self.env.info_dim, self.env.num_actions, cap), num_players=np_)
        self.strat_buffer = init_strategy_buffer(
            spec=strategy_buffer_spec(self.env.info_dim, self.env.num_actions, cap))
        self.hist_buffer = init_history_value_buffer(
            spec=history_value_buffer_spec(self.env.info_dim, self.env.history_dim,
                                           self.env.num_actions, cap))

        hidden = self.config.hidden_dim
        lr = self.config.learning_rate
        na = self.env.num_actions

        if cum_states is not None and inst_states is not None and hv_state is not None and avg_state is not None:
            self.cum_states = cum_states
            self.inst_states = inst_states
            self.hv_state = hv_state
            self.avg_state = avg_state
        else:
            self.key, *subkeys = jax.random.split(self.key, np_ * 2 + 3)
            self.cum_states = tuple(
                _make_train_state(CumulativeAdvantageNet(hidden, na), subkeys[i], lr, self.env)
                for i in range(np_)
            )
            self.inst_states = tuple(
                _make_train_state(InstantaneousAdvantageNet(hidden, na), subkeys[np_ + i], lr, self.env)
                for i in range(np_)
            )
            self.hv_state = _make_train_state(HistoryValueNet(hidden, na), subkeys[-2], lr, self.env)
            self.avg_state = _make_train_state(AverageStrategyNet(hidden, na), subkeys[-1], lr, self.env)

        self._prev_cum_params = tuple(s.params for s in self.cum_states)
        self._batched_scans = jax.jit(
            _batched_traversal_scans,
            static_argnames=("alpha", "epsilon", "reward_scale", "hidden_dim", "env"),
        )
        # MULTI-DEVICE SATURATION: if the process exposes >1 CPU device (set
        # --xla_force_host_platform_device_count=N before importing jax; see scripts/neural_cfr/
        # saturate_env.py), shard each iteration's traversals across the N devices via pmap so the
        # embarrassingly-parallel data-generation phase (the NLHE bottleneck, ~83% of wall time) lights
        # up all N devices' thread pools concurrently instead of the ~handful of cores a single device's
        # vmap reaches. pmap is built once here and reused; the static knobs are partial-applied so the
        # mapped call site only passes the dynamic arrays (keys, params, traverser, iteration).
        self._n_devices = jax.local_device_count()
        # Only pmap when there are >1 devices AND at least one traversal lands on each (k >= n_devices).
        # run_traversals additionally requires k % n_devices == 0 per call (falls back to vmap if not),
        # so for clean saturation set traversals_per_iter to a multiple of the device count.
        self._use_pmap = self._n_devices > 1 and self.config.traversals_per_iter >= self._n_devices
        if self._use_pmap:
            import functools
            self._pmapped_scans = functools.partial(
                _pmapped_traversal_scans,
                alpha=self.config.alpha, epsilon=self.config.epsilon,
                reward_scale=self.config.reward_scale, hidden_dim=self.config.hidden_dim,
                env=self.env,
            )

        # MULTIPROCESS SATURATION: spin up the persistent traversal worker pool if requested. Workers
        # generate outcome-sampled traversal samples data-parallel (independent => same estimator); the
        # parent keeps the single canonical buffer + grad path. This is the topology that actually
        # saturates a many-vCPU box (the in-process scan is ~1-2 cores no matter the thread count).
        self._trav_pool = None
        if self.config.traversal_workers and self.config.traversal_workers > 0:
            from pokerbot.training.mp_traversal import TraversalPool
            env_name = "leduc" if self.env.info_dim == 96 and self.env.num_actions == 3 else "nlhe"
            # Robust env tag: NLHE unless the spec matches the Leduc adapter dims exactly.
            try:
                from pokerbot.engine import jax_engine as _je
                env_name = "nlhe" if (self.env.info_dim == _je.INFO_FEATURE_DIM
                                      and self.env.num_actions == _je.NUM_ACTIONS) else "leduc"
            except Exception:
                pass
            self._trav_pool = TraversalPool(
                num_workers=int(self.config.traversal_workers), env_name=env_name,
                hidden_dim=self.config.hidden_dim, num_actions=self.env.num_actions,
                reward_scale=self.config.reward_scale, alpha=self.config.alpha,
                epsilon=self.config.epsilon,
            )

    def close(self) -> None:
        """Release the multiprocess traversal pool (no-op if not used). Call when done training."""
        if getattr(self, "_trav_pool", None) is not None:
            self._trav_pool.close()
            self._trav_pool = None

    def _shard_keys(self, keys: Array) -> Array:
        """Reshape a flat ``(k,)`` key batch to ``(n_devices, k//n_devices)`` for pmap. ``k`` MUST be
        divisible by n_devices (the trainer rounds traversals_per_iter up at config time)."""
        d = self._n_devices
        per = keys.shape[0] // d
        return keys[: d * per].reshape((d, per) + keys.shape[1:])

    def _insert_host_batch(self, traverser: int, res: dict) -> None:
        """Ingest worker-generated HOST (numpy) samples into the buffers (parent-side, single thread).

        Mirrors :meth:`_insert_traversal_batch` but the samples already arrive flattened on host; we
        move them to device once and run the same three batched buffer ops (identical mask semantics)."""
        adv_f = {k: jnp.asarray(v) for k, v in res["adv"].items()}
        strat_f = {k: jnp.asarray(v) for k, v in res["strat"].items()}
        hist_f = {k: jnp.asarray(v) for k, v in res["hist"].items()}
        active_f = jnp.asarray(res["active"])
        is_trav_f = jnp.asarray(res["is_trav"])
        adv_valid = jnp.logical_and(active_f, is_trav_f)
        strat_valid = jnp.logical_and(active_f, jnp.logical_not(is_trav_f))
        hist_valid = active_f

        new_adv_buf = linear_write_batch(self.adv_buffers.buffers[traverser], adv_f, adv_valid)
        self.adv_buffers = AdvantageBuffers(
            buffers=self.adv_buffers.buffers[:traverser] + (new_adv_buf,)
            + self.adv_buffers.buffers[traverser + 1 :]
        )
        self.key, ks = jax.random.split(self.key)
        self.strat_buffer = StrategyBuffer(
            buffer=reservoir_insert_batch(self.strat_buffer.buffer, ks, strat_f, strat_valid))
        self.key, kh = jax.random.split(self.key)
        self.hist_buffer = HistoryValueBuffer(
            buffer=reservoir_insert_batch(self.hist_buffer.buffer, kh, hist_f, hist_valid))

    def _train_player_cumulative(self, player: int, iteration: int, num_steps: int = 1) -> None:
        """Fit the player's cumulative-advantage net for ``num_steps`` fused Adam steps (one dispatch).

        ``num_steps`` minibatches are drawn inside the jitted scan over the (device-resident) advantage
        buffer; ``prev_params`` and ``iteration`` are held fixed across the scan exactly as the prior
        per-iteration loop held them. Statistically identical to ``num_steps`` of the old single-step
        call, with K-fold less dispatch/trace overhead."""
        cfg = self.config
        buf = self.adv_buffers.buffers[player]
        if int(buf.size) == 0:
            return
        self.key, k = jax.random.split(self.key)
        state = self.cum_states[player]
        new_state = _fused_cumulative_update(
            state, buf.data, buf.size, k, self._prev_cum_params[player], state.apply_fn,
            cfg.batch_size, int(num_steps), iteration, cfg.alpha,
        )
        self.cum_states = self.cum_states[:player] + (new_state,) + self.cum_states[player + 1 :]

    def _train_player_instantaneous(self, player: int, num_steps: int = 1) -> None:
        """Fit the player's instantaneous-advantage net for ``num_steps`` fused Adam steps."""
        cfg = self.config
        buf = self.adv_buffers.buffers[player]
        if int(buf.size) == 0:
            return
        self.key, k = jax.random.split(self.key)
        state = self.inst_states[player]
        new_state = _fused_instantaneous_update(
            state, buf.data, buf.size, k, state.apply_fn, cfg.batch_size, int(num_steps),
        )
        self.inst_states = self.inst_states[:player] + (new_state,) + self.inst_states[player + 1 :]

    def _train_history_value(self, iteration: int, num_steps: int = 1) -> None:
        """Fit the VR-baseline history-value net for ``num_steps`` fused Adam steps. The served
        average net (read by the Bellman bootstrap) is fixed across the scan, as before."""
        cfg = self.config
        buf = self.hist_buffer.buffer
        if int(buf.size) == 0:
            return
        self.key, k = jax.random.split(self.key)
        new_state = _fused_history_value_update(
            self.hv_state, buf.data, buf.size, k, self.avg_state.params, self.avg_state.apply_fn,
            self.hv_state.apply_fn, cfg.batch_size, int(num_steps),
        )
        self.hv_state = new_state

    def _train_average_strategy(self, iteration: int, num_steps: int = 1) -> None:
        """Fit the served average-strategy net for ``num_steps`` fused Adam steps."""
        cfg = self.config
        buf = self.strat_buffer.buffer
        if int(buf.size) == 0:
            return
        self.key, k = jax.random.split(self.key)
        state = self.avg_state
        new_state = _fused_average_strategy_update(
            state, buf.data, buf.size, k, state.apply_fn, cfg.batch_size, int(num_steps),
            cfg.total_iterations, cfg.gamma,
        )
        self.avg_state = new_state

    def _insert_traversal_batch(
        self,
        traverser: int,
        adv: dict[str, Array],
        strat: dict[str, Array],
        hist: dict[str, Array],
        is_trav: Array,
        active: Array,
    ) -> None:
        """Ingest a whole device rollout into the replay buffers in ONE pass.

        The old host-side ``(batch x steps)`` Python double loop did a per-sample ``device_get`` +
        per-sample jitted reservoir insert (~49k dispatches/iter), which serialized everything and
        dominated the iteration (~86% of wall-clock, profiled). Instead we keep the rollout on-device,
        flatten the ``(batch, steps)`` axes, and run three batched buffer ops -- one masked append into
        the iteration-local advantage buffer + two vectorized Algorithm-R reservoir inserts. Mask
        semantics are identical to the old loop:
          advantage <- active & is_traverser | strategy <- active & ~is_traverser | history <- active
        Reservoir sampling is preserved statistically (not bit-for-bit RNG). ~188x faster ingestion."""

        def _flatten(d: dict[str, Array]) -> dict[str, Array]:
            return {k: v.reshape((-1,) + v.shape[2:]) for k, v in d.items()}   # collapse (batch, steps)

        adv_f, strat_f, hist_f = _flatten(adv), _flatten(strat), _flatten(hist)
        active_f = active.reshape((-1,))
        is_trav_f = is_trav.reshape((-1,))
        adv_valid = jnp.logical_and(active_f, is_trav_f)
        strat_valid = jnp.logical_and(active_f, jnp.logical_not(is_trav_f))
        hist_valid = active_f

        new_adv_buf = linear_write_batch(self.adv_buffers.buffers[traverser], adv_f, adv_valid)
        self.adv_buffers = AdvantageBuffers(
            buffers=self.adv_buffers.buffers[:traverser] + (new_adv_buf,)
            + self.adv_buffers.buffers[traverser + 1 :]
        )
        self.key, ks = jax.random.split(self.key)
        self.strat_buffer = StrategyBuffer(
            buffer=reservoir_insert_batch(self.strat_buffer.buffer, ks, strat_f, strat_valid))
        self.key, kh = jax.random.split(self.key)
        self.hist_buffer = HistoryValueBuffer(
            buffer=reservoir_insert_batch(self.hist_buffer.buffer, kh, hist_f, hist_valid))

    def run_traversals(self, iteration: int) -> None:
        """Collect K outcome-sampled traversals for each player (scan + vmap)."""

        cfg = self.config
        self.adv_buffers = clear_all_advantage_buffers(self.adv_buffers)
        cum_params = tuple(s.params for s in self.cum_states)
        inst_params = tuple(s.params for s in self.inst_states)
        hv_params = self.hv_state.params
        iter_arr = jnp.int32(iteration)
        k = cfg.traversals_per_iter

        # MULTIPROCESS PATH: broadcast the current params (host numpy) to the worker pool, which
        # generates each traverser's traversals data-parallel; ingest the concatenated host samples.
        if self._trav_pool is not None:
            cum_host = jax.device_get(cum_params)
            inst_host = jax.device_get(inst_params)
            hv_host = jax.device_get(hv_params)
            for traverser in range(self.env.num_players):
                res = self._trav_pool.generate(
                    cum_host, inst_host, hv_host, traverser, iteration, k, int(cfg.seed))
                self._insert_host_batch(traverser, res)
            return

        for traverser in range(self.env.num_players):
            self.key, key_block = jax.random.split(self.key)
            keys = jax.random.split(key_block, k * 2)
            reset_keys = keys[:k]
            trav_keys = keys[k:]
            if self._use_pmap and k % self._n_devices == 0:
                out = self._pmapped_scans(
                    self._shard_keys(reset_keys),
                    self._shard_keys(trav_keys),
                    jnp.int32(traverser),
                    iter_arr,
                    cum_params,
                    inst_params,
                    hv_params,
                )
                # pmap output carries a leading device axis: (n_dev, per_dev, steps, ...). Collapse
                # (n_dev, per_dev) -> k so ingestion sees the same (k, steps, ...) layout as vmap.
                adv, strat, hist, is_trav, active = jax.tree_util.tree_map(
                    lambda x: x.reshape((-1,) + x.shape[2:]), out)
            else:
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
                    cfg.reward_scale,
                    cfg.hidden_dim,
                    self.env,
                )
            self._insert_traversal_batch(traverser, adv, strat, hist, is_trav, active)

    def train_iteration(self, iteration: int) -> dict[str, float]:
        """Single CFR iteration: traverse, then train all networks."""

        self._iteration = iteration
        self.run_traversals(iteration)

        steps = max(1, self.config.train_steps_per_iter)
        # Per-net overrides (None => uniform `steps`, exact legacy behavior).
        hv_steps = max(1, self.config.history_value_steps_per_iter
                       if self.config.history_value_steps_per_iter is not None else steps)
        avg_steps = max(1, self.config.avg_strategy_steps_per_iter
                        if self.config.avg_strategy_steps_per_iter is not None else steps)
        # Each call FUSES its K steps into one jitted scan over the device-resident buffer (one
        # dispatch + one trace per net per iter, vs the old K Python-dispatched single-step calls).
        # The advantage cum/inst nets previously updated in a STRIDED order (cum_step, inst_step, cum,
        # inst, ...). The cum net's prev_params and the inst loss are independent of the other net's
        # in-progress params, so fitting all K cum steps then all K inst steps yields the same per-net
        # sequence of updates (same minibatch sampler, same fixed prev_params/iteration) -- the served
        # nets land in the same regime.
        for player in range(self.env.num_players):
            self._train_player_cumulative(player, iteration, steps)
            self._train_player_instantaneous(player, steps)

        # The history-value net is the VR baseline that the avg-strategy net's bootstrap reads, so fit
        # it (>= its override) before the avg net. Both default to `steps` => identical to the old loop.
        self._train_history_value(iteration, hv_steps)
        self._train_average_strategy(iteration, avg_steps)

        self._prev_cum_params = tuple(s.params for s in self.cum_states)
        return {"iteration": float(iteration)}

    def train(self, *, log_every: int = 0, label: str = "VR-DeepPDCFR+",
              checkpoint_every: int = 0, checkpoint_path: str | None = None) -> NetworkParams:
        """Run full VR-DeepPDCFR+ training. ``log_every>0`` prints periodic ETA progress.

        ``checkpoint_every>0`` with ``checkpoint_path`` set saves a resumable full checkpoint every
        N iterations (ATOMICALLY: written to a temp path then ``os.replace``-promoted, so an
        interrupted write -- OOM, kill -- never corrupts the last good checkpoint). Essential for
        unattended overnight runs; a crash at hour 6 resumes from the last save, not from zero.
        """

        from pokerbot.training.train_logging import ProgressLogger

        start = self._start_iteration
        total = self.config.total_iterations
        plog = ProgressLogger(
            total, start_iter=start, every=log_every, label=label,
            extra=lambda: f"strat_buf={int(self.strat_buffer.buffer.size)} "
                          f"hist_buf={int(self.hist_buffer.buffer.size)}",
        ).begin()
        for t in range(start + 1, total + 1):
            self.train_iteration(t)
            plog.step(t)
            if checkpoint_every and checkpoint_path and (t % checkpoint_every == 0):
                _atomic_save_trainer_checkpoint(self, checkpoint_path)
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
        "train_steps_per_iter": config.train_steps_per_iter,
        "avg_strategy_steps_per_iter": config.avg_strategy_steps_per_iter,
        "history_value_steps_per_iter": config.history_value_steps_per_iter,
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

    from pokerbot.engine import jax_engine as _je

    cfg = config or VRDeepPDCFRConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "history_value_params": jax.device_get(params.history_value.params),
        "average_strategy_params": jax.device_get(params.average_strategy.params),
        "hidden_dim": cfg.hidden_dim,
        "total_iterations": cfg.total_iterations,
        "iteration": iteration,
        # encoding-flag stamp: the served info-state layout must match training (skew guard) -- a
        # checkpoint trained with POKERBOT_INFO_SUITS=1 served without it (or vice-versa) silently
        # feeds the net zeros where it expects suit features.
        "info_suits": bool(_je._INFO_SUITS),
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


def _atomic_save_trainer_checkpoint(trainer: VRDeepPDCFRPlusTrainer, path: str | Path) -> Path:
    """Crash-safe ``save_trainer_checkpoint``: write both files (inference + ``_full``) to a temp
    stem, then ``os.replace``-promote each to its final name. ``os.replace`` is atomic on the same
    filesystem, so an interrupted write leaves the previous good checkpoint untouched. Returns the
    promoted full-checkpoint path."""

    import os

    path = Path(path)
    tmp = path.with_name(f"{path.stem}.writing{path.suffix}")
    save_trainer_checkpoint(trainer, tmp)  # writes tmp + <tmp_stem>_full<suffix>
    tmp_full = tmp.with_name(f"{tmp.stem}_full{tmp.suffix}")
    final_full = path.with_name(f"{path.stem}_full{path.suffix}")
    os.replace(tmp, path)
    os.replace(tmp_full, final_full)
    return final_full


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
