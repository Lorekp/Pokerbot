"""JAX-compatible reservoir-sampled replay buffers for VR-DeepPDCFR+.

All buffers pre-allocate fixed-size arrays before training.  Insertion uses
reservoir sampling (Vitter's algorithm) so memory stays bounded while
approximating uniform samples over millions of transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from pokerbot.engine.jax_engine import HISTORY_FEATURE_DIM, INFO_FEATURE_DIM, NUM_ACTIONS, NUM_PLAYERS


class BufferState(NamedTuple):
    """Shared reservoir buffer state."""

    data: dict[str, Array]
    size: Array  # () int32 current logical size (capped at capacity)
    count: Array  # () int32 total inserts for reservoir sampling (x64 is off; exact to ~2.1e9 inserts)
    capacity: Array  # () int32


@dataclass(frozen=True)
class BufferSpec:
    """Schema for a pre-allocated replay buffer."""

    capacity: int
    fields: dict[str, tuple[tuple[int, ...], jnp.dtype]]


def _empty_field(shape: tuple[int, ...], dtype: jnp.dtype) -> Array:
    return jnp.zeros(shape, dtype=dtype)


def init_buffer(spec: BufferSpec) -> BufferState:
    """Allocate empty buffer arrays."""

    cap = jnp.int32(spec.capacity)
    data = {name: _empty_field((spec.capacity, *shape), dtype) for name, (shape, dtype) in spec.fields.items()}
    # count is int32 (x64 is off, so an int64 request was silently truncated to int32 anyway -- making it
    # explicit removes the JAX dtype warning). The reservoir math (p=count+j, randint(0,p)) is exact while
    # count < 2^31; at ~3e3 inserts/iter a 100k-iter run reaches ~3e8, ~6x under that ceiling.
    return BufferState(data=data, size=jnp.int32(0), count=jnp.int32(0), capacity=cap)


@jax.jit
def reservoir_insert(
    buf: BufferState,
    key: Array,
    sample: dict[str, Array],
) -> BufferState:
    """Insert one sample via reservoir sampling."""

    count = buf.count + 1
    size = buf.size
    cap = buf.capacity

    def insert_new(_: None) -> tuple[dict[str, Array], Array]:
        idx = size
        new_data = {k: v.at[idx].set(sample[k]) for k, v in buf.data.items()}
        return new_data, size + 1

    def maybe_replace(_: None) -> tuple[dict[str, Array], Array]:
        key_i, key_r = jax.random.split(key)
        j = jax.random.randint(key_i, (), 0, count)
        do_replace = j < cap
        idx = j.astype(jnp.int32)

        def replace(d: dict[str, Array]) -> dict[str, Array]:
            return {k: jax.lax.cond(do_replace, lambda x: x.at[idx].set(sample[k]), lambda x: x, v) for k, v in d.items()}

        return replace(buf.data), size

    new_data, new_size = jax.lax.cond(size < cap, insert_new, maybe_replace, None)
    return BufferState(data=new_data, size=jnp.minimum(new_size, cap), count=count, capacity=cap)


@jax.jit
def reservoir_insert_batch(
    buf: BufferState,
    key: Array,
    samples: dict[str, Array],
    valid: Array,
) -> BufferState:
    """Insert a masked batch via vectorized Algorithm-R reservoir sampling.

    Statistically equivalent to applying ``reservoir_insert`` to each active row in stream order (it
    does NOT reproduce the old per-sample RNG sequence -- statistical, not bit, equivalence). For each
    active row at 1-based stream position ``p = count + j``: if ``p <= capacity`` it fills slot
    ``p-1``; else draw ``r ~ U{0..p-1}`` and replace slot ``r`` iff ``r < capacity`` (Vitter's R).
    When several rows target the same slot the later row wins (sequential semantics)."""

    cap = buf.capacity
    count0 = buf.count
    size0 = buf.size
    m = valid.shape[0]
    valid_i = valid.astype(jnp.int32)
    n_active = jnp.sum(valid_i)

    order = jnp.argsort(jnp.logical_not(valid), stable=True)   # compact active rows to front, stable
    j = jnp.arange(m)
    active_rank = j < n_active
    p = count0 + j + 1
    filling = p <= cap.astype(p.dtype)
    fill_slot = (p - 1).astype(jnp.int32)
    r = jax.random.randint(key, (m,), 0, jnp.maximum(p.astype(jnp.int32), 1))
    do_replace = jnp.logical_and(jnp.logical_not(filling), r < cap)
    writes = jnp.logical_and(active_rank, jnp.logical_or(filling, do_replace))
    target = jnp.where(filling, fill_slot, r.astype(jnp.int32))
    target = jnp.clip(target, 0, cap - 1)

    # last-writer-wins without a capacity-sized array (capacity is traced): sort writing rows by
    # (target, rank); the final element of each contiguous target group survives. O(m log m), m static.
    sentinel = jnp.iinfo(jnp.int32).max
    sort_target = jnp.where(writes, target, sentinel)
    o1 = jnp.argsort(j, stable=True)
    o2 = jnp.argsort(sort_target[o1], stable=True)
    sorted_idx = o1[o2]
    st_sorted = sort_target[sorted_idx]
    is_last = jnp.concatenate([st_sorted[:-1] != st_sorted[1:], jnp.array([True])])
    survive_sorted = jnp.logical_and(is_last, st_sorted != sentinel)
    survives = jnp.zeros((m,), dtype=bool).at[sorted_idx].set(survive_sorted)
    scatter_idx = jnp.where(survives, target, cap)   # non-survivors routed out-of-bounds (drop)

    def write_field(buf_arr: Array, new: Array) -> Array:
        compacted = new[order]
        return buf_arr.at[scatter_idx].set(compacted, mode="drop")

    new_data = {k: write_field(v, samples[k]) for k, v in buf.data.items()}
    new_count = count0 + n_active.astype(count0.dtype)
    new_size = jnp.minimum(new_count.astype(jnp.int32), cap)
    new_size = jnp.maximum(new_size, size0)
    return BufferState(data=new_data, size=new_size, count=new_count, capacity=cap)


def sample_uniform(buf: BufferState, key: Array, batch_size: int) -> dict[str, Array]:
    """Draw ``batch_size`` uniform random samples (with replacement)."""

    size = buf.size.astype(jnp.int32)
    idx = jax.random.randint(key, (batch_size,), 0, jnp.maximum(size, 1))
    return {name: arr[idx] for name, arr in buf.data.items()}


# ---------------------------------------------------------------------------
# VR-DeepPDCFR+ buffer specifications
# ---------------------------------------------------------------------------

ADVANTAGE_BUFFER_SPEC = BufferSpec(
    capacity=2_000_000,
    fields={
        "info_state": ((INFO_FEATURE_DIM,), jnp.float32),
        "advantages": ((NUM_ACTIONS,), jnp.float32),
        "legal_mask": ((NUM_ACTIONS,), jnp.bool_),
        "iteration": ((), jnp.int32),
    },
)

STRATEGY_BUFFER_SPEC = BufferSpec(
    capacity=2_000_000,
    fields={
        "info_state": ((INFO_FEATURE_DIM,), jnp.float32),
        "strategy": ((NUM_ACTIONS,), jnp.float32),
        "legal_mask": ((NUM_ACTIONS,), jnp.bool_),
        "iteration": ((), jnp.int32),
    },
)

HISTORY_VALUE_BUFFER_SPEC = BufferSpec(
    capacity=2_000_000,
    fields={
        "history": ((HISTORY_FEATURE_DIM,), jnp.float32),
        "next_history": ((HISTORY_FEATURE_DIM,), jnp.float32),
        "action": ((), jnp.int32),
        "utility": ((), jnp.float32),
        "player": ((), jnp.int32),
        "iteration": ((), jnp.int32),
        "next_info_state": ((INFO_FEATURE_DIM,), jnp.float32),
        "next_legal_mask": ((NUM_ACTIONS,), jnp.bool_),
        "done": ((), jnp.float32),
        "next_seat": ((), jnp.int32),
    },
)


# --- dimension-parameterized spec factories (env-pluggable trainer: Leduc gate uses smaller dims) ---


def advantage_buffer_spec(info_dim: int, num_actions: int, capacity: int = 2_000_000) -> BufferSpec:
    return BufferSpec(capacity=capacity, fields={
        "info_state": ((info_dim,), jnp.float32),
        "advantages": ((num_actions,), jnp.float32),
        "legal_mask": ((num_actions,), jnp.bool_),
        "iteration": ((), jnp.int32),
    })


def strategy_buffer_spec(info_dim: int, num_actions: int, capacity: int = 2_000_000) -> BufferSpec:
    return BufferSpec(capacity=capacity, fields={
        "info_state": ((info_dim,), jnp.float32),
        "strategy": ((num_actions,), jnp.float32),
        "legal_mask": ((num_actions,), jnp.bool_),
        "iteration": ((), jnp.int32),
    })


def history_value_buffer_spec(info_dim: int, history_dim: int, num_actions: int,
                              capacity: int = 2_000_000) -> BufferSpec:
    return BufferSpec(capacity=capacity, fields={
        "history": ((history_dim,), jnp.float32),
        "next_history": ((history_dim,), jnp.float32),
        "action": ((), jnp.int32),
        "utility": ((), jnp.float32),
        "player": ((), jnp.int32),
        "iteration": ((), jnp.int32),
        "next_info_state": ((info_dim,), jnp.float32),
        "next_legal_mask": ((num_actions,), jnp.bool_),
        "done": ((), jnp.float32),
        "next_seat": ((), jnp.int32),
    })


class AdvantageBuffers(NamedTuple):
    """Per-player advantage buffers (cleared each CFR iteration)."""

    buffers: tuple[LinearBufferState, ...]


class StrategyBuffer(NamedTuple):
    """Reservoir-sampled strategy buffer B_Pi."""

    buffer: BufferState


class HistoryValueBuffer(NamedTuple):
    """History value buffer B_Q for variance reduction."""

    buffer: BufferState


class LinearBufferState(NamedTuple):
    """Iteration-local buffer with append-only writes (cleared each CFR iter)."""

    data: dict[str, Array]
    size: Array
    capacity: Array


def init_linear_buffer(spec: BufferSpec) -> LinearBufferState:
    cap = jnp.int32(spec.capacity)
    data = {name: _empty_field((spec.capacity, *shape), dtype) for name, (shape, dtype) in spec.fields.items()}
    return LinearBufferState(data=data, size=jnp.int32(0), capacity=cap)


@jax.jit
def linear_write(buf: LinearBufferState, sample: dict[str, Array]) -> LinearBufferState:
    idx = buf.size
    new_data = {k: v.at[idx].set(sample[k]) for k, v in buf.data.items()}
    return LinearBufferState(data=new_data, size=jnp.minimum(idx + 1, buf.capacity), capacity=buf.capacity)


@jax.jit
def linear_write_batch(
    buf: LinearBufferState,
    samples: dict[str, Array],
    valid: Array,
) -> LinearBufferState:
    """Append a masked batch into a linear (append-only) buffer. Active rows are written contiguously
    from the current ``size`` in stream order (same as repeated ``linear_write``); overflow is
    dropped and ``size`` saturates at capacity."""

    cap = buf.capacity
    base = buf.size
    m = valid.shape[0]
    n_active = jnp.sum(valid.astype(jnp.int32))
    order = jnp.argsort(jnp.logical_not(valid), stable=True)   # stable compaction, active to front
    pos = jnp.arange(m)
    dest = base + pos
    in_range = jnp.logical_and(pos < n_active, dest < cap)
    scatter_idx = jnp.where(in_range, dest, cap)               # overflow/invalid routed out (drop)

    def write_field(buf_arr: Array, new: Array) -> Array:
        compacted = new[order]
        return buf_arr.at[scatter_idx].set(compacted, mode="drop")

    new_data = {k: write_field(v, samples[k]) for k, v in buf.data.items()}
    new_size = jnp.minimum(base + n_active, cap)
    return LinearBufferState(data=new_data, size=new_size, capacity=cap)


def sample_linear(buf: LinearBufferState, key: Array, batch_size: int) -> dict[str, Array]:
    size = jnp.maximum(buf.size.astype(jnp.int32), 1)
    idx = jax.random.randint(key, (batch_size,), 0, size)
    return {name: arr[idx] for name, arr in buf.data.items()}


def init_advantage_buffers(capacity: int | None = None, *, spec: BufferSpec | None = None,
                           num_players: int = NUM_PLAYERS) -> AdvantageBuffers:
    spec = spec or ADVANTAGE_BUFFER_SPEC
    if capacity is not None:
        spec = BufferSpec(capacity=capacity, fields=spec.fields)
    return AdvantageBuffers(buffers=tuple(init_linear_buffer(spec) for _ in range(num_players)))


def init_strategy_buffer(capacity: int | None = None, *, spec: BufferSpec | None = None) -> StrategyBuffer:
    spec = spec or STRATEGY_BUFFER_SPEC
    if capacity is not None:
        spec = BufferSpec(capacity=capacity, fields=spec.fields)
    return StrategyBuffer(buffer=init_buffer(spec))


def init_history_value_buffer(capacity: int | None = None, *,
                              spec: BufferSpec | None = None) -> HistoryValueBuffer:
    spec = spec or HISTORY_VALUE_BUFFER_SPEC
    if capacity is not None:
        spec = BufferSpec(capacity=capacity, fields=spec.fields)
    return HistoryValueBuffer(buffer=init_buffer(spec))


@jax.jit
def clear_advantage_buffer(buf: LinearBufferState) -> LinearBufferState:
    """Clear iteration-local advantage buffer."""

    return LinearBufferState(data=buf.data, size=jnp.int32(0), capacity=buf.capacity)


def clear_all_advantage_buffers(buffers: AdvantageBuffers) -> AdvantageBuffers:
    return AdvantageBuffers(buffers=tuple(clear_advantage_buffer(b) for b in buffers.buffers))


@jax.jit
def insert_advantage_sample(
    buf: LinearBufferState,
    info_state: Array,
    advantages: Array,
    legal_mask: Array,
    iteration: Array,
) -> LinearBufferState:
    # 4.1 fix (CFR-AUDIT): clamp advantages to [-1, 1] at insert so a runaway regret value
    # (e.g. a single infoset that was visited O(10^4) times) cannot dominate the network's
    # MSE loss. The distillation loss in sd_cfr_nlhe._finish_iteration_after_traversal is
    # ``MSE(logits, advantages) * mask`` — a single un-clipped outlier 100x the median scales
    # the gradient 100x and the net never converges. Clamp is symmetric (the regret sign
    # carries the "better than alternative" signal); the loss penalty for a saturated
    # magnitude is a clean second-order regularizer, not a bias.
    advantages = jnp.clip(advantages, -1.0, 1.0)
    sample = {
        "info_state": info_state,
        "advantages": advantages,
        "legal_mask": legal_mask,
        "iteration": iteration,
    }
    return linear_write(buf, sample)


@jax.jit
def insert_strategy_sample(
    buf: BufferState,
    key: Array,
    info_state: Array,
    strategy: Array,
    legal_mask: Array,
    iteration: Array,
) -> BufferState:
    sample = {
        "info_state": info_state,
        "strategy": strategy,
        "legal_mask": legal_mask,
        "iteration": iteration,
    }
    return reservoir_insert(buf, key, sample)


@jax.jit
def insert_history_value_sample(
    buf: BufferState,
    key: Array,
    history: Array,
    next_history: Array,
    action: Array,
    utility: Array,
    player: Array,
    iteration: Array,
    next_info_state: Array,
    next_legal_mask: Array,
    done: Array,
    next_seat: Array | None = None,
) -> BufferState:
    sample = {
        "history": history,
        "next_history": next_history,
        "action": action,
        "utility": utility,
        "player": player,
        "iteration": iteration,
        "next_info_state": next_info_state,
        "next_legal_mask": next_legal_mask,
        "done": done,
        "next_seat": jnp.int32(0) if next_seat is None else next_seat,
    }
    return reservoir_insert(buf, key, sample)


def sample_advantage_batch(buf: LinearBufferState, key: Array, batch_size: int) -> dict[str, Array]:
    batch = sample_linear(buf, key, batch_size)
    # mask illegal actions in targets
    batch["advantages"] = batch["advantages"] * batch["legal_mask"]
    return batch


def sample_strategy_batch(buf: BufferState, key: Array, batch_size: int) -> dict[str, Array]:
    return sample_uniform(buf, key, batch_size)


def sample_history_value_batch(buf: BufferState, key: Array, batch_size: int) -> dict[str, Array]:
    return sample_uniform(buf, key, batch_size)
