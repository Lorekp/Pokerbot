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
    count: Array  # () int64 total inserts for reservoir sampling
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
    return BufferState(data=data, size=jnp.int32(0), count=jnp.int64(0), capacity=cap)


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
    },
)


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


def sample_linear(buf: LinearBufferState, key: Array, batch_size: int) -> dict[str, Array]:
    size = jnp.maximum(buf.size.astype(jnp.int32), 1)
    idx = jax.random.randint(key, (batch_size,), 0, size)
    return {name: arr[idx] for name, arr in buf.data.items()}


def init_advantage_buffers(capacity: int | None = None) -> AdvantageBuffers:
    spec = ADVANTAGE_BUFFER_SPEC
    if capacity is not None:
        spec = BufferSpec(capacity=capacity, fields=spec.fields)
    return AdvantageBuffers(buffers=tuple(init_linear_buffer(spec) for _ in range(NUM_PLAYERS)))


def init_strategy_buffer(capacity: int | None = None) -> StrategyBuffer:
    spec = STRATEGY_BUFFER_SPEC
    if capacity is not None:
        spec = BufferSpec(capacity=capacity, fields=spec.fields)
    return StrategyBuffer(buffer=init_buffer(spec))


def init_history_value_buffer(capacity: int | None = None) -> HistoryValueBuffer:
    spec = HISTORY_VALUE_BUFFER_SPEC
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
) -> BufferState:
    sample = {
        "history": history,
        "next_history": next_history,
        "action": action,
        "utility": utility,
        "player": player,
        "iteration": iteration,
        "next_info_state": next_info_state,
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
