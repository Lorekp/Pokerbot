#!/usr/bin/env python3
"""Benchmark JAX env apply_action throughput."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from pokerbot.engine.jax_engine import apply_action, batched_reset, reset_state

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    key = jax.random.PRNGKey(0)
    batch = 256
    keys = jax.random.split(key, batch)
    states = batched_reset(keys)
    actions = jnp.zeros(batch, dtype=jnp.int32)
    step_fn = jax.jit(jax.vmap(apply_action))

    # warmup
    step_fn(states, actions).stacks.block_until_ready()

    n = 500
    t0 = time.perf_counter()
    for _ in range(n):
        states = step_fn(states, actions)
    states.stacks.block_until_ready()
    elapsed = time.perf_counter() - t0
    steps_per_sec = n * batch / elapsed

    report = {
        "batch_size": batch,
        "steps_per_batch": n,
        "elapsed_sec": elapsed,
        "apply_actions_per_sec": steps_per_sec,
    }
    out = ROOT / "logs" / "jax_env_throughput.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
