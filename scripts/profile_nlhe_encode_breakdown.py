#!/usr/bin/env python3
"""Profile NLHE encode breakdown, backward target, and fast-path throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import (
    init_hand,
    is_chance_node,
    legal_action_mask,
    sample_chance,
    step_state,
)
from pokerbot.engine.jax_weva_encode import encode_nlhe_weva_bundle
from pokerbot.engine.jax_weva_fast import (
    default_weva_bucket_table,
    encode_nlhe_bundle_fast,
    encode_nlhe_bundles_batch_numpy,
    stack_env_states,
)
from pokerbot.engine.nlhe_bridge import game_state_from_jax
from pokerbot.encoding.tensors import encode_state
from pokerbot.encoding.weva_encoder import WEVAMapper

ROOT = Path(__file__).resolve().parents[1]
HU_OCC = (False, False, False, True, False, True)


def _collect_nodes(n_hands: int, seed: int) -> list[tuple]:
    occ = jnp.array(HU_OCC)
    key = jax.random.PRNGKey(seed)
    nodes: list[tuple] = []
    for _ in range(n_hands):
        key, k1 = jax.random.split(key)
        st = init_hand(k1, button=jnp.int32(3), occupancy=occ)
        for _ in range(200):
            if bool(st.done):
                break
            while bool(is_chance_node(st)):
                st = sample_chance(st)
            if bool(st.done):
                break
            seat = int(st.action_seat)
            if seat >= 0:
                nodes.append((st, seat))
            legal = np.flatnonzero(np.asarray(legal_action_mask(st), dtype=bool))
            if legal.size == 0:
                break
            key, sub = jax.random.split(key)
            pick = int(legal[int(jax.random.randint(sub, (), 0, legal.size))])
            st = step_state(st, jnp.int32(pick))
    return nodes


def _bench_ms(fn, nodes: list, *, warmup: int = 1, rounds: int = 5) -> float:
    for _ in range(warmup):
        for st, seat in nodes[: min(10, len(nodes))]:
            fn(st, seat)
    t0 = time.perf_counter()
    for _ in range(rounds):
        for st, seat in nodes:
            fn(st, seat)
    return (time.perf_counter() - t0) / (rounds * len(nodes)) * 1000.0


def breakdown(nodes: list) -> dict:
    mapper = WEVAMapper.default()
    table = default_weva_bucket_table()

    def full_slow(st, seat):
        encode_nlhe_weva_bundle(st, seat, mapper)

    def bridge(st, seat):
        game_state_from_jax(st)

    def encode_after_bridge(st, seat):
        gs = game_state_from_jax(st)
        encode_state(gs, seat, mapper)

    def weva_lookup(st, seat):
        gs = game_state_from_jax(st)
        h = gs.players[seat].hole
        if h:
            mapper.lookup(h, position=int(gs.players[seat].position))

    def fast_path(st, seat):
        encode_nlhe_bundle_fast(st, seat, weva_table=table)

    def batch_fast(st, seat):
        pass  # placeholder

    n = len(nodes)
    batch_ms = None
    if n >= 32:
        states = [st for st, _ in nodes]
        seats = np.array([seat for _, seat in nodes], dtype=np.int32)
        batch = stack_env_states(states)

        def run_batch():
            encode_nlhe_bundles_batch_numpy(batch, seats, weva_table=table)

        t0 = time.perf_counter()
        for _ in range(5):
            run_batch()
        batch_ms = (time.perf_counter() - t0) / (5 * n) * 1000.0

    return {
        "n_nodes": n,
        "slow_full_ms": _bench_ms(full_slow, nodes),
        "bridge_only_ms": _bench_ms(bridge, nodes),
        "encode_state_ms": _bench_ms(encode_after_bridge, nodes),
        "weva_lookup_ms": _bench_ms(weva_lookup, nodes),
        "fast_single_ms": _bench_ms(fast_path, nodes),
        "fast_batch_ms": batch_ms,
        "bridge_share_of_slow_pct": None,  # filled below
    }


def backward_target() -> dict:
    """
    Work backward from acceptable wall-clock to required encode throughput.

    Assumptions documented in report (tunable via CLI).
    """
    # Leduc reference
    leduc_iters = 200
    leduc_visits_per_iter = 1080  # ~6 sweeps × 180 traversals
    leduc_nodes_per_visit = 40  # rough external-sampling depth

    # HU: more nodes per traversal, more iters/visits TBD
    hu_iters = 80
    hu_traversals_per_iter = 32
    hu_deal_sweeps = 2
    hu_live_seats = 2
    hu_nodes_per_traversal = 80  # conservative vs max_depth 64 + branches
    acceptable_hours = 4.0

    visits_per_iter = hu_traversals_per_iter * hu_deal_sweeps * hu_live_seats
    encodes_per_run = hu_iters * visits_per_iter * hu_nodes_per_traversal
    target_sec = acceptable_hours * 3600.0
    required_encodes_per_sec = encodes_per_run / target_sec
    required_ms_per_encode = 1000.0 / required_encodes_per_sec

    return {
        "acceptable_wall_clock_hours": acceptable_hours,
        "assumed_hu_iters": hu_iters,
        "assumed_traversals_per_iter": hu_traversals_per_iter,
        "assumed_deal_sweeps": hu_deal_sweeps,
        "assumed_nodes_per_traversal": hu_nodes_per_traversal,
        "estimated_encodes_per_run": encodes_per_run,
        "required_encodes_per_sec": round(required_encodes_per_sec, 1),
        "required_ms_per_encode": round(required_ms_per_encode, 4),
        "leduc_reference": {
            "iters": leduc_iters,
            "visits_per_iter": leduc_visits_per_iter,
            "nodes_per_visit": leduc_nodes_per_visit,
        },
        "speedup_vs_slow_1_75ms": round(1.75 / required_ms_per_encode, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="NLHE encode profile + target")
    p.add_argument("--n-hands", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="logs/nlhe_encode_profile.json")
    args = p.parse_args()

    nodes = _collect_nodes(args.n_hands, args.seed)
    prof = breakdown(nodes)
    prof["bridge_share_of_slow_pct"] = round(
        100.0 * prof["bridge_only_ms"] / max(prof["slow_full_ms"], 1e-9), 1
    )
    prof["fast_speedup_vs_slow"] = round(prof["slow_full_ms"] / max(prof["fast_single_ms"], 1e-9), 2)
    if prof["fast_batch_ms"]:
        prof["batch_speedup_vs_slow"] = round(prof["slow_full_ms"] / max(prof["fast_batch_ms"], 1e-9), 2)

    target = backward_target()
    prof["backward_target"] = target
    prof["meets_target_single"] = prof["fast_single_ms"] <= target["required_ms_per_encode"]
    prof["meets_target_batch"] = (
        prof["fast_batch_ms"] is not None and prof["fast_batch_ms"] <= target["required_ms_per_encode"]
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prof, indent=2), encoding="utf-8")
    print(json.dumps(prof, indent=2))


if __name__ == "__main__":
    main()
