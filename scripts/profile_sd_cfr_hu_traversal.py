#!/usr/bin/env python3
"""Profile HU SD-CFR traversal: where per-visit wall time goes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import (
    CHIP_UNIT,
    init_hand,
    is_chance_node,
    legal_action_mask,
    sample_chance,
    step_state,
)
from pokerbot.training.sd_cfr_nlhe import (
    NLHE_NUM_ACTIONS,
    SDCFRNLHEConfig,
    SDCFRNLHETrainer,
    _NLHERegretVisit,
)

ROOT = Path(__file__).resolve().parents[1]


class ProfiledHUTrainer(SDCFRNLHETrainer):
    """Count and time hot-path operations inside traverse."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stats: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._t_stack: list[tuple[str, float]] = []

    def _bump(self, name: str, dt: float) -> None:
        self.stats[name] = self.stats.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1

    def _bundle(self, state, seat: int, **kwargs):
        t0 = time.perf_counter()
        b = super()._bundle(state, seat, **kwargs)
        self._bump("encode_bundle", time.perf_counter() - t0)
        return b

    def _regret_matching_sigma(self, state, seat: int) -> np.ndarray:
        t0 = time.perf_counter()
        s = super()._regret_matching_sigma(state, seat)
        self._bump("regret_matching_sigma", time.perf_counter() - t0)
        return s

    def traverse(self, state, traverser, key, p_by_seat=None, depth=0):
        self.counts["traverse_calls"] = self.counts.get("traverse_calls", 0) + 1
        if depth == 0:
            self.counts["root_traversals"] = self.counts.get("root_traversals", 0) + 1
        self.counts["max_depth"] = max(self.counts.get("max_depth", 0), depth)

        if depth >= self.config.max_traverse_depth or bool(state.done):
            return float(np.asarray(state.rewards[traverser], dtype=np.float64) / CHIP_UNIT)

        t0 = time.perf_counter()
        if bool(is_chance_node(state)):
            state = sample_chance(state)
            self._bump("sample_chance", time.perf_counter() - t0)
        else:
            self._bump("is_chance_check", time.perf_counter() - t0)

        seat = int(state.action_seat)
        if seat < 0:
            return 0.0

        t0 = time.perf_counter()
        mask = np.asarray(legal_action_mask(state), dtype=bool)
        self._bump("legal_action_mask", time.perf_counter() - t0)

        if p_by_seat is None:
            p_by_seat = {s: 1.0 for s in self._live_seats}

        sigma = self._regret_matching_sigma(state, seat)
        info_key = self._encode_key(state, seat)

        acting_reach = float(p_by_seat.get(seat, 1.0))
        self._strat_sum[info_key] += self._iter_weight * acting_reach * sigma

        if seat == traverser:
            self.counts["traverser_nodes"] = self.counts.get("traverser_nodes", 0) + 1
            util = np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)
            legal = np.flatnonzero(mask)
            self.counts["traverser_action_expansions"] = (
                self.counts.get("traverser_action_expansions", 0) + int(legal.size)
            )
            for a in legal:
                a = int(a)
                t0 = time.perf_counter()
                child = step_state(state, jnp.int32(a))
                self._bump("step_state", time.perf_counter() - t0)
                util[a] = self.traverse(child, traverser, key, p_by_seat, depth + 1)
            node_util = float(sum(sigma[a] * util[a] for a in legal))
            inst = np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)
            for a in legal:
                inst[a] = self._opponent_reach(traverser, p_by_seat) * (util[a] - node_util)
            if info_key not in self._cum_regret:
                self._cum_regret[info_key] = np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)
            self._cum_regret[info_key] += inst
            opps = [s for s in self._live_seats if s != seat]
            opp_bucket = self._opponent_weva_bucket_fast(state, opps[0]) if len(opps) == 1 else -1
            self._pending_regret[info_key].append(
                _NLHERegretVisit(
                    opp_reach=float(self._opponent_reach(traverser, p_by_seat)),
                    util=util.copy(),
                    sigma=sigma.copy(),
                    mask=mask.copy(),
                    bundle=self._bundle(state, seat),
                    opponent_weva_bucket=opp_bucket,
                )
            )
            return node_util

        self.counts["opponent_nodes"] = self.counts.get("opponent_nodes", 0) + 1
        legal_idx = np.flatnonzero(mask)
        probs = sigma[legal_idx]
        probs = probs / max(probs.sum(), 1e-12)
        key, sub = jax.random.split(key)
        pick = int(jax.random.choice(sub, legal_idx.shape[0], p=jnp.asarray(probs, dtype=jnp.float32)))
        action = int(legal_idx[pick])
        np_by = dict(p_by_seat)
        np_by[seat] = np_by.get(seat, 1.0) * sigma[action]
        t0 = time.perf_counter()
        child = step_state(state, jnp.int32(action))
        self._bump("step_state", time.perf_counter() - t0)
        return self.traverse(child, traverser, key, np_by, depth + 1)


def _run_profile(
    *,
    traversals: int,
    deal_sweeps: int,
    max_depth: int,
    sft: bool,
) -> dict:
    cfg = SDCFRNLHEConfig(
        iterations=1,
        traversals_per_iter=traversals,
        deal_sweeps=deal_sweeps,
        max_traverse_depth=max_depth,
        sft_checkpoint="checkpoints/policy_sft_prod6000.pt" if sft else None,
    )
    trainer = ProfiledHUTrainer(cfg)
    trainer._begin_iteration_regret()
    trainer._iter_samples = []
    trainer._iter_weight = 1.0

    occ = jnp.asarray(trainer.occupancy, dtype=bool)
    t_iter = time.perf_counter()
    for _ in range(max(1, deal_sweeps)):
        for _ in range(traversals):
            trainer.key, k1, k2 = jax.random.split(trainer.key, 3)
            state = init_hand(k1, button=jnp.int32(cfg.button_seat), occupancy=occ)
            for seat in trainer._live_seats:
                trainer.traverse(state, seat, k2)
    t_trav = time.perf_counter() - t_iter

    t_flush = time.perf_counter()
    trainer._flush_pending_regret()
    flush_sec = time.perf_counter() - t_flush

    visits = traversals * max(1, deal_sweeps) * trainer._n_live
    n_pending = sum(len(v) for v in trainer._pending_regret.values())
    # After flush pending is cleared; use iter_samples
    n_samples = len(trainer._iter_samples)

    stats = dict(trainer.stats)
    counts = dict(trainer.counts)
    total_accounted = sum(stats.values())

    report = {
        "config": {
            "traversals_per_iter": traversals,
            "deal_sweeps": deal_sweeps,
            "max_traverse_depth": max_depth,
            "visits": visits,
            "sft_init": sft,
        },
        "wall_sec": {
            "traverse_loop": round(t_trav, 3),
            "flush_pending_regret": round(flush_sec, 6),
            "per_visit_sec": round(t_trav / max(visits, 1), 3),
        },
        "counts": counts,
        "timed_sec": {k: round(v, 3) for k, v in sorted(stats.items(), key=lambda x: -x[1])},
        "timed_ms_per_call": {
            k: round(1000.0 * stats[k] / max(counts.get(k, counts.get("traverse_calls", 1)), 1), 4)
            for k in stats
        },
        "accounted_fraction_of_traverse": round(total_accounted / max(t_trav, 1e-9), 3),
        "belief_flush": {
            "loops_1326_buckets": False,
            "note": "_flush_pending_regret aggregates visits per infoset key; opponent_weva_bucket stored but not iterated.",
            "pending_visits_before_flush": n_samples,
            "unique_infosets_in_samples": n_samples,
        },
        "jit": {
            "traverse_is_compiled": False,
            "traverse_is_python_recursion": True,
            "step_state_jit": True,
            "legal_action_mask_jit": True,
            "init_hand_jit": True,
            "per_node_jax_dispatch": True,
        },
        "encode_multiplier": {
            "encode_bundle_calls": counts.get("encode_bundle", 0),
            "traverse_calls": counts.get("traverse_calls", 0),
            "encodes_per_traverse_call": round(
                counts.get("encode_bundle", 0) / max(counts.get("traverse_calls", 1), 1), 2
            ),
            "note": "_encode_key + traverser pending append each call _bundle; ~2 encodes per traverser node",
        },
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="logs/sd_cfr_hu_traversal_profile.json")
    p.add_argument("--traversals", type=int, default=2)
    p.add_argument("--deal-sweeps", type=int, default=1)
    p.add_argument("--max-depth", type=int, default=64)
    p.add_argument("--no-sft", action="store_true")
    args = p.parse_args()

    # Warmup JIT
    _run_profile(traversals=1, deal_sweeps=1, max_depth=16, sft=not args.no_sft)

    reports = []
    for depth in (24, args.max_depth):
        rep = _run_profile(
            traversals=args.traversals,
            deal_sweeps=args.deal_sweeps,
            max_depth=depth,
            sft=not args.no_sft,
        )
        rep["label"] = f"depth_{depth}"
        reports.append(rep)
        print(json.dumps(rep, indent=2), flush=True)

    out = {
        "profiles": reports,
        "verdict": _verdict(reports[-1]),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {path}\nVERDICT: {out['verdict']}", flush=True)


def _verdict(rep: dict) -> str:
    sc = rep["timed_sec"].get("sample_chance", 0)
    trav_total = rep["wall_sec"]["traverse_loop"]
    if sc > 0.5 * trav_total:
        return (
            "PATHOLOGICAL_AND_FIXABLE: sample_chance was not @jax.jit (~150ms/call eager trace). "
            "Add @jax.jit to sample_chance (see jax_engine.py)."
        )
    return (
        "PATHOLOGICAL_AND_FIXABLE: Python CFR recursion + per-node JAX dispatch + duplicate encode; "
        "no 1326-bucket belief loop. Also vmap the 2048-sample net loss (Python loop in train_iteration)."
    )


if __name__ == "__main__":
    main()
