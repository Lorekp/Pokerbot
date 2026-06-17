#!/usr/bin/env python3
"""
Quick HU SD-CFR cost estimate (~2–5 min): small visit probe + batched net timing + linear extrapolation.

Does NOT run a full 1080-visit iteration. Use for before/after pathology fixes and feasibility sanity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import init_hand
from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer

ROOT = Path(__file__).resolve().parents[1]
LEDUC_GATE_VISITS = 1080
LEDUC_GATE_ITERS = 200


class _ProfiledNLHETrainer(SDCFRNLHETrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_prof: dict[str, float] = {}
        self.encode_calls = 0

    def _bundle(self, state, seat: int, **kwargs):
        t0 = time.perf_counter()
        b = super()._bundle(state, seat, **kwargs)
        self.last_prof["encode_sec"] = self.last_prof.get("encode_sec", 0.0) + (
            time.perf_counter() - t0
        )
        self.encode_calls += 1
        return b

    def train_iteration(self, iteration_weight: float = 1.0) -> float:
        self.last_prof = {
            "encode_sec": 0.0,
            "traverse_sec": 0.0,
            "flush_sec": 0.0,
            "train_sec": 0.0,
            "iter_total_sec": 0.0,
        }
        self.encode_calls = 0
        t_iter = time.perf_counter()

        self._iter_weight = iteration_weight
        self._iter_samples = []
        self._begin_iteration_regret()
        occ = jnp.asarray(self.occupancy, dtype=bool)

        t_trav = time.perf_counter()
        for _ in range(max(1, self.config.deal_sweeps)):
            for _ in range(self.config.traversals_per_iter):
                self.key, k1, k2 = jax.random.split(self.key, 3)
                state = init_hand(
                    k1,
                    button=jnp.int32(self.config.button_seat),
                    occupancy=occ,
                )
                for seat in self._live_seats:
                    self.traverse(state, seat, k2)
        self.last_prof["traverse_sec"] = time.perf_counter() - t_trav

        t_flush = time.perf_counter()
        self._flush_pending_regret()
        self.last_prof["flush_sec"] = time.perf_counter() - t_flush

        if not self._iter_samples:
            self.last_prof["iter_total_sec"] = time.perf_counter() - t_iter
            return 0.0

        self._log_target_scale(self._iter_samples)
        for bundle, inst, mask in self._iter_samples:
            legal = np.flatnonzero(mask)
            a = int(legal[0]) if legal.size else 0
            self._reservoir_add(bundle, inst, mask, a)

        batch_n = min(self.config.train_batch_size, len(self._iter_samples))
        idx = np.random.randint(0, len(self._iter_samples), size=batch_n)
        batch = [self._iter_samples[i] for i in idx]
        weva = jnp.stack([jnp.int32(b[0].weva_id) for b in batch])
        cards = jnp.stack(
            [jnp.asarray(b[0].cards, dtype=jnp.float32).reshape(9, 4, 13) for b in batch]
        )
        actions = jnp.stack(
            [jnp.asarray(b[0].actions, dtype=jnp.float32).reshape(6, 4, 12) for b in batch]
        )
        positions = jnp.stack([jnp.asarray(b[0].policy_position, dtype=jnp.float32) for b in batch])
        masks = jnp.stack([jnp.asarray(b[2]) for b in batch])
        targets = jnp.stack([jnp.asarray(b[1], dtype=jnp.float32) for b in batch])

        def loss_fn(params):
            variables = {"params": params, "batch_stats": self._batch_stats}
            logits = self.state.apply_fn(
                variables,
                weva_ids=weva,
                cards=cards,
                actions=actions,
                position=positions,
                legal_mask=masks,
            )
            diff = (logits - targets) * masks
            per_sample = jnp.sum(jnp.square(diff), axis=-1) / jnp.maximum(
                jnp.sum(masks, axis=-1), 1.0
            )
            return jnp.mean(per_sample)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(self.state.params)
        jax.block_until_ready(loss)
        t_train = time.perf_counter()
        loss, grads = grad_fn(self.state.params)
        jax.block_until_ready(loss)
        self.state = self.state.apply_gradients(grads=grads)
        self.last_prof["train_sec"] = time.perf_counter() - t_train
        self.last_prof["iter_total_sec"] = time.perf_counter() - t_iter
        trav = self.last_prof["traverse_sec"]
        self.last_prof["traverse_non_encode_sec"] = max(0.0, trav - self.last_prof["encode_sec"])
        self.last_prof["n_iter_samples"] = float(len(self._iter_samples))
        self.last_prof["train_batch_n"] = float(batch_n)
        self.last_prof["encode_cache_hits"] = float(self._encode_stats.get("cache_hits", 0))
        self.last_prof["encode_cache_misses"] = float(self._encode_stats.get("cache_misses", 0))
        return float(loss)


def _visits(cfg: SDCFRNLHEConfig, n_live: int) -> int:
    return cfg.traversals_per_iter * max(1, cfg.deal_sweeps) * n_live


def _time_batched_net(trainer: SDCFRNLHETrainer, batch_n: int, repeats: int) -> dict:
    """Time loss+grad on a 2048 batch after JIT warmup (independent of visit count)."""
    n = len(trainer._iter_samples)
    if n < 32:
        return {"error": "need samples from probe iter", "n_samples": n}
    idx = np.random.randint(0, n, size=batch_n)
    batch = [trainer._iter_samples[i] for i in idx]
    weva = jnp.stack([jnp.int32(b[0].weva_id) for b in batch])
    cards = jnp.stack(
        [jnp.asarray(b[0].cards, dtype=jnp.float32).reshape(9, 4, 13) for b in batch]
    )
    actions = jnp.stack(
        [jnp.asarray(b[0].actions, dtype=jnp.float32).reshape(6, 4, 12) for b in batch]
    )
    positions = jnp.stack([jnp.asarray(b[0].policy_position, dtype=jnp.float32) for b in batch])
    masks = jnp.stack([jnp.asarray(b[2]) for b in batch])
    targets = jnp.stack([jnp.asarray(b[1], dtype=jnp.float32) for b in batch])

    def loss_fn(params):
        variables = {"params": params, "batch_stats": trainer._batch_stats}
        logits = trainer.state.apply_fn(
            variables,
            weva_ids=weva,
            cards=cards,
            actions=actions,
            position=positions,
            legal_mask=masks,
        )
        diff = (logits - targets) * masks
        per_sample = jnp.sum(jnp.square(diff), axis=-1) / jnp.maximum(
            jnp.sum(masks, axis=-1), 1.0
        )
        return jnp.mean(per_sample)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(trainer.state.params)
    jax.block_until_ready(loss)
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        loss, grads = grad_fn(trainer.state.params)
        jax.block_until_ready(loss)
        times.append(time.perf_counter() - t0)
    return {
        "batch_n": batch_n,
        "repeats": repeats,
        "sec": [round(t, 4) for t in times],
        "mean_sec": round(sum(times) / len(times), 4),
        "reference_python_loop_sec": 226.5,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Quick HU SD-CFR cost estimate (minutes, not hours)")
    p.add_argument("--probe-traversals", type=int, default=4)
    p.add_argument("--probe-sweeps", type=int, default=2)
    p.add_argument("--net-batch", type=int, default=2048)
    p.add_argument("--net-repeats", type=int, default=5)
    p.add_argument("--output", default="logs/sd_cfr_hu_quick_benchmark.json")
    args = p.parse_args()

    smoke = SDCFRNLHEConfig(
        iterations=1,
        traversals_per_iter=2,
        deal_sweeps=1,
        sft_checkpoint=None,
        seed=0,
    )
    print("JIT warmup (2 visits)...", flush=True)
    t0 = time.perf_counter()
    SDCFRNLHETrainer(smoke).train_iteration(1.0)
    print(f"  warmup {time.perf_counter() - t0:.1f}s", flush=True)

    probe_cfg = SDCFRNLHEConfig(
        iterations=1,
        traversals_per_iter=args.probe_traversals,
        deal_sweeps=args.probe_sweeps,
        sft_checkpoint=None,
        seed=1,
        train_batch_size=args.net_batch,
    )
    visits_probe = _visits(probe_cfg, 2)
    print(
        f"Probe: {visits_probe} visits "
        f"({args.probe_traversals} trav × {args.probe_sweeps} sweeps × 2 seats)...",
        flush=True,
    )
    trainer = _ProfiledNLHETrainer(probe_cfg)
    t0 = time.perf_counter()
    loss = trainer.train_iteration(1.0)
    probe_wall = time.perf_counter() - t0
    prof = dict(trainer.last_prof)

    net_micro = _time_batched_net(trainer, args.net_batch, args.net_repeats)

    # Linear extrapolation: traverse scales with visits; net ~ constant per iter
    scale = LEDUC_GATE_VISITS / max(visits_probe, 1)
    trav_extrap = prof["traverse_sec"] * scale
    flush_extrap = prof["flush_sec"] * scale  # usually tiny
    net_est = net_micro.get("mean_sec", prof["train_sec"])
    iter_extrap = trav_extrap + flush_extrap + net_est
    gate_hours = iter_extrap * LEDUC_GATE_ITERS / 3600.0

    report = {
        "probe": {
            "visits": visits_probe,
            "wall_sec": round(probe_wall, 2),
            "loss": loss,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in prof.items()},
            "encode_ms_per_call": round(
                1000.0 * prof["encode_sec"] / max(trainer.encode_calls, 1), 3
            ),
            "encode_calls": trainer.encode_calls,
        },
        "net_batched_micro": net_micro,
        "extrapolation_linear": {
            "target_visits_per_iter": LEDUC_GATE_VISITS,
            "scale_from_probe": scale,
            "traverse_sec": round(trav_extrap, 1),
            "flush_sec": round(flush_extrap, 3),
            "train_sec": round(net_est, 3),
            "iter_sec": round(iter_extrap, 1),
            "iter_minutes": round(iter_extrap / 60.0, 2),
            "gate_200_iters_hours": round(gate_hours, 2),
            "fits_local_debug_4h": gate_hours <= 4.0,
            "caveat": "Linear visit scaling; tree depth/sample variance not captured.",
        },
        "reference_pre_net_fix": {
            "net_train_sec_per_iter": 226.5,
            "source": "logs/sd_cfr_hu_traversal_profile_summary.json (8-trav probe, Python loss loop)",
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Quick comparison ---", flush=True)
    print(f"Probe ({visits_probe} visits): {probe_wall:.1f}s total", flush=True)
    print(
        f"  traverse {prof['traverse_sec']:.1f}s "
        f"(encode {prof['encode_sec']:.1f}s, dispatch {prof['traverse_non_encode_sec']:.1f}s) "
        f"flush {prof['flush_sec']:.3f}s train {prof['train_sec']:.3f}s",
        flush=True,
    )
    if "mean_sec" in net_micro:
        print(
            f"  net 2048-batch (JIT): {net_micro['mean_sec']:.3f}s "
            f"(was ~{net_micro['reference_python_loop_sec']:.0f}s Python loop)",
            flush=True,
        )
    print(
        f"Extrapolated @ {LEDUC_GATE_VISITS} visits/iter: "
        f"{report['extrapolation_linear']['iter_minutes']:.1f} min/iter, "
        f"{report['extrapolation_linear']['gate_200_iters_hours']:.1f} h for {LEDUC_GATE_ITERS} iters",
        flush=True,
    )
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
