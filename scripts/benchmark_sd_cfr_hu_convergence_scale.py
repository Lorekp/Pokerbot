#!/usr/bin/env python3
"""
Benchmark SD-CFR NLHE train_iteration at Leduc-gate-scale sampling (1080 visits/iter).

Reports full-iteration wall time after JIT warmup and back-of-envelope convergence hours.
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
from pokerbot.training.nlhe_spot_replay import check_gate_a_training_encode_parity_exhaustive
from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer

ROOT = Path(__file__).resolve().parents[1]

LEDUC_GATE_ITERS = 200
LEDUC_VISITS_PER_ITER = 1080


class _ProfiledNLHETrainer(SDCFRNLHETrainer):
    """Instrument encode / traverse / flush / net-update within one train_iteration."""

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

        t_train = time.perf_counter()
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

        loss, grads = jax.value_and_grad(loss_fn)(self.state.params)
        self.state = self.state.apply_gradients(grads=grads)
        self.last_prof["train_sec"] = time.perf_counter() - t_train
        self.last_prof["iter_total_sec"] = time.perf_counter() - t_iter
        enc = self.last_prof["encode_sec"]
        trav = self.last_prof["traverse_sec"]
        self.last_prof["traverse_non_encode_sec"] = max(0.0, trav - enc)
        self.last_prof["n_iter_samples"] = float(len(self._iter_samples))
        return float(loss)


def _visits_per_iter(cfg: SDCFRNLHEConfig, n_live: int) -> int:
    return cfg.traversals_per_iter * max(1, cfg.deal_sweeps) * n_live


def _leduc_analog_cfg(**overrides) -> SDCFRNLHEConfig:
    base = dict(
        iterations=1,
        traversals_per_iter=90,
        deal_sweeps=6,
        sft_checkpoint=None,
        seed=0,
        max_traverse_depth=64,
    )
    base.update(overrides)
    return SDCFRNLHEConfig(**base)


def main() -> None:
    p = argparse.ArgumentParser(description="HU SD-CFR convergence-scale iteration benchmark")
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--timed-iters", type=int, default=3)
    p.add_argument("--smoke-traversals", type=int, default=4)
    p.add_argument("--convergence-iters-estimate", type=int, default=LEDUC_GATE_ITERS)
    p.add_argument("--output", default="logs/sd_cfr_hu_convergence_scale_benchmark.json")
    p.add_argument("--skip-gate-parity", action="store_true")
    args = p.parse_args()

    report: dict = {
        "leduc_gate_reference": {
            "iterations": LEDUC_GATE_ITERS,
            "visits_per_iter": LEDUC_VISITS_PER_ITER,
        },
        "hu_convergence_scale": {
            "traversals_per_iter": 90,
            "deal_sweeps": 6,
            "live_seats": 2,
            "visits_per_iter": 1080,
            "note": "Leduc-analog visit count; NLHE tree depth >> Leduc so per-visit cost dominates.",
        },
    }

    if not args.skip_gate_parity:
        parity = check_gate_a_training_encode_parity_exhaustive()
        report["gate_a_training_encode_parity_exhaustive"] = parity
        print(
            f"Gate-A vs training encode (exhaustive): "
            f"{parity['rows_checked']} rows, mismatches={parity['n_mismatches']}, "
            f"errors={parity['n_replay_errors']}, passed={parity['passed']}",
            flush=True,
        )
        if not parity["passed"]:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise SystemExit(1)

    warmup_cfg = _leduc_analog_cfg(traversals_per_iter=args.smoke_traversals, deal_sweeps=1)
    trainer = _ProfiledNLHETrainer(warmup_cfg)
    print("JIT warmup (smoke scale)...", flush=True)
    for i in range(args.warmup_iters):
        trainer.train_iteration(iteration_weight=float(i + 1))
    print("Warmup done.", flush=True)

    conv_cfg = _leduc_analog_cfg()
    trainer = _ProfiledNLHETrainer(conv_cfg)
    trainer.train_iteration(iteration_weight=1.0)

    visits = _visits_per_iter(conv_cfg, trainer._n_live)
    iter_times: list[float] = []
    profiles: list[dict] = []

    print(
        f"Timing {args.timed_iters} iteration(s) at {visits} visits/iter "
        f"({conv_cfg.traversals_per_iter} trav × {conv_cfg.deal_sweeps} sweeps × {trainer._n_live} seats)...",
        flush=True,
    )
    for i in range(args.timed_iters):
        loss = trainer.train_iteration(iteration_weight=float(i + 10))
        elapsed = trainer.last_prof["iter_total_sec"]
        iter_times.append(elapsed)
        prof = dict(trainer.last_prof)
        prof["encode_calls"] = trainer.encode_calls
        prof["loss"] = loss
        profiles.append(prof)
        enc_ms = 1000.0 * prof["encode_sec"] / max(prof["encode_calls"], 1)
        print(
            f"  iter {i + 1}: {elapsed:.1f}s total, loss={loss:.4f}, "
            f"encode={prof['encode_sec']:.1f}s ({enc_ms:.3f} ms/call, n={prof['encode_calls']}), "
            f"traverse_non_enc={prof.get('traverse_non_encode_sec', 0):.1f}s, "
            f"flush={prof['flush_sec']:.3f}s, train={prof['train_sec']:.3f}s, "
            f"samples={int(prof.get('n_iter_samples', 0))}",
            flush=True,
        )

    mean_iter = sum(iter_times) / len(iter_times)
    est_hours = mean_iter * args.convergence_iters_estimate / 3600.0
    debug_loop_hours = 4.0
    last = profiles[-1] if profiles else {}
    enc_share = last.get("encode_sec", 0) / max(last.get("iter_total_sec", 1), 1e-9)

    report["iteration_timing"] = {
        "timed_iters": args.timed_iters,
        "iter_wall_sec": [round(t, 2) for t in iter_times],
        "mean_iter_sec": round(mean_iter, 2),
        "profiles": profiles,
        "encode_share_of_iter_last": round(enc_share, 3),
    }
    report["convergence_run_estimate"] = {
        "assumed_iterations": args.convergence_iters_estimate,
        "visits_per_iter": visits,
        "total_traverser_visits": visits * args.convergence_iters_estimate,
        "estimated_wall_hours": round(est_hours, 2),
        "fits_debug_loop_4h": est_hours <= debug_loop_hours,
        "debug_loop_hours": debug_loop_hours,
    }
    report["smoke_scale_comparison"] = {
        "smoke_visits_per_iter": args.smoke_traversals * 1 * 2,
        "convergence_visits_per_iter": visits,
        "visit_ratio": visits / max(args.smoke_traversals * 2, 1),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
