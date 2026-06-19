"""Saturation benchmark: measure core-utilization + throughput-scaling of the VR-DeepPDCFR+ trainer.

Sets the device/thread topology BEFORE importing jax (via saturate_env.configure), builds a trainer,
and times the traversal + grad phases while a background psutil sampler records per-core CPU %. Reports
traversals/sec, net-updates/sec, the traversal:train split, and mean "cores busy" (sum of per-core %
/ 100) during each phase -- the direct measure of how many cores the work actually lit up.

Run ONE topology per process (device count is import-frozen). Example:
    python -m scripts.neural_cfr.bench_saturation --game nlhe --topology fine --devices 6 --traversals 96
    python -m scripts.neural_cfr.bench_saturation --game leduc --topology single --traversals 256

The driver scripts/neural_cfr/scaling_curve.py runs this across a device sweep to produce the curve.
"""

from __future__ import annotations

import argparse
import json
import threading
import time

# --- configure threads/devices BEFORE importing jax -------------------------------------------------
from scripts.neural_cfr.saturate_env import configure  # noqa: E402


def _build(game: str, traversals: int, hidden: int, batch: int, train_steps: int):
    import jax  # noqa: F401  (import after configure)

    from pokerbot.training.cfr_env import leduc_env_spec, nlhe_env_spec
    from pokerbot.training.leduc_gate import LEDUC_REWARD_SCALE
    from pokerbot.training.neural_cfr import VRDeepPDCFRConfig, VRDeepPDCFRPlusTrainer

    if game == "nlhe":
        env = nlhe_env_spec()
        from pokerbot.engine.jax_engine import STARTING_STACK_CHIPS
        rscale = float(STARTING_STACK_CHIPS)
    else:
        env = leduc_env_spec()
        rscale = LEDUC_REWARD_SCALE
    cfg = VRDeepPDCFRConfig(
        total_iterations=1000, traversals_per_iter=traversals, batch_size=batch,
        hidden_dim=hidden, buffer_capacity=100_000, seed=0, reward_scale=rscale,
        train_steps_per_iter=train_steps,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg, env=env)
    return trainer


class _CoreSampler:
    """Background per-core CPU% sampler. mean 'cores busy' = sum(per-core %)/100 over the window."""

    def __init__(self, interval: float = 0.05):
        import psutil
        self._psutil = psutil
        self.interval = interval
        self._stop = threading.Event()
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._psutil.cpu_percent(percpu=True)  # prime
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            per = self._psutil.cpu_percent(percpu=True, interval=self.interval)
            self._samples.append(sum(per) / 100.0)

    def __exit__(self, *a):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def mean_cores(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    def max_cores(self) -> float:
        return max(self._samples) if self._samples else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["nlhe", "leduc"], default="nlhe")
    ap.add_argument("--topology", default="fine", choices=["fine", "coarse", "single"])
    ap.add_argument("--vcpus", type=int, default=None)
    ap.add_argument("--devices", type=int, default=None, help="override n_devices")
    ap.add_argument("--intra-op", type=int, default=None)
    ap.add_argument("--traversals", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--train-steps", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measure", type=int, default=4)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    sat = configure(vcpus=args.vcpus, topology=args.topology,
                    n_devices=args.devices, intra_op=args.intra_op)

    import jax

    hidden = args.hidden if args.hidden is not None else (256 if args.game == "nlhe" else 64)
    batch = args.batch if args.batch is not None else (512 if args.game == "nlhe" else 256)
    trainer = _build(args.game, args.traversals, hidden, batch, args.train_steps)
    n_dev = jax.local_device_count()

    def sync(x):
        jax.block_until_ready(x)

    # warmup (jit compile both phases)
    for it in range(1, args.warmup + 1):
        trainer.run_traversals(it)
        trainer._run_grad_phase(it) if hasattr(trainer, "_run_grad_phase") else _grad_phase(trainer, it)
    sync(trainer.avg_state.params)

    # ---- measure traversal phase ----
    trav_t = 0.0
    with _CoreSampler() as trav_sampler:
        for it in range(args.warmup + 1, args.warmup + args.measure + 1):
            t0 = time.perf_counter()
            trainer.run_traversals(it)
            sync((trainer.strat_buffer.buffer.data["info_state"],
                  trainer.hist_buffer.buffer.data["history"]))
            trav_t += time.perf_counter() - t0

    # ---- measure grad phase ----
    grad_t = 0.0
    n_updates = 0
    with _CoreSampler() as grad_sampler:
        for it in range(args.warmup + 1, args.warmup + args.measure + 1):
            t0 = time.perf_counter()
            n_updates += _grad_phase(trainer, it)
            sync(trainer.avg_state.params)
            grad_t += time.perf_counter() - t0

    n = args.measure
    trav_per_iter = args.traversals * trainer.env.num_players
    trav_per_sec = trav_per_iter * n / trav_t if trav_t else 0.0
    updates_per_sec = n_updates / grad_t if grad_t else 0.0
    total = trav_t + grad_t

    result = {
        "game": args.game,
        "topology": sat.topology,
        "n_devices": n_dev,
        "intra_op": sat.intra_op,
        "compute_threads": n_dev * sat.intra_op,
        "vcpus": sat.vcpus,
        "traversals_per_iter": args.traversals,
        "traversal_ms_iter": trav_t / n * 1e3,
        "grad_ms_iter": grad_t / n * 1e3,
        "traversal_pct": trav_t / total * 100 if total else 0.0,
        "traversals_per_sec": trav_per_sec,
        "net_updates_per_sec": updates_per_sec,
        "trav_mean_cores": trav_sampler.mean_cores(),
        "trav_max_cores": trav_sampler.max_cores(),
        "grad_mean_cores": grad_sampler.mean_cores(),
        "grad_max_cores": grad_sampler.max_cores(),
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(f"=== {args.game} topology={sat.topology} devices={n_dev} intra_op={sat.intra_op} "
              f"threads={n_dev*sat.intra_op} traversals={args.traversals} ===")
        print(f"  traversal : {result['traversal_ms_iter']:9.1f} ms/iter  "
              f"({result['traversals_per_sec']:8.1f} trav/s)  cores~{result['trav_mean_cores']:.1f} "
              f"(max {result['trav_max_cores']:.1f})")
        print(f"  grad      : {result['grad_ms_iter']:9.1f} ms/iter  "
              f"({result['net_updates_per_sec']:8.1f} upd/s)  cores~{result['grad_mean_cores']:.1f} "
              f"(max {result['grad_max_cores']:.1f})")
        print(f"  split     : traversal {result['traversal_pct']:.1f}%")


def _grad_phase(trainer, iteration: int) -> int:
    """Run the grad phase exactly as train_iteration does; return number of net updates performed."""
    steps = max(1, trainer.config.train_steps_per_iter)
    np_ = trainer.env.num_players
    for player in range(np_):
        for _ in range(steps):
            trainer._train_player_cumulative(player, iteration)
            trainer._train_player_instantaneous(player)
    for _ in range(steps):
        trainer._train_history_value(iteration)
        trainer._train_average_strategy(iteration)
    trainer._prev_cum_params = tuple(s.params for s in trainer.cum_states)
    # updates = (cum+inst per player) + (hv+avg) per step
    return steps * (2 * np_ + 2)


if __name__ == "__main__":
    main()
