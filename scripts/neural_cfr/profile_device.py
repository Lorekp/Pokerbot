"""Device profiler for the VR-DeepPDCFR+ trainer (GPU vs CPU).

Reports, for Leduc and (optionally) NLHE env shapes:
  * JAX backend + devices (so you know CPU vs CUDA),
  * traversal throughput (traversals/sec) at a fixed batch,
  * grad-step time for the cumulative-advantage net at a target batch size,
  * full per-iteration wall time (traverse + train) at a representative config,
  * peak device memory (GPU VRAM) where the backend exposes it.

All timings are warmed up (one untimed call to trigger XLA compile) then averaged
over a few timed reps with block_until_ready. Run inside the repo root, e.g.:
  ~/jaxgpu-venv/bin/python -m scripts.neural_cfr.profile_device --env both --batch 2048
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

from pokerbot.training.cfr_env import leduc_env_spec, nlhe_env_spec
from pokerbot.training.leduc_gate import LEDUC_REWARD_SCALE
from pokerbot.engine.jax_engine import STARTING_STACK_CHIPS
from pokerbot.training.neural_cfr import (
    VRDeepPDCFRConfig,
    VRDeepPDCFRPlusTrainer,
)


def _bench(fn, reps: int = 5):
    """Warmup once, then time `reps` calls; return mean seconds (block_until_ready)."""
    jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


def _mem_stats():
    try:
        dev = jax.devices()[0]
        st = dev.memory_stats()
        if st:
            peak = st.get("peak_bytes_in_use") or st.get("bytes_in_use")
            cur = st.get("bytes_in_use")
            return cur, peak
    except Exception:
        pass
    return None, None


def profile_env(name: str, env, reward_scale: float, batch: int, trav: int,
                hidden: int, reps: int):
    print(f"\n=== {name} (info_dim={env.info_dim} hist_dim={env.history_dim} "
          f"actions={env.num_actions} players={env.num_players} max_steps={env.max_steps}) ===",
          flush=True)
    cfg = VRDeepPDCFRConfig(
        total_iterations=10, traversals_per_iter=trav, batch_size=batch,
        hidden_dim=hidden, reward_scale=reward_scale, seed=0, train_steps_per_iter=1,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg, env=env)

    # 1) Traversal throughput: one player's batched scan at the configured trav count.
    def run_trav():
        trainer.run_traversals(1)
        return trainer.adv_buffers.buffers[0].size

    mean_t, std_t = _bench(run_trav, reps)
    # run_traversals loops over all players; throughput = total traversals (players*trav) / time.
    total_trav = trav * env.num_players
    print(f"traversal: {mean_t*1000:8.2f} ms/iter (+-{std_t*1000:.2f})  "
          f"=> {total_trav/mean_t:9.1f} trav/s  ({total_trav} trav/iter)", flush=True)

    # 2) Grad-step time at target batch (cumulative advantage net for player 0). Fill the buffer
    #    first (already done by run_trav) so sampling has data.
    def grad_step():
        trainer._train_player_cumulative(0, 5)
        return trainer.cum_states[0].params

    mean_g, std_g = _bench(grad_step, reps)
    print(f"grad-step (cum-adv, batch={batch}): {mean_g*1000:8.3f} ms (+-{std_g*1000:.3f})",
          flush=True)

    # 3) Full iteration wall time (traverse + train all nets, 1 step each).
    def full_iter():
        trainer.train_iteration(7)
        return trainer.avg_state.params

    mean_i, std_i = _bench(full_iter, reps)
    print(f"full iteration (traverse+train): {mean_i*1000:8.2f} ms (+-{std_i*1000:.2f})",
          flush=True)

    cur, peak = _mem_stats()
    if peak is not None:
        print(f"device mem: cur={cur/1e6:8.1f} MB  peak={peak/1e6:8.1f} MB", flush=True)
    else:
        print("device mem: (not reported by backend; check nvidia-smi externally for VRAM)",
              flush=True)
    trainer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["leduc", "nlhe", "both"], default="both")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--trav", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()

    print(f"JAX {jax.__version__}  backend={jax.default_backend()}")
    print(f"devices: {jax.devices()}", flush=True)

    if args.env in ("leduc", "both"):
        profile_env("Leduc", leduc_env_spec(), LEDUC_REWARD_SCALE, args.batch,
                    args.trav, args.hidden, args.reps)
    if args.env in ("nlhe", "both"):
        # NLHE uses the production hidden_dim=256 by default; keep the same knob for comparability
        # but note nlhe info/history dims are much larger than Leduc.
        profile_env("NLHE", nlhe_env_spec(), float(STARTING_STACK_CHIPS), args.batch,
                    args.trav, args.hidden, args.reps)


if __name__ == "__main__":
    main()
