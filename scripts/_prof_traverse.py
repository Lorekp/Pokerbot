"""Profile a single SD-CFR NLHE traversal to locate the per-node hot path.

Hypothesis to confirm/refute: the in-iteration strategy is tabular regret-matching (NOT the net),
so the per-node cost is the ENGINE ops (step_state / legal_action_mask / is_chance_node /
sample_chance / jax.random.choice) each dispatching to XLA in a Python recursion, plus _bundle
encoding. cProfile by cumulative + tottime tells us which.
"""
import cProfile
import io
import os
import pstats
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np  # noqa: E402

from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer  # noqa: E402
from pokerbot.training.sd_cfr_nlhe_distributed import (  # noqa: E402
    iteration_traversal_plan,
    run_traversal_unit,
)


def mk(depth):
    cfg = SDCFRNLHEConfig(iterations=1, traversals_per_iter=1, deal_sweeps=1, seed=11,
                          max_traverse_depth=depth, sft_checkpoint=None)
    return SDCFRNLHETrainer(cfg)


if __name__ == "__main__":
    DEPTH = int(os.environ.get("PROF_DEPTH", "14"))
    t = mk(DEPTH)
    t._iter_samples = []
    plan = iteration_traversal_plan(t.config, 1.0)
    t._begin_iteration_regret()
    pair = plan.pairs[0]
    print(f"depth={DEPTH} pairs={len(plan.pairs)} profiling unit {pair}", flush=True)

    # Warm up XLA compile (don't profile the one-time compile cost).
    t0 = time.perf_counter()
    run_traversal_unit(t, plan, pair[0], pair[1], 1)
    print(f"warmup unit wall = {time.perf_counter() - t0:.2f}s", flush=True)

    # Re-init regret state and profile a fresh unit (compile now cached).
    t._begin_iteration_regret()
    pr = cProfile.Profile()
    t1 = time.perf_counter()
    pr.enable()
    run_traversal_unit(t, plan, pair[0], pair[1], 1)
    pr.disable()
    wall = time.perf_counter() - t1
    print(f"profiled unit wall = {wall:.2f}s", flush=True)

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(30)
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("cumulative")
    ps2.print_stats(30)
    print(s2.getvalue())

    # caller attribution for the mystery JAX scalar-read/compare hotspots
    for fn in ("_operator_eq", "__bool__", "_value", "apply_primitive"):
        s3 = io.StringIO()
        pstats.Stats(pr, stream=s3).print_callers(fn)
        out = s3.getvalue()
        # trim to the matching block
        print(f"\n===== CALLERS of {fn} =====")
        for line in out.splitlines():
            if "Function" in line or fn in line:
                print(line)
