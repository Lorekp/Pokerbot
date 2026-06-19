"""Thread/device environment setup for saturating a many-vCPU box with JAX-on-CPU.

MUST be imported (or have :func:`configure` called) BEFORE ``import jax`` -- the XLA host-platform
device count and the Eigen intra-op thread-pool size are read once, at jax/jaxlib import time, and are
immutable afterwards. The trainer's saturation topology (pmap over N CPU devices) needs N devices to
exist, which only happens if ``--xla_force_host_platform_device_count=N`` is set in ``XLA_FLAGS`` at
import.

The whole point of this module is to make the per-process OS-thread count add up to EXACTLY the vCPU
budget with ZERO oversubscription:

    total compute threads  =  N_devices  x  intra_op_threads_per_device  (+ ~0 BLAS, all capped to 1)

We pin every numeric library's own threadpool (OMP/OpenBLAS/MKL/NumExpr) to 1 so they cannot spawn a
second ~Ncore pool underneath XLA (the classic N x 64 thrash). XLA's Eigen pool is the ONLY multi-thread
compute pool, and we size it explicitly.

Two coherent topologies (pick via env / CLI, both satisfy threads == vCPUs):
  * "fine"  : N_devices = vcpus, intra_op = 1   -> many thin single-thread devices (best for the
              latency-bound traversal scan, which does not vectorize across cores within one device).
  * "coarse": N_devices = D,     intra_op = vcpus // D  -> few fat multi-thread devices (better when a
              single device op is large enough to fill its Eigen pool, e.g. big grad matmuls).

Usage (wrapper sets env, THEN imports jax):
    from scripts.neural_cfr.saturate_env import configure
    cfg = configure(vcpus=64, topology="fine")   # call BEFORE importing jax anywhere
    import jax  # now jax.local_device_count() == cfg.n_devices
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SaturationConfig:
    vcpus: int
    n_devices: int
    intra_op: int
    topology: str

    @property
    def total_threads(self) -> int:
        # The only multi-threaded compute pool is XLA Eigen (intra_op) per device; BLAS pools pinned to 1.
        return self.n_devices * self.intra_op

    def describe(self) -> str:
        return (
            f"topology={self.topology} vcpus={self.vcpus} n_devices={self.n_devices} "
            f"intra_op={self.intra_op} -> compute_threads={self.total_threads} "
            f"(<= vcpus: {self.total_threads <= self.vcpus})"
        )


def _resolve(vcpus: int, topology: str, n_devices: int | None, intra_op: int | None) -> SaturationConfig:
    if topology == "fine":
        nd = n_devices if n_devices else vcpus
        io = 1
    elif topology == "coarse":
        nd = n_devices if n_devices else max(1, vcpus // 8)
        io = intra_op if intra_op else max(1, vcpus // nd)
    elif topology == "single":
        nd = 1
        io = intra_op if intra_op else vcpus
    else:
        raise ValueError(f"unknown topology {topology!r} (fine|coarse|single)")
    # Never let the product exceed the budget: shrink intra_op first.
    while nd * io > vcpus and io > 1:
        io -= 1
    return SaturationConfig(vcpus=vcpus, n_devices=nd, intra_op=io, topology=topology)


def configure(
    *,
    vcpus: int | None = None,
    topology: str | None = None,
    n_devices: int | None = None,
    intra_op: int | None = None,
) -> SaturationConfig:
    """Set XLA_FLAGS + thread caps in ``os.environ`` BEFORE jax import. Returns the resolved config.

    Reads defaults from env (so the cloud bootstrap can configure purely via env):
      POKERBOT_SAT_VCPUS, POKERBOT_SAT_TOPOLOGY, POKERBOT_SAT_NDEVICES, POKERBOT_SAT_INTRAOP.
    """

    if vcpus is None:
        vcpus = int(os.environ.get("POKERBOT_SAT_VCPUS", "0")) or (os.cpu_count() or 1)
    if topology is None:
        topology = os.environ.get("POKERBOT_SAT_TOPOLOGY", "fine")
    if n_devices is None:
        nd_env = os.environ.get("POKERBOT_SAT_NDEVICES")
        n_devices = int(nd_env) if nd_env else None
    if intra_op is None:
        io_env = os.environ.get("POKERBOT_SAT_INTRAOP")
        intra_op = int(io_env) if io_env else None

    cfg = _resolve(vcpus, topology, n_devices, intra_op)

    # JAX on CPU runs everything on the platform; force N host CPU devices so pmap can shard across them.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # XLA Eigen intra-op pool sizing. multi_thread_eigen=false forces a 1-thread pool (fine topology);
    # otherwise size it explicitly to intra_op. inter_op stays 1 (we drive parallelism via pmap+devices,
    # not XLA's op-graph scheduler).
    eigen = "false" if cfg.intra_op <= 1 else "true"
    xla_flags = [
        f"--xla_force_host_platform_device_count={cfg.n_devices}",
        f"--xla_cpu_multi_thread_eigen={eigen}",
        f"intra_op_parallelism_threads={cfg.intra_op}",
        "inter_op_parallelism_threads=1",
    ]
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (existing + " " + " ".join(xla_flags)).strip()

    # Pin EVERY other numeric threadpool to 1 so it cannot spawn a second ~Ncore pool under XLA.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NPY_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    return cfg


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print the resolved saturation thread arithmetic.")
    ap.add_argument("--vcpus", type=int, default=64)
    ap.add_argument("--topology", default="fine", choices=["fine", "coarse", "single"])
    ap.add_argument("--n-devices", type=int, default=None)
    ap.add_argument("--intra-op", type=int, default=None)
    args = ap.parse_args()
    cfg = configure(vcpus=args.vcpus, topology=args.topology, n_devices=args.n_devices, intra_op=args.intra_op)
    print(cfg.describe())
    print("XLA_FLAGS =", os.environ["XLA_FLAGS"])
