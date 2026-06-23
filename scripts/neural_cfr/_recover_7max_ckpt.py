#!/usr/bin/env python
"""Recover a resumable 7-max neural-CFR checkpoint from a power-cut-corrupted ``_full.pkl``.

The 2026-06-23 hard reboot zeroed the TAIL of ``seven_v1_full.pkl`` (the in-loop _atomic_save uses
os.replace but no fsync -> on a power cut the rename commits but the data blocks do not). The pickle is a
dict written PARAMS-FIRST (history_value, average_strategy, cumulative_advantage, instantaneous_advantage,
then opt_states/config/iteration/prng), so the intact ~10.4MB prefix holds the net params while the zeroed
~15MB tail holds only the (optional-on-load) Adam moments + config + iter + prng.

Strategy: partial-unpickle the corrupt prefix with the pure-Python ``pickle._Unpickler`` (it builds objects
onto an inspectable stack as it reads, then fails at the truncation) and harvest the param dict items that
were fully read. Take history_value/average_strategy params from the INTACT inference ``seven_v1.pkl`` (same
live params, guaranteed clean); take cumulative/instantaneous advantage params from the recovered prefix.
Rebuild a valid ``_full.pkl`` with a reconstructed config (hidden_dim must match to build the nets; the
resume path re-applies the CLI hyperparams anyway), iteration from the inference ckpt, and a fresh prng_key.
Omit opt_states (loader cold-starts Adam) and rely on the INTACT ``_buffers.pkl`` sidecar.

Run (inspect):  python scripts/neural_cfr/_recover_7max_ckpt.py
Run (write):    python scripts/neural_cfr/_recover_7max_ckpt.py --write
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import sys
from pathlib import Path

CKPT_DIR = Path("checkpoints/sd_cfr_7max_neural")
CORRUPT = CKPT_DIR / "seven_v1_full.CORRUPT.bak"      # backup of the zeroed-tail _full
INFER = CKPT_DIR / "seven_v1.pkl"                     # intact inference ckpt (hv + avg params + iter)
BUFFERS = CKPT_DIR / "seven_v1_buffers.pkl"           # intact reservoir sidecar
OUT_FULL = CKPT_DIR / "seven_v1_full.pkl"             # the resumable file to rebuild

# Known run hyperparams (babysit_7max_overnight.sh). Only hidden_dim is load-critical (net shapes); the
# resume path re-applies train/avg/hist steps + total_iterations via replace(cfg, **overrides).
HIDDEN_DIM = 256
LR = 1e-4
TRAVERSALS = 128
BATCH = 1024
TRAIN_STEPS = 16
AVG_STEPS = 128
HIST_STEPS = 32
SEED = 0
BUFFER_CAPACITY = 100_000


def _harvest(seq, out):
    """Collect {str_key: next_item} pairs from a partially-built unpickler stack segment."""
    for i in range(len(seq) - 1):
        if isinstance(seq[i], str):
            out.setdefault(seq[i], seq[i + 1])


def recover_advantage_params():
    data = CORRUPT.read_bytes()
    n = len(data)
    i = n
    while i > 0 and data[i - 1] == 0:
        i -= 1
    print(f"[recover] {CORRUPT.name}: size={n}  last_nonzero_off={i}  trailing_zeros={n - i}")
    u = pickle._Unpickler(io.BytesIO(data))
    err = None
    try:
        u.load()
    except Exception as e:  # noqa: BLE001 — truncation is EXPECTED; we mine the stack it left behind
        err = e
    print(f"[recover] unpickler stopped at: {type(err).__name__ if err else 'EOF'}: {err}")
    harvested: dict = {}
    for seg in list(u.metastack) + [u.stack]:
        _harvest(seg, harvested)
    print(f"[recover] keys harvested from prefix: {sorted(harvested)}")
    return harvested


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the rebuilt seven_v1_full.pkl (else inspect only)")
    args = ap.parse_args()

    import jax
    import numpy as np
    from pokerbot.training.neural_cfr import VRDeepPDCFRConfig, _config_to_dict

    harvested = recover_advantage_params()
    cum = harvested.get("cumulative_advantage_params")
    inst = harvested.get("instantaneous_advantage_params")
    if cum is None or inst is None:
        print("[FAIL] advantage params not in the recoverable prefix -> cannot reconstruct a resume ckpt.")
        return 2
    print(f"[recover] cumulative_advantage_params: list len={len(cum)} (expect 7 per-seat)")
    print(f"[recover] instantaneous_advantage_params: list len={len(inst)} (expect 7 per-seat)")

    infer = pickle.loads(INFER.read_bytes())
    iteration = int(infer["iteration"])
    print(f"[recover] inference ckpt iteration={iteration}  hidden={infer['hidden_dim']}")
    if infer["hidden_dim"] != HIDDEN_DIM:
        print(f"[WARN] inference hidden_dim {infer['hidden_dim']} != assumed {HIDDEN_DIM}")

    # Per-seat advantage net counts must equal num_players (7) for a 7-max resume.
    if len(cum) != 7 or len(inst) != 7:
        print(f"[FAIL] expected 7 per-seat advantage nets, got cum={len(cum)} inst={len(inst)}.")
        return 3

    cfg = VRDeepPDCFRConfig(
        total_iterations=100_000, traversals_per_iter=TRAVERSALS, batch_size=BATCH,
        learning_rate=LR, hidden_dim=HIDDEN_DIM, buffer_capacity=BUFFER_CAPACITY, seed=SEED,
        train_steps_per_iter=TRAIN_STEPS, avg_strategy_steps_per_iter=AVG_STEPS,
        history_value_steps_per_iter=HIST_STEPS, traversal_workers=0,
    )
    payload = {
        "history_value_params": infer["history_value_params"],      # clean, from inference ckpt
        "average_strategy_params": infer["average_strategy_params"],  # clean, from inference ckpt
        "cumulative_advantage_params": cum,                         # recovered from prefix
        "instantaneous_advantage_params": inst,                     # recovered from prefix
        "config": _config_to_dict(cfg),
        "iteration": iteration,
        "prng_key": np.asarray(jax.random.PRNGKey(SEED)),           # fresh stream (acceptable on resume)
        # opt_states / prev_cumulative omitted -> loader cold-starts Adam + prev==current (R1/R3 fallback);
        # reservoirs come from the INTACT _buffers.pkl sidecar.
    }
    print(f"[recover] rebuilt payload keys: {sorted(payload)}")

    if not args.write:
        print("[recover] inspect-only (pass --write to persist). Verifying it round-trips in-memory...")
        blob = pickle.dumps(payload, protocol=4)
        pickle.loads(blob)
        print(f"[recover] in-memory pickle round-trip OK ({len(blob)} bytes). Re-run with --write to persist.")
        return 0

    # Atomic + fsync write (the durability the original lacked): tmp -> fsync -> os.replace.
    tmp = OUT_FULL.with_suffix(".rebuild.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUT_FULL)
    print(f"[recover] WROTE {OUT_FULL} ({OUT_FULL.stat().st_size} bytes, fsync'd). iteration={iteration}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
