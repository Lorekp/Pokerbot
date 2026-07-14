#!/usr/bin/env python3
"""Preflop-only re-solve driver for the check-call-leaf pre-check (POKERBOT_PREFLOP_CHECKCALL_LEAF).

Trains NlheCFRTrainer in the DEPLOYED EMD-K160 preflop key space. With the flag ON the postflop is
replaced by the check-call-to-showdown leaf (trainer.rs::checkcall_showdown), so the PREFLOP is trained
against each reached hand realizing its equity (section-37 idealized field) and only preflop rows are
produced. Saves the full checkpoint (save_bytes_to_file -> .nlcfr) for serving via
audit_preflop_blueprint.py (AUDIT_PF_CHECKPOINT=<out>). Also used for the byte-identity check (flag OFF).

Env (set BEFORE running, matching the deployed key space):
  POKERBOT_PREFLOP_CHECKCALL_LEAF (0/1), POKERBOT_PREFLOP_LOSSLESS=1, POKERBOT_SNAPSHOT_AVG=1,
  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_POSTFLOP_NUM_BUCKETS=160
  POKERBOT_POSTFLOP_MODE=emd POKERBOT_POSTFLOP_EMD_FROZEN=<path>
  CCLEAF_ITERS CCLEAF_NDEALS CCLEAF_OUT CCLEAF_EMD CCLEAF_CHUNK
"""
import os
import sys
import time

ITERS = int(os.environ.get("CCLEAF_ITERS", "20000"))
NDEALS = int(os.environ.get("CCLEAF_NDEALS", "64"))
OUT = os.environ["CCLEAF_OUT"]
EMD = os.environ.get("CCLEAF_EMD") or None
CHUNK = int(os.environ.get("CCLEAF_CHUNK", "2000"))

import battery_native as bn  # noqa: E402

print(f"pyd: {bn.__file__}", flush=True)
t = bn.NlheCFRTrainer(160, [], NDEALS, 6, float("-inf"), False, EMD,
                      0, False, 1 << 21, False, False, -300000.0, 0.05, river_ochs_path=None)
print(f"checkcall_leaf={os.environ.get('POKERBOT_PREFLOP_CHECKCALL_LEAF')} iters={ITERS} "
      f"ndeals={NDEALS} snapshot_avg={os.environ.get('POKERBOT_SNAPSHOT_AVG')}", flush=True)
t0 = time.time()
while t.iters_done() < ITERS:
    t.train(min(CHUNK, ITERS - t.iters_done()))
    print(f"  iters={t.iters_done()}/{ITERS} infosets={t.n_infosets()} el={time.time() - t0:.0f}s", flush=True)
n = t.save_bytes_to_file(OUT)
print(f"SAVED {OUT} bytes={n} infosets={t.n_infosets()} wall={time.time() - t0:.0f}s", flush=True)
