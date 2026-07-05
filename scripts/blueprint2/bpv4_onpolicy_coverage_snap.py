#!/usr/bin/env python3
"""On-policy coverage diagnostic for an NLSNAP01 snapshot blob (v3 cum/current-strategy metric).

This is bpv4_onpolicy_coverage.py with the snapshot backend swapped from the NLCFR001 strat
section to an NLSNAP01 blob.  Under snapshot_avg the strat section of the checkpoint is
preflop-only + a frozen pre-snap_start residual (~3200 iters worth), which gives a misleadingly
LOW coverage number (that residual is mostly preflop postfix).  The NLSNAP01 file is the
regret-matched current strategy serialized from cum_regret — this is what the in-trainer
coverage_probe function (trainer.rs:coverage_probe) probes, and the basis on which the
pre-registered C1/C2/C3 gates were written.

Usage (v3 @20k, snap approach — the C1-equivalent metric):
  python scripts/blueprint2/bpv4_onpolicy_coverage_snap.py \
      --snap artifacts/bpv4_tmp/snap_020000.nlsnap \
      --snap-index artifacts/bpv4_tmp/idx_v3_snap20k \
      --hands 400 --key-mode fullhist --num-buckets 200 --mrr \
      --out artifacts/_bpv4_v3_onpolicy_snap20k.json
"""
from __future__ import annotations

import argparse
import json
import mmap as _mmap
import os
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import bp3_forced_response_probe as probe  # noqa: E402


NUM_ACTIONS = 11
SNAP_ROW = 16 + NUM_ACTIONS * 4  # 60 bytes


class _IndexedSnapFile:
    """mmap-backed lookup over a sorted NLSNAP01 index (built by bpv4_nlsnap_index.py).

    Same .get(key) contract as _IndexedSnap / in-RAM dict —
    returns an np.float64 array or None (hit/miss).
    Row format: 16-byte key + 11 f32 probabilities (LE).
    """
    _MISS = object()
    _CACHE_CAP = 500_000

    def __init__(self, index_dir: str):
        import numpy as np
        meta = json.load(open(os.path.join(index_dir, "meta.json"), encoding="utf-8"))
        assert meta.get("format") == "NLSNAP01", f"Expected NLSNAP01 index, got {meta.get('format')}"
        self._n = int(meta["n_rows"])
        self._np = np
        self._A = np.memmap(os.path.join(index_dir, "A.bin"), dtype="<u8", mode="r")
        self._B = np.memmap(os.path.join(index_dir, "B.bin"), dtype="<u8", mode="r")
        self._O = np.memmap(os.path.join(index_dir, "O.bin"), dtype="<u8", mode="r")
        assert len(self._A) == self._n == len(self._B) == len(self._O), "index length mismatch"
        f = open(meta["blob"], "rb")
        self._mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
        self._cache: dict[bytes, object] = {}
        self.iters = int(meta["iters"])

    def __len__(self) -> int:
        return self._n

    def get(self, key, default=None):
        v = self._cache.get(key, self._MISS)
        if v is not self._MISS:
            return default if v is None else v
        np = self._np
        # np.uint64 casts LOAD-BEARING — same NEP50 trap as _IndexedSnap
        a = np.uint64(int.from_bytes(key[:8], "big"))
        b = np.uint64(int.from_bytes(key[8:16], "big"))
        lo = int(np.searchsorted(self._A, a, "left"))
        hi = int(np.searchsorted(self._A, a, "right"))
        out = None
        if hi > lo:
            j = lo + int(np.searchsorted(self._B[lo:hi], b, "left"))
            if j < hi and int(self._B[j]) == b:
                off = int(self._O[j])
                # f32 probabilities (LE), already normalized by the Rust serializer
                row = struct.unpack_from(f"<{NUM_ACTIONS}f", self._mm, off + 16)
                tot = sum(row)
                if tot > 0.0:
                    out = np.asarray([x / tot for x in row], dtype=np.float64)
                else:
                    out = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float64)
        if len(self._cache) < self._CACHE_CAP:
            self._cache[key] = out
        return default if out is None else out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True, help="NLSNAP01 blob path")
    ap.add_argument("--snap-index", required=True, help="index dir from bpv4_nlsnap_index.py")
    ap.add_argument("--hands", type=int, default=400)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--key-mode", choices=("prefhist", "fullhist", "summary"), default="fullhist")
    ap.add_argument("--postflop-mode", choices=("emd", "fast"), default="fast")
    ap.add_argument("--num-buckets", type=int, default=200)
    ap.add_argument("--mrr", action="store_true")
    ap.add_argument("--emd-frozen",
                    default="data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn")
    ap.add_argument("--river-ochs",
                    default="data/abstraction/river_ochs_160.rochs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # env BEFORE any pokerbot import (standing rule; same logic as _configure_env)
    probe._configure_env(args)
    import random

    import numpy as np

    from pokerbot.engine.game import GameState, play_hand
    from pokerbot.eval import lbr as _lbr
    from pokerbot.eval.blueprint_ranking import TabularBlueprintAgent

    snap = _IndexedSnapFile(args.snap_index)
    print(f"snap {len(snap):,} rows  iters={snap.iters}  key-mode={args.key_mode}  mrr={args.mrr}",
          flush=True)

    agents = {s: TabularBlueprintAgent(snap, random.Random(1000 + s)) for s in range(6)}
    t0 = time.time()
    end_street: dict[str, int] = {}
    for h in range(args.hands):
        rng = random.Random(args.seed * 77 + h)
        state = GameState.new_hand(rng=rng)
        for s, ag in agents.items():
            ag.reseed(args.seed * 1_000_003 + h * 31 + s)
        final, _ = play_hand({s: agents[s] for s in range(6)}, state=state)
        es = len(final.board)
        end_street[str(es)] = end_street.get(str(es), 0) + 1

    cov = _lbr.bp_coverage_report()
    fb = sum(a.fallback_count for a in agents.values())
    dc = sum(a.decision_count for a in agents.values())
    pf_h = sum(cov.get(s, {}).get("hits", 0) for s in ("flop", "turn", "river"))
    pf_m = sum(cov.get(s, {}).get("misses", 0) for s in ("flop", "turn", "river"))
    res = {
        "mode": "onpolicy_coverage_snap",
        "snap": args.snap,
        "snap_index": args.snap_index,
        "hands": args.hands,
        "seed": args.seed,
        "key_mode": args.key_mode,
        "mrr": bool(args.mrr),
        "adapter_fallback_rate": fb / max(1, dc),
        "bp_coverage": cov,
        "postflop_miss_rate": pf_m / max(1, pf_h + pf_m),
        "end_street_board_cards": end_street,
        "wall_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("adapter_fallback_rate", "postflop_miss_rate",
                                          "bp_coverage", "end_street_board_cards")}, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
