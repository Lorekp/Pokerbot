#!/usr/bin/env python3
"""ISOLATED-ARM policy worker for scripts/engine_ab_h2h.py (the `worker_env` knob).

One worker process == one arm == one POKERBOT_* keying regime. The parent sets this process's
environment BEFORE python starts (subprocess env=), so every import-latched Python gate and every
OnceLock-latched native gate (FULLHIST/PRESUM/RIVER_OCHS/CLOSE_CALLED_STREET/PRE_KEEP/EMD blob/...)
configures exactly as a production serve process would — the certified "env before import"
configuration unit, per arm.

Protocol (stdio, length-prefixed pickle frames, little-endian u32 length):
    ("hello", None)         -> ("ready", sorted(native_caps))
    ("act", (env, seat))    -> ("row", np.float64[NUM_ACTIONS])   # pf(env, seat), the in-process contract
                             | ("err", "ExcType: message")
    ("stats", None)         -> ("stats", {"acts": int})
    ("quit", None)          -> (closes)

fd hygiene: fd1 is duplicated to a private frame pipe and then redirected to fd2, so ANY textual
print — Python or native — lands on stderr and can never corrupt the frame stream.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import struct
import sys

# --- fd hygiene BEFORE any pokerbot import (imports print banners) ---
_FRAME_FD = os.dup(1)          # the real pipe back to the parent
os.dup2(2, 1)                  # fd1 (and every C-level println) now goes to stderr
sys.stdout = sys.stderr        # Python-level prints too
_FRAME_OUT = os.fdopen(_FRAME_FD, "wb", buffering=0)
_FRAME_IN = sys.stdin.buffer


def _send(obj) -> None:
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _FRAME_OUT.write(struct.pack("<I", len(b)))
    _FRAME_OUT.write(b)


def _recv():
    hdr = _FRAME_IN.read(4)
    if len(hdr) < 4:
        return None
    n = struct.unpack("<I", hdr)[0]
    buf = b""
    while len(buf) < n:
        chunk = _FRAME_IN.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="arm knob-spec JSON (worker_env already stripped)")
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--repo-root", required=True)
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    # POKERBOT_* env came from the parent's spawn env (set BEFORE this process started); the
    # engine_ab_h2h module setdefaults never overwrite pinned values.
    import numpy as np  # noqa: E402
    import engine_ab_h2h as h2h  # noqa: E402  (module import applies the deployed setdefaults)

    spec = json.loads(args.spec)
    print(f"[arm_worker] pid={os.getpid()} blueprint={args.blueprint} spec={spec}", flush=True)
    print(f"[arm_worker] keying env: " + " ".join(
        f"{k}={os.environ.get(k, '')}" for k in (
            "POKERBOT_FULLHIST_KEY", "POKERBOT_PRESUM_KEY", "POKERBOT_MRR",
            "POKERBOT_CLOSE_CALLED_STREET", "POKERBOT_PRE_KEEP", "POKERBOT_PRE_RR_KEEP",
            "POKERBOT_RIVER_OCHS", "POKERBOT_POSTFLOP_MODE", "POKERBOT_POSTFLOP_EMD_FROZEN",
            "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "POKERBOT_SETTLE_SHOWDOWN_LEAF")), flush=True)

    snap = h2h.load_snapshot(args.blueprint)
    print(f"[arm_worker] blueprint rows={len(snap)}", flush=True)

    # Blueprint lookup MISS instrument (Python-side lookups only — the hero-preflop surface when
    # search_preflop is off; native playout lookups hit the marshaled HashMap and are not counted).
    # A fullhist-keyed blueprint MISSES on any history its trainer never generated (off-menu
    # opponent action) and the policy degrades to uniform — this counter makes that rate a
    # MEASURED number in every cert log instead of a guess.
    class _CountingSnap(dict):
        hits = 0
        misses = 0

        def get(self, k, default=None):
            v = dict.get(self, k, default)
            if v is default and k not in self:
                _CountingSnap.misses += 1
            else:
                _CountingSnap.hits += 1
            return v

        def __getitem__(self, k):
            try:
                v = dict.__getitem__(self, k)
                _CountingSnap.hits += 1
                return v
            except KeyError:
                _CountingSnap.misses += 1
                raise

        def __contains__(self, k):
            c = dict.__contains__(self, k)
            if c:
                _CountingSnap.hits += 1
            else:
                _CountingSnap.misses += 1
            return c

    class _CountingWrap:
        """Counting facade for NON-dict serve objects (engine_ab_h2h.SortedSnapshotDir — the cert6
        full-table mmap index). Same class-level counters as _CountingSnap; delegates get/len."""

        def __init__(self, inner):
            self._inner = inner

        def get(self, k, default=None):
            v = self._inner.get(k, default)
            if v is None and default is None:
                _CountingSnap.misses += 1
            else:
                _CountingSnap.hits += 1
            return v

        def __contains__(self, k):
            c = k in self._inner
            if c:
                _CountingSnap.hits += 1
            else:
                _CountingSnap.misses += 1
            return c

        def __len__(self):
            return len(self._inner)

    snap = _CountingSnap(snap) if isinstance(snap, dict) else _CountingWrap(snap)

    from pokerbot.search import native_solver as _ns  # noqa: E402
    from pokerbot.search.pluribus_search import set_native_subgame  # noqa: E402
    if not _ns.native_subgame_available():
        _ns.enable()
    set_native_subgame(True)
    caps = sorted(_ns._EXT.native_caps()) if (_ns._EXT is not None
                                              and hasattr(_ns._EXT, "native_caps")) else []
    print(f"[arm_worker] native available={_ns.available()} "
          f"subgame={_ns.native_subgame_available()} caps={caps}", flush=True)

    pf = h2h._build_policy(spec, snap, args.seed, repo_root)

    n_acts = 0
    while True:
        msg = _recv()
        if msg is None:
            break
        kind, payload = msg
        if kind == "hello":
            _send(("ready", caps))
        elif kind == "act":
            env, seat = payload
            try:
                row = np.asarray(pf(env, int(seat)), dtype=np.float64)
                n_acts += 1
                _send(("row", row))
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                _send(("err", f"{type(e).__name__}: {e}"))
        elif kind == "stats":
            # CERT9 proof-of-search: ship the full runtime counter block back to the parent so
            # every shard checkpoint/result JSON carries it (decisions_total, per-street served
            # tags + wall p50/p95, fallback counters, blueprint hit/miss, native caps).
            payload_out = {
                "acts": n_acts,
                "bp_hits": int(_CountingSnap.hits),
                "bp_misses": int(_CountingSnap.misses),
                "native_caps": caps,
                "strict_env": os.environ.get("POKERBOT_SEARCH_STRICT", ""),
            }
            st = getattr(pf, "search_stats", None)
            if st is not None:
                from pokerbot.eval.solver_exploitability import summarize_search_stats
                payload_out["search_stats"] = summarize_search_stats(st)
            _send(("stats", payload_out))
        elif kind == "quit":
            break
        else:
            _send(("err", f"unknown frame kind {kind!r}"))
    print(f"[arm_worker] exiting after {n_acts} acts; blueprint lookups: "
          f"hits={_CountingSnap.hits} misses={_CountingSnap.misses} "
          f"miss_rate={_CountingSnap.misses / max(1, _CountingSnap.hits + _CountingSnap.misses):.4f}",
          flush=True)
    # Per-street coverage (opt-in POKERBOT_BP_COVERAGE=1 in this arm's worker_env): the lbr
    # instrument counts a masked-to-zero row as a miss too — the honest serve-quality surface.
    try:
        from pokerbot.eval.lbr import bp_coverage_report  # noqa: E402
        rep = bp_coverage_report()
        if rep:
            print(f"[arm_worker] bp_coverage per-street: {json.dumps(rep)}", flush=True)
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
