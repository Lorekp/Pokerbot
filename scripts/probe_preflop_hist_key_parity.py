#!/usr/bin/env python3
"""PARITY GATE for the preflop HISTORY key (POKERBOT_PREFLOP_HIST_KEY=1, blueprint-v3 serve twin).

Walks GAME-FLOW states ONLY (real hands dealt through the pure-Python engine; standing rule: never
hand-build nodes). For every PREFLOP decision state the acting seat's blueprint key is computed
three ways and all three must be byte-identical:

  * py_key  — ``jax_weva_fast.fingerprint_fast`` (the serve twin under test);
  * nat_key — ``battery_native.fingerprint_fast`` (the Rust trainer/serve keying; reads
              POKERBOT_PREFLOP_HIST_KEY itself via OnceLock, so this process's env drives it);
  * ref_key — an INDEPENDENT in-script reimplementation of the exact message layout
              blake2b16(b"w" + LE32(pos*169+class) [+ b"H" + hist3] + b"S" + betting6), with
              hist3/betting6 recomputed from the raw state (pins the layout, not just twin==twin).

Run with --hist 1 (the bp3 key space) AND --hist 0 (must be byte-identical to the deployed layout,
i.e. ref_key WITHOUT the H segment). One env state per process — the flags are read at import
(Python) / first use (Rust OnceLock) — so the pytest wrapper (tests/test_preflop_hist_key_parity.py)
runs this twice via subprocess.

Coverage floors guarantee the walk actually exercises the aliasing-critical histories: limped pots,
multi-limper pots, single-raised, 3-bet, 4-bet+ pots, several opener positions, blind-vs-blind.

Usage:
  python scripts/probe_preflop_hist_key_parity.py --hist 1 --min-states 512 \
      --out artifacts/prefhist_parity_on.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# ---------------------------------------------------------------------------
# Env MUST be resolved before any pokerbot import (flags are read at import; the Rust side reads
# POKERBOT_PREFLOP_HIST_KEY once via OnceLock at first fingerprint call in this same process).
# Preflop keys are postflop-abstraction-independent, so the bounded made-hand config (the
# fingerprint.rs header config) is the right minimal parity regime.
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--hist", type=int, choices=(0, 1), required=True,
                 help="1 = POKERBOT_PREFLOP_HIST_KEY=1 (bp3 key space); 0 = deployed layout")
_ap.add_argument("--min-states", type=int, default=512)
_ap.add_argument("--max-hands", type=int, default=6000)
_ap.add_argument("--seed", type=int, default=7)
_ap.add_argument("--out", default="")
ARGS = _ap.parse_args()

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _k, _v in {
    "POKERBOT_BETTING_ABSTRACTION": "1",
    "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "200",
    "POKERBOT_POSTFLOP_MODE": "fast",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
    "POKERBOT_ALLOW_MADE_HAND_BUCKETS": "1",  # parity probe, not a production train
}.items():
    os.environ.setdefault(_k, _v)
if ARGS.hist == 1:
    os.environ["POKERBOT_PREFLOP_HIST_KEY"] = "1"
else:
    os.environ.pop("POKERBOT_PREFLOP_HIST_KEY", None)

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Independent reference layout (mirrors fingerprint.rs, NOT the twin under test).
# ---------------------------------------------------------------------------
_POS_BY_OFFSET_REF = (3, 4, 5, 0, 1, 2)
_POT_EDGES = (3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0)
_TOCALL_EDGES = (0.01, 0.34, 0.67, 1.01, 2.01)
_SPR_EDGES = (1.0, 2.5, 5.0, 10.0, 20.0)
_CALL, _BET_BUCKET_START, _ALL_IN = 2, 3, 10


def _bucket(x: float, edges) -> int:
    b = 0
    for e in edges:
        if x >= e:
            b += 1
        else:
            break
    return b


def _canon_class(c0: int, c1: int) -> int:
    r0, s0 = c0 // 4, c0 % 4
    r1, s1 = c1 // 4, c1 % 4
    if r0 == r1:
        return r0
    hr, lr = (r0, r1) if r0 > r1 else (r1, r0)
    tri = hr * (hr - 1) // 2 + lr
    return (13 + tri) if s0 == s1 else (13 + 78 + tri)


def _ref_betting6(env, seat: int) -> bytes:
    committed = np.asarray(env.committed_street)
    seat_status = np.asarray(env.seat_status)
    stacks = np.asarray(env.stacks)
    pot = int(np.asarray(env.pot_dead)) + int(committed.sum())
    to_call = max(0, int(np.asarray(env.current_bet)) - int(committed[seat]))
    live = [int(stacks[s]) for s in range(6) if int(seat_status[s]) not in (0, 1)]
    eff = (min(live) if live else 0) / 100.0
    pot_bb = pot / 100.0
    last_agg = int(np.asarray(env.last_aggressor))
    return bytes((
        int(np.asarray(env.street)) & 7,
        _bucket(pot_bb, _POT_EDGES),
        _bucket(to_call / max(pot, 1), _TOCALL_EDGES),
        _bucket(eff / max(pot_bb, 1e-6), _SPR_EDGES),
        1 if seat == int(np.asarray(env.button)) else 0,
        1 if (last_agg >= 0 and last_agg != seat) else 0,
    ))


def _ref_hist3(env) -> bytes:
    hl = max(0, min(int(np.asarray(env.hist_len)), 64))
    act = np.asarray(env.hist_action)
    seats = np.asarray(env.hist_seat)
    n_calls = n_raises = 0
    opener_seat = -1
    for i in range(hl):
        a = int(act[i])
        if a == _CALL:
            if n_raises == 0 and n_calls < 3:
                n_calls += 1
        elif _BET_BUCKET_START <= a <= _ALL_IN:
            if n_raises == 0:
                opener_seat = int(seats[i])
            if n_raises < 3:
                n_raises += 1
    opener_pos = (
        _POS_BY_OFFSET_REF[(opener_seat - int(np.asarray(env.button))) % 6] if opener_seat >= 0 else -1
    )
    return bytes((n_calls, n_raises, opener_pos + 1))


def _ref_key(env, seat: int, hist_on: bool) -> tuple[bytes, bytes]:
    holes = np.asarray(env.holes)
    h0, h1 = int(holes[seat, 0]), int(holes[seat, 1])
    lo, hi = (h0, h1) if h0 <= h1 else (h1, h0)
    pos_enum = _POS_BY_OFFSET_REF[(seat - int(np.asarray(env.button))) % 6]
    weva_id = pos_enum * 169 + _canon_class(lo, hi)
    hist3 = _ref_hist3(env)
    msg = b"w" + struct.pack("<i", weva_id)
    if hist_on:
        msg += b"H" + hist3
    msg += b"S" + _ref_betting6(env, seat)
    return hashlib.blake2b(msg, digest_size=16).digest(), hist3


def _pick_action(rng: random.Random, legal_set: set[int]) -> int:
    """Game-flow policy tuned for HISTORY variety: calls (limps/flat-3bets), sized raises
    (opens/3bets/4bets), a little jam, some folds (so blind-vs-blind pots occur)."""
    u = rng.random()
    sized = [a for a in legal_set if 3 <= a <= 9]
    if u < 0.26 and 0 in legal_set:
        return 0
    if u < 0.64:
        for a in (2, 1):
            if a in legal_set:
                return a
    if u < 0.95 and sized:
        return rng.choice(sized)
    if 10 in legal_set and rng.random() < 0.4:
        return 10
    return rng.choice(sorted(legal_set))


def main() -> int:
    import jax
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, is_chance_node_np, legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.engine.jax_weva_fast import fingerprint_fast as py_fingerprint
    from pokerbot.search import native_solver as _ns

    hist_on = ARGS.hist == 1
    if not _ns.available():
        print("FATAL: battery_native extension unavailable (rebuild + install the .pyd)", flush=True)
        return 2
    ext = _ns._EXT
    caps = set(ext.native_caps()) if hasattr(ext, "native_caps") else set()
    if hist_on and "preflop_hist_key" not in caps:
        print(f"FATAL: battery_native binary predates the preflop-hist-key port (caps={sorted(caps)}); "
              "rebuild --features python and reinstall the .pyd", flush=True)
        return 2
    print(f"hist={ARGS.hist} native_caps={sorted(caps)}", flush=True)

    rng = random.Random(ARGS.seed)
    total = 0
    fails = 0
    mismatches = []
    cov = {"limp1": 0, "limp2": 0, "raise1": 0, "raise2": 0, "raise3": 0, "unopened": 0, "bvb": 0}
    opener_positions = set()
    hands = 0
    t0 = time.time()

    def covered() -> bool:
        return (total >= ARGS.min_states and cov["limp1"] >= 30 and cov["limp2"] >= 8
                and cov["raise1"] >= 60 and cov["raise2"] >= 12 and cov["raise3"] >= 4
                and cov["unopened"] >= 30 and cov["bvb"] >= 3 and len(opener_positions) >= 4)

    while hands < ARGS.max_hands and not covered():
        key = jax.random.PRNGKey(rng.randrange(0, 2**31 - 1))
        env = to_numpy_state(init_hand(key, button=hands % 6))
        hands += 1
        depth = 0
        while not bool(env.done) and depth < 60:
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            if int((np.asarray(env.board) >= 0).sum()) > 0:
                break  # left preflop — later streets are not this probe's subject
            seat = int(env.action_seat)
            if seat < 0:
                break
            mask = legal_action_mask_np(env, seat)
            legal = set(int(a) for a in np.flatnonzero(mask))
            if not legal:
                break
            # --- the three-way parity check on this GAME-FLOW preflop decision state ---
            py_key = bytes(py_fingerprint(env, seat))
            nat_key = bytes(ext.fingerprint_fast(_ns._to_flat(env).tobytes(), seat, 200))
            ref_key, hist3 = _ref_key(env, seat, hist_on)
            total += 1
            n_limps, n_raises, opener_b = hist3[0], hist3[1], hist3[2]
            if n_limps >= 1:
                cov["limp1"] += 1
            if n_limps >= 2:
                cov["limp2"] += 1
            if n_raises == 1:
                cov["raise1"] += 1
            if n_raises == 2:
                cov["raise2"] += 1
            if n_raises >= 3:
                cov["raise3"] += 1
            if n_raises == 0 and n_limps == 0:
                cov["unopened"] += 1
            if opener_b:
                opener_positions.add(opener_b - 1)
            # blind-vs-blind: everyone but SB/BB already folded, hero is a blind.
            ss = np.asarray(env.seat_status)
            btn = int(np.asarray(env.button))
            sb, bb = (btn + 1) % 6, (btn + 2) % 6
            if seat in (sb, bb) and all(int(ss[s]) == 1 for s in range(6) if s not in (sb, bb)):
                cov["bvb"] += 1
            if not (py_key == nat_key == ref_key):
                fails += 1
                if len(mismatches) < 20:
                    mismatches.append({
                        "seat": seat, "hist3": list(hist3),
                        "py": py_key.hex(), "nat": nat_key.hex(), "ref": ref_key.hex(),
                        "hist_seq": [int(x) for x in np.asarray(env.hist_action)[:int(env.hist_len)]],
                    })
            env = apply_action_np(env, _pick_action(rng, legal))
            depth += 1
        if hands % 500 == 0:
            print(f"[progress] hands={hands} states={total} cov={cov} openers={sorted(opener_positions)} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    ok = fails == 0 and covered()
    print(f"\n=== PREFLOP HIST-KEY PARITY (hist={ARGS.hist}) ===")
    print(f"hands={hands} preflop_states={total} mismatches={fails}")
    print(f"coverage: {cov}  opener_positions={sorted(opener_positions)} "
          f"(floors: limp1>=30 limp2>=8 raise1>=60 raise2>=12 raise3>=4 unopened>=30 bvb>=3 openers>=4)")
    print(f"PARITY GATE: {'PASS' if ok else 'FAIL'}", flush=True)
    result = {"hist": ARGS.hist, "hands": hands, "states": total, "mismatches": fails,
              "coverage": cov, "opener_positions": sorted(opener_positions),
              "covered": covered(), "pass": bool(ok), "mismatch_samples": mismatches,
              "seed": ARGS.seed}
    if ARGS.out:
        os.makedirs(os.path.dirname(ARGS.out) or ".", exist_ok=True)
        with open(ARGS.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {ARGS.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
