"""Parity GATE for fingerprint_native: native fp bytes == Python fingerprint_fast bytes for EVERY
sampled (EnvState, seat), across preflop+flop+turn+river, all 6 seats, in BOTH config regimes.

The abstraction flags (POKERBOT_PREFLOP_LOSSLESS / _BETTING / POKERBOT_BETTING_ABSTRACTION /
POKERBOT_POSTFLOP_BUCKETS) are read at IMPORT, so each regime is exercised in a SEPARATE subprocess
(this script re-execs itself with the right env per regime). REQUIRES 0 mismatches; reports per
config/street counts. Also benchmarks Python vs native per-fingerprint time per path.

Run (driver):   python tools/fingerprint_native/verify_parity.py --n 20000
A single regime worker is invoked internally with --worker and the env preset by the driver.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEV_VEC_DIR = Path(__file__).resolve().parent / "dev_vectors"

# Config regimes. Betting-abstraction is REQUIRED for the native fast path (the Python oracle's
# fingerprint_fast itself only abstracts then). We sweep preflop-lossless and postflop-buckets on/off.
REGIMES = [
    {
        "name": "abstraction_only",
        "env": {
            "POKERBOT_BETTING_ABSTRACTION": "1",
            "POKERBOT_PREFLOP_LOSSLESS": "0",
            "POKERBOT_PREFLOP_LOSSLESS_BETTING": "0",
            "POKERBOT_POSTFLOP_BUCKETS": "0",
        },
    },
    {
        "name": "lossless_pf+postflop_buckets",
        "env": {
            "POKERBOT_BETTING_ABSTRACTION": "1",
            "POKERBOT_PREFLOP_LOSSLESS": "1",
            "POKERBOT_PREFLOP_LOSSLESS_BETTING": "0",
            "POKERBOT_POSTFLOP_BUCKETS": "1",
            "POKERBOT_POSTFLOP_MODE": "fast",
            "POKERBOT_POSTFLOP_NUM_BUCKETS": "200",
        },
    },
    {
        "name": "lossless_pf+lossless_betting+postflop_buckets",
        "env": {
            "POKERBOT_BETTING_ABSTRACTION": "1",
            "POKERBOT_PREFLOP_LOSSLESS": "1",
            "POKERBOT_PREFLOP_LOSSLESS_BETTING": "1",
            "POKERBOT_POSTFLOP_BUCKETS": "1",
            "POKERBOT_POSTFLOP_MODE": "fast",
            "POKERBOT_POSTFLOP_NUM_BUCKETS": "200",
        },
    },
]


def _street_of(state) -> int:
    import numpy as np

    return int(np.asarray(state.street))


def _sample_states(n_hands: int, seed: int):
    """Play random hands through np_engine; yield (state) snapshots across all streets."""
    import numpy as np
    import jax

    from pokerbot.engine.jax_engine import reset_state
    from pokerbot.engine import np_engine as npe

    rng = np.random.default_rng(seed)
    states = []
    for hi in range(n_hands):
        key = jax.random.PRNGKey(seed * 1_000_003 + hi)
        js = reset_state(key)
        st = npe.to_numpy_state(js)
        steps = 0
        while not npe.is_terminal_np(st) and steps < 400:
            states.append(st)
            if npe.is_chance_node_np(st):
                # deal next street deterministically via the engine's own sampler
                st = npe.sample_chance_np(st)
            else:
                mask = npe.legal_action_mask_np(st)
                legal = [i for i, m in enumerate(mask) if m]
                if not legal:
                    break
                # Mixed policy: mostly check(1)/call(2) so betting rounds CLOSE and hands reach
                # flop/turn/river (postflop coverage); a small fraction picks any legal action
                # (bets/raises/all-in/occasional fold) so betting histories vary. Fold (0) is rare.
                r = rng.random()
                pref = [a for a in (1, 2) if a in legal]
                if pref and r > 0.18:
                    a = int(rng.choice(pref))
                elif r > 0.04:
                    nonfold = [a for a in legal if a != 0] or legal
                    a = int(rng.choice(nonfold))
                else:
                    a = int(rng.choice(legal))
                st = npe.apply_action_np(st, a)
            steps += 1
        states.append(st)
    return states


def _worker(regime_name: str, n_hands: int, seed: int, out_path: str) -> int:
    import numpy as np

    from pokerbot.engine.jax_weva_fast import fingerprint_fast as py_fp, default_weva_bucket_table
    from pokerbot.encoding import fingerprint_native as fn
    from pokerbot.engine import jax_weva_encode as enc

    if not fn.available():
        print(json.dumps({"error": "extension not built"}))
        return 2
    if not enc._BETTING_ABSTRACTION:
        print(json.dumps({"error": "betting abstraction must be ON for this regime"}))
        return 2

    table = default_weva_bucket_table()
    states = _sample_states(n_hands, seed)

    per_street_total = defaultdict(int)
    per_street_mismatch = defaultdict(int)
    mismatches = []
    # bench accumulators per path: preflop (bucket<0) vs postflop (bucket>=0)
    n_seats = 6

    for st in states:
        street = _street_of(st)
        for seat in range(n_seats):
            want = py_fp(st, seat, weva_table=table)
            got = fn.fingerprint_native(st, seat, weva_table=table)
            per_street_total[street] += 1
            if want != got:
                per_street_mismatch[street] += 1
                if len(mismatches) < 8:
                    mismatches.append(
                        {
                            "street": street,
                            "seat": seat,
                            "want": want.hex(),
                            "got": got.hex(),
                            "holes": np.asarray(st.holes).tolist(),
                            "board": np.asarray(st.board).tolist(),
                            "hist_len": int(np.asarray(st.hist_len)),
                        }
                    )

    # Benchmark: time Python vs native over a fixed subset (cap to keep it bounded).
    bench_states = states[: min(len(states), 4000)]
    # warm
    for st in bench_states[:50]:
        py_fp(st, 0, weva_table=table)
        fn.fingerprint_native(st, 0, weva_table=table)

    def time_fn(fn_call, reps):
        t0 = time.perf_counter()
        cnt = 0
        for _ in range(reps):
            for st in bench_states:
                fn_call(st, 0, table)
                cnt += 1
        return (time.perf_counter() - t0) / max(cnt, 1)

    reps = max(1, 8000 // max(len(bench_states), 1))
    py_t = time_fn(lambda s, seat, t: py_fp(s, seat, weva_table=t), reps)
    nat_t = time_fn(lambda s, seat, t: fn.fingerprint_native(s, seat, weva_table=t), reps)

    total = sum(per_street_total.values())
    total_mm = sum(per_street_mismatch.values())
    result = {
        "regime": regime_name,
        "states": len(states),
        "comparisons": total,
        "mismatches": total_mm,
        "per_street_total": {str(k): v for k, v in sorted(per_street_total.items())},
        "per_street_mismatch": {str(k): v for k, v in sorted(per_street_mismatch.items())},
        "sample_mismatches": mismatches,
        "bench_py_us": py_t * 1e6,
        "bench_native_us": nat_t * 1e6,
        "speedup": (py_t / nat_t) if nat_t > 0 else 0.0,
    }
    Path(out_path).write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sample_mismatches"}))
    return 0 if total_mm == 0 else 1


def _driver(n_hands: int, seed: int) -> int:
    DEV_VEC_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    summary = []
    for reg in REGIMES:
        env = dict(os.environ)
        env.update(reg["env"])
        out_path = str(DEV_VEC_DIR / f"parity_{reg['name']}.json")
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            reg["name"],
            "--n",
            str(n_hands),
            "--seed",
            str(seed),
            "--out",
            out_path,
        ]
        print(f"\n=== regime: {reg['name']} ===")
        proc = subprocess.run(cmd, env=env, cwd=str(REPO))
        if proc.returncode != 0:
            all_ok = False
        try:
            res = json.loads(Path(out_path).read_text(encoding="utf-8"))
            summary.append(res)
        except Exception:
            all_ok = False

    print("\n===== PARITY SUMMARY =====")
    for r in summary:
        status = "PASS" if r["mismatches"] == 0 else f"FAIL ({r['mismatches']} mm)"
        print(
            f"  {r['regime']:48s} states={r['states']:6d} cmp={r['comparisons']:7d} "
            f"{status:14s} py={r['bench_py_us']:.2f}us nat={r['bench_native_us']:.2f}us "
            f"x{r['speedup']:.2f}"
        )
        if r["mismatches"] != 0:
            all_ok = False
            for mm in r.get("sample_mismatches", [])[:3]:
                print(f"      MM street={mm['street']} seat={mm['seat']} want={mm['want']} got={mm['got']}")
    print(f"\nGATE: {'PASS (0 mismatches across all regimes)' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None, help="internal: run one regime worker")
    ap.add_argument("--n", type=int, default=20000, help="hands per regime (states ~ 5-15x this)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.worker is not None:
        return _worker(args.worker, args.n, args.seed, args.out or "parity_out.json")
    return _driver(args.n, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
