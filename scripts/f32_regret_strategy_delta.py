#!/usr/bin/env python3
"""f32_regret vs f64 6-max average-strategy PATH-DIVERGENCE DIAGNOSTIC (NOT a quality gate).

Parses two NLCFR001 checkpoints (examples/ab_train.rs built with vs without --features f32_regret, at a
MATCHED seed) and reports the per-infoset average-strategy L1 distance distribution.

*** READ THIS BEFORE TRUSTING THE NUMBER ***
Per-infoset strategy-L1 at a matched seed is an UNSOUND quality metric for this trainer, and the
empirical run proves it:
  * CONTROL (f64 vs f64, same seed): bit-identical, L1 = 0 everywhere -> the trainer is deterministic.
  * f32 vs f64 (same seed): median L1 ~= 0.53 -> a LARGE divergence.
6-max ES-MCCFR has NO unique equilibrium and is chaotically seed-sensitive: the first f32 rounding that
flips a sampled action sends the two runs down DIFFERENT-but-equally-valid strategy paths. So a large L1
means "f32 landed on a different equilibrium", NOT "f32 is worse". The QUALITY question is answered by
LBR EXPLOITABILITY (scripts/f32_regret_lbr_ab.py), where the two blueprints are statistically identical
(paired diff +5.65 bb/100, 95% CI [-27,+38], t=0.36) — equally good despite different strategies.

This script is kept as a DIAGNOSTIC: the L1 distribution quantifies how chaos-sensitive the trainer is,
and the f64-vs-f64 path confirms determinism. It does NOT emit a quality PASS/FAIL. On-disk rows are f64
for both builds, so no native import is needed — we parse the blob directly.

NLCFR001 layout (little-endian):
    magic "NLCFR001" (8) | iters_done u64 (8)
    cum  : n u64 (8) | n * [ key 16 bytes | 11 * f64 ]
    strat: n u64 (8) | n * [ key 16 bytes | 11 * f64 ]
"""
import struct
import sys

NUM_ACTIONS = 11
ROW_BYTES = 16 + NUM_ACTIONS * 8


def _read_map(buf, p):
    (n,) = struct.unpack_from("<Q", buf, p)
    p += 8
    out = {}
    for _ in range(n):
        key = bytes(buf[p : p + 16])
        p += 16
        row = struct.unpack_from("<11d", buf, p)
        p += NUM_ACTIONS * 8
        out[key] = row
    return out, p


def load_checkpoint(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != b"NLCFR001":
        raise ValueError(f"{path}: bad magic {buf[:8]!r}")
    p = 8
    (iters_done,) = struct.unpack_from("<Q", buf, p)
    p += 8
    cum, p = _read_map(buf, p)
    strat, p = _read_map(buf, p)
    return {"iters_done": iters_done, "cum": cum, "strat": strat}


def avg_strategy(row):
    """Row-normalized average strategy (uniform fallback for an all-zero row) — mirrors trainer::snapshot."""
    tot = sum(row)
    if tot > 0.0:
        return [v / tot for v in row]
    return [1.0 / NUM_ACTIONS] * NUM_ACTIONS


def main():
    if len(sys.argv) < 3:
        print("usage: f32_regret_strategy_delta.py <f64_blob> <f32_blob>", file=sys.stderr)
        sys.exit(2)
    a = load_checkpoint(sys.argv[1])  # f64 baseline
    b = load_checkpoint(sys.argv[2])  # f32_regret
    print(f"f64 : iters={a['iters_done']} cum_rows={len(a['cum'])} strat_rows={len(a['strat'])}")
    print(f"f32 : iters={b['iters_done']} cum_rows={len(b['cum'])} strat_rows={len(b['strat'])}")

    ka, kb = set(a["strat"]), set(b["strat"])
    shared = ka & kb
    only_a, only_b = ka - kb, kb - ka
    print(
        f"strat keys: shared={len(shared)} only_f64={len(only_a)} only_f32={len(only_b)} "
        f"(jaccard={len(shared) / max(1, len(ka | kb)):.6f})"
    )

    l1s = []
    worst = (0.0, None)
    # edge-bearing collapse: an action carrying served prob > prob_floor whose prob shifts > shift_tol.
    prob_floor, shift_tol = 1e-2, 1e-3
    collapses = 0
    collapse_examples = []
    for k in shared:
        sa = avg_strategy(a["strat"][k])
        sb = avg_strategy(b["strat"][k])
        l1 = sum(abs(x - y) for x, y in zip(sa, sb))
        l1s.append(l1)
        if l1 > worst[0]:
            worst = (l1, k)
        for x, y in zip(sa, sb):
            if (x > prob_floor or y > prob_floor) and abs(x - y) > shift_tol:
                collapses += 1
                if len(collapse_examples) < 5:
                    collapse_examples.append((k.hex(), round(x, 5), round(y, 5)))
                break

    l1s.sort()
    n = len(l1s)
    if n == 0:
        print("NO SHARED KEYS — cannot compare (did both runs train?)")
        sys.exit(1)

    def pct(q):
        return l1s[min(n - 1, int(q * n))]

    median = pct(0.50)
    p90, p99 = pct(0.90), pct(0.99)
    mx = l1s[-1]
    mean = sum(l1s) / n
    print("\n=== per-infoset average-strategy L1 delta (f32_regret vs f64) ===")
    print(f"  n={n}  mean={mean:.3e}  median={median:.3e}  p90={p90:.3e}  p99={p99:.3e}  max={mx:.3e}")
    print(f"  worst-infoset L1={worst[0]:.3e} key={worst[1].hex() if worst[1] else None}")
    print(f"  edge-bearing collapses (prob>{prob_floor} shifted >{shift_tol}): {collapses}")
    for ex in collapse_examples:
        print(f"      e.g. key={ex[0]} f64_p={ex[1]} f32_p={ex[2]}")

    # DIAGNOSTIC ONLY — no quality PASS/FAIL (see the module docstring). A large L1 here is EXPECTED and
    # benign: it measures chaotic re-equilibration in a no-unique-equilibrium game, not degradation. The
    # quality verdict lives in scripts/f32_regret_lbr_ab.py (LBR exploitability CI-overlap).
    print("\n=== interpretation ===")
    if median < 1e-6:
        print("  median L1 ~ 0 -> runs are (near-)identical: either f64-vs-f64 control, or no f32 divergence.")
    else:
        print(f"  median L1 = {median:.3e} -> the two runs landed on DIFFERENT strategy profiles (chaotic")
        print("  seed-sensitivity of 6-max ES-MCCFR). This is NOT a quality signal — see f32_regret_lbr_ab.py")
        print("  for the exploitability comparison, which is the real gate.")
    sys.exit(0)


if __name__ == "__main__":
    main()
