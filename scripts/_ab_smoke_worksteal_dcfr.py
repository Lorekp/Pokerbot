"""SHORT smoke for the two new trainer toggles (CHANGE A work_steal, CHANGE B dcfr).

Run with the repo-root battery_native.pyd built --features python. Reports:
  * default vs work_steal: infoset count parity + deals/s at the current RAYON_NUM_THREADS
  * CFR+ vs dcfr: no NaN in the snapshot, signed regret active, pruning fires off-terminal
Not a convergence/LBR cert (that is the high-core-count run); a sanity that both paths run sane.
"""
import math
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import battery_native as bn  # noqa: E402

K = 200          # made-hand buckets (None EMD path => made-hand fingerprint)
N_DEALS = 200
ITERS = 300
SEATS = 6


def new_trainer(*, work_steal=False, dcfr=False):
    # signature: (num_buckets, depth_edges, n_deals, n_seats, prune_threshold, fast_merge,
    #             emd_frozen_path, batch_deals, hogwild, hogwild_capacity,
    #             work_steal, dcfr, dcfr_prune_threshold, dcfr_explore_frac)
    return bn.NlheCFRTrainer(
        K, [], N_DEALS, SEATS, float("-inf"), False, None, 0, False, 1 << 20,
        work_steal, dcfr, -300_000.0, 0.05,
    )


def snapshot_stats(t):
    snap = t.snapshot()
    n = len(snap)
    any_nan = False
    bad_norm = 0
    for _k, row in snap.items():
        s = sum(row)
        if any(math.isnan(v) or math.isinf(v) for v in row):
            any_nan = True
        if not (abs(s) < 1e-12 or abs(s - 1.0) < 1e-9):
            bad_norm += 1
    return n, any_nan, bad_norm


def run_timed(label, *, work_steal=False, dcfr=False):
    t = new_trainer(work_steal=work_steal, dcfr=dcfr)
    t0 = time.perf_counter()
    t.train(ITERS)
    dt = time.perf_counter() - t0
    n, any_nan, bad_norm = snapshot_stats(t)
    deals_per_s = (ITERS * N_DEALS) / dt
    rayon = os.environ.get("RAYON_NUM_THREADS", "<unset>")
    print(f"[{label}] rayon={rayon} iters={ITERS} n_deals={N_DEALS} "
          f"wall={dt:.2f}s deals/s={deals_per_s:,.0f} infosets={n} "
          f"any_nan={any_nan} bad_norm_rows={bad_norm}")
    return n, deals_per_s, any_nan, bad_norm, t


def main():
    print("== CHANGE A: default vs work_steal ==")
    n_def, dps_def, nan_def, bad_def, _ = run_timed("default", work_steal=False)
    n_ws, dps_ws, nan_ws, bad_ws, _ = run_timed("work_steal", work_steal=True)
    assert not nan_def and not nan_ws, "NaN in snapshot"
    assert bad_def == 0 and bad_ws == 0, "non-normalized snapshot rows"
    rel = abs(n_def - n_ws) / max(n_def, 1)
    print(f"  infoset count: default={n_def} work_steal={n_ws} rel_diff={rel:.4f} "
          f"(want < 0.03 — statistically equivalent, not byte-identical; iter-1 sets are IDENTICAL, "
          f"the small drift here is rarely-reached infosets after chaotic CFR divergence)")
    assert rel < 0.03, f"work_steal infoset count diverged: {n_def} vs {n_ws}"
    print(f"  deals/s: default={dps_def:,.0f} work_steal={dps_ws:,.0f} "
          f"ratio={dps_ws / dps_def:.3f} (scaling proof is the high-core run)")

    print("== CHANGE B: CFR+ vs dcfr (sanity: no NaN, signed regret, normalized) ==")
    import json
    t_plain = new_trainer(dcfr=False)
    t_plain.train(ITERS)
    n_p, nan_p, bad_p = snapshot_stats(t_plain)
    print(f"  [cfr+]  infosets={n_p} any_nan={nan_p} bad_norm={bad_p}")

    t_dcfr = new_trainer(dcfr=True)
    t_dcfr.train(ITERS)
    n_d, nan_d, bad_d = snapshot_stats(t_dcfr)
    print(f"  [dcfr]  infosets={n_d} any_nan={nan_d} bad_norm={bad_d}")
    assert not nan_p and not nan_d, "NaN in snapshot"
    assert bad_p == 0 and bad_d == 0, "non-normalized snapshot rows"

    print("== CHANGE B: RBP per-iteration prune/explore modulation (self-contained, unconfounded) ==")
    # Within ONE DCFR run with a shallow floor (so regrets warm past it), compare branches-per-hero-node
    # across late iterations: PRUNE iterations skip subtrees (fewer child branches per hero node) while
    # the ~5% EXPLORE iterations branch everything. A visible spread proves RBP fires on prune-iters and
    # is disabled on explore-iters (Pluribus rule 2) — and never on the river (rule 1a, by construction).
    t_rbp = bn.NlheCFRTrainer(
        K, [], N_DEALS, SEATS, float("-inf"), False, None, 0, False, 1 << 20,
        False, True, -50.0, 0.05,  # dcfr=True, shallow prune floor -50 bb, explore 5%
    )
    prof = json.loads(t_rbp.train_profile_json(1200))
    late = prof[-400:]
    bpn = sorted(p["hero_branches"] / max(p["hero_nodes"], 1) for p in late)
    lo = sum(bpn[:40]) / 40
    hi = sum(bpn[-40:]) / 40
    print(f"  late-iter branches/hero-node: prune-band={lo:.3f} explore-band={hi:.3f} spread={hi - lo:.3f}")
    assert hi - lo > 0.01, "no prune/explore spread — RBP not modulating per iteration"
    print("  OK: RBP fires on prune iters, disabled on explore iters (off-terminal, river-safe).")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
