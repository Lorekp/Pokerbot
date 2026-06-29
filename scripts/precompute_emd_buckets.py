#!/usr/bin/env python3
"""Fit the potential-aware EMD postflop bucketer centroids for POKERBOT_POSTFLOP_MODE=emd.

The deployed blueprint buckets postflop by MADE-HAND RANK only (POKERBOT_POSTFLOP_MODE=fast), which
ignores draw equity — a flush draw is bucketed by its weak current rank, so the blueprint can't size or
continue it correctly. This fits the draw-aware refinement: cluster (hole, board) spots by their
hand-strength DISTRIBUTION under Earth Mover's Distance, so draws and made hands of equal current rank
land in different buckets (postflop_bucketer.EMDBucketer). Run offline (AWS box) BEFORE a retrain; point
training + serve at the resulting table via POKERBOT_POSTFLOP_EMD_TABLE.

    python scripts/precompute_emd_buckets.py --num-buckets 200 --samples 60000 \
        --n-runouts 200 --out data/abstraction/emd_buckets_200.json

Sampling note: centroids are fit over a representative SAMPLE of legal (hole, board) flop+turn spots
(river collapses to a made-hand spike, no potential left). More samples + runouts = sharper clusters.
This is an abstraction precompute, not training (no game tree); it is the cloud-scale analogue of the
WEVA/marginals precomputes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# K-cap fail-fast guard. K=500 overfit the EMD bucketer on run 2026-06-25
# (real -140 SB regression). Hard cap K_MAX=200 across the EMD pipeline.
# EMD_K_OVERRIDE env var CANNOT bypass the cap (rejection is at import time
# so it fires before numpy / sklearn / the k-means fit, fail-fast in CI).
_K_MAX = 200
_emd_k_override = os.environ.get("EMD_K_OVERRIDE")
if _emd_k_override is not None and _emd_k_override != "":
    # The override VALUE is echoed only via f-string concatenation (no shell, no
    # os.system, no eval); a malicious env var cannot delete a sentinel file.
    sys.stderr.write(
        f"FATAL: EMD_K_OVERRIDE is not honored: env var value is {_emd_k_override!r}. "
        f"Production K is hard-capped at {_K_MAX}. "
        f"The K=500 incident (run 2026-06-25) overfit the EMD bucketer "
        f"and produced a -140 SB regression. To bump the cap, edit this file.\n"
    )
    sys.exit(1)

import numpy as np  # noqa: E402  -- must come AFTER the EMD_K_OVERRIDE guard

from pokerbot.encoding.postflop_bucketer import EMDBucketer, strength_histogram  # noqa: E402


def _sample_spot(rng: np.random.Generator, n_board: int):
    """A random legal (hole_ids, board_ids): 2 hole + n_board board, distinct, card_id = rank*4+suit."""
    cards = rng.choice(52, size=2 + n_board, replace=False)
    return tuple(int(c) for c in cards[:2]), tuple(int(c) for c in cards[2:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-buckets", type=int, default=200)
    ap.add_argument("--samples", type=int, default=60000, help="(hole,board) spots to fit over")
    ap.add_argument("--n-bins", type=int, default=20, help="strength-histogram resolution")
    ap.add_argument("--n-runouts", type=int, default=200, help="board completions sampled per spot")
    ap.add_argument("--kmeans-iters", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-max", type=int, default=_K_MAX, help=f"hard cap on num_buckets (default {_K_MAX})")
    ap.add_argument("--out", default="data/abstraction/emd_buckets_200.json")
    a = ap.parse_args()
    # Post-parse K-cap guard. K=500 overfit the bucketer on 2026-06-25 (-140 SB).
    if a.num_buckets > a.k_max:
        sys.stderr.write(
            f"FATAL: --num-buckets={a.num_buckets} exceeds --k-max={a.k_max}. "
            f"K_MAX={_K_MAX} is the hard cap. K=500 is BANNED. Use K<=200 for production.\n"
        )
        sys.exit(1)

    rng = np.random.default_rng(a.seed)
    # Mix flop (3) and turn (4) spots; river has no potential so it is excluded from fitting.
    n_each = a.samples // 2
    t0 = time.time()
    hists = np.empty((2 * n_each, a.n_bins), dtype=np.float64)
    for i in range(2 * n_each):
        n_board = 3 if i < n_each else 4
        hole, board = _sample_spot(rng, n_board)
        hists[i] = strength_histogram(hole, board, n_bins=a.n_bins, n_runouts=a.n_runouts,
                                      seed=int(rng.integers(1 << 30)))
        if (i + 1) % 5000 == 0:
            print(f"  histograms {i + 1}/{2 * n_each}  ({(time.time() - t0):.0f}s)", flush=True)

    print(f"fitting {a.num_buckets} EMD buckets over {hists.shape[0]} spots…", flush=True)
    bk = EMDBucketer(num_buckets=a.num_buckets, n_bins=a.n_bins, n_runouts=a.n_runouts)
    bk.fit(hists, iters=a.kmeans_iters, seed=a.seed)
    bk.save(a.out)
    print(f"wrote {a.out}  ({(time.time() - t0):.0f}s total). "
          f"Set POKERBOT_POSTFLOP_MODE=emd POKERBOT_POSTFLOP_EMD_TABLE={a.out} for train+serve.")


if __name__ == "__main__":
    main()
