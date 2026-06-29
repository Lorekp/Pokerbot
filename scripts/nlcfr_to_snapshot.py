#!/usr/bin/env python3
"""Convert a native NLCFR001 trainer checkpoint (binary blob from examples/ab_train.rs or the production
trainer) into a raw snapshot pickle (dict[bytes16 -> np.ndarray of the normalized average strategy]) that
`scripts/eval/validate_blueprint_native.py` accepts. The average strategy is strat_sum row-normalized
(uniform fallback for an all-zero row) — identical to `trainer::snapshot`.

usage: nlcfr_to_snapshot.py <in.bin> <out.pkl>
"""
import pickle
import sys

import numpy as np

from f32_regret_strategy_delta import NUM_ACTIONS, avg_strategy, load_checkpoint  # type: ignore


def main():
    if len(sys.argv) < 3:
        print("usage: nlcfr_to_snapshot.py <in.bin> <out.pkl>", file=sys.stderr)
        sys.exit(2)
    ckpt = load_checkpoint(sys.argv[1])
    snap = {}
    for k, row in ckpt["strat"].items():
        snap[k] = np.asarray(avg_strategy(row), dtype=np.float64)
    with open(sys.argv[2], "wb") as f:
        pickle.dump(snap, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {sys.argv[2]}: {len(snap):,} infosets (iters_done={ckpt['iters_done']})")


if __name__ == "__main__":
    main()
