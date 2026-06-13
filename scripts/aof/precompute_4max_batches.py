"""Precompute the 4-max AoF Monte-Carlo deal-batch payoff tensors to disk.

The 4-max exploit/EV path (`aof_cfr.exploit_router._router_batch`) builds an 8000-deal MC payoff
tensor per stack depth on first use — ~194s each. That is the entire 4-max "warmup" cost. The
tensor is deterministic (fixed seed) and pure numpy, so we precompute it ONCE here and write
`data/aof/cfr_base/mc_batch_4p_d{depth}_n{n_deals}_s{seed}.npz`; `_router_batch` then loads it in
~tens of ms instead of recomputing. After running this, `AoFEngine.warmup()` (and the first live
4-max hand) is effectively instant.

Run once (offline, ~3 min/depth -> ~22 min for the default 7 depths):
    python scripts/aof/precompute_4max_batches.py
    python scripts/aof/precompute_4max_batches.py --depths 8 10 --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aof_cfr.exploit_router import batch_disk_path  # noqa: E402
from aof_cfr.fp_4max_vec import precompute_payoffs, sample_deals  # noqa: E402
from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default  # noqa: E402

# The infoset depth buckets the engine uses; precompute these so live effective stacks
# (snapped to the nearest bucket) always hit a precomputed batch.
DEFAULT_DEPTHS = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
N_DEALS = 8000   # must match exploit_router.route_4max / engine.py defaults
SEED = 12345     # ditto


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depths", type=float, nargs="*", default=list(DEFAULT_DEPTHS),
                    help="stack depths (bb) to precompute")
    ap.add_argument("--n-deals", type=int, default=N_DEALS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    ap.add_argument("--game-model", type=int, default=None,
                    help="terminal chip model to embed (default: the shipped manifest's)")
    ap.add_argument("--flat-fee-bb", type=float, default=None,
                    help="showdown fee to embed in the payoffs (default: the shipped "
                         "manifest's flat_fee_bb, else 0)")
    args = ap.parse_args()

    import os
    gm = args.game_model if args.game_model is not None else game_model_default()
    fee = args.flat_fee_bb if args.flat_fee_bb is not None else flat_fee_default()
    os.environ["AOF_FLAT_FEE_BB"] = str(fee)   # pin the payoff layer's fee for this build
    out_dir = batch_disk_path(8.0, args.n_deals, args.seed).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[precompute] {len(args.depths)} depth(s) -> {out_dir}  "
          f"(n_deals={args.n_deals}, seed={args.seed}, game_model={gm}, flat_fee={fee})")

    for depth in args.depths:
        path = batch_disk_path(depth, args.n_deals, args.seed)
        if path.is_file() and not args.force:
            data = np.load(path, allow_pickle=False)
            old_gm = int(data["game_model"]) if "game_model" in data else 1
            old_fee = float(data["flat_fee"]) if "flat_fee" in data else 0.0
            if old_gm == gm and abs(old_fee - fee) < 1e-9:
                print(f"  d{depth:g}: exists (game model {gm}, fee {fee}), skip ({path.name})")
                continue
            print(f"  d{depth:g}: exists but wrong game model/fee -> rebuilding")
        t = time.time()
        buckets, holes = sample_deals(args.n_deals, args.seed)
        payoffs = precompute_payoffs(holes, float(depth), game_model=gm)
        # atomic write: tmp then replace. Tmp name MUST end in .npz or np.savez appends it.
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, buckets=buckets, payoffs=payoffs, game_model=np.int64(gm),
                            flat_fee=np.float64(fee))
        tmp.replace(path)
        print(f"  d{depth:g}: built in {time.time()-t:.0f}s -> {path.name} "
              f"({path.stat().st_size/1e6:.1f} MB, payoffs {payoffs.shape})")

    print("[precompute] done. _router_batch will now load these instead of recomputing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
