#!/usr/bin/env python3
"""Export the S1-style subgame corpus + deployed blueprint/config for the NATIVE bench harness.

READ-ONLY against the main tree (loads pickles + blueprint, builds the same corpus as
scripts/track2_s1_ksweep.py); writes binary bench inputs into THIS worktree only:
  bench_data/blueprint.bin  [u64 n] then n * (16B key + 11 * f64 LE row)
  bench_data/states.bin     [u64 n] then n * (u8 cls, u8 hero, 291 * i64 LE flat state)
  bench_data/config.json    masks + knobs + state ids (cls 0=river 1=live_probe 2=mw_flop)
"""
import json
import os
import struct
import sys
from pathlib import Path

MAIN = Path(r".")
OUT = Path(__file__).resolve().parent.parent / "bench_data"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "scripts"))
os.chdir(MAIN)  # blueprint auto-detect + relative paths behave like the S1 runner

import track2_s1_ksweep as t2  # noqa: E402  (sets the deployed EMD serve env at import)
import numpy as np  # noqa: E402


def main():
    snap = t2.load_snapshot()
    print(f"snapshot infosets: {len(snap):,}")
    river = t2.build_river_corpus()
    live = t2.build_live_probe_corpus(15)
    mw = t2.build_multiway_flop_corpus(snap, want=12, max_hands=600, seed=77)
    corpus = river + live + mw
    print(f"corpus: {len(corpus)} (river={len(river)} live={len(live)} mw={len(mw)})")

    sv = t2.build_solver(snap)
    adapter = sv.adapter
    from pokerbot.search import native_solver as ns
    from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS

    allowed_pre = [bool(x) for x in adapter._allowed_pre]
    allowed_post = [bool(x) for x in adapter._allowed_post]

    # blueprint.bin — exactly the dict the SubgameSolver ctor receives.
    with open(OUT / "blueprint.bin", "wb") as f:
        items = list(adapter.snapshot.items())
        f.write(struct.pack("<Q", len(items)))
        for k, v in items:
            kb = bytes(k)
            assert len(kb) == 16
            row = np.asarray(v, dtype=np.float64).reshape(-1)[:11]
            f.write(kb)
            f.write(struct.pack("<11d", *[float(x) for x in row]))
    print(f"blueprint.bin: {len(items):,} rows")

    # states.bin — the exact flat buffers native SubgameSolver.solve receives.
    cls_map = {"river": 0, "live_probe": 1, "mw_flop": 2}
    ids = []
    with open(OUT / "states.bin", "wb") as f:
        f.write(struct.pack("<Q", len(corpus)))
        for it in corpus:
            root = it["root"]
            env, hero = root.env, int(root.hero_seat)
            flat = ns._to_flat(env)
            assert flat.dtype == np.int64 and flat.size == ns._FLAT_LEN
            f.write(struct.pack("<BB", cls_map[it["cls"]], hero))
            f.write(flat.tobytes())
            ids.append(it["id"])
    print(f"states.bin: {len(corpus)} states, flat_len={ns._FLAT_LEN}")

    cfg = {
        "flat_len": int(ns._FLAT_LEN),
        "ids": ids,
        "allowed_pre": allowed_pre,
        "allowed_post": allowed_post,
        "bias_mult": float(adapter.bias_mult),
        "bias_floor": float(adapter.bias_floor),
        "num_buckets": int(_POSTFLOP_NUM_BUCKETS),
        "leaf_max_playout_depth": int(sv.config.leaf.max_playout_depth),
        "emd_frozen_path": str(MAIN / "data" / "abstraction" / "emd_frozen_flop_turn_160_r400.emdfrzn"),
        "emd_miss_made_hand": True,
        # deployed solve knobs (engine_ab_h2h._DEPLOYED_BASELINE resolved through the solver config)
        "solve": {
            "max_iterations": 2000,
            "max_depth": int(sv.config.max_depth),
            "blueprint_mix": float(sv.config.blueprint_mix),
            "rollout_samples": int(sv.config.leaf.rollout_samples),
            "resample_board": bool(getattr(sv.config.leaf, "resample_board", True)),
            "linear_averaging": bool(sv.config.linear_averaging),
            "repair_deck": bool(sv.config.repair_deck),
            "two_sided": bool(sv.config.two_sided),
            "decouple_sampler": bool(sv.config.decouple_sampler),
            "range_aware": bool(sv.config.range_aware),
            "range_uniform_floor": float(sv.config.range_uniform_floor),
            "range_reach_full": bool(sv.config.range_reach_full),
            "ensemble_streams": int(sv.config.ensemble_streams),
            "leaf_cache_worlds": int(sv.config.leaf_cache_worlds),
            "avg_skip_frac": float(sv.config.avg_skip_frac),
            "leaf_biased_continuations": bool(sv.config.leaf_biased_continuations),
            "leaf_bias_mult": float(sv.config.leaf_bias_mult),
            "serve_final_iterate": bool(sv.config.serve_final_iterate),
            "root_unmask": 0,
        },
    }
    with open(OUT / "config.json", "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"wrote {OUT}/config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
