#!/usr/bin/env python3
"""Build full HU showdown table and write score2 + toolchain sidecar for cross-OS repro."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aof_equity import warmup
from aof_equity_showdown import build_showdown_table_v2, load_canonical_keys
from scripts.hu_showdown_toolchain import collect_toolchain_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    warmup()
    keys = load_canonical_keys()
    _, score2, _ = build_showdown_table_v2(keys, threads=args.threads)
    score2 = np.asarray(score2, dtype=np.uint32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "score2.uint32.npy", score2)
    sha = hashlib.sha256(score2.tobytes()).hexdigest()
    meta = {
        "n_keys": int(len(score2)),
        "score2_sha256": sha,
        **collect_toolchain_meta(portable_build=True),
    }
    (args.out_dir / "score2.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"score2_sha256={sha}")


if __name__ == "__main__":
    main()
