#!/usr/bin/env python3
import json
from pathlib import Path
from pokerbot.solver.label_completion import spot_key_hash

root = Path(__file__).resolve().parents[1]
keys = [l.strip() for l in (root / "logs/ab_spot_keys_20.txt").read_text().splitlines() if l.strip()]
manifest = json.loads((root / "data/solver/manifest_prod6000.json").read_text())
manifest_keys = set(manifest.get("spot_keys", []))
shard_dir = root / "data/solver/train_prod6000_shards"
for k in keys[:5]:
    h = spot_key_hash(k)
    print(k, h, k in manifest_keys, (shard_dir / f"{h}.parquet").is_file())
for mk in sorted(manifest_keys):
    if "3sKc6h" in mk and "sb_vs_bb" in mk:
        print("manifest", mk, spot_key_hash(mk), (shard_dir / f"{spot_key_hash(mk)}.parquet").is_file())
print("manifest_sample", list(sorted(manifest_keys))[:3])
print("manifest_total", len(manifest_keys))
