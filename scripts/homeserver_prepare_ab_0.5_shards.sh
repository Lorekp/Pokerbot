#!/usr/bin/env bash
# After 1.0% AB solves finish: build label sets and run SFT/KL gate locally (CUDA).
set -euo pipefail
ROOT="."
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT"

WORK=data/solver/exploitability_ab
mkdir -p "$WORK/shards_0.5" "$WORK/shards_1.0"

python3 - <<'PY'
from pathlib import Path
import shutil
from pokerbot.solver.label_completion import spot_key_hash

root = Path(".")
keys = [l.strip() for l in (root / "logs/ab_spot_keys_20.txt").read_text().splitlines() if l.strip()]
src_dir = root / "data/solver/train_prod6000_shards"
out05 = root / "data/solver/exploitability_ab/shards_0.5"
out10 = root / "data/solver/exploitability_ab/shards_1.0"
copied = 0
for sk in keys:
    h = spot_key_hash(sk)
    src = src_dir / f"{h}.parquet"
    if src.is_file():
        shutil.copy2(src, out05 / f"{h}.parquet")
        copied += 1
print(f"copied_0.5pct_shards_from_prod6000={copied}/{len(keys)}")
PY

echo "Run KL gate locally after 1.0% solves complete (homeserver GPU driver too old for SFT)."
