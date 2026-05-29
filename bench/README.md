# HU showdown baseline benchmarks (Phase 1)

Trustworthy measurements for the native rewrite. See
[`docs/native_rewrite/01_baseline.md`](../docs/native_rewrite/01_baseline.md).

## Commands

```bash
# Regenerate fixtures (deterministic, seed=0)
python -m bench.datasets.generate --out-dir bench/datasets/fixtures --seed 0

# Micro — isolated eval7 / single exact key (NOT build ETA)
python -m bench.micro --json-out docs/native_rewrite/baseline/micro_run.json

# Meso — primary ETA (s/key on fixed key chunk)
python -m bench.meso --fixture stratified --repeat 2
python -m bench.meso --fixture smoke --repeat 2

# Macro — build-driver dedupe overhead
python -m bench.macro --dedupe-loop --keys 1000

# Profile
python -m cProfile -o docs/native_rewrite/baseline/cprofile_stratified.prof -m bench.meso --fixture stratified --repeat 1
```
