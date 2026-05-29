# OMPEval offline CLI + Phase 2 native oracle

Build-only helpers for combo-exact HU showdown tables. **Not used in the training hot path.**

## Prerequisites (Windows)

- Visual Studio 2022 Build Tools (C++ workload)
- CMake 3.16+ (bundled with VS Build Tools)
- Python 3.11+ with development headers
- `pip install pybind11`

## Build (OMPEval + hu_table_native)

```powershell
cd tools/ompeval_cli
git submodule update --init vendor/OMPEval
.\build.ps1
```

Outputs:

- `build/Release/enumerate_hu_equity.exe` — legacy single-matchup float CLI
- `build/Release/hu_table_native.cp314-win_amd64.pyd` — pybind oracle + batch `build_scores`
- `build/Release/hu_table_build.exe` — Phase 3 one-shot score2 builder
- `build/Release/bench_native.exe` — Google Benchmark scaling harness

Verify:

```powershell
python -c "from tools.ompeval_cli.native import native_backend_available; print(native_backend_available())"
```

## Sentinel bootstrap

Gate A (AcAd vs KhKd) score2 constant:

```bash
python scripts/bootstrap_sentinel_score2.py
```

Hardcoded in `hu_table_native.cpp` and `tools/ompeval_cli/native.py`.

## Python import

```python
from tools.ompeval_cli.native import load_native_module, native_backend_available, build_scores

mod = load_native_module()
score2 = mod.enumerate_score2_one(key)
scores = build_scores(keys, threads=4)
```

## Phase 3 CLI

```powershell
.\build\Release\hu_table_build.exe `
  --keys ..\..\bench\datasets\fixtures\full_keys.uint64.npy `
  --out build\full_scores.bin `
  --threads 4 `
  --format raw
```

Startup runs sentinel self-test; output is raw little-endian `uint32 score2[]` in key order.

Card ids 0..51 (phevaluator / pokerkit convention: `rank*4+suit`).

## Tests

```bash
python -m pytest tests/test_hu_table_native.py tests/test_hu_table_build_cli.py -q
python -m pytest tests/test_hu_table_native.py tests/test_hu_table_build_cli.py -q -m slow
```

## Benchmark

```bash
python -m bench.meso_native --fixture stratified --repeat 6 --scaling-sweep
python -m bench.macro_native --fixture stratified --threads 4
```
