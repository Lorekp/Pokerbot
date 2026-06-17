# Phase A CFR base — run heavy jobs one at a time.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "."
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMBA_NUM_THREADS = "1"
# OMP_NUM_THREADS set per-step below (native trio uses all physical cores)
$log = "data/aof/cfr_base/sequential_run.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

Log "=== Phase A sequential pipeline start (FP pivot) ==="

Log "Step 0: build + validate bucket row 0 (quality gate before full tensor)"
$env:OMP_NUM_THREADS = [string]([Math]::Max(1, [Environment]::ProcessorCount / 2))
python scripts/build_3way_eq_tensor_npz.py --only-row 0 --out data/aof/cfr_base/trio_row0.npz --mc-boards 2048 --validate-samples 15
if ($LASTEXITCODE -ne 0) { Log "FAILED row 0 build/validate"; exit $LASTEXITCODE }

Log "Step 0b: hard-pre audit_terminal_ev"
python -c "from aof_cfr.pushfold_payoffs import audit_terminal_ev; audit_terminal_ev(); print('audit_terminal_ev OK')"
if ($LASTEXITCODE -ne 0) { Log "FAILED terminal audit"; exit $LASTEXITCODE }

Log "Step 1/5: build trio_eq_tensor.npz (adaptive budget=2048, native OpenMP per row)"
$env:OMP_NUM_THREADS = [string]([Math]::Max(1, [Environment]::ProcessorCount / 2))
python scripts/build_3way_eq_tensor_npz.py --mc-boards 2048 --chunk-rows 1
if ($LASTEXITCODE -ne 0) { Log "FAILED step 1"; exit $LASTEXITCODE }

Log "Step 2/5: validate tensor spot-checks"
python scripts/validate_trio_tensor.py --tensor data/aof/cfr_base/trio_eq_tensor.npz --samples 15
if ($LASTEXITCODE -ne 0) { Log "FAILED step 2"; exit $LASTEXITCODE }

Log "Step 3/5: solve 3-max FP"
python scripts/solve_aof_cfr_base.py --only 3 --solver fp --fp-rounds 200
if ($LASTEXITCODE -ne 0) { Log "FAILED step 3"; exit $LASTEXITCODE }

Log "Step 4/5: solve 4-max FP"
python scripts/solve_aof_cfr_base.py --only 4 --solver fp --fp-rounds 20 --fp-4-br-samples 2000
if ($LASTEXITCODE -ne 0) { Log "FAILED step 4"; exit $LASTEXITCODE }

Log "Step 5/5: verify artifacts"
python scripts/verify_aof_cfr_base.py --base-dir data/aof/cfr_base
if ($LASTEXITCODE -ne 0) { Log "FAILED step 5"; exit $LASTEXITCODE }

Log "=== Phase A sequential pipeline complete ==="
