# Fresh SD-CFR Leduc ruler v4 — 40 iters, ckpt 10/20/40, single process, 6 threads.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:OMP_NUM_THREADS = "6"
$env:OPENBLAS_NUM_THREADS = "6"
$env:MKL_NUM_THREADS = "6"
$env:JAX_PLATFORMS = "cpu"
$env:PYTHONPATH = $Root

New-Item -ItemType Directory -Force -Path logs | Out-Null

python scripts/run_sd_cfr_leduc_ruler.py 2>&1 | Tee-Object -FilePath logs/sd_cfr_leduc_ruler_v4.log
