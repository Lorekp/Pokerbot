# Local SD-CFR Leduc short ruler — single process, 6 OpenMP threads.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:OMP_NUM_THREADS = "6"
$env:OPENBLAS_NUM_THREADS = "6"
$env:MKL_NUM_THREADS = "6"
$env:JAX_PLATFORMS = "cpu"
$env:PYTHONPATH = $Root

$ckptDir = "checkpoints/sd_cfr_leduc_local_v3"
New-Item -ItemType Directory -Force -Path $ckptDir, logs | Out-Null

python scripts/train_sd_cfr_leduc.py `
  --iterations 40 `
  --traversals 32 `
  --seed 2 `
  --checkpoint-dir $ckptDir `
  --checkpoint-every 10 `
  --output "$ckptDir/final.pkl" `
  2>&1 | Tee-Object -FilePath logs/sd_cfr_leduc_local_v3.log
