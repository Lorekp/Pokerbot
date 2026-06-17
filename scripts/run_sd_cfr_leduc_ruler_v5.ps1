# v5: sample-starvation test — 6× full deal enumeration (1080 visits/iter) vs v4's 32 random.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:OMP_NUM_THREADS = "6"
$env:OPENBLAS_NUM_THREADS = "6"
$env:MKL_NUM_THREADS = "6"
$env:JAX_PLATFORMS = "cpu"
$env:PYTHONPATH = $Root

New-Item -ItemType Directory -Force -Path logs | Out-Null

python scripts/run_sd_cfr_leduc_ruler.py `
  --enumerate-deals `
  --deal-sweeps 6 `
  --iterations 40 `
  --seed 2 `
  --checkpoint-dir checkpoints/sd_cfr_leduc_ruler_v5_highsample `
  --checkpoint-every 10 `
  2>&1 | Tee-Object -FilePath logs/sd_cfr_leduc_ruler_v5_highsample.log
