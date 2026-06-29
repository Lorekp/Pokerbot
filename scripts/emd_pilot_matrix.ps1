# Dry-run helper for fixed-budget EMD pilot CFR runs.
#
# This is the second filter after scripts/emd_abstraction_sweep.py:
#   1. sweep K/runouts cheaply on held-out EMD inertia + occupancy;
#   2. run one or more fixed-budget CFR pilots for the survivors;
#   3. only then launch the multi-day marathon.
#
# Default is DRY-RUN. For safety, -Launch requires exactly one K in -KGrid.
param(
  [string]$KGrid = "120,160,200",
  [string]$RunPrefix = "run_emd_pilot",
  [string]$Stamp = "",
  [int]$Iters = 220000,
  [int]$NDeals = 200,
  [int]$NRunouts = 200,
  [int]$CheckpointEvery = 5000,
  [int]$MaxHours = 30,
  [int]$EvalSeeds = 64,
  [int]$EvalHands = 4000,
  [int]$MaxLaunch = 12,
  [int]$MinRelaunchGapSec = 600,
  [int]$QuotaDrainSec = 600,
  [switch]$OnDemand,
  [switch]$Launch
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not $Stamp) { $Stamp = Get-Date -Format "yyyyMMdd_HHmm" }

$ks = @()
foreach ($part in ($KGrid -split ",")) {
  $p = $part.Trim()
  if ($p) { $ks += [int]$p }
}
if (-not $ks.Count) { throw "-KGrid must contain at least one K" }
if (($ks | Measure-Object -Maximum).Maximum -gt 200) {
  throw "K > 200 is banned in this pipeline. K=500 overfit; use the abstraction sweep before raising the cap."
}
if ($Launch -and $ks.Count -ne 1) {
  throw "-Launch is allowed only with exactly one K. Dry-run the full matrix, then launch the chosen K with -KGrid <K> -Launch."
}

function Show-Cmd([string[]]$ArgsList, [string[]]$WatcherArgsList) {
  Write-Host "Command:" -ForegroundColor Cyan
  Write-Host ("powershell " + ($ArgsList -join " "))
  Write-Host "Watcher:" -ForegroundColor Cyan
  Write-Host ("powershell " + ($WatcherArgsList -join " "))
}

foreach ($k in $ks) {
  $centroids = "data/abstraction/emd_buckets_${k}.json"
  $run = "${RunPrefix}_k${k}_${Stamp}"
  $ship = "emd_k${k}_pilot_${Stamp}"
  Write-Host ""
  Write-Host "=== EMD pilot K=$k  run=$run ===" -ForegroundColor Cyan
  if (-not (Test-Path $centroids)) {
    Write-Host "Missing $centroids" -ForegroundColor Yellow
    Write-Host "Create it first, for example:"
    Write-Host ("  python scripts\precompute_emd_buckets.py --num-buckets {0} --samples 60000 --n-runouts {1} --out {2}" -f $k, $NRunouts, $centroids)
    continue
  }
  $args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\emd_pipeline_babysitter_v2.ps1",
    "-Run", $run,
    "-Iters", "$Iters",
    "-NDeals", "$NDeals",
    "-NumBuckets", "$k",
    "-Centroids", $centroids,
    "-NRunouts", "$NRunouts",
    "-CheckpointEvery", "$CheckpointEvery",
    "-MaxHours", "$MaxHours",
    "-CrateSrc", "runs/bp_crate_emd/battery_native",
    "-ConvergeLogOnly",
    "-ShipName", $ship,
    "-EvalSeeds", "$EvalSeeds",
    "-EvalHands", "$EvalHands",
    "-MaxLaunch", "$MaxLaunch",
    "-MinRelaunchGapSec", "$MinRelaunchGapSec",
    "-QuotaDrainSec", "$QuotaDrainSec",
    "-Spotpick"
  )
  if ($OnDemand) { $args += "-OnDemand" }
  $watchArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\watch_emd_pipeline_v2.ps1",
    "-Run", $run,
    "-Iters", "$Iters",
    "-NumBuckets", "$k",
    "-Centroids", $centroids,
    "-NRunouts", "$NRunouts",
    "-ShipName", $ship,
    "-ExpectTableMB", "153",
    "-Refresh", "60"
  )
  Show-Cmd $args $watchArgs
  if ($Launch) {
    Write-Host ""
    Write-Host "Launching one selected pilot..." -ForegroundColor Green
    & powershell @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
}

if (-not $Launch) {
  Write-Host ""
  Write-Host "DRY RUN only. To launch one pilot: re-run with -KGrid <one K> -Launch." -ForegroundColor Yellow
}
