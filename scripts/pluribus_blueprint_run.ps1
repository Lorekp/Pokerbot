# Dry-run/launch wrapper for a Pluribus-level 6-max blueprint training run.
#
# Default is DRY-RUN: it prints the exact command. Add -Launch to spend cloud
# money. Production recommendation depends on the experiment:
#   - emd200-marathon: K200 EMD run with V2 guardrails. Strategy-delta convergence
#     is logged only, while a native CRN eval-margin probe can stop once further
#     activation-score gains are marginal.
#   - nb800 / nb1000: made-hand control runs (denser, stable blueprint priors).
# EMD K=200 ONLY. K=500 is BANNED (overfit caused a blind regression on
# run_emd_k500_pluribus_20260625).
#
# SCRIPTS-LIVE-AUDIT #2 (CRITICAL): the K=500 mode was previously exposed
# via `-Mode emd500` with a docstring that said K=500 was BANNED — a
# contradiction that invited a re-run of the k500 regression. The mode is
# now ONLY accessible via `-Mode emd500-LEGACY-K-BANNED` and prints a
# loud warning at the top. The supported production paths are
# `emd200-marathon`, `calibrate-nb800`, `nb800`, `nb1000`.
#
# The wrapper also now surfaces the babysitter's storm-guard caps
# (`-MaxLaunch`, `-MinRelaunchGapSec`, `-QuotaDrainSec`) as its own
# `-MaxLaunch` / `-MinRelaunchGapSec` parameters so the operator can
# see what the babysitter will use BEFORE invoking it (previously the
# wrapper delegated silently).
param(
  [ValidateSet("emd200-marathon", "calibrate-nb800", "nb800", "nb1000", "emd500-LEGACY-K-BANNED")]
  [string]$Mode = "emd200-marathon",
  [string]$Run = "",
  [int]$Vcpus = 128,
  [int]$MaxLaunch = 32,
  [int]$MinRelaunchGapSec = 600,
  [int]$QuotaDrainSec = 600,
  [switch]$OnDemand,
  [switch]$Force,
  [switch]$Launch
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Show-And-Maybe-Run([string]$Title, [string[]]$ArgsList, [string[]]$WatcherArgsList = @()) {
  Write-Host "=== $Title ===" -ForegroundColor Cyan
  Write-Host "Babysitter storm guards:" -ForegroundColor Cyan
  Write-Host ("  MaxLaunch            = $MaxLaunch")
  Write-Host ("  MinRelaunchGapSec    = $MinRelaunchGapSec")
  Write-Host ("  QuotaDrainSec        = $QuotaDrainSec")
  Write-Host ("  Vcpus                = $Vcpus")
  Write-Host "Command:"
  Write-Host ("powershell " + ($ArgsList -join " "))
  if ($WatcherArgsList.Count) {
    Write-Host "Watcher:"
    Write-Host ("powershell " + ($WatcherArgsList -join " "))
  }
  if (-not $Launch) {
    Write-Host ""
    Write-Host "DRY RUN only. Re-run this wrapper with -Launch to start EC2." -ForegroundColor Yellow
    return
  }
  & python -c "import boto3" 2>$null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Python cannot import boto3. Install/check boto3 in the same terminal before launching." -ForegroundColor Red
    exit 2
  }
  & powershell @ArgsList
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Scale-IterBudgetForNDeals([int]$ItersAtNDeals200, [int]$NDeals, [int]$RoundUpTo = 1) {
  if ($NDeals -le 0) { return $ItersAtNDeals200 }
  $scaled = [int][Math]::Ceiling(([double]$ItersAtNDeals200 * 200.0) / [double]$NDeals)
  if ($RoundUpTo -gt 1) {
    $scaled = [int]([Math]::Ceiling([double]$scaled / [double]$RoundUpTo) * $RoundUpTo)
  }
  return [Math]::Max(1, $scaled)
}

# SCRIPTS-LIVE-AUDIT #2: if a caller passes the legacy mode, scream loudly
# before the wrapper does anything destructive. The k500 incident produced
# an SB regression of -140 bb/100; the EVAL gate caught it, but the cloud
# spend on that run was wasted. We refuse without an explicit --force flag.
if ($Mode -eq "emd500-LEGACY-K-BANNED") {
  Write-Host ""
  Write-Host "====================================================================" -ForegroundColor Red
  Write-Host "  WARNING: -Mode emd500-LEGACY-K-BANNED" -ForegroundColor Red
  Write-Host "  EMD K=500 produced a real -140 SB regression on 2026-06-25" -ForegroundColor Red
  Write-Host "  (artifacts/run_emd_k500_pluribus_20260625). The CRN-collision" -ForegroundColor Red
  Write-Host "  bug + K=500 overfit caused the verdict to read GREEN while the" -ForegroundColor Red
  Write-Host "  blueprint was WORSE than iter62000 in the blinds." -ForegroundColor Red
  Write-Host "" -ForegroundColor Red
  Write-Host "  Re-introduce this mode only if you are debugging the k500 regression" -ForegroundColor Red
  Write-Host "  itself. Production EMD is K=200." -ForegroundColor Red
  Write-Host "====================================================================" -ForegroundColor Red
  Write-Host ""
  if (-not $env:POKERBOT_FORCE_EMD500 -and -not $Force) {
    Write-Host "Refusing to launch. Set POKERBOT_FORCE_EMD500=1 in the env to override." -ForegroundColor Red
    exit 3
  }
}

$EMD200_MARATHON_NDEALS = [Math]::Max(200, $Vcpus * 8)
$EMD200_MARATHON_ITERS_AT_NDEALS200 = 4200000
$EMD200_MARATHON_ITERS = Scale-IterBudgetForNDeals $EMD200_MARATHON_ITERS_AT_NDEALS200 $EMD200_MARATHON_NDEALS 1000
$EMD200_MARATHON_CHECKPOINT_EVERY = Scale-IterBudgetForNDeals 10000 $EMD200_MARATHON_NDEALS 100
$EMD200_MARATHON_MAX_HOURS = 220
$EMD200_MARATHON_EVAL_SEEDS = 128
$EMD200_MARATHON_EVAL_HANDS = 8000
$EMD200_EVAL_CONV_MIN_ITERS = Scale-IterBudgetForNDeals 300000 $EMD200_MARATHON_NDEALS 1000
$EMD200_EVAL_CONV_EVERY_ITERS = Scale-IterBudgetForNDeals 50000 $EMD200_MARATHON_NDEALS 1000
$EMD200_EVAL_CONV_SEEDS = 24
$EMD200_EVAL_CONV_HANDS = 4000
$EMD200_EVAL_CONV_MIN_GAIN = 3.0
$EMD200_EVAL_CONV_K = 3
$EMD200_EVAL_CONV_MIN_EVALS = 4

if ($Mode -eq "emd200-marathon") {
  $centroids = "data/abstraction/emd_buckets_200.json"
  if (-not (Test-Path $centroids)) {
    Write-Host "Missing $centroids" -ForegroundColor Red
    Write-Host "Precompute it first:"
    Write-Host "  python scripts\precompute_emd_buckets.py --num-buckets 200 --samples 60000 --n-runouts 200 --out data\abstraction\emd_buckets_200.json"
    exit 2
  }
  if (-not $Run) { $Run = "run_emd_k200_pluribus_marathon_" + (Get-Date -Format "yyyyMMdd_HHmm") }
  $shipName = "emd_k200_pluribus_marathon"
  $fanoutPerVcpu = [double]$EMD200_MARATHON_NDEALS / [Math]::Max(1, $Vcpus)
  Write-Host ("CPU fanout: NDeals={0} for {1} vCPU ({2:0.0} deal tasks/vCPU); iters scaled {3} -> {4} to preserve sampled-deal budget." -f `
    $EMD200_MARATHON_NDEALS, $Vcpus, $fanoutPerVcpu, $EMD200_MARATHON_ITERS_AT_NDEALS200, $EMD200_MARATHON_ITERS) -ForegroundColor Cyan
  $args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\emd_pipeline_babysitter_v2.ps1",
    "-Run", $Run,
    "-Iters", "$EMD200_MARATHON_ITERS",
    "-NDeals", "$EMD200_MARATHON_NDEALS",
    "-NumBuckets", "200",
    "-Centroids", $centroids,
    "-NRunouts", "200",
    "-CheckpointEvery", "$EMD200_MARATHON_CHECKPOINT_EVERY",
    "-Vcpus", "$Vcpus",
    "-MaxHours", "$EMD200_MARATHON_MAX_HOURS",
    "-CrateSrc", "runs/bp_crate_emd/battery_native",
    "-ConvergeLogOnly",
    "-EvalConvergeStop",
    "-EvalConvergeMinIters", "$EMD200_EVAL_CONV_MIN_ITERS",
    "-EvalConvergeEveryIters", "$EMD200_EVAL_CONV_EVERY_ITERS",
    "-EvalConvergeSeeds", "$EMD200_EVAL_CONV_SEEDS",
    "-EvalConvergeHands", "$EMD200_EVAL_CONV_HANDS",
    "-EvalConvergeMinGain", "$EMD200_EVAL_CONV_MIN_GAIN",
    "-EvalConvergeK", "$EMD200_EVAL_CONV_K",
    "-EvalConvergeMinEvals", "$EMD200_EVAL_CONV_MIN_EVALS",
    "-ShipName", $shipName,
    "-EvalSeeds", "$EMD200_MARATHON_EVAL_SEEDS",
    "-EvalHands", "$EMD200_MARATHON_EVAL_HANDS",
    "-MaxLaunch", "$MaxLaunch",
    "-MinRelaunchGapSec", "$MinRelaunchGapSec",
    "-QuotaDrainSec", "$QuotaDrainSec",
    "-Spotpick"
  )
  if ($OnDemand) { $args += "-OnDemand" }
  $watchArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\watch_emd_pipeline_v2.ps1",
    "-Run", $Run,
    "-Iters", "$EMD200_MARATHON_ITERS",
    "-NDeals", "$EMD200_MARATHON_NDEALS",
    "-NumBuckets", "200",
    "-Centroids", $centroids,
    "-NRunouts", "200",
    "-ShipName", $shipName,
    "-ExpectTableMB", "153",
    "-Refresh", "60"
  )
  Show-And-Maybe-Run "EMD K=200 Pluribus-grade marathon (eval-marginal convergence stop)" $args $watchArgs
  return
}

$commonNative = @(
  "-ExecutionPolicy", "Bypass",
  "-File", "scripts\native_cfr_babysitter.ps1",
  "-NDeals", "200",
  "-Vcpus", "$Vcpus",
  "-CrateSrc", "runs/bp_crate_frozen/battery_native",
  "-Spotpick",
  "-MaxLaunch", "$MaxLaunch",
  "-MinRelaunchGapSec", "$MinRelaunchGapSec",
  "-QuotaDrainSec", "$QuotaDrainSec"
)
if ($OnDemand) { $commonNative += "-OnDemand" }

if ($Mode -eq "calibrate-nb800") {
  if (-not $Run) { $Run = "run_bp_pluribus_nb800_cal" }
  $args = $commonNative + @(
    "-Run", $Run,
    "-Iters", "6000",
    "-NumBuckets", "800",
    "-CheckpointEvery", "500",
    "-MaxHours", "3",
    "-NoConvergeStop"
  )
  Show-And-Maybe-Run "NB=800 calibration (cheap sizing probe)" $args
  return
}

if ($Mode -eq "nb800") {
  if (-not $Run) { $Run = "run_bp_pluribus_nb800" }
  $args = $commonNative + @(
    "-Run", $Run,
    "-Iters", "300000",
    "-NumBuckets", "800",
    "-CheckpointEvery", "2000",
    "-MaxHours", "36",
    "-ConvThreshold", "0.022",
    "-ConvK", "3",
    "-ConvNumBuckets", "800"
  )
  Show-And-Maybe-Run "NB=800 Pluribus-level made-hand blueprint train" $args
  return
}

if ($Mode -eq "nb1000") {
  if (-not $Run) { $Run = "run_bp_pluribus_nb1000" }
  $args = $commonNative + @(
    "-Run", $Run,
    "-Iters", "400000",
    "-NumBuckets", "1000",
    "-CheckpointEvery", "2000",
    "-MaxHours", "48",
    "-ConvThreshold", "0.023",
    "-ConvK", "3",
    "-ConvNumBuckets", "1000"
  )
  Show-And-Maybe-Run "NB=1000 high-ceiling made-hand blueprint train" $args
  return
}

if ($Mode -eq "emd500-LEGACY-K-BANNED") {
  $centroids = "data/abstraction/emd_buckets_500.json"
  if (-not (Test-Path $centroids)) {
    Write-Host "Missing $centroids" -ForegroundColor Yellow
    Write-Host "Fit it first, for example with the bundled Python:"
    Write-Host "  python scripts\precompute_emd_buckets.py --num-buckets 500 --samples 60000 --n-runouts 200 --out data\abstraction\emd_buckets_500.json"
    exit 2
  }
  if (-not $Run) { $Run = "run_emd_pluribus_k500" }
  $args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\emd_pipeline_babysitter.ps1",
    "-Run", $Run,
    "-Iters", "320000",
    "-NDeals", "200",
    "-NumBuckets", "500",
    "-Centroids", $centroids,
    "-NRunouts", "200",
    "-CheckpointEvery", "2000",
    "-Vcpus", "$Vcpus",
    "-MaxHours", "40",
    "-CrateSrc", "runs/bp_crate_emd/battery_native",
    "-ConvThreshold", "0.022",
    "-ConvK", "3",
    "-ShipName", "emd_k500_pluribus",
    "-EvalSeeds", "24",
    "-EvalHands", "4000",
    "-MaxLaunch", "$MaxLaunch",
    "-MinRelaunchGapSec", "$MinRelaunchGapSec",
    "-QuotaDrainSec", "$QuotaDrainSec"
  )
  if ($OnDemand) { $args += "-OnDemand" }
  $watchArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\watch_emd_pipeline.ps1",
    "-Run", $Run,
    "-Iters", "320000",
    "-NumBuckets", "500",
    "-Centroids", $centroids,
    "-NRunouts", "200",
    "-ShipName", "emd_k500_pluribus",
    "-ExpectTableMB", "160"
  )
  Show-And-Maybe-Run "EMD K=500 (LEGACY, BANNED) draw-aware full pipeline" $args $watchArgs
  return
}
