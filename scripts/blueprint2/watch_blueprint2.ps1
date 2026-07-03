# Read-only dashboard for the blueprint-v2 run — thin wrapper over the proven
# scripts\watch_blueprint_run.ps1 (progress bar, it/s, infosets, ETA, stall alarm, spot cost)
# with the bp2 defaults baked in. NEVER terminates a box. Run in its own terminal:
#
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\watch_blueprint2.ps1
#
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
param(
  [string]$Run = "bp2_riverochs_1",
  [int]$Iters = 62000,
  [int]$Refresh = 45,
  [int]$StallMin = 20,
  [string]$Region = "eu-central-1",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr"
)
$watcher = Join-Path $PSScriptRoot "..\watch_blueprint_run.ps1"
& powershell -ExecutionPolicy Bypass -File $watcher `
    -Run $Run -Iters $Iters -Refresh $Refresh -StallMin $StallMin `
    -Region $Region -Bucket $Bucket -Prefix $Prefix
