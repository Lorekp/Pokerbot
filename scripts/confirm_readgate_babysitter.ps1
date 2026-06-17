# Babysitter for the READ-GATED MIXER confirm (G1): gated arm B (ON) vs pure search (OFF) on
# station/lag/tag at 18 reps. Smaller than the full suite (108 jobs). Same caps/resume mechanics.
# Run from your terminal (your AWS creds):
#   powershell -ExecutionPolicy Bypass -File scripts\confirm_readgate_babysitter.ps1
# ASCII ONLY (PS 5.1 + BOM-less UTF-8 mangles non-ASCII).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

$REGION = "eu-central-1"
$RUN    = "run_readgate_confirm1"
$BUCKET = "video-processing-6789"
$PREFIX = "pokerbot/prod6000/eval_suite/$RUN"
$MAXSEC = 10 * 3600
$MAXLAUNCH = 6
$start = Get-Date
$launches = 0
$prevBox = ""

function Test-Done {
  aws s3 ls "s3://$BUCKET/$PREFIX/eval_suite_DONE.marker" --region $REGION 2>$null | Out-Null
  return $?
}
function Get-Boxes([string]$states) {
  $r = (aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=$states" --query "Reservations[].Instances[].[InstanceId,LaunchTime]" --output text 2>$null)
  if ($r) { return (($r -join " ") -replace "\s+", " ").Trim() } else { return "" }
}
function Stop-Run {
  $ids = (Get-Boxes "running,pending,stopping")
  if ($ids) {
    $idList = @($ids.Split(" ") | Where-Object { $_ -like "i-*" })
    if ($idList.Count) { aws ec2 terminate-instances --instance-ids $idList --region $REGION | Out-Null; Write-Host "terminated: $idList" }
  }
}

Write-Host "=== read-gate confirm babysitter: $RUN (caps: $([int]($MAXSEC/3600))h / $MAXLAUNCH launches) ==="
while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  $out = (aws s3 ls "s3://$BUCKET/$PREFIX/results/" --region $REGION 2>$null)
  $n = if ($out) { @($out | Select-String "\.json" | Where-Object { $_ -notmatch "SUMMARY" }).Count } else { 0 }
  $fails = if ($out) { @($out | Select-String "FAIL_").Count } else { 0 }
  $live = Get-Boxes "running,pending"
  $boxId = if ($live) { (@($live.Split(" ") | Where-Object { $_ -like "i-*" }) -join ",") } else { "" }
  Write-Host "[t+${el}s] results=$n/108 live='$boxId' launches=$launches fails=$fails"
  if ($prevBox -and $boxId -and $boxId -ne $prevBox) { Write-Host "          BOX CHANGED: $prevBox -> $boxId (reclaim)" -ForegroundColor Yellow }
  if ($boxId) { $prevBox = $boxId }
  if (Test-Done) { Write-Host "DONE"; Stop-Run; break }
  if ($el -ge $MAXSEC) { Write-Host "STOP: time cap"; Stop-Run; break }
  if ($live) { Start-Sleep -Seconds 180; continue }
  if ($launches -ge $MAXLAUNCH) { Write-Host "STOP: max launches"; break }
  Write-Host "  no live box -> launching #$($launches + 1)"
  python scripts\launch_eval_suite_spot.py --run-subdir $RUN --mode confirm --seeds 18
  $launches++
  Start-Sleep -Seconds 180
}

New-Item -ItemType Directory -Force "runs/readgate_confirm" | Out-Null
aws s3 cp "s3://$BUCKET/$PREFIX/results/" runs/readgate_confirm/ --recursive --region $REGION --no-progress
Write-Host ""
Write-Host "=== SUMMARY (gated ON vs pure search OFF; paired_lag_delta is the do-no-harm number) ==="
Get-Content runs/readgate_confirm/SUMMARY.json -ErrorAction SilentlyContinue
Write-Host "BABYSITTER EXIT after ${el}s, $launches launches"
