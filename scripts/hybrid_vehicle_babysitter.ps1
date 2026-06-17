# Babysitter for the HYBRIDAGENT VEHICLE validation (the chosen live exploitation path): production
# hero exploit ON vs pure-GTO OFF vs the 4 fair archetypes + do-no-harm vs counter/blueprint, 18 reps.
# 6 families x 18 = 108 jobs. Production hero (neural net forward + mixer) is FASTER than the search
# hero (no solve), so this is a short run. Run from your terminal (your AWS creds):
#   powershell -ExecutionPolicy Bypass -File scripts\hybrid_vehicle_babysitter.ps1
# ASCII ONLY (PS 5.1 + BOM-less UTF-8 mangles non-ASCII).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

$REGION = "eu-central-1"
$RUN    = "run_hybrid_vehicle1"
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

Write-Host "=== hybrid-vehicle babysitter: $RUN (caps: $([int]($MAXSEC/3600))h / $MAXLAUNCH launches) ==="
# Clear a stale DONE marker from a prior failed run so this run can actually start (the driver only
# writes DONE on success now, but an old all-failed run may have left one). Per-job result JSONs are
# preserved (resume); only the marker is cleared.
aws s3 rm "s3://$BUCKET/$PREFIX/eval_suite_DONE.marker" --region $REGION 2>$null | Out-Null
while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  $out = (aws s3 ls "s3://$BUCKET/$PREFIX/results/" --region $REGION 2>$null)
  $n = if ($out) { @($out | Select-String "\.json" | Where-Object { $_ -notmatch "SUMMARY" }).Count } else { 0 }
  $fails = if ($out) { @($out | Select-String "FAIL_").Count } else { 0 }
  $fam = ""
  if ($out) {
    $rows = @($out | Select-String "\.json" | Where-Object { $_ -notmatch "SUMMARY" })
    $parts = foreach ($k in "hy_hs_station","hy_hs_nit","hy_hs_lag","hy_hs_tag","hycounter","hybpbot") {
      $c = @($rows | Where-Object { $_ -match $k }).Count; if ($c) { "${k}:$c" }
    }
    $fam = ($parts -join " ")
  }
  $live = Get-Boxes "running,pending"
  $boxId = if ($live) { (@($live.Split(" ") | Where-Object { $_ -like "i-*" }) -join ",") } else { "" }
  Write-Host "[t+${el}s] results=$n/108 live='$boxId' launches=$launches fails=$fails"
  if ($fam) { Write-Host "          $fam" }
  if ($prevBox -and $boxId -and $boxId -ne $prevBox) { Write-Host "          BOX CHANGED: $prevBox -> $boxId (reclaim)" -ForegroundColor Yellow }
  if ($boxId) { $prevBox = $boxId }
  if (Test-Done) { Write-Host "DONE"; Stop-Run; break }
  if ($el -ge $MAXSEC) { Write-Host "STOP: time cap"; Stop-Run; break }
  if ($live) { Start-Sleep -Seconds 180; continue }
  if ($launches -ge $MAXLAUNCH) { Write-Host "STOP: max launches"; break }
  Write-Host "  no live box -> launching #$($launches + 1)"
  python scripts\launch_eval_suite_spot.py --run-subdir $RUN --mode hybrid --seeds 18
  $launches++
  Start-Sleep -Seconds 180
}

New-Item -ItemType Directory -Force "runs/hybrid_vehicle" | Out-Null
aws s3 cp "s3://$BUCKET/$PREFIX/results/" runs/hybrid_vehicle/ --recursive --region $REGION --no-progress
Write-Host ""
Write-Host "=== SUMMARY (HybridAgent exploit ON vs pure-GTO OFF; the live-vehicle validation) ==="
Get-Content runs/hybrid_vehicle/SUMMARY.json -ErrorAction SilentlyContinue
Write-Host "BABYSITTER EXIT after ${el}s, $launches launches"
