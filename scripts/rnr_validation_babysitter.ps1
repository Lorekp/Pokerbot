# Autonomous babysitter for the RNR-restriction crush-retention validation (--mode rnr_validation, #29).
# Measures the SOLVER RNR cap's effect on the exploit crush: per (field, arm) the pluribus frozen-BR +
# overlays exploit A/B with POKERBOT_RNR_SOLVER_EVAL OFF vs ON, over soft_table / realistic_table /
# pro_table. ON retaining most of OFF's uplift => the 0.5 6-max cap does NOT over-clamp the fishy field
# (default-ON live is safe); a large drop on the soft/realistic fields => over-clamp (ship the per-punisher
# effective-N, task #32). Relaunches the spot box on reclaim until eval_suite_DONE.marker, then downloads
# SUMMARY.json and prints it.
# Run from your terminal (your AWS creds):
#   powershell -ExecutionPolicy Bypass -File scripts\rnr_validation_babysitter.ps1
# ASCII ONLY (PS 5.1 + BOM-less UTF-8 mangles non-ASCII into string-breaking bytes).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
# Fresh prefix each iteration because run_job skips rnr_*.json already in S3 (resume). Bump run_rnrN to
# force a clean re-measure.
$RUN     = "run_rnr1"
$SEEDS   = 12
$HANDS   = 3000
$PREFIX = "pokerbot/prod6000/eval_suite/$RUN"
$MARKER = "$PREFIX/eval_suite_DONE.marker"
$RESULT = "$PREFIX/results/SUMMARY.json"
$LOCAL  = "runs/cloud_rnr_validation/SUMMARY.json"
$MAXSEC = 12 * 3600
$MAXLAUNCH = 8
$start = Get-Date
$launches = 0
$prevBox = ""
$tracked = ""   # instance id of the box we last launched (verified by id, not the laggy tag filter)

function Test-Done {
  aws s3 ls "s3://$BUCKET/$MARKER" --region $REGION 2>$null | Out-Null
  return $?
}
function Get-Boxes([string]$states) {
  $r = (aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=$states" --query "Reservations[].Instances[].[InstanceId,LaunchTime]" --output text 2>$null)
  if ($r) { return (($r -join " ") -replace "\s+", " ").Trim() } else { return "" }
}
function Get-StateById([string]$id) {
  if (-not $id) { return "" }
  $s = (aws ec2 describe-instances --region $REGION --instance-ids $id --query "Reservations[].Instances[].State.Name" --output text 2>$null)
  if ($s) { return $s.Trim() } else { return "" }
}
function Stop-Run {
  $ids = (Get-Boxes "running,pending,stopping")
  if ($ids) {
    $idList = @($ids.Split(" ") | Where-Object { $_ -like "i-*" })
    if ($idList.Count) { aws ec2 terminate-instances --instance-ids $idList --region $REGION | Out-Null; Write-Host "terminated: $idList" }
  }
}

$capH = [int]($MAXSEC / 3600)
Write-Host "=== RNR-validation babysitter: $RUN (caps: ${capH}h / $MAXLAUNCH launches) ==="
if (-not (Test-Done)) {
  aws s3 rm "s3://$BUCKET/$PREFIX/results/" --region $REGION --recursive --exclude "*" --include "FAIL_*" --include "SUMMARY.json" 2>$null | Out-Null
}
while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  $done = Test-Done
  $live = Get-Boxes "running,pending"
  if (-not $live -and $tracked) {
    $st = Get-StateById $tracked
    if ($st -eq "running" -or $st -eq "pending") { $live = "$tracked" }
    else { $tracked = "" }
  }
  $boxId = ""; $up = ""
  if ($live) {
    $boxId = (@($live.Split(" ") | Where-Object { $_ -like "i-*" }) -join ",")
    if ($live -match "(\d{4}-\d{2}-\d{2}T[\d:.]+[+\-]\d{2}:?\d{2})") {
      try { $mins = [int](((Get-Date).ToUniversalTime() - [datetimeoffset]::Parse($Matches[1]).UtcDateTime).TotalMinutes); $up = " (up ${mins}m)" } catch {}
    }
  }
  Write-Host "[t+${el}s] done=$(if ($done) {'yes'} else {'no'}) live='$boxId$up' launches=$launches"
  if ($prevBox -and $boxId -and $boxId -ne $prevBox) {
    Write-Host "          BOX CHANGED: $prevBox -> $boxId (spot reclaim + relaunch)" -ForegroundColor Yellow
  }
  if ($boxId) { $prevBox = $boxId }

  if ($done) { Write-Host "DONE: eval_suite marker found"; Stop-Run; break }
  if ($el -ge $MAXSEC) { Write-Host "STOP: time cap reached"; Stop-Run; break }
  if ($live) { Start-Sleep -Seconds 180; continue }
  if ($launches -ge $MAXLAUNCH) { Write-Host "STOP: max launches ($MAXLAUNCH) hit"; break }
  $next = $launches + 1
  Write-Host "  no live box -> launching #$next (--mode rnr_validation seeds=$SEEDS hands=$HANDS)"
  $out = (& python scripts\launch_eval_suite_spot.py --run-subdir $RUN --mode rnr_validation --seeds $SEEDS --hands $HANDS 2>&1)
  $out | ForEach-Object { Write-Host $_ }
  $txt = ($out | Out-String)
  if ($txt -match "LAUNCHED (i-[0-9a-f]+)") {
    $tracked = $Matches[1]; $launches++
    Write-Host "  tracking $tracked; waiting for boot before any relaunch"
    Start-Sleep -Seconds 300
  } else {
    Write-Host "  launch did not start a box (quota/capacity?) - backing off 180s"
    Start-Sleep -Seconds 180
  }
}

if (Test-Done) {
  New-Item -ItemType Directory -Force "runs/cloud_rnr_validation" | Out-Null
  aws s3 cp "s3://$BUCKET/$RESULT" $LOCAL --region $REGION --no-progress
  Write-Host ""
  Write-Host "=== RNR-VALIDATION SUMMARY (downloaded -> $LOCAL) ==="
  Get-Content $LOCAL
  Write-Host ""
  Write-Host "Read: per field, compare arms.off.uplift_bb vs arms.on.uplift_bb (rnr_uplift_retained_frac)."
  Write-Host "  soft_table/realistic_table retained ~>=0.7 => RNR default-ON is safe vs the fishy field."
  Write-Host "  a large drop => over-clamp; ship the per-punisher effective-N (task #32)."
} else {
  Write-Host "No DONE marker - the run did not complete (check the console / FAIL_*.log)."
}
Write-Host "BABYSITTER EXIT after ${el}s, $launches launches"
