# Autonomous babysitter for the PHASE-2 VALUE-EXTRACTION gauntlet (--mode valueextract).
# Tests whether the UNOPENED value-bet exploit can be turned ON do-no-harm on the blueprint+search base,
# vs the realistic IRC villains (pokerbot/eval/irc_villains.py). Relaunches the spot box on reclaim until
# eval_suite_DONE.marker appears, then downloads SUMMARY.json and prints it. The on-box driver fans every
# (field,seed) gauntlet cell across a thread pool and self-uploads each result + the marker.
# Run from your terminal (your AWS creds):
#   powershell -ExecutionPolicy Bypass -File scripts\valueextract_babysitter.ps1
# ASCII ONLY (PS 5.1 + BOM-less UTF-8 mangles non-ASCII into string-breaking bytes).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
# uniform leaf is cheap => high hands OK. The gauntlet cells are (6 fields x SEEDS), one base arm.
# run_ve2 = the FISH-GATED unopened exploit (run_ve1 was the ungated baseline; a fresh prefix is required
# because run_job skips ve_*.json cells already in S3, so reusing run_ve1 would re-pool the old numbers).
$RUN     = "run_ve2"
$VECONF  = "10:uniform"
$SEEDS   = 12
$HANDS   = 3000
$PREFIX = "pokerbot/prod6000/eval_suite/$RUN"
$MARKER = "$PREFIX/eval_suite_DONE.marker"
$RESULT = "$PREFIX/results/SUMMARY.json"
$LOCAL  = "runs/cloud_valueextract/SUMMARY.json"
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
  # Strongly-consistent check by instance id (the tag:Run filter lags ~minutes for a fresh spot box,
  # which caused a relaunch storm - track the launched id and verify it directly instead).
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
Write-Host "=== value-extraction babysitter: $RUN (caps: ${capH}h / $MAXLAUNCH launches) ==="
# One-time cleanup of stale FAIL_* logs + empty SUMMARY from a PRIOR failed run so the dashboard
# reflects only this run (ve_*.json results are kept + resumed; the DONE marker is left alone).
if (-not (Test-Done)) {
  aws s3 rm "s3://$BUCKET/$PREFIX/results/" --region $REGION --recursive --exclude "*" --include "FAIL_*" --include "SUMMARY.json" 2>$null | Out-Null
}
while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  $done = Test-Done
  $live = Get-Boxes "running,pending"
  if (-not $live -and $tracked) {
    # Tag filter hasn't caught up yet - verify the box we launched is still alive by id.
    $st = Get-StateById $tracked
    if ($st -eq "running" -or $st -eq "pending") { $live = "$tracked" }
    else { $tracked = "" }  # it's gone (reclaimed/terminated) - allow a relaunch
  }
  $boxId = ""; $up = ""
  if ($live) {
    $boxId = (@($live.Split(" ") | Where-Object { $_ -like "i-*" }) -join ",")
    if ($live -match "(\d{4}-\d{2}-\d{2}T[\d:.]+[+\-]\d{2}:?\d{2})") {
      # Full LaunchTime incl. +00:00 offset via DateTimeOffset, else the UTC stamp is read as local and
      # uptime is inflated by the local offset (the phantom "+120m" bug).
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
  Write-Host "  no live box -> launching #$next (--mode valueextract ve-config $VECONF)"
  $out = (& python scripts\launch_eval_suite_spot.py --run-subdir $RUN --mode valueextract --ve-config $VECONF --seeds $SEEDS --hands $HANDS 2>&1)
  $out | ForEach-Object { Write-Host $_ }
  $txt = ($out | Out-String)
  if ($txt -match "LAUNCHED (i-[0-9a-f]+)") {
    $tracked = $Matches[1]; $launches++
    Write-Host "  tracking $tracked; waiting for boot before any relaunch"
    Start-Sleep -Seconds 300   # boot + tag-propagation grace so we never double-launch
  } else {
    # launch FAILED (e.g. MaxSpotInstanceCountExceeded) - do NOT count it; back off and retry.
    Write-Host "  launch did not start a box (quota/capacity?) - backing off 180s"
    Start-Sleep -Seconds 180
  }
}

if (Test-Done) {
  New-Item -ItemType Directory -Force "runs/cloud_valueextract" | Out-Null
  aws s3 cp "s3://$BUCKET/$RESULT" $LOCAL --region $REGION --no-progress
  Write-Host ""
  Write-Host "=== VALUE-EXTRACTION SUMMARY (downloaded -> $LOCAL) ==="
  Get-Content $LOCAL
  Write-Host ""
  Write-Host "value_vs_weak > 0 AND harm_vs_tough >= -0.05  =>  flip root_mixer_unopened ON (gated)."
} else {
  Write-Host "No DONE marker - the run did not complete (check the console / FAIL_*.log)."
}
Write-Host "BABYSITTER EXIT after ${el}s, $launches launches"
