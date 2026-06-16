# Autonomous babysitter for the 6-MAX LBR EXPLOITABILITY run (--mode lbr6).
# Runs the certified 6-max Local Best Response (pokerbot/eval/lbr6.py) at scale: SEEDS independent
# best-response cells vs the blueprint foundation, pooled into ONE tight CI (stack-off variance needs
# large N). Two calibration cells (foldbot, station) prove on THIS run that the adversary detects leaks.
# Relaunches the spot box on reclaim until eval_suite_DONE.marker appears, then downloads + prints
# SUMMARY.json (blueprint_pooled CI + calibration + verdict). Run from your terminal (your AWS creds):
#   powershell -ExecutionPolicy Bypass -File scripts\lbr6_babysitter.ps1
# ASCII ONLY (PS 5.1 + BOM-less UTF-8 mangles non-ASCII into string-breaking bytes).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
# 120 seeds x 3000 hands = 360k best-response hands -> pooled CI ~ +/-33 bb/100 (enough to rule out any
# meaningful exploit; per-hand sigma ~100 bb makes a tighter CI cost ~10x). Cells fan across the box's
# thread pool + self-upload lbr6_bp_s<seed>.json; the driver pools them into SUMMARY.json. Fresh prefix
# per study because run_job skips lbr6_*.json already in S3.
# GRADED Lisy-Bowling protocol with the CORRECT analytic 1-ply LBR: sweeps ALL THREE action menus
# (fc floor / fcpa / full) in one run + a per-seat deviation-gain breakdown, to MAP how the +127 bb/100
# exploit grows with the adversary's betting options and which seats leak most. run_lbr6_2 (single fcpa)
# already found BEATABLE +127. This maps the full shape. 3 menus x SEEDS cells + 2 calibration cells.
$RUN      = "run_lbr6_graded1"
$SEEDS    = 60
$HANDS    = 3000
$PREFIX = "pokerbot/prod6000/eval_suite/$RUN"
$MARKER = "$PREFIX/eval_suite_DONE.marker"
$RESULT = "$PREFIX/results/SUMMARY.json"
$LOCAL  = "runs/cloud_lbr6/SUMMARY.json"
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
Write-Host "=== lbr6 babysitter: $RUN ($SEEDS seeds x $HANDS hands, caps: ${capH}h / $MAXLAUNCH launches) ==="
# One-time cleanup of stale FAIL_* logs + empty SUMMARY from a PRIOR failed run so the dashboard
# reflects only this run (lbr6_*.json results are kept + resumed; the DONE marker is left alone).
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
  Write-Host "  no live box -> launching #$next (--mode lbr6 analytic GRADED fc/fcpa/full)"
  $out = (& python scripts\launch_eval_suite_spot.py --run-subdir $RUN --mode lbr6 --seeds $SEEDS --hands $HANDS 2>&1)
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
  New-Item -ItemType Directory -Force "runs/cloud_lbr6" | Out-Null
  aws s3 cp "s3://$BUCKET/$RESULT" $LOCAL --region $REGION --no-progress
  Write-Host ""
  Write-Host "=== LBR6 SUMMARY (downloaded -> $LOCAL) ==="
  Get-Content $LOCAL
  Write-Host ""
  Write-Host "calibration foldbot/station must be >> 0 (adversary competent); blueprint_pooled.ci_lo <= 0"
  Write-Host "=> a competent best-responder cannot prove a positive exploit of the 6-max foundation."
} else {
  Write-Host "No DONE marker - the run did not complete (check the console / FAIL_*.log)."
}
Write-Host "BABYSITTER EXIT after ${el}s, $launches launches"
