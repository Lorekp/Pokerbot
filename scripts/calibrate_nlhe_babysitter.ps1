# Autonomous babysitter for the NLHE skill-tier CALIBRATION cloud job (--mode calibrate).
# Relaunches the spot box on reclaim until calibrate_DONE.marker appears, then downloads the fitted
# nlhe_skill_tiers.json into the repo (backing up the old one) and prints it. The calibrate driver
# fans (tier,seed) across a process pool on the box and self-uploads the result + marker.
# Heartbeat every 180s in the old babysitter style:
#   [t+1234s] done=no live='i-0abc (up 12m)' launches=1
# Run from your terminal (your AWS creds):
#   powershell -ExecutionPolicy Bypass -File scripts\calibrate_nlhe_babysitter.ps1
# ASCII ONLY (PS 5.1 + BOM-less UTF-8 mangles non-ASCII into string-breaking bytes).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

$REGION = "eu-central-1"
$RUN    = "run_nlhetiers2"   # v2: CRN-paired leak-cost measurement (v1 used flawed absolute-vs-blueprint)
$BUCKET = "video-processing-6789"
$PREFIX = "pokerbot/prod6000/eval_suite/$RUN"
$MARKER = "$PREFIX/calibrate_DONE.marker"
$RESULT = "$PREFIX/results/nlhe_skill_tiers.json"
$REPO_JSON = "data/player_profiles/nlhe_skill_tiers.json"
$MAXSEC = 8 * 3600      # hard wall-clock cap ~8h (calibrate is ~15-45min/run; cap covers reclaims)
$MAXLAUNCH = 6
$start = Get-Date
$launches = 0
$prevBox = ""

function Test-Done {
  aws s3 ls "s3://$BUCKET/$MARKER" --region $REGION 2>$null | Out-Null
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

$capH = [int]($MAXSEC / 3600)
Write-Host "=== nlhe-tier calibration babysitter: $RUN (caps: ${capH}h / $MAXLAUNCH launches) ==="
while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  $done = Test-Done
  $live = Get-Boxes "running,pending"
  $boxId = ""; $up = ""
  if ($live) {
    $boxId = (@($live.Split(" ") | Where-Object { $_ -like "i-*" }) -join ",")
    if ($live -match "(\d{4}-\d{2}-\d{2}T[\d:.]+)") {
      try { $mins = [int](((Get-Date).ToUniversalTime() - [datetime]::Parse($Matches[1]).ToUniversalTime()).TotalMinutes); $up = " (up ${mins}m)" } catch {}
    }
  }
  Write-Host "[t+${el}s] done=$(if ($done) {'yes'} else {'no'}) live='$boxId$up' launches=$launches"
  if ($prevBox -and $boxId -and $boxId -ne $prevBox) {
    Write-Host "          BOX CHANGED: $prevBox -> $boxId (spot reclaim + relaunch)" -ForegroundColor Yellow
  }
  if ($boxId) { $prevBox = $boxId }

  if ($done) { Write-Host "DONE: calibrate marker found"; Stop-Run; break }
  if ($el -ge $MAXSEC) { Write-Host "STOP: time cap reached"; Stop-Run; break }
  if ($live) { Start-Sleep -Seconds 180; continue }
  if ($launches -ge $MAXLAUNCH) { Write-Host "STOP: max launches ($MAXLAUNCH) hit"; break }
  $next = $launches + 1
  Write-Host "  no live box -> launching #$next (--mode calibrate)"
  python scripts\launch_eval_suite_spot.py --run-subdir $RUN --mode calibrate
  $launches++
  Start-Sleep -Seconds 180
}

# Ingest the fitted ladder into the repo (back up the old file first), then print it.
if (Test-Done) {
  if (Test-Path $REPO_JSON) { Copy-Item $REPO_JSON "$REPO_JSON.bak" -Force; Write-Host "backed up -> $REPO_JSON.bak" }
  New-Item -ItemType Directory -Force "runs/cloud_calibrate" | Out-Null
  aws s3 cp "s3://$BUCKET/$RESULT" "runs/cloud_calibrate/nlhe_skill_tiers.json" --region $REGION --no-progress
  Copy-Item "runs/cloud_calibrate/nlhe_skill_tiers.json" $REPO_JSON -Force
  Write-Host ""
  Write-Host "=== FITTED LADDER (ingested -> $REPO_JSON) ==="
  Get-Content $REPO_JSON
} else {
  Write-Host "No DONE marker - calibration did not complete (check the watcher / console output)."
}
Write-Host "BABYSITTER EXIT after ${el}s, $launches launches"
