# Autonomous babysitter for the BASE-STRENGTH cloud job (--mode basestrength).
# Relaunches the spot box on reclaim until eval_suite_DONE.marker appears, then downloads SUMMARY.json
# (the absolute AIVAT EV of BOTH bases -- SFT net + blueprint+search -- vs weak/strong/realistic mixed
# fields) into the repo and prints it. The on-box driver fans every (base,cell,seed) across a thread
# pool and self-uploads each result + the marker; a reclaim loses only in-flight jobs.
# Heartbeat every 180s:
#   [t+1234s] done=no live='i-0abc (up 12m)' launches=1
# Run from your terminal (your AWS creds):
#   powershell -ExecutionPolicy Bypass -File scripts\basestrength_babysitter.ps1
# ASCII ONLY (PS 5.1 + BOM-less UTF-8 mangles non-ASCII into string-breaking bytes).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
# ---- pick ONE config block ----
# DEFAULT: PER-SEAT exploit re-fit confirm. Same coverage-fixed 10:uniform base, but the mixed-table
# +exploit arm now TARGETS THE AGGRESSOR (one profile per villain seat) instead of one pooled blend --
# the fix for the ~0.8 bb/100 +exploit regression on realistic/soft/tough. The OFF (base) number is
# unchanged (reproduces +9.12); the A/B is this run's ON vs run_bs_sweep1's pooled ON. Per-seat is the
# DEFAULT now (no flag needed; pass --bs-pooled-profiles to reproduce the legacy pooled measurement).
$RUN     = "run_bs_perseat1"
$CONFIGS = "10:uniform"
$SEEDS   = 12
$HANDS   = 3000
# GADGET variant (run SEPARATELY, edit the 4 lines above to these): gadget leaf is ~30s/DECISION, so a
# gadget cell of N hands ~= N*10*30s; keep HANDS LOW (<=150) or it exceeds the on-box 4h job timeout and
# every gadget job is killed (the run looks "stuck"). Pool the low-hand gadget cells across more seeds.
#   $RUN="run_bs_gadget1"; $CONFIGS="50:gadget"; $SEEDS=16; $HANDS=120
$PREFIX = "pokerbot/prod6000/eval_suite/$RUN"
$MARKER = "$PREFIX/eval_suite_DONE.marker"
$RESULT = "$PREFIX/results/SUMMARY.json"
$LOCAL  = "runs/cloud_basestrength/SUMMARY.json"
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
Write-Host "=== base-strength babysitter: $RUN (caps: ${capH}h / $MAXLAUNCH launches) ==="
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
    if ($live -match "(\d{4}-\d{2}-\d{2}T[\d:.]+)") {
      try { $mins = [int](((Get-Date).ToUniversalTime() - [datetime]::Parse($Matches[1]).ToUniversalTime()).TotalMinutes); $up = " (up ${mins}m)" } catch {}
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
  Write-Host "  no live box -> launching #$next (--mode basestrength sweep $CONFIGS)"
  $out = (& python scripts\launch_eval_suite_spot.py --run-subdir $RUN --mode basestrength --bs-configs $CONFIGS --bs-skip-sft --seeds $SEEDS --hands $HANDS 2>&1)
  $out | ForEach-Object { Write-Host $_ }
  $txt = ($out | Out-String)
  if ($txt -match "LAUNCHED (i-[0-9a-f]+)") {
    $tracked = $Matches[1]; $launches++
    Write-Host "  tracking $tracked; waiting for boot before any relaunch"
    Start-Sleep -Seconds 300   # boot + tag-propagation grace so we never double-launch
  } else {
    Write-Host "  launch did not start a box (quota/capacity?) - backing off 180s"
    Start-Sleep -Seconds 180
  }
}

# Ingest the SUMMARY into the repo and print it.
if (Test-Done) {
  New-Item -ItemType Directory -Force "runs/cloud_basestrength" | Out-Null
  aws s3 cp "s3://$BUCKET/$RESULT" $LOCAL --region $REGION --no-progress
  Write-Host ""
  Write-Host "=== BASE-STRENGTH SUMMARY (downloaded -> $LOCAL) ==="
  Get-Content $LOCAL
  Write-Host ""
  Write-Host "Next: python scripts\decide_base_routing.py $LOCAL   (applies the pre-registered rule)"
} else {
  Write-Host "No DONE marker - the run did not complete (check the watcher / console output)."
}
Write-Host "BABYSITTER EXIT after ${el}s, $launches launches"
