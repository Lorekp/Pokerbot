# Autonomous babysitter for the vast.ai 7-max neural-CFR RESUME training. Relaunches the bid box on
# preemption/termination (the box pulls the latest S3 checkpoint and resumes warm) until the S3 DONE
# marker appears, with relaunch-STORM protection and a price-ladder fallback (the launcher already
# walks a capacity-diverse offer chain; the babysitter caps how MANY times it relaunches).
#
# Mirrors scripts/eval_suite_babysitter.ps1 (AWS) but swaps EC2 calls for the vast.ai CLI:
#   describe-instances  -> vastai show instances   (find our labeled box + actual_status)
#   run_instances       -> python scripts/launch_neural_cfr_7max_vast.py ...
#   terminate-instances -> vastai destroy instance <id>
# The S3-poll half (DONE marker, heartbeat, caps) is reused verbatim.
#
# Run from your terminal (your AWS + VAST creds in env):
#   $env:AWS_ACCESS_KEY_ID="..."; $env:AWS_SECRET_ACCESS_KEY="..."; $env:VAST_API_KEY="..."
#   powershell -ExecutionPolicy Bypass -File scripts\neural_cfr_7max_vast_babysitter.ps1 -Run run_1700000000
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
param(
  [string]$Run = "",
  [string]$Bucket = "video-processing-6789",
  [string]$Region = "eu-central-1",
  [string]$PrefixBase = "pokerbot/prod6000/neural_cfr/vast_7max",
  [double]$MaxBudget = 71.0,
  [int]$MaxLaunch = 6,            # relaunch-storm cap: never create more than this many boxes
  [int]$MaxSec = 11 * 24 * 3600, # absolute wall-clock cap (~11 days; budget usually bites first)
  [int]$PollSec = 120,
  [string]$Label = ""
)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not $Run) { Write-Host "ERROR: -Run <run_subdir> is required (the launcher prints it)."; exit 1 }
if (-not $Label) { $Label = "pokerbot-7max-$Run" }
$PREFIX = "$PrefixBase/$Run"
$start = Get-Date
$launches = 0
$prevBox = ""
$lastAlive = Get-Date
$fastRestarts = New-Object System.Collections.ArrayList   # epochs of recent relaunches (storm guard)
$FAST_WINDOW = 1800   # if >= 3 relaunches within 30 min -> structural, abort
$FAST_MAX = 3

function Test-Done {
  aws s3 ls "s3://$Bucket/$PREFIX/sevenmax_DONE.marker" --region $Region 2>$null | Out-Null
  return $?
}
function Get-VastBox {
  # Returns "<id> <actual_status>" for OUR labeled instance, or "" if none. Defensive JSON parse.
  $raw = (vastai show instances --raw 2>$null)
  if (-not $raw) { return "" }
  try { $arr = $raw | ConvertFrom-Json } catch { return "" }
  foreach ($inst in $arr) {
    $lbl = "$($inst.label)"
    if ($lbl -eq $Label) {
      return ("{0} {1}" -f $inst.id, $inst.actual_status)
    }
  }
  return ""
}
function Destroy-VastBox([string]$id) {
  if ($id) {
    Write-Host "  destroying vast instance $id"
    vastai destroy instance $id 2>$null | Out-Null
  }
}
function Get-Heartbeat {
  $raw = (aws s3 cp "s3://$Bucket/$PREFIX/state/heartbeat.json" - --region $Region 2>$null)
  if ($raw) { try { return ($raw | ConvertFrom-Json) } catch { return $null } }
  return $null
}

$capH = [int]($MaxSec / 3600)
Write-Host "=== vast 7-max babysitter: $Run (label=$Label caps: ${capH}h / $MaxLaunch launches / `$$MaxBudget) ==="

while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  $box = Get-VastBox
  $boxId = ""; $boxStatus = ""
  if ($box) { $parts = $box.Split(" "); $boxId = $parts[0]; $boxStatus = $parts[1] }
  $alive = ($boxStatus -eq "running")
  if ($alive) { $lastAlive = Get-Date }

  $hb = Get-Heartbeat
  $hbTxt = ""
  if ($hb) {
    $it = if ($null -ne $hb.iter) { $hb.iter } else { "?" }
    $tot = if ($null -ne $hb.total) { $hb.total } else { "?" }
    $ms = if ($null -ne $hb.ms_per_iter) { ("{0:N0}ms/it" -f $hb.ms_per_iter) } else { "" }
    $hbTxt = "iter=$it/$tot $ms"
  }
  $spent = ""
  if ($hb -and $hb.start_epoch) {
    # rough $ spent estimate uses the runtime cap rate baked into max_runtime / budget is the cap.
    $upH = ((Get-Date).ToUniversalTime() - ([datetimeoffset]::FromUnixTimeSeconds([long]$hb.start_epoch)).UtcDateTime).TotalHours
    $spent = ("up {0:N1}h" -f $upH)
  }
  Write-Host "[t+${el}s] box='$boxId $boxStatus' $hbTxt $spent launches=$launches"

  if (Test-Done) {
    Write-Host "DONE: sevenmax_DONE.marker present (gate met / budget cap / iters done). Cleaning up."
    Destroy-VastBox $boxId
    break
  }
  if ($el -ge $MaxSec) { Write-Host "STOP: wall-clock cap reached"; Destroy-VastBox $boxId; break }

  if ($prevBox -and $boxId -and ($boxId -ne $prevBox)) {
    Write-Host "  BOX CHANGED: $prevBox -> $boxId (preempt + relaunch)" -ForegroundColor Yellow
  }
  if ($boxId) { $prevBox = $boxId }

  if ($alive) { Start-Sleep -Seconds $PollSec; continue }

  # No live box. It was preempted/destroyed/never-launched. Reap any stopped husk (storage bills!).
  if ($boxId -and $boxStatus -ne "running") {
    Write-Host "  box $boxId is '$boxStatus' (not running) -> destroying husk to stop storage billing"
    Destroy-VastBox $boxId
    $prevBox = ""
  }

  # storm guard
  $now = [int][double]::Parse((Get-Date -UFormat %s))
  $null = $fastRestarts.Add($now)
  $fastRestarts = [System.Collections.ArrayList]@($fastRestarts | Where-Object { ($now - $_) -le $FAST_WINDOW })
  if ($fastRestarts.Count -ge $FAST_MAX) {
    Write-Host "ABORT: $FAST_MAX relaunches within ${FAST_WINDOW}s -> structural failure (bad creds / image / offer). Fix the cause." -ForegroundColor Red
    break
  }
  if ($launches -ge $MaxLaunch) { Write-Host "STOP: max launches ($MaxLaunch) hit"; break }

  $next = $launches + 1
  Write-Host "  no live box -> launching #$next (resumes warm from S3 checkpoint)"
  # The launcher re-uploads the code tarball + re-seeds checkpoints only if S3 has none; on a relaunch
  # S3 already has the latest checkpoint so the box resumes from it. Reuse the SAME run subdir.
  python scripts/launch_neural_cfr_7max_vast.py --run-subdir $Run --bucket $Bucket --region $Region `
    --max-budget $MaxBudget --no-seed-checkpoint
  if ($LASTEXITCODE -ne 0) {
    Write-Host "  launcher returned $LASTEXITCODE (offer supply / CLI?). Backing off ${PollSec}s." -ForegroundColor Yellow
  }
  $launches++
  Start-Sleep -Seconds $PollSec
}

Write-Host ""
Write-Host "=== babysitter exit after ${el}s, $launches launches ==="
# Final: pull the latest checkpoint + the last gate result so the operator has the converged net.
New-Item -ItemType Directory -Force "runs/vast_7max/$Run" | Out-Null
aws s3 sync "s3://$Bucket/$PREFIX/checkpoints/" "runs/vast_7max/$Run/checkpoints/" --region $Region --no-progress 2>$null
aws s3 cp "s3://$Bucket/$PREFIX/state/tournament_eval_net.json" "runs/vast_7max/$Run/tournament_eval_net.json" --region $Region 2>$null
Write-Host "Downloaded latest checkpoint + gate JSON to runs/vast_7max/$Run/"
if (Test-Path "runs/vast_7max/$Run/tournament_eval_net.json") {
  Write-Host "--- gate verdict ---"
  python scripts/vast/gate_decider.py --json "runs/vast_7max/$Run/tournament_eval_net.json" --tier 1
}
