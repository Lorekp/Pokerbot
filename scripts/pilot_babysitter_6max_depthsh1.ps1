# Autonomous babysitter for the DEPTH-AWARE + SHORT-HANDED 6-max blueprint retrain (A2/A3 + C1).
# Mirrors pilot_babysitter_6max_depth.ps1 (hogwild + DCFR + VR-MCCFR + preflop-lossless + depth flags) and
# ADDS short-handed coverage:
#   --depth-buckets / --depth-bb 30,200 -> depth-aware PREFLOP key + per-deal shared log-uniform depth.
#   --occupancy-mix 2,6 -> POKERBOT_TRAIN_OCCUPANCY_MIX=2,6: randomize each deal's LIVE-SEAT COUNT over 2..6
#                          (weighted toward fuller tables), so the blueprint prior also covers 5/4/3-handed
#                          and heads-up. No abstraction-key change (position/history already separate seat
#                          counts); the engine + real-time search already PLAY any seat count correctly --
#                          this only adds the trained warm-start the deployed bp6max (HU + full-ring only) lacks.
# BUDGET: in PERSISTENT hogwild --traversals is INERT; the real knob is (TARGET iters) x
# (--checkpoint-interval-sec, default 180s). The proven single-depth lossless run was 200 iters ~14h. Depth
# alone used 300; depth + occupancy spreads deals across BOTH the x4 depth-keyed preflop slice AND the
# short-handed position/history regions, so this run uses 400 iters (~28h) for headroom. FINAL gate is a
# coverage check stratified by BOTH the _DEPTH_EDGES bands AND the live-seat count (bp_coverage_report is
# per-STREET only): stop early if all cells saturate before 400, EXTEND (bump $TARGET, rerun -- the trainer
# resumes from S3) if a deep-bucket / short-handed cell reads thin. FRESH prefix run_6max_depthsh1; the
# checkpoint stamps depth_bucketed=True so serve cross-checks it (lbr.check_blueprint_serve_env). SERVE MUST
# set POKERBOT_DEPTH_BUCKETS=1. HARD caps so it can never run away. Run from YOUR terminal (your AWS creds):
#   powershell -ExecutionPolicy Bypass -File scripts\pilot_babysitter_6max_depthsh1.ps1
# NOTE: stop any other pilot/training babysitter first (the single spot slot is shared) -- in particular
# terminate the depth-only run_6max_depth1 box before launching this (do NOT resume that prefix; it lacks
# occupancy randomization).
# ASCII ONLY (PS 5.1 mangles non-ASCII).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root (script lives in scripts/)

$REGION = "eu-central-1"
$RUN    = "run_6max_depthsh1"   # depth-aware + short-handed 6-max blueprint, hogwild DCFR + VR + preflop-lossless
$BUCKET = "video-processing-6789"
$PREFIX = "pokerbot/prod6000/sd_cfr_hu/$RUN"
$MAXSEC = 48 * 3600     # hard wall-clock cap ~48h (400 iters ~28h training + spot-interrupt slack)
$TARGET = 400           # stop once this many checkpoints (iterations) exist (depth + short-handed keyspace)
$MAXLAUNCH = 24         # never launch more than this many boxes total (longer run = more spot churn)
$start = Get-Date
$launches = 0

function Get-LatestIter {
  $out = (aws s3 ls "s3://$BUCKET/$PREFIX/" --recursive --region $REGION 2>$null)
  $m = [regex]::Matches(($out -join "`n"), "iter_(\d+)")
  if ($m.Count -gt 0) { return ($m | ForEach-Object { [int]$_.Groups[1].Value } | Measure-Object -Maximum).Maximum }
  return 0
}
function Get-Boxes([string]$states) {
  $r = (aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=$states" --query "Reservations[].Instances[].InstanceId" --output text 2>$null)
  if ($r) { return ($r -replace "\s+", " ").Trim() } else { return "" }
}
function Stop-Run {
  $ids = Get-Boxes "running,pending,stopping"
  if ($ids) { aws ec2 terminate-instances --instance-ids $ids.Split(" ") --region $REGION | Out-Null; Write-Host "terminated: $ids" }
}

while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  $iter = Get-LatestIter
  $live = Get-Boxes "running,pending"
  $dying = Get-Boxes "shutting-down,stopping"
  Write-Host "[t+${el}s] iter=$iter live='$live' dying='$dying' launches=$launches"
  if ($iter -ge $TARGET) { Write-Host "DONE: target iter $TARGET reached"; Stop-Run; break }
  if ($el -ge $MAXSEC)   { Write-Host "DONE: time cap reached"; Stop-Run; break }
  if ($live)  { Start-Sleep -Seconds 150; continue }            # healthy -> wait
  if ($dying) { Write-Host "  box terminating; waiting for quota..."; Start-Sleep -Seconds 45; continue }
  if ($launches -ge $MAXLAUNCH) { Write-Host "STOP: max launches ($MAXLAUNCH) hit"; break }
  Write-Host "  no live box -> relaunch (resume from S3)"
  python scripts\launch_sd_cfr_training_spot.py --instance-type m7g.16xlarge --arch arm64 --workers 56 --iterations 400 --traversals 2000 --deal-sweeps 1 --max-depth 64 --betting-abstraction --postflop-buckets 200 --client-bet-sizing --trainer hogwild --hogwild-capacity 4194304 --no-sft --occupancy 6max --cfr-variant dcfr --vr-baseline --preflop-lossless --depth-buckets --depth-bb 30,200 --occupancy-mix 2,6 --run-subdir $RUN
  $launches++
  Start-Sleep -Seconds 150
}
Write-Host "BABYSITTER EXIT after ${el}s, $launches launches, latest iter=$iter"
