# Autonomous babysitter for the 6-MAX Pluribus-foundation pilot — DCFR + VR-MCCFR, THROUGHPUT-FIXED v2: --traversals 2000 (v1 ran 20 = smoke-test value => ~all checkpoint overhead, ~0% postflop coverage).
# Fresh prefix run_6max_dcfrvr2 (clean start, NOT a resume of the CFR+ run_6max_v1). HARD caps so it can
# never run away. Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\pilot_babysitter_6max_dcfrvr.ps1
# NOTE: stop any other pilot babysitter first (single spot slot is shared).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root (script lives in scripts/)

$REGION = "eu-central-1"
$RUN    = "run_6max_dcfrvr2"   # 6-max blueprint, FIXED engine, Discounted-CFR + VR-MCCFR baseline
$BUCKET = "video-processing-6789"
$PREFIX = "pokerbot/prod6000/sd_cfr_hu/$RUN"
$MAXSEC = 14 * 3600     # hard wall-clock cap ~14h (~$18 spot) — under the $40 budget
$TARGET = 200           # stop once this many 180s checkpoints exist (~10h continuous)
$MAXLAUNCH = 10         # never launch more than this many boxes total
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
  python scripts\launch_sd_cfr_training_spot.py --instance-type m7g.16xlarge --arch arm64 --workers 56 --iterations 200 --traversals 2000 --deal-sweeps 1 --max-depth 64 --betting-abstraction --postflop-buckets 200 --client-bet-sizing --trainer hogwild --hogwild-capacity 4194304 --no-sft --occupancy 6max --cfr-variant dcfr --vr-baseline --run-subdir $RUN
  $launches++
  Start-Sleep -Seconds 150
}
Write-Host "BABYSITTER EXIT after ${el}s, $launches launches, latest iter=$iter"
