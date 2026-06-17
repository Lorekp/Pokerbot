# Autonomous babysitter for the cloud exploitation-eval suite (Exploitation Program v2 batteries).
# Relaunches the spot box on reclaim until the DONE marker appears (per-job S3 resume makes
# reclaims cheap), then downloads all results to runs/cloud_eval/ and prints the pooled summary.
# HARD caps so it can never run away. Live heartbeat every 180s in the old babysitter style:
#   [t+1234s] results=151/342 (+9, 2.8/min) live='i-0abc (up 47m)' launches=1 fails=0
# Run from your terminal (your AWS creds):
#   powershell -ExecutionPolicy Bypass -File scripts\eval_suite_babysitter.ps1
# ASCII ONLY in this file (PS 5.1 + BOM-less UTF-8 mangles non-ASCII into string-breaking bytes).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

$REGION = "eu-central-1"
$RUN    = "run_evalsuite1"
$BUCKET = "video-processing-6789"
$PREFIX = "pokerbot/prod6000/eval_suite/$RUN"
$MAXSEC = 18 * 3600     # hard wall-clock cap ~18h (m7g.16xlarge spot ~1.3 USD/h -> ~24 USD worst case)
$MAXLAUNCH = 8
$start = Get-Date
$launches = 0
$prevBox = ""
$prevN = 0
$prevT = Get-Date
$lastNew = Get-Date

function Get-Results {
  $out = (aws s3 ls "s3://$BUCKET/$PREFIX/results/" --region $REGION 2>$null)
  $n = 0; $fails = 0; $fam = ""
  if ($out) {
    $rows = @($out | Select-String "\.json" | Where-Object { $_ -notmatch "SUMMARY" })
    $n = $rows.Count
    $fails = @($out | Select-String "FAIL_").Count
    $parts = foreach ($k in "rmA","rmB","sw_","pab_","bluff","weakmix","counter","bpbot") {
      $c = @($rows | Where-Object { $_ -match $k }).Count
      if ($c) { "${k}:$c" }
    }
    $fam = ($parts -join " ")
  }
  return @($n, $fails, $fam)
}
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
    if ($idList.Count) {
      aws ec2 terminate-instances --instance-ids $idList --region $REGION | Out-Null
      Write-Host "terminated: $idList"
    }
  }
}

$capH = [int]($MAXSEC / 3600)
Write-Host "=== eval-suite babysitter: $RUN (caps: ${capH}h / $MAXLAUNCH launches) ==="
while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  $r = Get-Results
  $n = $r[0]; $fails = $r[1]; $fam = $r[2]
  $live = Get-Boxes "running,pending"
  $boxId = ""
  $up = ""
  if ($live) {
    $boxId = (@($live.Split(" ") | Where-Object { $_ -like "i-*" }) -join ",")
    if ($live -match "(\d{4}-\d{2}-\d{2}T[\d:.]+)") {
      try {
        $mins = [int](((Get-Date).ToUniversalTime() - [datetime]::Parse($Matches[1]).ToUniversalTime()).TotalMinutes)
        $up = " (up ${mins}m)"
      } catch {}
    }
  }
  $dMin = [math]::Max(0.1, ((Get-Date) - $prevT).TotalMinutes)
  $delta = $n - $prevN
  if ($delta -gt 0) { $lastNew = Get-Date }
  $rate = "{0:n1}" -f ($delta / $dMin)
  $staleMin = [int]((Get-Date) - $lastNew).TotalMinutes
  $target = if ($n -ge 342) { 468 } else { 342 }
  Write-Host "[t+${el}s] results=$n/$target (+$delta, $rate/min) live='$boxId$up' launches=$launches fails=$fails"
  if ($fam) { Write-Host "          $fam" }
  if ($boxId -and $staleMin -ge 15 -and $n -gt 0) {
    Write-Host "          WARNING: no new result for ${staleMin}min with a live box (stall?)" -ForegroundColor Yellow
  }
  if ($prevBox -and $boxId -and $boxId -ne $prevBox) {
    Write-Host "          BOX CHANGED: $prevBox -> $boxId (spot reclaim + relaunch)" -ForegroundColor Yellow
  }
  if ($boxId) { $prevBox = $boxId }
  $prevN = $n; $prevT = Get-Date

  if (Test-Done) { Write-Host "DONE: suite marker found"; Stop-Run; break }
  if ($el -ge $MAXSEC) { Write-Host "STOP: time cap reached"; Stop-Run; break }
  if ($live) { Start-Sleep -Seconds 180; continue }
  if ($launches -ge $MAXLAUNCH) { Write-Host "STOP: max launches ($MAXLAUNCH) hit"; break }
  $next = $launches + 1
  Write-Host "  no live box -> launching #$next (per-job S3 resume - nothing already done re-runs)"
  python scripts\launch_eval_suite_spot.py --run-subdir $RUN
  $launches++
  Start-Sleep -Seconds 180
}

# Download everything + pooled verdicts.
New-Item -ItemType Directory -Force "runs/cloud_eval" | Out-Null
aws s3 cp "s3://$BUCKET/$PREFIX/results/" runs/cloud_eval/ --recursive --region $REGION --no-progress
Write-Host ""
Write-Host "=== SUMMARY ==="
Get-Content runs/cloud_eval/SUMMARY.json -ErrorAction SilentlyContinue
python scripts/summarize_exploit_battery.py "runs/cloud_eval/eb3000_rmA_*.json" "runs/cloud_eval/eb3000_rmB_*.json" "runs/cloud_eval/eb3000_bluff*_*.json" "runs/cloud_eval/eb3000_weakmix*.json" "runs/cloud_eval/eb3000_counter*.json" "runs/cloud_eval/eb3000_bpbot*.json"
Write-Host "BABYSITTER EXIT after ${el}s, $launches launches"
