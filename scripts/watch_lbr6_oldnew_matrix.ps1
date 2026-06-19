# READ-ONLY dashboard for the OLD-vs-NEW fc-LBR matrix (scripts\lbr6_oldnew_matrix_babysitter.ps1).
# Never launches or terminates anything -- just polls S3 + EC2 and prints per-cell status every $Every
# seconds. Run in a second terminal alongside the babysitter:
#   powershell -ExecutionPolicy Bypass -File scripts\watch_lbr6_oldnew_matrix.ps1
# ASCII ONLY.
param(
  [string]$RunTag = "lbr6_oldnew1",
  [string]$Menu   = "fc",
  [int]$Every     = 60
)
$ErrorActionPreference = "Continue"
$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
$BASE   = "pokerbot/prod6000/eval_suite"
$cells = @(
  @{ name = "cell1_OLD_fc";          run = "run_${RunTag}_cell1" },
  @{ name = "cell2_NEW_fc_fast";     run = "run_${RunTag}_cell2" },
  @{ name = "cell3_NEW_fc_robust";   run = "run_${RunTag}_cell3" },
  @{ name = "cell4_NEW_fcpa_fast";   run = "run_${RunTag}_cell4" },
  @{ name = "cell5_NEW_fcpa_robust"; run = "run_${RunTag}_cell5" },
  @{ name = "cell6_NEW_fcpa_raw";    run = "run_${RunTag}_cell6" }
)

function Test-Done([string]$run) {
  aws s3 ls "s3://$BUCKET/$BASE/$run/eval_suite_DONE.marker" --region $REGION 2>$null | Out-Null
  return $?
}
function Count-Results([string]$run) {
  $o = (aws s3 ls "s3://$BUCKET/$BASE/$run/results/" --region $REGION 2>$null)
  if (-not $o) { return @(0, 0) }
  $res = ([regex]::Matches(($o -join "`n"), "lbr6_[^ ]+\.json")).Count
  $fail = ([regex]::Matches(($o -join "`n"), "FAIL_[^ ]+")).Count
  return @($res, $fail)
}
function Box-State([string]$run) {
  $r = (aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$run" "Name=instance-state-name,Values=running,pending" --query "Reservations[].Instances[].[InstanceId,LaunchTime]" --output text 2>$null)
  if (-not $r) { return "" }
  $flat = (($r -join " ") -replace "\s+", " ").Trim()
  $id = (@($flat.Split(" ") | Where-Object { $_ -like "i-*" }) -join ",")
  $up = ""
  if ($flat -match "(\d{4}-\d{2}-\d{2}T[\d:.]+[+\-]\d{2}:?\d{2})") {
    try { $mins = [int](((Get-Date).ToUniversalTime() - [datetimeoffset]::Parse($Matches[1]).UtcDateTime).TotalMinutes); $up = " up${mins}m" } catch {}
  }
  return "$id$up"
}

Write-Host "=== WATCH OLD-vs-NEW fc-LBR matrix (tag=$RunTag, menu=$Menu, refresh ${Every}s) -- Ctrl+C to stop ==="
while ($true) {
  $stamp = (Get-Date).ToString("HH:mm:ss")
  Write-Host ""
  Write-Host "[$stamp]"
  foreach ($c in $cells) {
    $done = Test-Done $c.run
    $rc = Count-Results $c.run
    $box = Box-State $c.run
    $bb = ""
    if ($done) {
      $tmp = Join-Path $env:TEMP "watch_$($c.run)_SUMMARY.json"
      aws s3 cp "s3://$BUCKET/$BASE/$($c.run)/results/SUMMARY.json" $tmp --region $REGION --no-progress 2>$null | Out-Null
      if (Test-Path $tmp) {
        try { $j = Get-Content $tmp -Raw | ConvertFrom-Json; $m = $j.menus.$Menu; if ($m) { $bb = "  bb/100=$([math]::Round($m.bb_per_100,1)) ci[$([math]::Round($m.ci_lo,1)),$([math]::Round($m.ci_hi,1))]" } } catch {}
      }
    }
    $state = if ($done) { "DONE" } elseif ($box) { "RUN " } else { "wait" }
    Write-Host ("  {0,-20} {1}  results={2} fails={3} box='{4}'{5}" -f $c.name, $state, $rc[0], $rc[1], $box, $bb)
  }
  Start-Sleep -Seconds $Every
}
