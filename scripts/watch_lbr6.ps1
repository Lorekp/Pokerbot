# At-a-glance dashboard for the 6-MAX LBR exploitability cloud run (--mode lbr6). READ-ONLY.
#   powershell -ExecutionPolicy Bypass -File scripts\watch_lbr6.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\watch_lbr6.ps1 -Run run_lbr6_1 -Seeds 120 -Refresh 45
# Shows the spot box, blueprint-seed progress + the 2 calibration cells, fail count, ETA, and the
# DONE/SUMMARY status. Pair with lbr6_babysitter.ps1 (which launches). ASCII ONLY.
param(
  [string]$Run = "run_lbr6_graded1",
  [int]$Seeds = 60,
  [int]$Refresh = 60
)
$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
$PREFIX = "pokerbot/prod6000/eval_suite/$Run"
$MENUS  = @("fc", "fcpa", "full")
$TOTAL  = $Seeds * $MENUS.Count + 2   # 3 menus x seeds + foldbot + station calibration
$t0 = Get-Date
$baseDone = -1
$baseTime = $t0

function Bar([int]$d, [int]$n) {
  if ($d -gt $n) { $d = $n }
  $w = 40
  $f = [int]($w * $d / [Math]::Max($n, 1))
  return ("#" * $f) + ("-" * ($w - $f))
}

while ($true) {
  $boxRaw = aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$Run" "Name=instance-state-name,Values=running,pending" --query "Reservations[].Instances[].[InstanceId,State.Name,InstanceType,LaunchTime]" --output text 2>$null
  $boxLine = "(no live box - babysitter idle, reclaimed, or finished)"
  if ($boxRaw) {
    $f = ($boxRaw -split "\s+")
    $up = ""
    if ($boxRaw -match "(\d{4}-\d{2}-\d{2}T[\d:.]+[+\-]\d{2}:?\d{2})") {
      try { $m = [int](((Get-Date).ToUniversalTime() - [datetimeoffset]::Parse($Matches[1]).UtcDateTime).TotalMinutes); $up = "up ${m}m" } catch {}
    }
    $boxLine = ("{0}  {1}  {2}  {3}" -f $f[0], $f[1], $f[2], $up)
  }

  $ls = aws s3 ls "s3://$BUCKET/$PREFIX/results/" --region $REGION 2>$null
  $names = @()
  if ($ls) { $names = @($ls | ForEach-Object { ($_ -split "\s+")[-1] }) }
  $done = @{}; foreach ($m in $MENUS) { $done[$m] = 0 }
  $cal = 0; $fail = 0
  foreach ($nm in $names) {
    if ($nm -match "^lbr6_(fc|fcpa|full)_s\d+\.json$") { $done[$Matches[1]]++ }
    elseif ($nm -match "^lbr6_cal_[a-z]+_s0\.json$") { $cal++ }
    elseif ($nm -match "^FAIL_lbr6_") { $fail++ }
  }
  $menuDone = 0; foreach ($m in $MENUS) { $menuDone += $done[$m] }
  $totalDone = $menuDone + $cal

  if ($baseDone -lt 0) { $baseDone = $totalDone; $baseTime = Get-Date }
  $mins = ((Get-Date) - $baseTime).TotalMinutes
  $rate = if ($mins -gt 0.1) { ($totalDone - $baseDone) / $mins } else { 0 }
  $eta = if ($rate -gt 0.01) { "~{0}m" -f [int](($TOTAL - $totalDone) / $rate) } else { "--" }

  $sum = aws s3 ls "s3://$BUCKET/$PREFIX/results/SUMMARY.json" --region $REGION 2>$null
  aws s3 ls "s3://$BUCKET/$PREFIX/eval_suite_DONE.marker" --region $REGION 2>$null | Out-Null
  $isDone = $?

  Clear-Host
  Write-Host ("=== {0} 6-max LBR exploitability   {1}   (watching {2}m) ===" -f $Run, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), [int]((Get-Date) - $t0).TotalMinutes)
  Write-Host ("box:   {0}" -f $boxLine)
  $failTxt = if ($fail -gt 0) { "$fail fail" } else { "0 fail" }
  Write-Host ("cells: {0}/{1} done   {2}   rate {3:N1}/min   ETA {4}" -f $totalDone, $TOTAL, $failTxt, $rate, $eta)
  Write-Host ""
  foreach ($m in $MENUS) {
    Write-Host ("  menu {0,-5} [{1}] {2,3}/{3}" -f $m, (Bar $done[$m] $Seeds), $done[$m], $Seeds)
  }
  Write-Host ("  calibration [{0}] {1,3}/2   (foldbot + station competence check)" -f (Bar $cal 2), $cal)
  Write-Host ""
  if ($fail -gt 0) {
    Write-Host ("  WARNING: {0} FAIL_lbr6_*.log present (check: aws s3 cp s3://$BUCKET/$PREFIX/results/FAIL_lbr6_bp_s0.log -)" -f $fail) -ForegroundColor Yellow
  }
  $sumTxt = if ($sum) { "written" } else { "not yet" }
  $doneTxt = if ($isDone) { "YES - run COMPLETE" } else { "no" }
  Write-Host ("SUMMARY: {0}    DONE: {1}" -f $sumTxt, $doneTxt)
  if ($isDone) {
    Write-Host ""
    Write-Host "Fetch it:  aws s3 cp s3://$BUCKET/$PREFIX/results/SUMMARY.json -" -ForegroundColor Green
    break
  }
  Write-Host ("(refresh {0}s; Ctrl-C to stop)" -f $Refresh)
  Start-Sleep -Seconds $Refresh
}
