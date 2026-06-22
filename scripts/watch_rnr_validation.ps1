# At-a-glance dashboard for the RNR-restriction validation cloud run (--mode rnr_validation). READ-ONLY.
#   powershell -ExecutionPolicy Bypass -File scripts\watch_rnr_validation.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\watch_rnr_validation.ps1 -Run run_rnr1 -Refresh 45 -Seeds 12
# Shows the spot box, per-(field,arm) progress (3 fields x {off,on} x SEEDS), cells done, fail count,
# rate + ETA, and DONE/SUMMARY status. Pair with rnr_validation_babysitter.ps1 (which launches).
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252; non-ASCII bytes break strings).
param(
  [string]$Run = "run_rnr1",
  [int]$Refresh = 60,
  [int]$Seeds = 12
)
$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
$PREFIX = "pokerbot/prod6000/eval_suite/$Run"
$FIELDS = @("soft_table", "realistic_table", "pro_table")
$ARMS   = @("off", "on")
$TOTAL  = $FIELDS.Count * $ARMS.Count * $Seeds
$t0 = Get-Date
$baseDone = -1
$baseTime = $t0

function Bar([int]$d, [int]$n) {
  if ($d -gt $n) { $d = $n }
  return ("#" * $d) + ("-" * ($n - $d))
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
  $done = @{}; foreach ($fld in $FIELDS) { foreach ($arm in $ARMS) { $done["${arm}_${fld}"] = 0 } }
  $fail = 0
  foreach ($nm in $names) {
    if ($nm -match "^rnr_(off|on)_([a-z_]+)_s\d+\.json$") {
      $key = "$($Matches[1])_$($Matches[2])"
      if ($done.ContainsKey($key)) { $done[$key]++ }
    }
    elseif ($nm -match "^FAIL_rnr_") { $fail++ }
  }
  $totalDone = 0; foreach ($k in $done.Keys) { $totalDone += $done[$k] }

  if ($baseDone -lt 0) { $baseDone = $totalDone; $baseTime = Get-Date }
  $mins = ((Get-Date) - $baseTime).TotalMinutes
  $rate = if ($mins -gt 0.1) { ($totalDone - $baseDone) / $mins } else { 0 }
  $eta = if ($rate -gt 0.01) { "~{0}m" -f [int](($TOTAL - $totalDone) / $rate) } else { "--" }

  $sum = aws s3 ls "s3://$BUCKET/$PREFIX/results/SUMMARY.json" --region $REGION 2>$null
  aws s3 ls "s3://$BUCKET/$PREFIX/eval_suite_DONE.marker" --region $REGION 2>$null | Out-Null
  $isDone = $?

  Clear-Host
  Write-Host ("=== {0} rnr_validation   {1}   (watching {2}m) ===" -f $Run, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), [int]((Get-Date) - $t0).TotalMinutes)
  Write-Host ("box:   {0}" -f $boxLine)
  $failTxt = if ($fail -gt 0) { "$fail fail" } else { "0 fail" }
  Write-Host ("cells: {0}/{1} done   {2}   rate {3:N1}/min   ETA {4}" -f $totalDone, $TOTAL, $failTxt, $rate, $eta)
  Write-Host ""
  foreach ($fld in $FIELDS) {
    foreach ($arm in $ARMS) {
      $key = "${arm}_${fld}"
      Write-Host ("  {0,-22} [{1}] {2,2}/{3}" -f $key, (Bar $done[$key] $Seeds), $done[$key], $Seeds)
    }
  }
  Write-Host ""
  if ($fail -gt 0) {
    Write-Host ("  WARNING: {0} FAIL_rnr_*.log present (check: aws s3 cp s3://$BUCKET/$PREFIX/results/FAIL_rnr_off_soft_table_s0.log -)" -f $fail) -ForegroundColor Yellow
  }
  $sumTxt = if ($sum) { "written" } else { "not yet" }
  $doneTxt = if ($isDone) { "YES - run COMPLETE" } else { "no" }
  Write-Host ("SUMMARY: {0}    DONE: {1}" -f $sumTxt, $doneTxt)
  if ($isDone) {
    Write-Host ""
    Write-Host "Done. The babysitter downloads runs\cloud_rnr_validation\SUMMARY.json and prints the verdict." -ForegroundColor Green
    break
  }
  Write-Host ("(refresh {0}s; Ctrl-C to stop)" -f $Refresh)
  Start-Sleep -Seconds $Refresh
}
