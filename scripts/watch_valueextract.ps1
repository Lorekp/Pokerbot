# At-a-glance dashboard for the PHASE-2 VALUE-EXTRACTION cloud run (--mode valueextract). READ-ONLY.
#   powershell -ExecutionPolicy Bypass -File scripts\watch_valueextract.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\watch_valueextract.ps1 -Run run_ve1 -Refresh 45
# Shows the spot box, a per-field progress bar (5 IRC fields x SEEDS), total cells done, fail count,
# completion rate + ETA, and the DONE/SUMMARY status. Pair with valueextract_babysitter.ps1 (which does
# the launching). ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252; non-ASCII bytes break strings).
param(
  [string]$Run = "run_ve1",
  [int]$Refresh = 60,
  [int]$Seeds = 12
)
$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
$PREFIX = "pokerbot/prod6000/eval_suite/$Run"
$FIELDS = @("irc_soft", "irc_station", "irc_mixed", "irc_tough", "irc_lag")
$WEAK   = @("irc_soft", "irc_station", "irc_mixed")
$TOTAL  = $FIELDS.Count * $Seeds
$t0 = Get-Date
$baseDone = -1
$baseTime = $t0

function Bar([int]$d, [int]$n) {
  if ($d -gt $n) { $d = $n }
  return ("#" * $d) + ("-" * ($n - $d))
}

while ($true) {
  # --- box state (tag:Run; may lag a few min for a fresh box) ---
  $boxRaw = aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$Run" "Name=instance-state-name,Values=running,pending" --query "Reservations[].Instances[].[InstanceId,State.Name,InstanceType,LaunchTime]" --output text 2>$null
  $boxLine = "(no live box - babysitter idle, reclaimed, or finished)"
  if ($boxRaw) {
    $f = ($boxRaw -split "\s+")
    $up = ""
    if ($boxRaw -match "(\d{4}-\d{2}-\d{2}T[\d:.]+[+\-]\d{2}:?\d{2})") {
      # Capture the FULL LaunchTime incl. the +00:00 offset and parse as DateTimeOffset, else the UTC
      # stamp is read as local and uptime is inflated by the local offset (the phantom "+120m" bug).
      try { $m = [int](((Get-Date).ToUniversalTime() - [datetimeoffset]::Parse($Matches[1]).UtcDateTime).TotalMinutes); $up = "up ${m}m" } catch {}
    }
    $boxLine = ("{0}  {1}  {2}  {3}" -f $f[0], $f[1], $f[2], $up)
  }

  # --- cell listing ---
  $ls = aws s3 ls "s3://$BUCKET/$PREFIX/results/" --region $REGION 2>$null
  $names = @()
  if ($ls) { $names = @($ls | ForEach-Object { ($_ -split "\s+")[-1] }) }
  $done = @{}; foreach ($fld in $FIELDS) { $done[$fld] = 0 }
  $fail = 0
  foreach ($nm in $names) {
    if ($nm -match "^ve_(irc_[a-z]+)_s\d+\.json$") { if ($done.ContainsKey($Matches[1])) { $done[$Matches[1]]++ } }
    elseif ($nm -match "^FAIL_ve_") { $fail++ }
  }
  $totalDone = 0; foreach ($fld in $FIELDS) { $totalDone += $done[$fld] }

  # --- rate + ETA (anchored at first poll) ---
  if ($baseDone -lt 0) { $baseDone = $totalDone; $baseTime = Get-Date }
  $mins = ((Get-Date) - $baseTime).TotalMinutes
  $rate = if ($mins -gt 0.1) { ($totalDone - $baseDone) / $mins } else { 0 }
  $eta = if ($rate -gt 0.01) { "~{0}m" -f [int](($TOTAL - $totalDone) / $rate) } else { "--" }

  # --- status flags ---
  $sum = aws s3 ls "s3://$BUCKET/$PREFIX/results/SUMMARY.json" --region $REGION 2>$null
  aws s3 ls "s3://$BUCKET/$PREFIX/eval_suite_DONE.marker" --region $REGION 2>$null | Out-Null
  $isDone = $?

  Clear-Host
  Write-Host ("=== {0} value-extraction   {1}   (watching {2}m) ===" -f $Run, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), [int]((Get-Date) - $t0).TotalMinutes)
  Write-Host ("box:   {0}" -f $boxLine)
  $failTxt = if ($fail -gt 0) { "$fail fail" } else { "0 fail" }
  Write-Host ("cells: {0}/{1} done   {2}   rate {3:N1}/min   ETA {4}" -f $totalDone, $TOTAL, $failTxt, $rate, $eta)
  Write-Host ""
  foreach ($fld in $FIELDS) {
    $tag = if ($WEAK -contains $fld) { "weak " } else { "tough" }
    Write-Host ("  {0,-12} [{1}] {2,2}/{3}  {4}" -f $fld, (Bar $done[$fld] $Seeds), $done[$fld], $Seeds, $tag)
  }
  Write-Host ""
  if ($fail -gt 0) {
    Write-Host ("  WARNING: {0} FAIL_ve_*.log present - cells erroring (check a log: aws s3 cp s3://$BUCKET/$PREFIX/results/FAIL_ve_irc_soft_s0.log -)" -f $fail) -ForegroundColor Yellow
  }
  $sumTxt = if ($sum) { "written" } else { "not yet" }
  $doneTxt = if ($isDone) { "YES - run COMPLETE" } else { "no" }
  Write-Host ("SUMMARY: {0}    DONE: {1}" -f $sumTxt, $doneTxt)
  if ($isDone) {
    Write-Host ""
    Write-Host "Pool it:  python scripts\pool_valueextract_local.py runs\cloud_ve1" -ForegroundColor Green
    break
  }
  Write-Host ("(refresh {0}s; Ctrl-C to stop)" -f $Refresh)
  Start-Sleep -Seconds $Refresh
}
