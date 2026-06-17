# Live monitor for the cloud exploitation-eval suite - refreshes every 60s (read-only, safe).
#   powershell -ExecutionPolicy Bypass -File scripts\watch_eval_suite.ps1
# ASCII ONLY in this file: PS 5.1 reads BOM-less .ps1 as cp1252; UTF-8 em dashes decode into
# smart-quote bytes that TERMINATE strings early (parse errors far from the real cause).
$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
$RUN    = "run_evalsuite1"
$PREFIX = "pokerbot/prod6000/eval_suite/$RUN"

# family -> expected count (phase B families appear only after the automated gate passes)
$FAM = [ordered]@{
  "rmA"     = 72;  "rmB"     = 72;   # P0 arms: 4 archetypes x 18 seeds (pluribus hero)
  "sw_"     = 162; "pab_"    = 36;   # G4 sweep + G3 prior A/B (heuristic hero)
  "bluff"   = 72;                    # P2 (phase B)
  "weakmix" = 18;  "counter" = 18;  "bpbot" = 18   # G3 gauntlet + P5 (phase B)
}
$TOTAL_A = 342; $TOTAL = 468        # phase A (144 P0 + 162 sweep + 36 priorAB) vs + phase B (126)
$prevN = -1; $lastChange = Get-Date; $t0 = Get-Date; $n0 = $null

while ($true) {
  $res = (aws s3 ls "s3://$BUCKET/$PREFIX/results/" --region $REGION 2>$null)
  $rows = if ($res) { @($res | Select-String "\.json" | Where-Object { $_ -notmatch "SUMMARY" }) } else { @() }
  $n = $rows.Count
  $fails = if ($res) { @($res | Select-String "FAIL_").Count } else { 0 }
  if ($null -eq $n0) { $n0 = $n }
  if ($n -ne $prevN) { $prevN = $n; $lastChange = Get-Date }
  $stale = [int]((Get-Date) - $lastChange).TotalSeconds
  $elMin = [math]::Max(0.1, ((Get-Date) - $t0).TotalMinutes)
  $rate = ($n - $n0) / $elMin
  $gateDone = ($n -ge $TOTAL_A)
  $target = if ($gateDone) { $TOTAL } else { $TOTAL_A }
  $phase = if ($gateDone) { "phase B live" } else { "phase A" }
  $eta = if ($rate -gt 0.05) { "~$([int](($target - $n) / $rate))min" } else { "-" }

  Clear-Host
  Write-Host ("=== eval suite: $RUN  @ {0} ===" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
  Write-Host ("results: {0}/{1} ({2})   fails: {3}   rate: {4:n1}/min   ETA: {5}" -f $n, $target, $phase, $fails, $rate, $eta)
  Write-Host ("last new result: {0}s ago" -f $stale)
  if ($stale -gt 900 -and $n -gt 0) { Write-Host "  ^ WARNING: >15min without a new result (box reclaimed/stalled?)" -ForegroundColor Yellow }
  if ($n -eq 0) { Write-Host "  (no results yet - bootstrap + first 3000-hand jobs take ~30-60min after launch)" }
  Write-Host ""
  foreach ($k in $FAM.Keys) {
    $c = @($rows | Where-Object { $_ -match $k }).Count
    $w = [int](20 * [math]::Min(1.0, $c / $FAM[$k]))
    $bar = ("#" * $w) + ("-" * (20 - $w))
    Write-Host ("  {0,-8} [{1}] {2,3}/{3}" -f $k, $bar, $c, $FAM[$k])
  }
  if ($rows.Count -gt 0) {
    Write-Host ""
    Write-Host "recent:"
    $rows | Select-Object -Last 5 | ForEach-Object { Write-Host ("  " + $_.Line.Trim()) }
  }
  $box = aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=running,pending,shutting-down" --query "Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]" --output text 2>$null
  Write-Host ""
  Write-Host ("box: {0}" -f $(if ($box) { $box } else { "(none running)" }))
  $summary = aws s3 ls "s3://$BUCKET/$PREFIX/results/SUMMARY.json" --region $REGION 2>$null
  if ($summary) { Write-Host "GATE/SUMMARY: written (suite finishing or done)" -ForegroundColor Green }
  aws s3 ls "s3://$BUCKET/$PREFIX/eval_suite_DONE.marker" --region $REGION 2>$null | Out-Null
  if ($?) { Write-Host "DONE marker present - suite COMPLETE" -ForegroundColor Green }
  Write-Host "(refresh 60s; Ctrl-C to stop)"
  Start-Sleep -Seconds 60
}
