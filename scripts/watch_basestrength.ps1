# Live monitor for the BASE-STRENGTH cloud job - refreshes every 60s (read-only, safe).
#   powershell -ExecutionPolicy Bypass -File scripts\watch_basestrength.ps1
# Shows the box state + the latest per-cell completion lines from the EC2 serial console
# (best-effort: console output lags ~1-3min) + the SUMMARY/DONE markers. The on-box driver logs
# "[HH:MM:SS] bs_<base>_<cell>_s<seed> rc=0" as each (base,cell,seed) job finishes.
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252; non-ASCII bytes terminate strings early).
$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
$RUN    = "run_bs_uniform1"   # match the babysitter's active $RUN (run_bs_gadget1 for the gadget variant)
$PREFIX = "pokerbot/prod6000/eval_suite/$RUN"
$t0 = Get-Date

while ($true) {
  $box = aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=running,pending,shutting-down" --query "Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]" --output text 2>$null
  $boxId = ""
  if ($box -and $box -match "(i-[0-9a-f]+)") { $boxId = $Matches[1] }

  Clear-Host
  Write-Host ("=== base-strength: $RUN  @ {0} ===" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
  Write-Host ("elapsed: {0}min   box: {1}" -f [int]((Get-Date) - $t0).TotalMinutes, $(if ($box) { $box } else { "(none running)" }))
  Write-Host ""

  if ($boxId) {
    $con = aws ec2 get-console-output --instance-id $boxId --region $REGION --latest --output text 2>$null
    if ($con) {
      $lines = @($con -split "`n" | Select-String -Pattern "basestrength:|bs_.* rc=|BASESTRENGTH|Traceback|Error|uploaded |IMPORTS OK|suite exit")
      if ($lines.Count) {
        Write-Host "recent progress (console, lags ~1-3min):"
        $lines | Select-Object -Last 18 | ForEach-Object { Write-Host ("  " + $_.Line.Trim()) }
      } else {
        Write-Host "  (box up; no basestrength lines on the console yet - bootstrap + model load takes ~5-15min)"
      }
    } else {
      Write-Host "  (console output not available yet)"
    }
  } else {
    Write-Host "  (no live box - babysitter will (re)launch, or the run finished)"
  }

  Write-Host ""
  $done = $false
  aws s3 ls "s3://$BUCKET/$PREFIX/eval_suite_DONE.marker" --region $REGION 2>$null | Out-Null
  if ($?) { $done = $true }
  $res = aws s3 ls "s3://$BUCKET/$PREFIX/results/SUMMARY.json" --region $REGION 2>$null
  if ($res) { Write-Host "RESULT: SUMMARY.json written to S3" -ForegroundColor Green }
  # Count completed per-cell result objects (bs_*.json) so you can watch the 2 x 9 x seeds grid fill.
  $n = (aws s3 ls "s3://$BUCKET/$PREFIX/results/" --region $REGION 2>$null | Select-String -Pattern "bs_.*\.json").Count
  Write-Host ("completed cells: {0} result objects" -f $n)
  if ($done) { Write-Host "DONE marker present - run COMPLETE" -ForegroundColor Green }
  Write-Host "(refresh 60s; Ctrl-C to stop)"
  Start-Sleep -Seconds 60
}
