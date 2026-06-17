# Live monitor for the NLHE skill-tier calibration job - refreshes every 60s (read-only, safe).
#   powershell -ExecutionPolicy Bypass -File scripts\watch_calibrate_nlhe.ps1
# Shows the box state + the latest [measure]/[fit] progress lines from the EC2 serial console
# (best-effort: console output lags ~1-3min) + the DONE marker. The on-box driver streams
# "[fit] <tier> iter <i> scale=<s> -> <bb/100>" as it bisects each leaky tier.
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252; non-ASCII bytes terminate strings early).
$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
$RUN    = "run_nlhetiers2"
$PREFIX = "pokerbot/prod6000/eval_suite/$RUN"
$t0 = Get-Date

while ($true) {
  $box = aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=running,pending,shutting-down" --query "Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]" --output text 2>$null
  $boxId = ""
  if ($box -and $box -match "(i-[0-9a-f]+)") { $boxId = $Matches[1] }

  Clear-Host
  Write-Host ("=== nlhe-tier calibration: $RUN  @ {0} ===" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
  Write-Host ("elapsed: {0}min   box: {1}" -f [int]((Get-Date) - $t0).TotalMinutes, $(if ($box) { $box } else { "(none running)" }))
  Write-Host ""

  if ($boxId) {
    $con = aws ec2 get-console-output --instance-id $boxId --region $REGION --latest --output text 2>$null
    if ($con) {
      $lines = @($con -split "`n" | Select-String -Pattern "\[measure\]|\[fit\]|Traceback|Error|wrote |uploaded |IMPORTS OK|suite exit")
      if ($lines.Count) {
        Write-Host "recent progress (console, lags ~1-3min):"
        $lines | Select-Object -Last 16 | ForEach-Object { Write-Host ("  " + $_.Line.Trim()) }
      } else {
        Write-Host "  (box up; no [measure]/[fit] lines on the console yet - bootstrap + model load takes ~5-15min)"
      }
    } else {
      Write-Host "  (console output not available yet)"
    }
  } else {
    Write-Host "  (no live box - babysitter will (re)launch, or the run finished)"
  }

  Write-Host ""
  $res = aws s3 ls "s3://$BUCKET/$PREFIX/results/nlhe_skill_tiers.json" --region $REGION 2>$null
  if ($res) { Write-Host "RESULT: nlhe_skill_tiers.json written to S3" -ForegroundColor Green }
  aws s3 ls "s3://$BUCKET/$PREFIX/calibrate_DONE.marker" --region $REGION 2>$null | Out-Null
  if ($?) { Write-Host "DONE marker present - calibration COMPLETE" -ForegroundColor Green }
  Write-Host "(refresh 60s; Ctrl-C to stop)"
  Start-Sleep -Seconds 60
}
