# Live monitor for the 6-MAX DCFR+VR pilot — refreshes every 60s.
#   powershell -ExecutionPolicy Bypass -File scripts\watch_pilot_6max_dcfrvr.ps1
$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
$RUN    = "run_6max_dcfrvr2"
$PREFIX = "pokerbot/prod6000/sd_cfr_hu/$RUN"
$prevMax = -1
$lastChange = Get-Date

while ($true) {
  Clear-Host
  Write-Host ("=== 6max DCFR+VR pilot: $RUN  @ {0} ===" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
  $lines = aws s3 ls "s3://$BUCKET/$PREFIX/" --recursive --region $REGION 2>$null | Select-String "iter_\d+/.*\.pkl"
  if ($lines) {
    $iters = [regex]::Matches(($lines -join "`n"), "iter_(\d+)") | ForEach-Object { [int]$_.Groups[1].Value }
    $max = ($iters | Measure-Object -Maximum).Maximum
    Write-Host ("checkpoints: {0}   latest: iter_{1}" -f $lines.Count, $max)
    if ($max -ne $prevMax) { $prevMax = $max; $lastChange = Get-Date }
    $stale = [int]((Get-Date) - $lastChange).TotalSeconds
    Write-Host ("last new checkpoint: {0}s ago" -f $stale)
    if ($stale -gt 360) { Write-Host "  ^ WARNING: >6min since a new checkpoint (box reclaimed/stalled?)" -ForegroundColor Yellow }
    $lines | Select-Object -Last 6 | ForEach-Object { Write-Host ("  " + $_.Line) }
  } else {
    Write-Host "no checkpoints yet"
  }
  $box = aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=running,pending,shutting-down" --query "Reservations[].Instances[].[InstanceId,State.Name]" --output text 2>$null
  Write-Host ("box: {0}" -f $(if ($box) { $box } else { "(none running)" }))
  Write-Host "(refresh 60s; Ctrl-C to stop)"
  Start-Sleep -Seconds 60
}
