# Read-only dashboard for scripts\lossless_betting_pipeline_babysitter.ps1. Refreshes every 30s.
#   powershell -ExecutionPolicy Bypass -File scripts\watch_lossless_betting_pipeline.ps1
# ASCII ONLY.
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

$REGION  = "eu-central-1"
$RUN     = "run_6max_lossless3"
$BUCKET  = "video-processing-6789"
$PREFIX  = "pokerbot/prod6000/sd_cfr_hu/$RUN"
$OUT     = "runs\lossless_betting_pipeline"
$STATUS  = "$OUT\STATUS.txt"
$GATELOG = "$OUT\gate.log"
$DONE    = "$OUT\PIPELINE_DONE.txt"
$TARGET  = 300

function Get-LatestIter {
  $out = (aws s3 ls "s3://$BUCKET/$PREFIX/" --recursive --region $REGION 2>$null)
  $m = [regex]::Matches(($out -join "`n"), "sd_cfr_hu_iter_(\d+)\.pkl")
  if ($m.Count -gt 0) { return ($m | ForEach-Object { [int]$_.Groups[1].Value } | Measure-Object -Maximum).Maximum }
  return 0
}
function Get-Box {
  $r = (aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=running,pending" --query "Reservations[].Instances[].[InstanceId,State.Name]" --output text 2>$null)
  if ($r) { return ($r -replace "\s+", " ").Trim() } else { return "(none)" }
}

while ($true) {
  Clear-Host
  Write-Host "=== LOSSLESS-PREFLOP-BETTING PIPELINE WATCHER  $((Get-Date).ToString('u')) ===" -ForegroundColor Cyan
  Write-Host "run=$RUN  region=$REGION"
  Write-Host ""
  Write-Host "--- babysitter STATUS ---" -ForegroundColor Yellow
  if (Test-Path $STATUS) { Get-Content $STATUS } else { Write-Host "(no STATUS yet - babysitter not started)" }
  Write-Host ""
  Write-Host "--- live (S3 / EC2) ---" -ForegroundColor Yellow
  Write-Host ("S3 latest iter : {0}/{1}" -f (Get-LatestIter), $TARGET)
  Write-Host ("training box   : {0}" -f (Get-Box))
  if (Test-Path $GATELOG) {
    Write-Host ""
    Write-Host "--- gate.log (tail) ---" -ForegroundColor Yellow
    Get-Content $GATELOG -Tail 16
  }
  if (Test-Path $DONE) {
    Write-Host ""
    Write-Host "--- FINAL VERDICT ---" -ForegroundColor Green
    Get-Content $DONE
    Write-Host ""
    Write-Host "Pipeline finished. Stopping watcher."
    break
  }
  Start-Sleep -Seconds 30
}
