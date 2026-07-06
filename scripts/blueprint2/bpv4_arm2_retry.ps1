# Spot-storm retry supervisor for the arm-2 babysitter (user decision 2026-07-06: KEEP TRYING SPOT).
# Relaunches bpv4_arm2_babysitter.ps1 -ReuseCode whenever it exits without reaching the 40k target,
# with a cool-down between cycles (each failed cycle costs <= 3 short-lived boxes, ~2-4 USD).
# Stop: create artifacts\_arm2_retry.stop . Hard wall: 24h. ASCII ONLY.
# 72h wall: 64-vCPU boxes run ~48 min/1000 iters, so 40k needs ~32h of TRAINING time before churn.
param([int]$CooldownSec = 2700, [int]$MaxHours = 72)
Set-Location (Join-Path $PSScriptRoot "..\..")
$deadline = (Get-Date).AddHours($MaxHours)
$cycle = 0
function Get-Iters {
  $keys = (aws s3api list-objects-v2 --bucket video-processing-6789 --prefix "pokerbot/prod6000/native_cfr/bpv4_h200_arm2/checkpoints/" --query "Contents[].Key" --output text --region eu-central-1 2>$null)
  $best = [int64]-1
  if ($keys) { foreach ($k in ($keys -split "\s+")) { if ($k -match "iter_(\d+)\.nlcfr\.ok$") { $v=[int64]$Matches[1]; if ($v -gt $best) { $best=$v } } } }
  return $best
}
while ((Get-Date) -lt $deadline) {
  if (Test-Path "artifacts\_arm2_retry.stop") { Write-Output "[retry] stop sentinel -> exit"; break }
  $it = Get-Iters
  if ($it -ge 40000) { Write-Output "[retry] target reached ($it) -> exit"; break }
  $cycle++
  Write-Output "[retry] cycle $cycle starting at $(Get-Date -Format HH:mm:ss) (iters=$it)"
  & powershell -ExecutionPolicy Bypass -File "scripts\blueprint2\bpv4_arm2_babysitter.ps1" -ReuseCode 2>&1 |
    Out-File -Append "artifacts\_arm2_babysitter_run3.log" -Encoding utf8
  $it2 = Get-Iters
  Write-Output "[retry] cycle $cycle babysitter exited at $(Get-Date -Format HH:mm:ss) (iters $it -> $it2)"
  if ($it2 -ge 40000) { Write-Output "[retry] target reached -> exit"; break }
  Write-Output "[retry] cooldown $CooldownSec s (pool may calm)"
  $slept = 0
  while ($slept -lt $CooldownSec) {
    if (Test-Path "artifacts\_arm2_retry.stop") { Write-Output "[retry] stop sentinel during cooldown -> exit"; exit }
    Start-Sleep -Seconds 60; $slept += 60
  }
}
Write-Output "[retry] done at $(Get-Date -Format HH:mm:ss)"
