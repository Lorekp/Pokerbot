# Retry supervisor for the arm-2b babysitter. Original role: spot-storm cycling. Current role
# (2026-07-06 ~12:00): ON-DEMAND QUOTA WATCHER -- the account's on-demand vCPU quota (16) blocks
# the c6a.32xlarge launch until the user's Service Quotas increase lands; failed RunInstances
# attempts are free, so this loop just re-runs the babysitter until a launch sticks and training
# reaches 40k. Each babysitter cycle probes ~3 launches over ~30 min, then a short cooldown.
# Stop: create artifacts\_arm2_retry.stop . ASCII ONLY.
# 72h wall covers quota-approval latency + the ~10-16h run itself.
param([int]$CooldownSec = 1200, [int]$MaxHours = 72)
Set-Location (Join-Path $PSScriptRoot "..\..")
$deadline = (Get-Date).AddHours($MaxHours)
$cycle = 0
function Get-Iters {
  $keys = (aws s3api list-objects-v2 --bucket video-processing-6789 --prefix "pokerbot/prod6000/native_cfr/bpv4_emd160_arm2b/checkpoints/" --query "Contents[].Key" --output text --region eu-central-1 2>$null)
  $best = [int64]-1
  if ($keys) { foreach ($k in ($keys -split "\s+")) { if ($k -match "iter_(\d+)\.(nlcfr|nldlt)\.ok$") { $v=[int64]$Matches[1]; if ($v -gt $best) { $best=$v } } } }
  return $best
}
while ((Get-Date) -lt $deadline) {
  if (Test-Path "artifacts\_arm2_retry.stop") { Write-Output "[retry] stop sentinel -> exit"; break }
  $it = Get-Iters
  if ($it -ge 40000) { Write-Output "[retry] target reached ($it) -> exit"; break }
  $cycle++
  Write-Output "[retry] cycle $cycle starting at $(Get-Date -Format HH:mm:ss) (iters=$it)"
  # No -ReuseCode: each cycle re-uploads the tarball so the 2026-07-06 perf merge (and nothing
  # stale) is what the box builds; the tree is stable now that the perf worktrees are merged.
  & powershell -ExecutionPolicy Bypass -File "scripts\blueprint2\bpv4_arm2_babysitter.ps1" 2>&1 |
    Out-File -Append "artifacts\_arm2b_babysitter.log" -Encoding utf8
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
