# Retry supervisor for the arm-2b babysitter. Original role: spot-storm cycling. Current role
# (2026-07-06 ~12:00): ON-DEMAND QUOTA WATCHER -- the account's on-demand vCPU quota (16) blocks
# the c6a.32xlarge launch until the user's Service Quotas increase lands; failed RunInstances
# attempts are free, so this loop just re-runs the babysitter until a launch sticks and training
# reaches 40k. Each babysitter cycle probes ~3 launches over ~30 min, then a short cooldown.
# Stop: create artifacts\_arm2_retry.stop . ASCII ONLY.
# 72h wall covers quota-approval latency + the ~10-16h run itself.
#
# 2026-07-06 adaptive cooldown: if iters advanced during a cycle, cool down only 120s (productive
# cycle; a new box should be relaunched quickly). Only keep the full $CooldownSec (1200s) when the
# cycle was dry (churn-abort or no box launched at all). Branch logged with "[retry] cooldown".
param([int]$CooldownSec = 1200, [int]$MaxHours = 72)
Set-Location (Join-Path $PSScriptRoot "..\..")
$deadline = (Get-Date).AddHours($MaxHours)
$cycleNum = 0
function Get-Iters {
  $keys = (aws s3api list-objects-v2 --bucket video-processing-6789 --prefix "pokerbot/prod6000/native_cfr/bpv4_emd160_arm2b/checkpoints/" --query "Contents[].Key" --output text --region eu-central-1 2>$null)
  $best = [int64]-1
  if ($keys) { foreach ($k in ($keys -split "\s+")) { if ($k -match "iter_(\d+)\.(nlcfr|nldlt)\.ok$") { $v=[int64]$Matches[1]; if ($v -gt $best) { $best=$v } } } }
  return $best
}
while ((Get-Date) -lt $deadline) {
  if (Test-Path "artifacts\_arm2_retry.stop") { Write-Output "[retry] stop sentinel -> exit"; break }
  $itBefore = Get-Iters
  if ($itBefore -ge 40000) { Write-Output "[retry] target reached ($itBefore) -> exit"; break }
  $cycleNum++
  Write-Output "[retry] cycle $cycleNum starting at $(Get-Date -Format HH:mm:ss) (iters=$itBefore)"
  # No -ReuseCode: each cycle re-uploads the tarball so the 2026-07-06 perf merge (and nothing
  # stale) is what the box builds; the tree is stable now that the perf worktrees are merged.
  & powershell -ExecutionPolicy Bypass -File "scripts\blueprint2\bpv4_arm2_babysitter.ps1" 2>&1 |
    Out-File -Append "artifacts\_arm2b_babysitter.log" -Encoding utf8
  $itAfter = Get-Iters
  Write-Output "[retry] cycle $cycleNum babysitter exited at $(Get-Date -Format HH:mm:ss) (iters $itBefore -> $itAfter)"
  if ($itAfter -ge 40000) { Write-Output "[retry] target reached -> exit"; break }
  # 2026-07-06 adaptive cooldown: short sleep when cycle was productive, full cooldown when dry.
  if ($itAfter -gt $itBefore) {
    $sleepSec = 120
    Write-Output "[retry] cooldown ADAPTIVE 120s (cycle made progress: $itBefore -> $itAfter)"
  } else {
    $sleepSec = $CooldownSec
    Write-Output "[retry] cooldown FULL ${CooldownSec}s (dry cycle: iters unchanged at $itAfter)"
  }
  $slept = 0
  while ($slept -lt $sleepSec) {
    if (Test-Path "artifacts\_arm2_retry.stop") { Write-Output "[retry] stop sentinel during cooldown -> exit"; exit }
    Start-Sleep -Seconds 60; $slept += 60
  }
}
Write-Output "[retry] done at $(Get-Date -Format HH:mm:ss)"
