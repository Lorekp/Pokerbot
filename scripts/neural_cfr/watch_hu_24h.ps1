# PowerShell watcher / dashboard for the local heads-up NLHE neural-CFR run (paired with
# babysit_hu_24h.ps1). Read-only: shows babysitter + training liveness, GPU, latest progress, and the
# newest checkpoints. Reads logs/checkpoints straight off the Windows filesystem; uses WSL only for the
# process-liveness check and nvidia-smi.
#
#   powershell -ExecutionPolicy Bypass -File scripts\neural_cfr\watch_hu_24h.ps1        # one-shot
#   powershell -ExecutionPolicy Bypass -File scripts\neural_cfr\watch_hu_24h.ps1 30     # refresh every 30s
# ASCII ONLY (PS 5.1).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..")
$RepoWin = (Get-Location).Path
$Distro  = "Debian"
$RunDir  = Join-Path $RepoWin "runs\hu_neural"
$CkptDir = Join-Path $RepoWin "checkpoints\sd_cfr_hu_neural"
$Every   = if ($args.Count -ge 1) { [int]$args[0] } else { 0 }

# NB: not 'Wsl' -- PS is case-insensitive, so that would shadow wsl.exe and recurse. Call wsl.exe.
function Invoke-Wsl([string]$bash) { & wsl.exe -d $Distro -- bash -lc $bash }

function Show {
  Write-Host ("=== HU neural-CFR watch  [{0}] ===" -f (Get-Date -Format "HH:mm:ss"))

  # babysitter
  $bstat = if (Test-Path (Join-Path $RunDir "ps_babysit.status")) { Get-Content (Join-Path $RunDir "ps_babysit.status") -ErrorAction SilentlyContinue } else { "not started" }
  $rst   = if (Test-Path (Join-Path $RunDir "babysit.restarts")) { Get-Content (Join-Path $RunDir "babysit.restarts") -ErrorAction SilentlyContinue } else { "0" }
  Write-Host "babysitter: $bstat   launches: $rst"

  # training process (liveness via pgrep -f; [t] bracket excludes pgrep's own bash wrapper)
  $info = Invoke-Wsl "set -f; pgrep -af [t]rain_neural_cfr.py | head -1"
  $alive = [bool]$info
  Write-Host ("training:   {0}{1}" -f $(if ($alive) { "ALIVE  " } else { "not running" }), $info)

  # GPU
  Write-Host "--- GPU ---"
  $gpu = Invoke-Wsl "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null"
  if ($gpu) { Write-Host "  $gpu" } else { Write-Host "  (nvidia-smi unavailable)" }

  # latest progress
  Write-Host "--- progress ---"
  $curlog = Join-Path $RunDir "hu_train.curlog"
  $logPath = if (Test-Path $curlog) { Get-Content $curlog -ErrorAction SilentlyContinue | Select-Object -First 1 } else { $null }
  if ($logPath -and (Test-Path $logPath)) {
    $hits = Select-String -Path $logPath -Pattern "iter \d+/|Saved|Resuming|Training game|Traceback|Error|Killed|non-finite" -ErrorAction SilentlyContinue | Select-Object -Last 8
    if ($hits) { $hits | ForEach-Object { Write-Host ("  " + $_.Line) } }
    else { Get-Content $logPath -Tail 6 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ("  " + $_) } }
  } else { Write-Host "  (no training log yet)" }

  # checkpoints
  Write-Host "--- checkpoints ---"
  if (Test-Path $CkptDir) {
    Get-ChildItem (Join-Path $CkptDir "*_full.pkl") -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending | Select-Object -First 3 |
      ForEach-Object { Write-Host ("  {0:yyyy-MM-dd HH:mm:ss}  {1}  ({2:N1} MB)" -f $_.LastWriteTime, $_.Name, ($_.Length / 1MB)) }
  } else { Write-Host "  (none yet)" }

  # babysitter log tail
  $blog = Join-Path $RunDir "ps_babysit.log"
  if (Test-Path $blog) {
    Write-Host "--- babysitter log (last 4) ---"
    Get-Content $blog -Tail 4 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ("  " + $_) }
  }
}

if ($Every -gt 0) {
  Write-Host "(refresh ${Every}s, Ctrl+C to stop)"
  while ($true) { Show; Write-Host ""; Start-Sleep -Seconds $Every }
} else {
  Show
}
