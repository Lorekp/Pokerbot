# Rich PowerShell watcher / dashboard for the local HU v3 neural-CFR run (paired with
# start_hu_v3_overnight.ps1 + babysit_hu_overnight.sh). Read-only: shows babysitter + training
# liveness, GPU, rich progress (iter/600000 + %, ms/iter, throughput iters/min, ETA to 600k,
# elapsed), restart count, the newest v3 checkpoints, and a STALL detector on the last checkpoint
# write. Reads logs/checkpoints straight off the Windows filesystem; uses WSL only for nvidia-smi.
# Robust to the run not having started yet (files missing). Mirrors watch_hu_24h.ps1.
#
#   powershell -ExecutionPolicy Bypass -File scripts\neural_cfr\watch_hu_v3.ps1        # one-shot
#   powershell -ExecutionPolicy Bypass -File scripts\neural_cfr\watch_hu_v3.ps1 30     # refresh every 30s
# ASCII ONLY (PS 5.1).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..")
$RepoWin = (Get-Location).Path
$Distro  = "Debian"
$RunDir  = Join-Path $RepoWin "runs\hu_neural"
$CkptDir = Join-Path $RepoWin "checkpoints\sd_cfr_hu_neural"
$CoreCkpt = Join-Path $CkptDir "hu_vrpdcfr_v3.pkl"     # atomic core ckpt; mtime = stall detector
$TotalIters = 600000
$StallWarnSec = 300   # warn if no core-checkpoint write for > 5 min while training is alive
$Every   = if ($args.Count -ge 1) { [int]$args[0] } else { 0 }

# NB: not 'Wsl' -- PS is case-insensitive, so that would shadow wsl.exe and recurse. Call wsl.exe.
function Invoke-Wsl([string]$bash) { & wsl.exe -d $Distro -- bash -lc $bash }

function Fmt-Dur([double]$sec) {
  if ($sec -lt 0) { return "?" }
  $t = [TimeSpan]::FromSeconds([math]::Round($sec))
  if ($t.TotalDays -ge 1) { return ("{0}d{1:00}h{2:00}m" -f [int]$t.TotalDays, $t.Hours, $t.Minutes) }
  if ($t.TotalHours -ge 1) { return ("{0}h{1:00}m{2:00}s" -f [int]$t.TotalHours, $t.Minutes, $t.Seconds) }
  if ($t.TotalMinutes -ge 1) { return ("{0}m{1:00}s" -f [int]$t.TotalMinutes, $t.Seconds) }
  return ("{0}s" -f [int]$t.TotalSeconds)
}

function Show {
  Write-Host ("=== HU v3 neural-CFR watch  [{0}] ===" -f (Get-Date -Format "HH:mm:ss"))

  # --- babysitter ---
  $statPath = Join-Path $RunDir "babysit.status"
  $rstPath  = Join-Path $RunDir "babysit.restarts"
  $bpidPath = Join-Path $RunDir "babysit.pid"
  $bstat = if (Test-Path $statPath) { (Get-Content $statPath -ErrorAction SilentlyContinue | Select-Object -First 1) } else { "" }
  $rst   = if (Test-Path $rstPath)  { (Get-Content $rstPath  -ErrorAction SilentlyContinue | Select-Object -First 1) } else { "0" }
  $bpid  = if (Test-Path $bpidPath) { (Get-Content $bpidPath -ErrorAction SilentlyContinue | Select-Object -First 1) } else { "" }
  $bstate = if ($bstat) { "$bstat (exited)" } elseif ($bpid) { "running (PID $bpid)" } else { "not started" }
  Write-Host ("babysitter: {0}   restarts: {1}" -f $bstate, $rst)

  # --- training process (liveness via pgrep -f; [t] bracket excludes pgrep's own bash wrapper) ---
  $info = Invoke-Wsl "set -f; pgrep -af [t]rain_neural_cfr.py | head -1"
  $alive = [bool]$info
  Write-Host ("training:   {0}{1}" -f $(if ($alive) { "ALIVE  " } else { "not running  " }), $info)

  # --- GPU ---
  Write-Host "--- GPU ---"
  $gpu = Invoke-Wsl "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null"
  if ($gpu) { Write-Host "  $gpu" } else { Write-Host "  (nvidia-smi unavailable)" }

  # --- progress (parse the live train log for rich metrics) ---
  Write-Host "--- progress ---"
  $curlog  = Join-Path $RunDir "hu_train.curlog"
  $logPath = if (Test-Path $curlog) { Get-Content $curlog -ErrorAction SilentlyContinue | Select-Object -First 1 } else { $null }
  if (-not ($logPath -and (Test-Path $logPath))) {
    # fall back to newest hu_train_*.log if curlog missing/stale
    $newest = Get-ChildItem (Join-Path $RunDir "hu_train_*.log") -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newest) { $logPath = $newest.FullName }
  }
  if ($logPath -and (Test-Path $logPath)) {
    # Pull the last two "iter N/T ... ms/iter ..." lines to compute % done + live ms/iter + ETA.
    $iterLines = Select-String -Path $logPath -Pattern "iter \d+/\d+" -ErrorAction SilentlyContinue |
                 Select-Object -Last 2 | ForEach-Object { $_.Line }
    if ($iterLines) {
      $last = $iterLines[-1]
      if ($last -match "iter\s+(\d+)/(\d+)") {
        $it = [double]$Matches[1]; $tot = [double]$Matches[2]
        if ($tot -le 0) { $tot = $TotalIters }
        $pct = if ($tot -gt 0) { 100.0 * $it / $tot } else { 0 }
        # ms/iter as reported (cumulative avg over this process lifetime)
        $msRep = if ($last -match "(\d+(?:\.\d+)?)\s*ms/iter") { [double]$Matches[1] } else { $null }
        # live ms/iter: diff the two iter lines using the log file's own line write times is not
        # available from Select-String; instead diff the embedded "elapsed" if present.
        $msLive = $null
        if ($iterLines.Count -eq 2) {
          $a = $iterLines[0]; $b = $iterLines[1]
          if (($a -match "iter\s+(\d+)/") -and $true) { $itA = [double]$Matches[1] }
          if (($b -match "iter\s+(\d+)/") -and $true) { $itB = [double]$Matches[1] }
          function Parse-Elapsed([string]$ln) {
            if ($ln -match "elapsed\s+(\S+)") {
              $e = $Matches[1]; $s = 0.0
              if ($e -match "(\d+)d") { $s += [double]$Matches[1]*86400 }
              if ($e -match "(\d+)h") { $s += [double]$Matches[1]*3600 }
              if ($e -match "(\d+)m") { $s += [double]$Matches[1]*60 }
              if ($e -match "(\d+)s") { $s += [double]$Matches[1] }
              return $s
            }
            return $null
          }
          $eA = Parse-Elapsed $a; $eB = Parse-Elapsed $b
          if (($null -ne $eA) -and ($null -ne $eB) -and ($itB -gt $itA) -and ($eB -gt $eA)) {
            $msLive = 1000.0 * ($eB - $eA) / ($itB - $itA)
          }
        }
        $msUse = if ($msLive) { $msLive } else { $msRep }
        $ipm = if ($msUse -and $msUse -gt 0) { 60000.0 / $msUse } else { $null }
        $etaSec = if ($msUse -and $msUse -gt 0) { ($tot - $it) * $msUse / 1000.0 } else { -1 }
        Write-Host ("  iter {0}/{1}  ({2:N2}% )" -f [int]$it, [int]$tot, $pct)
        $msTxt  = if ($msUse) { ("{0:N0} ms/iter{1}" -f $msUse, $(if ($msLive) { " (live)" } else { " (avg)" })) } else { "ms/iter ?" }
        $ipmTxt = if ($ipm)   { ("{0:N1} iter/min" -f $ipm) } else { "iter/min ?" }
        Write-Host ("  {0}   {1}   ETA-to-{2}: {3}" -f $msTxt, $ipmTxt, $TotalIters, (Fmt-Dur $etaSec))
        if ($last -match "elapsed\s+(\S+)") { Write-Host ("  elapsed(this process): {0}" -f $Matches[1]) }
      }
    } else {
      Write-Host "  (no iter line yet -- XLA compile takes ~3-4 min at start)"
    }
    # surface the most recent notable events too
    $hits = Select-String -Path $logPath -Pattern "Saved|Resuming|Training game|Traceback|Error|Killed|non-finite" -ErrorAction SilentlyContinue | Select-Object -Last 4
    if ($hits) { $hits | ForEach-Object { Write-Host ("  . " + $_.Line) } }
  } else {
    Write-Host "  (no training log yet -- run not started?)"
  }

  # --- checkpoints + STALL detector ---
  Write-Host "--- checkpoints ---"
  if (Test-Path $CoreCkpt) {
    $f = Get-Item $CoreCkpt
    $ageSec = (New-TimeSpan -Start $f.LastWriteTime -End (Get-Date)).TotalSeconds
    $stall = if (($ageSec -gt $StallWarnSec) -and $alive) { "  *** STALL? no ckpt write in {0} ***" -f (Fmt-Dur $ageSec) } else { "" }
    Write-Host ("  core: {0:yyyy-MM-dd HH:mm:ss}  ({1} ago, {2:N1} MB){3}" -f $f.LastWriteTime, (Fmt-Dur $ageSec), ($f.Length / 1MB), $stall)
  } else {
    Write-Host "  core: (hu_vrpdcfr_v3.pkl not written yet)"
  }
  if (Test-Path $CkptDir) {
    Get-ChildItem (Join-Path $CkptDir "hu_vrpdcfr_v3*_full.pkl") -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending | Select-Object -First 2 |
      ForEach-Object { Write-Host ("  full: {0:yyyy-MM-dd HH:mm:ss}  {1}  ({2:N1} MB)" -f $_.LastWriteTime, $_.Name, ($_.Length / 1MB)) }
  }

  # --- babysitter log tail ---
  $blog = Join-Path $RunDir "babysit.log"
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
