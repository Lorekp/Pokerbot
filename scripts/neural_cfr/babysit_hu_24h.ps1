# PowerShell babysitter for the LOCAL heads-up NLHE neural-CFR run (tournament HU: shallow depth +
# antes, chip-EV; ICM applied at serve time). The training runs on the GPU inside WSL (Debian, CUDA
# jaxgpu-venv); this script supervises it from Windows -- mirrors the AWS pilot babysitters: a foreground
# poll loop that tracks iterations, relaunches (resuming from the last atomic checkpoint) when no live
# training process is found, and stops at a target iteration or wall-clock cap, with a restart-storm guard.
#
# Run from YOUR PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\neural_cfr\babysit_hu_24h.ps1
# Watch (separate window):
#   powershell -ExecutionPolicy Bypass -File scripts\neural_cfr\watch_hu_24h.ps1
#
# Notes:
#  - FRESH stem hu_vrpdcfr_v2: the prior ~7h hu_vrpdcfr.pkl was trained before the chance-node + ante
#    fixes and at the old epsilon -> do NOT resume it. Override $Ckpt to resume a specific run.
#  - The training is launched detached (nohup) inside WSL, so it keeps running if you close this window;
#    you just lose auto-restart until you relaunch this babysitter (it will resume from the checkpoint).
#  - ASCII ONLY (PS 5.1 mangles non-ASCII).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..")   # repo root (script lives in scripts/neural_cfr/)
$RepoWin = (Get-Location).Path
$WslRepo = "."

# --- config (override inline: $env:VAR before launch, or edit here) --------------------------------
$Distro       = "Debian"
$Ckpt         = if ($env:HU_CKPT) { $env:HU_CKPT } else { "checkpoints/sd_cfr_hu_neural/hu_vrpdcfr_v2.pkl" }
$Iters        = 100000          # effective "run until killed"; the wall-clock cap is the real stop
$Trav         = 128
$Batch        = 1024
$Hidden       = 256
$TrainSteps   = 16
$AvgSteps     = 128
$HistSteps    = 32
$Lr           = "1e-4"
$Seed         = 0
$CkptEvery    = 25              # cheap resumable core checkpoint cadence (params + Adam moments)
$BufCkptEvery = 500            # heavy reservoir snapshot cadence (~300MB; kept off the slow /mnt/c hot path)
$LogEvery     = 25             # progress-line cadence so the watcher shows throughput promptly
$DepthBB      = if ($env:POKERBOT_HU_DEPTH_BB) { $env:POKERBOT_HU_DEPTH_BB } else { "5,100" }
$AnteBB       = if ($env:POKERBOT_HU_ANTE_BB) { $env:POKERBOT_HU_ANTE_BB } else { "0.125" }
$AnteZeroProb = if ($env:POKERBOT_HU_ANTE_ZERO_PROB) { $env:POKERBOT_HU_ANTE_ZERO_PROB } else { "0.2" }

$MaxSec    = 24 * 3600          # hard wall-clock cap (~24h); kills training + exits
$TargetIter = 100000            # stop once this many checkpoints (iterations) exist
$PollSec   = 120                # status/poll cadence
$MaxFast   = 5                  # restart-storm guard: >= this many restarts within...
$FastWindow = 900               # ...this many seconds -> structural failure, abort
$BackoffSec = 20

# --- derived paths -------------------------------------------------------------------------------
$RunDir    = Join-Path $RepoWin "runs\hu_neural"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$CkptFullWin = Join-Path $RepoWin ($Ckpt -replace "/", "\" -replace "\.pkl$", "_full.pkl")
$CurLog    = Join-Path $RunDir "hu_train.curlog"
# The training runs as a FOREGROUND wsl.exe child of this script (Start-Process -PassThru). A detached
# WSL process does NOT survive wsl.exe returning -- WSL reaps a one-shot invocation's process tree (even
# under setsid/nohup) -- so we hold the wsl.exe handle for the whole run and track it directly. Killing
# the handle propagates to the WSL bash->python tree (verified). Trade-off: closing THIS window stops
# training; for a fully detached "set and forget" run use the bash babysitter (babysit_hu_overnight.sh).
$script:TrainProc = $null
$Status    = Join-Path $RunDir "ps_babysit.status"
$BabyLog   = Join-Path $RunDir "ps_babysit.log"

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $BabyLog -Value $line -Encoding ASCII
}

# NB: do not name this 'Wsl' -- PowerShell is case-insensitive, so a function 'Wsl' would shadow the
# wsl.exe binary and '& wsl' would recurse into itself (call-depth overflow). Call wsl.exe explicitly.
function Invoke-Wsl([string]$bash) { & wsl.exe -d $Distro -- bash -lc $bash }

function Train-Alive {
  return ($script:TrainProc -ne $null -and -not $script:TrainProc.HasExited)
}

function Get-LatestIter {
  if (-not (Test-Path $CurLog)) { return 0 }
  $logPath = (Get-Content $CurLog -ErrorAction SilentlyContinue | Select-Object -First 1)
  if (-not $logPath -or -not (Test-Path $logPath)) { return 0 }
  $m = [regex]::Matches((Get-Content $logPath -Tail 200 -ErrorAction SilentlyContinue) -join "`n", "iter (\d+)/")
  if ($m.Count -gt 0) { return ($m | ForEach-Object { [int]$_.Groups[1].Value } | Measure-Object -Maximum).Maximum }
  return 0
}

function Start-Train {
  $stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
  $logWin = Join-Path $RunDir "hu_train_$stamp.log"
  $logWsl = "$WslRepo/runs/hu_neural/hu_train_$stamp.log"
  Set-Content -Path $CurLog -Value $logWin -Encoding ASCII
  $resume = ""
  if (Test-Path $CkptFullWin) { $resume = "--resume $Ckpt"; Log "resuming from $CkptFullWin" }
  else { Log "fresh start (no checkpoint at $CkptFullWin)" }

  $envline = "export PYTHONPATH=. POKERBOT_INFO_SUITS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false " +
             "XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4 " +
             "OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 POKERBOT_HU_DEPTH_BB=$DepthBB " +
             "POKERBOT_HU_ANTE_BB=$AnteBB POKERBOT_HU_ANTE_ZERO_PROB=$AnteZeroProb"
  # exec so wsl.exe -> bash becomes python (clean tree; killing the handle kills python). Foreground:
  # wsl.exe stays alive until python exits, which is how we keep the WSL session (and the GPU job) up.
  $py = "exec python -u scripts/train_neural_cfr.py --game nlhe-hu --iterations $Iters " +
        "--traversals $Trav --batch-size $Batch --hidden-dim $Hidden --train-steps $TrainSteps " +
        "--avg-steps $AvgSteps --hist-steps $HistSteps --lr $Lr --seed $Seed --checkpoint $Ckpt " +
        "--checkpoint-every $CkptEvery --buffer-checkpoint-every $BufCkptEvery --log-every $LogEvery " +
        "--traversal-workers 0 $resume >> $logWsl 2>&1"
  $bash = "cd $WslRepo; source `$HOME/jaxgpu-venv/bin/activate; $envline; $py"
  $script:TrainProc = Start-Process -FilePath "wsl.exe" `
    -ArgumentList @("-d", $Distro, "--", "bash", "-lc", $bash) -PassThru -WindowStyle Hidden
  Start-Sleep -Seconds 3
  Log ("launched training (wsl.exe pid {0}) -> {1}" -f $script:TrainProc.Id, $logWin)
}

function Stop-Train {
  if ($script:TrainProc -ne $null -and -not $script:TrainProc.HasExited) {
    try { $script:TrainProc.Kill() } catch {}
    Log "stopped training (killed wsl.exe pid $($script:TrainProc.Id))"
  }
}

# --- supervise -----------------------------------------------------------------------------------
Set-Content -Path $Status -Value "RUNNING" -Encoding ASCII
Log "babysitter up. ckpt=$Ckpt depth=$DepthBB ante=$AnteBB/$AnteZeroProb steps=$TrainSteps/$AvgSteps/$HistSteps maxsec=$MaxSec target=$TargetIter"
$start = Get-Date
$restarts = New-Object System.Collections.ArrayList
$launches = 0

while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  $iter = Get-LatestIter
  $alive = Train-Alive
  Set-Content -Path (Join-Path $RunDir "babysit.restarts") -Value $launches -Encoding ASCII
  Log ("t+{0}s iter={1} alive={2} launches={3}" -f $el, $iter, $alive, $launches)

  if ($iter -ge $TargetIter) { Log "DONE: target iter $TargetIter reached"; Stop-Train; break }
  if ($el -ge $MaxSec)       { Log "DONE: wall-clock cap ${MaxSec}s reached"; Stop-Train; break }
  if ($alive) { Start-Sleep -Seconds $PollSec; continue }

  # not alive -> crashed/never-started. storm guard, then relaunch (resume).
  $now = [int]((Get-Date) - (Get-Date "1970-01-01")).TotalSeconds
  [void]$restarts.Add($now)
  $recent = @($restarts | Where-Object { ($now - $_) -le $FastWindow })
  $restarts = [System.Collections.ArrayList]@($recent)
  if ($recent.Count -ge $MaxFast) {
    Log "ABORT: $MaxFast restarts within ${FastWindow}s -> structural failure. Check the latest hu_train_*.log."
    Set-Content -Path $Status -Value "ABORTED" -Encoding ASCII
    break
  }
  if ($launches -gt 0) {
    Log "training not alive; backing off ${BackoffSec}s then resuming (restart $($recent.Count)/$MaxFast in window)"
    Start-Sleep -Seconds $BackoffSec
  }
  Start-Train
  $launches++
  Start-Sleep -Seconds $PollSec
}

Set-Content -Path $Status -Value "DONE" -Encoding ASCII
Log "babysitter exit after ${el}s, $launches launches, latest iter=$iter"
