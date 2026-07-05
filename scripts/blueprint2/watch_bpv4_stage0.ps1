# READ-ONLY dashboard for the LIVE blueprint-v4 Stage-0 FULLHIST + M-RR run (bpv4_h200).
#
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\watch_bpv4_stage0.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\watch_bpv4_stage0.ps1 -Refresh 30 -StallMin 90
#
# Pairs with scripts\blueprint2\bpv4_stage0_babysitter.ps1 (which OWNS launch/resume/terminate). This
# script NEVER mutates AWS: it uses ONLY `aws ec2 describe-instances`, `aws ec2 get-console-output`,
# and `aws s3 ls`. It never terminates/modifies/put/cp/downloads. Kill decisions are YOURS (kill-gate).
#
# Why the bp2 watcher is inadequate here (bpv4 deltas baked in below):
#   * KEY METRIC is strat-rows, not iters: the driver prints `PROGRESS iters_done=X/ITERS infosets=Y ...`
#     to the box console. We parse infosets=Y (rows) from the LATEST console PROGRESS line via
#     get-console-output -- we do NOT download the 30-55 GB checkpoint blobs. Rows are shown vs the 400M
#     G8 budget (POKERBOT_ROW_BUDGET); a ROW_BUDGET_STOP ends the run EARLY with partial evidence.
#   * Dual ETA: to 45000 iters AND to the 400M-row budget; the run ends at min(iters, row-budget).
#   * STALL threshold ~90 min (a fullhist 10k-iter chunk + a ~50 GB multipart upload is slow); stall is
#     based on NO new console PROGRESS line, not on checkpoint cadence (checkpoints every 10k iters).
#   * Surfaces bpv4 console milestones (WALL_GUARD, FULLHIST_KEY, MRR, RAYON_NUM_THREADS, CKPT multipart,
#     ROW_BUDGET_STOP, RESUMED) + kill-gate reminders at ~20k / ~30k iters.
#
# Runbook: scripts\blueprint2\README_LAUNCH.md, "Blueprint-v4 Stage-0 -- cloud FULLHIST arms".
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
param(
  [string]$Run = "bpv4_h200",
  [int]$Iters = 45000,                       # --iters backstop (kill-gates at 20k/30k; design v2 6.3)
  [long]$RowBudget = 400000000,              # POKERBOT_ROW_BUDGET G8 RAM backstop (strat rows); 0 = off
  [int]$Refresh = 45,
  [double]$SpotPrice = 2.50,                  # approx spot $/hr fallback for c6i.32xlarge (spotpick overrides)
  [int]$StallMin = 90,                        # alarm if NO new console PROGRESS line in this many minutes
  [string]$Region = "eu-central-1",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr"
)
$RUN_PREFIX  = "$Prefix/$Run"
$CKPT_PREFIX = "$RUN_PREFIX/checkpoints/"
$t0 = Get-Date

# --- persistent state across the refresh loop ---
$lastProgIter       = -1            # max iters_done seen on a console PROGRESS line
$lastProgChangeTime = Get-Date      # when that last advanced (stall clock)
$lastCkptSeen       = -1            # max S3 checkpoint iter seen (2nd liveness signal for the stall clock)
$prevIter           = -1            # previous PROGRESS sample (for instantaneous it/s + rows/iter)
$prevElapsed        = -1.0
$prevRows           = -1.0
$curItersPerSec     = 0.0
$curRowsPerIter     = 0.0
$kill20Shown        = $false
$kill30Shown        = $false

function Bar([double]$frac, [int]$width = 40) {
  if ($frac -lt 0) { $frac = 0 }; if ($frac -gt 1) { $frac = 1 }
  $d = [int]([math]::Round($frac * $width))
  return ("#" * $d) + ("-" * ($width - $d))
}
function EtaStr([double]$sec) {
  if ($sec -lt 0) { return "--" }
  if ($sec -lt 5400) { return ("~{0}m" -f [int]($sec / 60)) }
  return ("~{0:N1}h" -f ($sec / 3600.0))
}
$priceCache = @{}
function Get-SpotpickPrice([string]$it) {
  if (-not $it) { return $SpotPrice }
  if ($priceCache.ContainsKey($it)) { return $priceCache[$it] }
  $p = $SpotPrice
  try {
    $out = (& python -c "import sys;sys.path.insert(0,'.');import spotpick;print({i.name:i.spot for i in spotpick.load()}.get('$it',''))" 2>$null)
    if ($out) { $out = ("$out").Trim(); if ($out -match '^[0-9.]+$') { $p = [double]$out } }
  } catch {}
  $priceCache[$it] = $p
  return $p
}

while ($true) {
  # --- live box (READ-ONLY describe-instances; same idiom as watch_blueprint_run.ps1) ---
  $boxRaw = aws ec2 describe-instances --region $Region `
    --filters "Name=tag:Run,Values=$Run" "Name=instance-state-name,Values=running,pending" `
    --query "Reservations[].Instances[].[InstanceId,State.Name,InstanceType,LaunchTime]" --output text 2>$null
  $boxLine = "(no live box - launching, reclaimed, or finished; babysitter owns relaunch)"
  $upMin = 0
  $boxId = ""
  $instType = ""
  if ($boxRaw) {
    $f = ($boxRaw -split "\s+")
    $boxId = $f[0]
    $instType = $f[2]
    if ($boxRaw -match "(\d{4}-\d{2}-\d{2}T[\d:.]+[+\-]\d{2}:?\d{2})") {
      try { $upMin = [int](((Get-Date).ToUniversalTime() - [datetimeoffset]::Parse($Matches[1]).UtcDateTime).TotalMinutes) } catch {}
    }
    $boxLine = ("{0}  {1}  {2}  up {3}m" -f $f[0], $f[1], $f[2], $upMin)
  }
  $livePrice = Get-SpotpickPrice $instType

  # --- console output (READ-ONLY get-console-output): the PRIMARY signal (PROGRESS rows + milestones) ---
  $console = ""
  if ($boxId) {
    $console = (aws ec2 get-console-output --instance-id $boxId --region $Region --latest `
                  --query "Output" --output text 2>$null | Out-String)
  }

  $iters = -1; $rows = -1.0; $elapsed = -1.0; $logRate = 0.0; $haveProgress = $false; $isDone = $false
  if ($console) {
    $mProg = [regex]::Matches($console, "PROGRESS iters_done=(\d+)/(\d+) infosets=(\d+) elapsed=([0-9.]+)s rate=([0-9.]+) it/s")
    if ($mProg.Count) {
      $g = $mProg[$mProg.Count - 1].Groups
      $iters = [int]$g[1].Value
      $rows = [double]$g[3].Value
      $elapsed = [double]$g[4].Value
      $logRate = [double]$g[5].Value
      $haveProgress = $true
    }
    $mDone = [regex]::Match($console, "DONE iters_done=(\d+) infosets=(\d+)")
    if ($mDone.Success) {
      $isDone = $true
      $di = [int]$mDone.Groups[1].Value
      if ($di -gt $iters) { $iters = $di; $rows = [double]$mDone.Groups[2].Value }
    }
  }

  # --- milestones surfaced from the console (bpv4-specific) ---
  $ms = @()
  if ($console) {
    if ($console -match "WALL_GUARD armed") { $ms += "WALL_GUARD" }
    if ($console -match "FULLHIST_KEY=1")   { $ms += "FULLHIST_KEY=1" }
    if ($console -match "MRR=1")            { $ms += "MRR=1" }
    if ($console -match "RAYON_NUM_THREADS=(\d+)") { $ms += ("RAYON=" + $Matches[1]) }
    $mResume = [regex]::Matches($console, "RESUMED from latest checkpoint at iters_done=(\d+)")
    if ($mResume.Count) { $ms += ("RESUMED@" + $mResume[$mResume.Count - 1].Groups[1].Value) }
  }
  $ckptLine = ""
  if ($console) {
    $mCkpt = [regex]::Matches($console, "CKPT iters_done=\d+ -> [^\r\n]*multipart[^\r\n]*")
    if ($mCkpt.Count) { $ckptLine = $mCkpt[$mCkpt.Count - 1].Value.Trim() }
  }
  $rowStopLine = ""
  if ($console) {
    $mStop = [regex]::Match($console, "ROW_BUDGET_STOP[^\r\n]*")
    if ($mStop.Success) { $rowStopLine = $mStop.Value.Trim() }
  }

  # --- checkpoint listing (heartbeat only; NOT the stall clock) ---
  $ls = aws s3 ls "s3://$Bucket/$CKPT_PREFIX" --region $Region 2>$null
  $ckptIters = @()
  if ($ls) {
    foreach ($line in $ls) {
      $name = ($line -split "\s+")[-1]
      if ($name -match "^iter_(\d+)\.nlcfr$") { $ckptIters += [int]$Matches[1] }
    }
  }
  $maxCkptIter = if ($ckptIters.Count) { ($ckptIters | Measure-Object -Maximum).Maximum } else { -1 }
  $nCkpt = $ckptIters.Count

  # --- instantaneous it/s + rows/iter from successive PROGRESS samples (chunk-to-chunk) ---
  if ($haveProgress -and $iters -gt $prevIter) {
    if ($prevIter -ge 0 -and $elapsed -gt $prevElapsed -and $iters -gt $prevIter) {
      $dIter = $iters - $prevIter
      $dEl   = $elapsed - $prevElapsed
      $dRows = $rows - $prevRows
      if ($dEl -gt 0)   { $curItersPerSec = $dIter / $dEl }
      if ($dIter -gt 0) { $curRowsPerIter = $dRows / $dIter }
    }
    $prevIter = $iters; $prevElapsed = $elapsed; $prevRows = $rows
  }
  # fallbacks (first sample / just started): cumulative rate + cumulative rows/iter
  $itersPerSec = if ($curItersPerSec -gt 1e-9) { $curItersPerSec } elseif ($logRate -gt 1e-9) { $logRate } else { 0.0 }
  $rowsPerIter = if ($curRowsPerIter -gt 1e-9) { $curRowsPerIter } elseif ($iters -gt 0 -and $rows -gt 0) { $rows / $iters } else { 0.0 }

  # --- stall clock: resets on a new PROGRESS iters OR a new S3 checkpoint (both are per-10k-chunk
  # liveness). The alarm threshold AUTO-SCALES to the arm's real throughput: a 10k-iter chunk emits ONE
  # PROGRESS line, so at low it/s a healthy chunk legitimately exceeds a fixed 90m and would false-fire.
  # Predict the chunk time from the observed it/s and alarm only past 2.5x it (floor $StallMin); unknown
  # rate => floor. Keeps the alarm from crying wolf on a slow-but-healthy fullhist chunk. ---
  if ($iters -gt $lastProgIter) { $lastProgIter = $iters; $lastProgChangeTime = Get-Date }
  if ($maxCkptIter -gt $lastCkptSeen) { $lastCkptSeen = $maxCkptIter; $lastProgChangeTime = Get-Date }
  $stallMinObs = [int]((Get-Date) - $lastProgChangeTime).TotalMinutes
  $expectChunkMin = if ($itersPerSec -gt 1e-9) { 10000.0 / $itersPerSec / 60.0 } else { 0.0 }
  $effStall = [int]([math]::Max([double]$StallMin, 2.5 * $expectChunkMin))
  $stalled = ($haveProgress -and $stallMinObs -ge $effStall -and -not $isDone)

  # --- dual ETA: to iters AND to the 400M row budget; the run ends at min(iters, row-budget) ---
  $etaItersSec = -1.0; $etaRowsSec = -1.0
  if ($itersPerSec -gt 1e-9 -and $iters -ge 0 -and $iters -lt $Iters) {
    $etaItersSec = ($Iters - $iters) / $itersPerSec
  }
  if ($RowBudget -gt 0 -and $rows -ge 0 -and $rows -lt $RowBudget -and $rowsPerIter -gt 1e-9 -and $itersPerSec -gt 1e-9) {
    $itersToBudget = ($RowBudget - $rows) / $rowsPerIter
    $etaRowsSec = $itersToBudget / $itersPerSec
  }
  # end = whichever bound is nearer
  $endSec = -1.0; $endWhich = ""
  if ($etaItersSec -ge 0 -and $etaRowsSec -ge 0) {
    if ($etaRowsSec -lt $etaItersSec) { $endSec = $etaRowsSec; $endWhich = "ROW-BUDGET" } else { $endSec = $etaItersSec; $endWhich = "iters" }
  } elseif ($etaItersSec -ge 0) { $endSec = $etaItersSec; $endWhich = "iters" }
  elseif ($etaRowsSec -ge 0) { $endSec = $etaRowsSec; $endWhich = "ROW-BUDGET" }

  $iFrac = if ($Iters -gt 0 -and $iters -ge 0) { [double]$iters / $Iters } else { 0.0 }
  $rFrac = if ($RowBudget -gt 0 -and $rows -ge 0) { [double]$rows / $RowBudget } else { 0.0 }
  $spent = ($upMin / 60.0) * $livePrice
  $ecc = if ($endSec -ge 0) { ($endSec / 3600.0) * $livePrice } else { -1.0 }

  # ================================ RENDER ================================
  Clear-Host
  Write-Host ("=== bpv4 Stage-0 FULLHIST+M-RR: {0}   {1}   (watching {2}m) ===" -f $Run, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), [int]((Get-Date) - $t0).TotalMinutes) -ForegroundColor Cyan
  $boxColor = if ($boxRaw -match "running") { "Green" } elseif ($boxRaw) { "Yellow" } else { "DarkGray" }
  Write-Host ("box:    {0}" -f $boxLine) -ForegroundColor $boxColor

  # just-started / booting grace: live box but no PROGRESS and no checkpoints yet
  $booting = ($boxId -and -not $haveProgress -and $maxCkptIter -lt 0)
  if ($booting) {
    $msTxt = if ($ms.Count) { ($ms -join " ") } else { "(no console milestones yet)" }
    Write-Host ("state:  booting/building... box up {0}m, no console PROGRESS or checkpoint yet" -f $upMin) -ForegroundColor Yellow
    Write-Host ("        (apt + AWS CLI v2 install + crate build + first {0}-iter chunk take a while)" -f 10000) -ForegroundColor DarkGray
    Write-Host ("boot:   {0}" -f $msTxt) -ForegroundColor DarkCyan
  } elseif (-not $boxId -and $maxCkptIter -lt 0 -and -not $haveProgress) {
    Write-Host "state:  no live box and no checkpoint yet - initial launch pending (babysitter relaunches)" -ForegroundColor DarkGray
  }

  if ($ms.Count) { Write-Host ("milestones: {0}" -f ($ms -join "  ")) -ForegroundColor Green }

  # iters progress
  $idTxt = if ($iters -ge 0) { "$iters" } else { "?" }
  $iBarColor = if ($iFrac -ge 0.99) { "Green" } elseif ($iFrac -ge 0.5) { "Cyan" } else { "Yellow" }
  Write-Host ("iters:  {0}/{1}  [{2}] {3:P1}" -f $idTxt, $Iters, (Bar $iFrac), $iFrac) -ForegroundColor $iBarColor

  # rows = THE KEY METRIC, vs the 400M G8 budget
  if ($RowBudget -gt 0) {
    $rowsTxt = if ($rows -ge 0) { "{0:N0}" -f $rows } else { "(no PROGRESS line yet)" }
    $rBarColor = if ($rFrac -ge 0.80) { "Red" } elseif ($rFrac -ge 0.5) { "Yellow" } else { "Cyan" }
    Write-Host ("rows:   {0} / {1:N0}  [{2}] {3:P1} of G8 budget" -f $rowsTxt, $RowBudget, (Bar $rFrac), $rFrac) -ForegroundColor $rBarColor
    if ($rFrac -ge 0.80 -and $rows -ge 0) {
      Write-Host ("  WARNING: strat-rows at {0:P1} of the 400M ROW_BUDGET -- a ROW_BUDGET_STOP will END THE RUN" -f $rFrac) -ForegroundColor Red
      Write-Host ("           EARLY with PARTIAL evidence (writes a FINAL checkpoint + latest, then stops).") -ForegroundColor Red
    }
  } else {
    $rowsTxt = if ($rows -ge 0) { "{0:N0}" -f $rows } else { "(no PROGRESS line yet)" }
    Write-Host ("rows:   {0}  (ROW_BUDGET disabled)" -f $rowsTxt) -ForegroundColor Cyan
  }

  # rates + dual ETA
  $ipsTxt = if ($curItersPerSec -gt 1e-9) { "{0:N4} it/s" -f $itersPerSec } elseif ($itersPerSec -gt 1e-9) { "{0:N4} it/s (cum)" -f $itersPerSec } else { "--" }
  $rpiTxt = if ($rowsPerIter -gt 1e-9) { "{0:N0} rows/iter" -f $rowsPerIter } else { "--" }
  Write-Host ("rate:   {0}   {1}" -f $ipsTxt, $rpiTxt) -ForegroundColor Yellow
  Write-Host ("ETA:    to 45k iters {0}   |   to 400M rows {1}   |   RUN ENDS {2} (bound: {3})" -f `
      (EtaStr $etaItersSec), (EtaStr $etaRowsSec), (EtaStr $endSec), $(if ($endWhich) { $endWhich } else { "--" })) -ForegroundColor Yellow

  # checkpoints (heartbeat) + last multipart upload
  Write-Host ("ckpts:  {0} versioned (max iter {1}); every 10000 iters, ~30-55 GB multipart blobs" -f $nCkpt, $maxCkptIter) -ForegroundColor DarkCyan
  if ($ckptLine) { Write-Host ("  {0}" -f $ckptLine) -ForegroundColor DarkGray }

  # cost
  $eccTxt = if ($ecc -ge 0) { ("~`${0:N2}" -f $ecc) } else { "--" }
  $itTxt = if ($instType) { $instType } else { "c6i.32xlarge" }
  Write-Host ("cost:   spent ~`${0:N2} this box   |   ECC {1} to run-end   (@ `${2:N3}/hr {3})" -f $spent, $eccTxt, $livePrice, $itTxt) -ForegroundColor Magenta

  # stall alarm (auto-scaled to throughput; resets on new PROGRESS OR checkpoint; cross-refs box state)
  if ($haveProgress) {
    if ($stalled) {
      $runTxt = if ($boxRaw -match "running") { "box is STILL 'running' -> most likely a slow chunk + ~50 GB upload, NOT a hang: check get-console-output for an error BEFORE killing" } else { "box is NOT 'running' (reclaimed/terminated) -> babysitter should relaunch + RESUME" }
      Write-Host ("STALL?: no new PROGRESS/checkpoint in {0}m (alarm {1}m = max({2}m floor, 2.5x ~{3}m/chunk))." -f $stallMinObs, $effStall, $StallMin, [int]$expectChunkMin) -ForegroundColor Red
      Write-Host ("        {0}" -f $runTxt) -ForegroundColor Red
    } else {
      Write-Host ("prog:   last advance {0}m ago (alarm at {1}m, auto-scaled to ~{2}m/10k-chunk)" -f $stallMinObs, $effStall, [int]$expectChunkMin) -ForegroundColor DarkGray
    }
  } elseif (-not $booting) {
    Write-Host ("prog:   no console PROGRESS line seen yet (stall clock not armed)") -ForegroundColor DarkGray
  }

  # ROW_BUDGET_STOP milestone (terminal, partial evidence)
  if ($rowStopLine) {
    Write-Host ""
    Write-Host ("ROW_BUDGET_STOP: {0}" -f $rowStopLine) -ForegroundColor Red
    Write-Host ("  The run stopped EARLY at the 400M-row backstop (PARTIAL evidence). FINAL checkpoint written.") -ForegroundColor Red
  }

  # kill-gate reminders (design v2 6.3): ~20k (~`$5 sunk) and ~30k (~`$12 sunk) decision points
  Write-Host ""
  if ($iters -ge 30000) {
    if (-not $kill30Shown) { $kill30Shown = $true }
    Write-Host ("KILL-GATE 30k REACHED (~`$12 sunk): decide GO/KILL. In a second terminal run:") -ForegroundColor Cyan
    Write-Host ("  python scripts\blueprint2\bpv4_kill_gate.py --run-subdir $Run --iter 30000") -ForegroundColor Cyan
  } elseif ($iters -ge 20000) {
    if (-not $kill20Shown) { $kill20Shown = $true }
    Write-Host ("KILL-GATE 20k REACHED (~`$5 sunk): decide GO/KILL. In a second terminal run:") -ForegroundColor Cyan
    Write-Host ("  python scripts\blueprint2\bpv4_kill_gate.py --run-subdir $Run --iter 20000") -ForegroundColor Cyan
    Write-Host ("  (next decision point at ~30000 iters, ~`$12 sunk)") -ForegroundColor DarkGray
  } elseif ($iters -ge 0) {
    Write-Host ("kill-gate: next decision point at ~20000 iters (~`$5 sunk); then ~30000 (~`$12).") -ForegroundColor DarkGray
  }

  # completion
  if (($isDone -or ($iters -ge $Iters -and $iters -ge 0)) -and -not $rowStopLine) {
    Write-Host ""
    Write-Host "RUN COMPLETE (backstop reached). The babysitter terminates the box; follow README_LAUNCH.md 'V4/V5'." -ForegroundColor Green
    break
  }
  if ($rowStopLine) {
    Write-Host ""
    Write-Host "RUN ENDED via ROW_BUDGET_STOP. The babysitter terminates the box; evaluate the partial FINAL checkpoint." -ForegroundColor Yellow
    break
  }

  Write-Host ""
  Write-Host ("READ-ONLY (describe / get-console-output / s3 ls). refresh {0}s; Ctrl-C to stop." -f $Refresh) -ForegroundColor DarkGray
  Start-Sleep -Seconds $Refresh
}
