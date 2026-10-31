# READ-ONLY dashboard for the LIVE blueprint-v4 **v3** mini-proof arm (bpv4_h200_v3) -- the
# Pluribus-faithful TRAINER-SCHEDULE arm: FULLHIST + M-RR + LCFR + hard negative-regret pruning +
# snapshot averaging + coverage-as-first-class-metric.
#
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\watch_bpv4_v3.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\watch_bpv4_v3.ps1 -Refresh 30 -StallMin 75
#
# Pairs with scripts\blueprint2\bpv4_v3_babysitter.ps1 (which OWNS launch/resume/terminate). This script
# NEVER mutates AWS: it uses ONLY `aws ec2 describe-instances`, `aws ec2 get-console-output`, and
# `aws s3 ls`. It never terminates/modifies/put/cp/downloads. Kill decisions are YOURS (bpv4_kill_gate.py
# + the COVERAGE C-gates below). Design of record: artifacts\blueprint_v4_amendment_v3.md.
#
# Why the Stage-0 watcher is inadequate here (the v3 deltas this script bakes in):
#   * TWO row metrics, not one. Under -V3SnapshotAvg the driver DROPS the postflop strat_sum, so the
#     `PROGRESS ... infosets=Y` field is the PREFLOP-BOUNDED strat-row count (plateaus ~60M, NOT the RAM
#     lever). The RAM-driving cum_regret rows (~330-350M @40k, the G8 400M backstop) are printed on the
#     `REGSTATS ... rows=R` line. ROW_BUDGET_STOP fires on max(strat, cum) == cum here. We track BOTH.
#   * COVERAGE is a FIRST-CLASS gate (the decisive Stage-0 finding: 71.7% on-policy postflop miss @20k).
#     We parse the COVERAGE lines' postflop miss% at every checkpoint and evaluate the pre-registered
#     C-gates (C1<=55%@20k, C2<=45%@30k, C3<=35%@40k + flop<=30% + pre-allin-runout<=35% @40k) and the
#     K3 monotonicity kill (postflop miss RISING >5pt over the previous checkpoint at any iter>=15k).
#   * REGSTATS neg_p999 SANITY CLAUSE: at iter 5000 (prune_start), BEFORE pruning fires, neg_p999 must be
#     in [150, 1500] bb -- else the -30000bb prune-C is miscalibrated and the run should relaunch with C
#     rescaled (amendment doc section 2.2/6). We surface the 5k neg_p999 with a PASS/RESCALE verdict.
#   * SNAPSHOTS (snap_start 3200 = 8% of 40k; one per checkpoint => 8 by 40k) are the OFFLINE-AVERAGE
#     inputs -- the final blueprint = snap_avg.exe over checkpoints+snapshots. We count them (console +
#     s3 ls snapshots/) so a silent snapshot-upload failure is visible before the run ends.
#   * Checkpoints every 5000 iters (8 total); kill/decision points at 10k / 20k / 30k (trend clauses HARD
#     from 20k). STALL is NO new console PROGRESS line, auto-scaled to the arm's real it/s.
#
# Runbook: scripts\blueprint2\README_LAUNCH.md. ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
param(
  [string]$Run = "bpv4_h200_v3",
  [int]$Iters = 40000,                       # --iters backstop (spec measurement point; kills at 10/20/30k)
  [long]$RowBudget = 400000000,              # POKERBOT_ROW_BUDGET G8 RAM backstop (CUM-regret rows); 0 = off
  [int]$CheckpointEvery = 5000,              # 8 checkpoints; COVERAGE/REGSTATS/SNAPSHOT fire at this cadence
  [int]$PruneStart = 5000,                   # -V3PruneStart: pruning fires from here; neg_p999 read at 5k
  [double]$PruneC = -30000,                  # -V3PruneC (bb); for the neg_p999 [150,1500] sanity display
  [int]$SnapStart = 3200,                    # -V3SnapStart default (8% of 40k); first snapshot at 5k ckpt
  [int]$StratPlateau = 65000000,             # expected preflop-bounded strat rows under snapshot_avg
  [int]$Refresh = 45,
  [double]$SpotPrice = 2.50,                 # approx spot $/hr fallback for c6i.32xlarge (spotpick overrides)
  [int]$StallMin = 60,                       # alarm if NO new console PROGRESS line in this many minutes
  [string]$Region = "eu-central-1",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr"
)
$RUN_PREFIX  = "$Prefix/$Run"
$CKPT_PREFIX = "$RUN_PREFIX/checkpoints/"
$SNAP_PREFIX = "$RUN_PREFIX/snapshots/"
$t0 = Get-Date

# Pre-registered coverage C-gates (amendment doc section 6.2): postflop on-policy miss% caps by checkpoint.
$C1_ITER = 20000; $C1_MAX = 55.0
$C2_ITER = 30000; $C2_MAX = 45.0
$C3_ITER = 40000; $C3_MAX = 35.0; $C3_FLOP_MAX = 30.0; $C3_ALLIN_MAX = 35.0
$H200_BASELINE_20K = 71.7    # Stage-0 H200-CFR+ postflop miss @20k (the coverage bar this arm must beat)
$NEGP999_LO = 150.0; $NEGP999_HI = 1500.0   # prune-C calibration sanity band (bb) at iter 5000

# --- persistent state across the refresh loop ---
$lastProgIter       = -1            # max iters_done seen on a console PROGRESS line
$lastProgChangeTime = Get-Date      # when that last advanced (stall clock)
$lastCkptSeen       = -1            # max S3 checkpoint iter seen (2nd liveness signal for the stall clock)
$prevIter           = -1            # previous PROGRESS sample (for instantaneous it/s)
$prevElapsed        = -1.0
$curItersPerSec     = 0.0
$covHist            = @{}           # ckpt-iter -> @{post;flop;turn;river;pre;allin} (retained past console scroll)
$regHist            = @{}           # ckpt-iter -> @{rows;negp999;negmax}  (cum-rows + prune-C sanity)
$kill10Shown        = $false
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
    $out = (& python -c "import os,sys;root=os.environ.get('SPOTPICK_ROOT');sys.path.insert(0,root) if root else None;import spotpick;print({i.name:i.spot for i in spotpick.load()}.get('$it',''))" 2>$null)
    if ($out) { $out = ("$out").Trim(); if ($out -match '^[0-9.]+$') { $p = [double]$out } }
  } catch {}
  $priceCache[$it] = $p
  return $p
}

while ($true) {
  # --- live box (READ-ONLY describe-instances) ---
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

  # --- console output (READ-ONLY get-console-output): PRIMARY signal (PROGRESS/COVERAGE/REGSTATS/milestones) ---
  $console = ""
  if ($boxId) {
    $console = (aws ec2 get-console-output --instance-id $boxId --region $Region --latest `
                  --query "Output" --output text 2>$null | Out-String)
  }

  # NOTE: the parsed current-iteration local MUST NOT be named $iters -- PowerShell variable names are
  # CASE-INSENSITIVE, so $iters would alias the $Iters target param and clobber it to -1 every loop.
  $curIter = -1; $stratRows = -1.0; $elapsed = -1.0; $logRate = 0.0; $haveProgress = $false; $isDone = $false
  if ($console) {
    $mProg = [regex]::Matches($console, "PROGRESS iters_done=(\d+)/(\d+) infosets=(\d+) elapsed=([0-9.]+)s rate=([0-9.]+) it/s")
    if ($mProg.Count) {
      $g = $mProg[$mProg.Count - 1].Groups
      $curIter = [int]$g[1].Value
      $stratRows = [double]$g[3].Value
      $elapsed = [double]$g[4].Value
      $logRate = [double]$g[5].Value
      $haveProgress = $true
    }
    $mDone = [regex]::Match($console, "DONE iters_done=(\d+) infosets=(\d+)")
    if ($mDone.Success) {
      $isDone = $true
      $di = [int]$mDone.Groups[1].Value
      if ($di -gt $curIter) { $curIter = $di; $stratRows = [double]$mDone.Groups[2].Value }
    }
    # COVERAGE lines -> upsert into $covHist (postflop miss% is THE first-class v3 gate metric; the
    # raised_* fields are the presum K-COV subclass -- opener-faced raised-pot coverage, design 36).
    # The raised_* fields sit BETWEEN postflop= and pre_allin_runout=, so match them explicitly (an
    # adjacency regex silently stops matching every current-code COVERAGE line).
    foreach ($m in [regex]::Matches($console, "COVERAGE iters_done=(\d+) hands=\d+ miss% pre=([0-9.]+) flop=([0-9.]+) turn=([0-9.]+) river=([0-9.]+) postflop=([0-9.]+) raised_postflop=([0-9.]+) raised_flop=[0-9.]+ raised_turn=[0-9.]+ raised_river=[0-9.]+ raised_decisions=(\d+) pre_allin_runout=([0-9.]+)")) {
      $ci = [int]$m.Groups[1].Value
      $covHist[$ci] = @{ pre=[double]$m.Groups[2].Value; flop=[double]$m.Groups[3].Value;
                         turn=[double]$m.Groups[4].Value; river=[double]$m.Groups[5].Value;
                         post=[double]$m.Groups[6].Value; rpost=[double]$m.Groups[7].Value;
                         rdec=[int]$m.Groups[8].Value; allin=[double]$m.Groups[9].Value }
    }
    # REGSTATS lines -> cum-regret rows (the RAM lever) + neg_p999 (prune-C sanity)
    foreach ($m in [regex]::Matches($console, "REGSTATS iters_done=(\d+) rows=(\d+) neg_vals=\d+ pos_vals=\d+ neg_p50=[0-9.eE+-]+ neg_p90=[0-9.eE+-]+ neg_p99=[0-9.eE+-]+ neg_p999=([0-9.eE+-]+) neg_max=([0-9.eE+-]+)")) {
      $ri = [int]$m.Groups[1].Value
      $regHist[$ri] = @{ rows=[double]$m.Groups[2].Value; negp999=[double]$m.Groups[3].Value; negmax=[double]$m.Groups[4].Value }
    }
  }

  # --- milestones surfaced from the console (v3-specific) ---
  $ms = @()
  if ($console) {
    if ($console -match "FULLHIST_KEY=1")   { $ms += "FULLHIST_KEY=1" }
    if ($console -match "MRR=1")            { $ms += "MRR=1" }
    if ($console -match "LCFR=1")           { $ms += "LCFR=1" }
    if ($console -match "SNAPSHOT_AVG=1")   { $ms += "SNAPSHOT_AVG=1" }
    if ($console -match "WALL_GUARD armed") { $ms += "WALL_GUARD" }
    if ($console -match "RAYON_NUM_THREADS=(\d+)") { $ms += ("RAYON=" + $Matches[1]) }
    $mResume = [regex]::Matches($console, "RESUMED from latest checkpoint at iters_done=(\d+)")
    if ($mResume.Count) { $ms += ("RESUMED@" + $mResume[$mResume.Count - 1].Groups[1].Value) }
    if ($console -match "SPOT_RECLAIM_NOTICE") { $ms += "SPOT_RECLAIM" }
  }
  $snapAvgOn = ($console -and $console -match "SNAPSHOT_AVG=1")
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

  # --- S3 listings (heartbeat + snapshot completeness; NOT the stall clock) ---
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

  $lss = aws s3 ls "s3://$Bucket/$SNAP_PREFIX" --region $Region 2>$null
  $snapIters = @()
  if ($lss) {
    foreach ($line in $lss) {
      $name = ($line -split "\s+")[-1]
      if ($name -match "^snap_(\d+)\.nlsnap$") { $snapIters += [int]$Matches[1] }
    }
  }
  $maxSnapIter = if ($snapIters.Count) { ($snapIters | Measure-Object -Maximum).Maximum } else { -1 }
  $nSnap = $snapIters.Count

  # --- display iters: the AWS serial console (get-console-output) lags several minutes and is size-capped,
  # so PROGRESS often trails the S3 checkpoint the babysitter already sees. Fall back to the latest S3
  # checkpoint iter for the iters/progress/kill-gate readout (the babysitter itself keys off S3 latest.nlcfr).
  # $curIter (PROGRESS, elapsed-paired) still drives rate/ETA; $dispIter drives what the operator reads. ---
  $dispIter = [math]::Max($curIter, $maxCkptIter)
  $dispSrc = if ($curIter -ge 0 -and $curIter -ge $maxCkptIter) { "console" } elseif ($maxCkptIter -ge 0) { "s3-ckpt" } else { "" }

  # snapshots expected so far: one per checkpoint at iter >= SnapStart, up to the current iter
  $snapExpected = 0
  if ($dispIter -ge 0) {
    for ($k = $CheckpointEvery; $k -le $dispIter; $k += $CheckpointEvery) { if ($k -ge $SnapStart) { $snapExpected++ } }
  }

  # --- cum-regret rows (RAM lever) from the latest REGSTATS; slope from the two most recent REGSTATS ---
  $regIters = @($regHist.Keys | Sort-Object)
  $cumRows = -1.0; $cumRowsPerIter = 0.0; $curNegP999 = -1.0
  if ($regIters.Count -ge 1) {
    $hi = $regIters[-1]
    $cumRows = $regHist[$hi].rows
    $curNegP999 = $regHist[$hi].negp999
    if ($regIters.Count -ge 2) {
      $lo = $regIters[-2]
      $dI = $hi - $lo
      $dR = $regHist[$hi].rows - $regHist[$lo].rows
      if ($dI -gt 0) { $cumRowsPerIter = $dR / $dI }
    } elseif ($hi -gt 0) { $cumRowsPerIter = $cumRows / $hi }
  }

  # --- instantaneous it/s from successive PROGRESS samples (chunk-to-chunk) ---
  if ($haveProgress -and $curIter -gt $prevIter) {
    if ($prevIter -ge 0 -and $elapsed -gt $prevElapsed) {
      $dIter = $curIter - $prevIter
      $dEl   = $elapsed - $prevElapsed
      if ($dEl -gt 0) { $curItersPerSec = $dIter / $dEl }
    }
    $prevIter = $curIter; $prevElapsed = $elapsed
  }
  $itersPerSec = if ($curItersPerSec -gt 1e-9) { $curItersPerSec } elseif ($logRate -gt 1e-9) { $logRate } else { 0.0 }

  # --- stall clock: resets on a new PROGRESS iter OR a new S3 checkpoint; auto-scaled to real it/s ---
  if ($curIter -gt $lastProgIter) { $lastProgIter = $curIter; $lastProgChangeTime = Get-Date }
  if ($maxCkptIter -gt $lastCkptSeen) { $lastCkptSeen = $maxCkptIter; $lastProgChangeTime = Get-Date }
  $stallMinObs = [int]((Get-Date) - $lastProgChangeTime).TotalMinutes
  $expectChunkMin = if ($itersPerSec -gt 1e-9) { $CheckpointEvery / $itersPerSec / 60.0 } else { 0.0 }
  $effStall = [int]([math]::Max([double]$StallMin, 2.5 * $expectChunkMin))
  $stalled = ($haveProgress -and $stallMinObs -ge $effStall -and -not $isDone)

  # --- dual ETA: to 40k iters AND to the 400M CUM-row budget; the run ends at min(iters, cum-row-budget) ---
  $etaItersSec = -1.0; $etaRowsSec = -1.0
  if ($itersPerSec -gt 1e-9 -and $dispIter -ge 0 -and $dispIter -lt $Iters) {
    $etaItersSec = ($Iters - $dispIter) / $itersPerSec
  }
  if ($RowBudget -gt 0 -and $cumRows -ge 0 -and $cumRows -lt $RowBudget -and $cumRowsPerIter -gt 1e-9 -and $itersPerSec -gt 1e-9) {
    $etaRowsSec = (($RowBudget - $cumRows) / $cumRowsPerIter) / $itersPerSec
  }
  $endSec = -1.0; $endWhich = ""
  if ($etaItersSec -ge 0 -and $etaRowsSec -ge 0) {
    if ($etaRowsSec -lt $etaItersSec) { $endSec = $etaRowsSec; $endWhich = "CUM-ROW-BUDGET" } else { $endSec = $etaItersSec; $endWhich = "iters" }
  } elseif ($etaItersSec -ge 0) { $endSec = $etaItersSec; $endWhich = "iters" }
  elseif ($etaRowsSec -ge 0) { $endSec = $etaRowsSec; $endWhich = "CUM-ROW-BUDGET" }

  $iFrac = if ($Iters -gt 0 -and $dispIter -ge 0) { [double]$dispIter / $Iters } else { 0.0 }
  $rFrac = if ($RowBudget -gt 0 -and $cumRows -ge 0) { [double]$cumRows / $RowBudget } else { 0.0 }
  $spent = ($upMin / 60.0) * $livePrice
  $ecc = if ($endSec -ge 0) { ($endSec / 3600.0) * $livePrice } else { -1.0 }

  # ================================ RENDER ================================
  Clear-Host
  Write-Host ("=== bpv4 v3 (LCFR+prune+snap_avg, FULLHIST+M-RR): {0}   {1}   (watching {2}m) ===" -f $Run, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), [int]((Get-Date) - $t0).TotalMinutes) -ForegroundColor Cyan
  $boxColor = if ($boxRaw -match "running") { "Green" } elseif ($boxRaw) { "Yellow" } else { "DarkGray" }
  Write-Host ("box:    {0}" -f $boxLine) -ForegroundColor $boxColor

  $booting = ($boxId -and -not $haveProgress -and $maxCkptIter -lt 0)
  if ($booting) {
    $msTxt = if ($ms.Count) { ($ms -join " ") } else { "(no console milestones yet)" }
    Write-Host ("state:  booting/building... box up {0}m, no console PROGRESS or checkpoint yet" -f $upMin) -ForegroundColor Yellow
    Write-Host ("        (apt + AWS CLI v2 install + crate build + first {0}-iter chunk take a while)" -f $CheckpointEvery) -ForegroundColor DarkGray
    Write-Host ("boot:   {0}" -f $msTxt) -ForegroundColor DarkCyan
  } elseif (-not $boxId -and $maxCkptIter -lt 0 -and -not $haveProgress) {
    Write-Host "state:  no live box and no checkpoint yet - initial launch pending (babysitter relaunches)" -ForegroundColor DarkGray
  }

  if ($ms.Count) { Write-Host ("milestones: {0}" -f ($ms -join "  ")) -ForegroundColor Green }
  # v3 schedule (derived): LCFR window, prune-from, snap-from -- so the config is visible at a glance
  $pruneState = if ($dispIter -ge $PruneStart) { ("PRUNING (C={0}bb, from {1})" -f $PruneC, $PruneStart) } elseif ($dispIter -ge 0) { ("prune pending (starts {0})" -f $PruneStart) } else { ("prune from {0}" -f $PruneStart) }
  Write-Host ("sched:  LCFR D=250/W=10000 (40 discount events over first 25%)  |  {0}  |  snap from {1} (8%)" -f $pruneState, $SnapStart) -ForegroundColor DarkCyan

  # iters progress ($dispIter = max(console PROGRESS, S3 checkpoint); source shown so a lagging console is obvious)
  $idTxt = if ($dispIter -ge 0) { "$dispIter" } else { "?" }
  $srcTxt = if ($dispSrc) { " (via $dispSrc)" } else { "" }
  $iBarColor = if ($iFrac -ge 0.99) { "Green" } elseif ($iFrac -ge 0.5) { "Cyan" } else { "Yellow" }
  Write-Host ("iters:  {0}/{1}  [{2}] {3:P1}{4}" -f $idTxt, $Iters, (Bar $iFrac), $iFrac, $srcTxt) -ForegroundColor $iBarColor

  # --- rows: TWO metrics. CUM-regret = the RAM lever (vs 400M G8 budget); strat = preflop-bounded info ---
  if ($RowBudget -gt 0) {
    if ($cumRows -ge 0) {
      $rBarColor = if ($rFrac -ge 0.80) { "Red" } elseif ($rFrac -ge 0.60) { "Yellow" } else { "Cyan" }
      Write-Host ("cum-rows: {0:N0} / {1:N0}  [{2}] {3:P1} of G8 budget  (RAM lever; ROW_BUDGET_STOP watches max(strat,cum))" -f $cumRows, $RowBudget, (Bar $rFrac), $rFrac) -ForegroundColor $rBarColor
      if ($rFrac -ge 0.80) {
        Write-Host ("  WARNING: cum_regret rows at {0:P1} of 400M -- a ROW_BUDGET_STOP will END THE RUN EARLY with" -f $rFrac) -ForegroundColor Red
        Write-Host ("           PARTIAL evidence (writes a FINAL checkpoint + latest, then stops). Memory math projects") -ForegroundColor Red
        Write-Host ("           ~330-350M @40k (< 400M); crossing 80% early means growth is faster than projected.") -ForegroundColor Red
      }
    } else {
      Write-Host ("cum-rows: (no REGSTATS line yet -- prints at the first {0}-iter checkpoint)  vs {1:N0} G8 budget" -f $CheckpointEvery, $RowBudget) -ForegroundColor DarkGray
    }
  }
  if ($stratRows -ge 0) {
    $stratTxt = "{0:N0}" -f $stratRows
    $stratWarn = ($stratRows -gt ($StratPlateau * 1.35))   # >~88M => strat not preflop-bounded
    $stratColor = if ($stratWarn) { "Red" } else { "DarkGray" }
    $stratNote = if ($snapAvgOn) { "preflop-bounded (snapshot_avg drops postflop strat_sum); expected plateau ~{0:N0}" -f $StratPlateau } else { "SNAPSHOT_AVG not yet seen on console" }
    Write-Host ("strat:  {0}  ({1})" -f $stratTxt, $stratNote) -ForegroundColor $stratColor
    if ($stratWarn) {
      Write-Host ("  WARNING: strat rows exceed ~1.35x the {0:N0} preflop bound -- snapshot_avg may NOT be dropping" -f $StratPlateau) -ForegroundColor Red
      Write-Host ("           postflop strat_sum; RSS could blow past the 60% guard. Verify SNAPSHOT_AVG=1 on the box.") -ForegroundColor Red
    }
  }

  # rate + dual ETA
  $ipsTxt = if ($curItersPerSec -gt 1e-9) { "{0:N4} it/s" -f $itersPerSec } elseif ($itersPerSec -gt 1e-9) { "{0:N4} it/s (cum)" -f $itersPerSec } else { "--" }
  $rpiTxt = if ($cumRowsPerIter -gt 1e-9) { "{0:N0} cum-rows/iter" -f $cumRowsPerIter } else { "--" }
  Write-Host ("rate:   {0}   {1}" -f $ipsTxt, $rpiTxt) -ForegroundColor Yellow
  Write-Host ("ETA:    to 40k iters {0}   |   to 400M cum-rows {1}   |   RUN ENDS {2} (bound: {3})" -f `
      (EtaStr $etaItersSec), (EtaStr $etaRowsSec), (EtaStr $endSec), $(if ($endWhich) { $endWhich } else { "--" })) -ForegroundColor Yellow

  # ============ COVERAGE panel: THE first-class v3 gate (on-policy postflop miss%) ============
  Write-Host ""
  Write-Host ("COVERAGE (on-policy postflop miss% -- the decisive v3 gate; Stage-0 H200-CFR+ was {0:N1}% @20k):" -f $H200_BASELINE_20K) -ForegroundColor Cyan
  $covIters = @($covHist.Keys | Sort-Object)
  if ($covIters.Count -eq 0) {
    Write-Host "  (no COVERAGE line yet -- prints at each checkpoint from -V3CovHands 2048)" -ForegroundColor DarkGray
  } else {
    # per-checkpoint trend + hard C-gate verdicts
    foreach ($ci in $covIters) {
      $post = $covHist[$ci].post
      $gate = ""; $gcolor = "DarkGray"; $pass = $true
      if ($ci -ge $C3_ITER) {
        $flopOk = ($covHist[$ci].flop -le $C3_FLOP_MAX); $allinOk = ($covHist[$ci].allin -le $C3_ALLIN_MAX); $postOk = ($post -le $C3_MAX)
        $pass = ($postOk -and $flopOk -and $allinOk)
        $gate = ("C3<=35% (flop<=30 allin<=35): {0}" -f $(if ($pass) { "PASS" } else { "FAIL" }))
      } elseif ($ci -ge $C2_ITER) {
        $pass = ($post -le $C2_MAX); $gate = ("C2<=45%: {0}" -f $(if ($pass) { "PASS" } else { "FAIL" }))
      } elseif ($ci -ge $C1_ITER) {
        $pass = ($post -le $C1_MAX); $gate = ("C1<=55%: {0}" -f $(if ($pass) { "PASS" } else { "FAIL" }))
      } else {
        $gate = "(pre-C1: trend only)"
      }
      $gcolor = if ($ci -lt $C1_ITER) { "DarkGray" } elseif ($pass) { "Green" } else { "Red" }
      # K-COV (design 36): raised-pot ("opener-faced") coverage = 100 - raised_postflop miss. The
      # presum thesis is that THIS is high where fullhist is ~5%; shown as coverage% (n = raised-pot
      # decisions sampled; <2000 is an INSUFFICIENT_DATA read for the kill-gate).
      $rpost = $covHist[$ci].rpost; $rcov = if ($null -ne $rpost) { 100.0 - $rpost } else { $null }
      $kcovTxt = if ($null -ne $rcov) { "  K-COV(raised) {0,5:N1}% cov (n={1})" -f $rcov, $covHist[$ci].rdec } else { "" }
      Write-Host (("  iter {0,6}:  postflop {1,5:N1}%   flop {2,5:N1}%   allin-runout {3,5:N1}%   {4}" -f `
          $ci, $post, $covHist[$ci].flop, $covHist[$ci].allin, $gate) + $kcovTxt) -ForegroundColor $gcolor
    }
    # K3 monotonicity kill: postflop miss RISING >5pt over the previous checkpoint at any iter>=15k
    if ($covIters.Count -ge 2) {
      $a = $covIters[-2]; $b = $covIters[-1]
      $rise = $covHist[$b].post - $covHist[$a].post
      if ($b -ge 15000 -and $rise -gt 5.0) {
        Write-Host ("  K3 KILL: postflop miss ROSE {0:N1}pt ({1:N1}% -> {2:N1}%) from iter {3} to {4} (>5pt at iter>=15k)." -f $rise, $covHist[$a].post, $covHist[$b].post, $a, $b) -ForegroundColor Red
        Write-Host ("           Pre-registered K3: coverage is diverging, not converging -> KILL (amendment 6.2).") -ForegroundColor Red
      } elseif ($b -ge 15000) {
        Write-Host ("  K3 ok:  postflop miss {0:N1}pt over last ckpt ({1} -> {2}); needs to keep FALLING toward C3<=35%." -f $rise, $a, $b) -ForegroundColor DarkGray
      }
    }
  }

  # ============ REGSTATS: cum-rows already shown; here the neg_p999 prune-C sanity clause ============
  if ($regHist.ContainsKey($PruneStart)) {
    $np = $regHist[$PruneStart].negp999
    $ok = ($np -ge $NEGP999_LO -and $np -le $NEGP999_HI)
    $v = if ($ok) { "PASS -- prune-C -30000bb well-calibrated (C/p999 ~= {0:N0}x, target-ish 64x)" -f ($(if ($np -gt 0) { [math]::Abs($PruneC) / $np } else { 0 })) } else { "RESCALE -- relaunch with prune-C re-derived (amendment 2.2)" }
    $vc = if ($ok) { "Green" } else { "Red" }
    Write-Host ""
    Write-Host ("REGSTATS: neg_p999 @{0} (pre-prune) = {1:N1} bb   band [{2:N0},{3:N0}]   => {4}" -f $PruneStart, $np, $NEGP999_LO, $NEGP999_HI, $v) -ForegroundColor $vc
  } elseif ($curNegP999 -ge 0) {
    Write-Host ""
    Write-Host ("REGSTATS: latest neg_p999 = {0:N1} bb (5k pre-prune read pending; sanity band [{1:N0},{2:N0}])" -f $curNegP999, $NEGP999_LO, $NEGP999_HI) -ForegroundColor DarkGray
  }

  # snapshots (offline-average inputs) + checkpoints (heartbeat)
  Write-Host ""
  $snapColor = if ($snapExpected -gt 0 -and $nSnap -lt $snapExpected) { "Yellow" } else { "DarkCyan" }
  Write-Host ("snaps:  {0} in s3 (max iter {1}); expected {2} by now (one per ckpt from {3}); 8 total by 40k -> snap_avg.exe inputs" -f $nSnap, $maxSnapIter, $snapExpected, $SnapStart) -ForegroundColor $snapColor
  if ($snapExpected -gt 0 -and $nSnap -lt $snapExpected) {
    Write-Host ("  NOTE: fewer snapshots than expected -- check the console for 'SNAPSHOT UPLOAD FAIL/GAVE UP' (offline avg needs them).") -ForegroundColor Yellow
  }
  Write-Host ("ckpts:  {0} versioned (max iter {1}); every {2} iters, ~half the Stage-0 blob under snapshot_avg" -f $nCkpt, $maxCkptIter, $CheckpointEvery) -ForegroundColor DarkCyan
  if ($ckptLine) { Write-Host ("  {0}" -f $ckptLine) -ForegroundColor DarkGray }

  # cost
  $eccTxt = if ($ecc -ge 0) { ("~`${0:N2}" -f $ecc) } else { "--" }
  $itTxt = if ($instType) { $instType } else { "c6i.32xlarge" }
  Write-Host ("cost:   spent ~`${0:N2} this box   |   ECC {1} to run-end   (@ `${2:N3}/hr {3})" -f $spent, $eccTxt, $livePrice, $itTxt) -ForegroundColor Magenta

  # stall alarm (auto-scaled; resets on new PROGRESS OR checkpoint; cross-refs box state)
  if ($haveProgress) {
    if ($stalled) {
      $runTxt = if ($boxRaw -match "running") { "box is STILL 'running' -> most likely a slow chunk + upload, NOT a hang: read get-console-output for an error BEFORE killing" } else { "box is NOT 'running' (reclaimed/terminated) -> babysitter should relaunch + RESUME" }
      Write-Host ("STALL?: no new PROGRESS/checkpoint in {0}m (alarm {1}m = max({2}m floor, 2.5x ~{3}m/chunk))." -f $stallMinObs, $effStall, $StallMin, [int]$expectChunkMin) -ForegroundColor Red
      Write-Host ("        {0}" -f $runTxt) -ForegroundColor Red
    } else {
      Write-Host ("prog:   last advance {0}m ago (alarm at {1}m, auto-scaled to ~{2}m/{3}-chunk)" -f $stallMinObs, $effStall, [int]$expectChunkMin, $CheckpointEvery) -ForegroundColor DarkGray
    }
  } elseif (-not $booting) {
    Write-Host ("prog:   no console PROGRESS line seen yet (stall clock not armed)") -ForegroundColor DarkGray
  }

  # ROW_BUDGET_STOP milestone (terminal, partial evidence)
  if ($rowStopLine) {
    Write-Host ""
    Write-Host ("ROW_BUDGET_STOP: {0}" -f $rowStopLine) -ForegroundColor Red
    Write-Host ("  The run stopped EARLY at the 400M cum-row backstop (PARTIAL evidence). FINAL checkpoint written.") -ForegroundColor Red
  }

  # kill/decision reminders (amendment 6.2 / 24.3): 10k / 20k / 30k. Shape via bpv4_kill_gate.py; coverage via COVERAGE above.
  # arm-3 (widened preflop menu, run subdir bpv4_emd160_arm3) MUST pass --pre-menu arm3 to every kill-gate call:
  # a default-menu scan of an arm-3 blob drops the id-5/6 open mass and biases every gate toward a FALSE KILL
  # (the kill-gate now fail-closed ABORTS rc2 on that mismatch, but printing the flag keeps the copy-paste correct).
  $preMenuArg = if ($Run -eq "bpv4_emd160_arm3") { " --pre-menu arm3" } else { "" }
  Write-Host ""
  if ($dispIter -ge 30000) {
    if (-not $kill30Shown) { $kill30Shown = $true }
    Write-Host ("KILL-GATE 30k (K2, HARD level caps + trend + C2<=45%): decide GO/KILL. In a second terminal:") -ForegroundColor Cyan
    Write-Host ("  python scripts\blueprint2\bpv4_kill_gate.py --run-subdir $Run --iter 30000$preMenuArg") -ForegroundColor Cyan
    Write-Host ("  + eyeball C2 (postflop miss <=45%) on the COVERAGE panel above.") -ForegroundColor DarkGray
  } elseif ($dispIter -ge 20000) {
    if (-not $kill20Shown) { $kill20Shown = $true }
    Write-Host ("KILL-GATE 20k (K1, BINDING all-clause: G1 mono + G2 floor + G3 AA-limp<=30% + G4 + G6 all HARD): decide GO/KILL. Run:") -ForegroundColor Cyan
    Write-Host ("  python scripts\blueprint2\bpv4_kill_gate.py --run-subdir $Run --iter 20000$preMenuArg") -ForegroundColor Cyan
    Write-Host ("  + eyeball C1 (postflop miss <=55%) above.  (next: ~30k K2; then 40k spec point)") -ForegroundColor DarkGray
  } elseif ($dispIter -ge 10000) {
    if (-not $kill10Shown) { $kill10Shown = $true }
    Write-Host ("KILL-GATE 10k (K1a EARLY-KILL, amendment 24.3): AA-BTN-limp>=50% => KILL (sole HARD clause; all else advisory). Run:") -ForegroundColor Cyan
    Write-Host ("  python scripts\blueprint2\bpv4_kill_gate.py --run-subdir $Run --iter 10000$preMenuArg") -ForegroundColor Cyan
  } elseif ($dispIter -ge 0) {
    Write-Host ("kill-gate: next decision points at ~10k (K1a early-kill), ~20k (K1+C1), ~30k (K2+C2), 40k (spec).") -ForegroundColor DarkGray
  }

  # completion (S3 checkpoint at the target counts even if the console DONE line is still lagging)
  if (($isDone -or ($dispIter -ge $Iters -and $dispIter -ge 0)) -and -not $rowStopLine) {
    Write-Host ""
    Write-Host "RUN COMPLETE (40k backstop reached). Babysitter terminates the box. AFTER THE RUN:" -ForegroundColor Green
    Write-Host "  download checkpoints/iter_040000.nlcfr + snapshots/snap_*.nlsnap, then snap_avg.exe -> final blueprint." -ForegroundColor Green
    Write-Host "  Eval MUST set POKERBOT_FULLHIST_KEY=1 AND POKERBOT_MRR=1 (gates G1-G4/G6 + G5'-checkcall + C1-C3)." -ForegroundColor Green
    break
  }
  if ($rowStopLine) {
    Write-Host ""
    Write-Host "RUN ENDED via ROW_BUDGET_STOP. Babysitter terminates the box; evaluate the partial FINAL checkpoint." -ForegroundColor Yellow
    break
  }

  Write-Host ""
  Write-Host ("READ-ONLY (describe / get-console-output / s3 ls). refresh {0}s; Ctrl-C to stop." -f $Refresh) -ForegroundColor DarkGray
  Start-Sleep -Seconds $Refresh
}
