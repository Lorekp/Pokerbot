# Autonomous babysitter for the BLUEPRINT-V2 river-OCHS retrain (bp2_riverochs_1).
#
# Fork of the proven scripts/blueprint_run_babysitter.ps1 (same resume-from-S3, relaunch-storm guards,
# spot-quota guard, spotpick auto-pick, convergence stop, on-box dead-man cost caps) with the two
# blueprint-v2 changes:
#   * -RiverOchs <table>  : passes --river-ochs to the launcher so the trainer keys the RIVER with the
#     OCHS nearest-centroid bucket (design: artifacts/blueprint2_design.md, Change A).
#   * -ConvMinIters 40000 : the convergence stop may NOT fire before this many iters (Change B — the
#     bp_emd_k160_1 run stopped prematurely at 18k because the aggregate strategy-delta plateaus as soon
#     as the preflop mass converges, while postflop rows are still undertrained).
#
# Run (your AWS creds; same boto3 default profile the launcher uses):
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\blueprint2_babysitter.ps1 -WorkSteal
#
# Pairs with scripts\blueprint2\watch_blueprint2.ps1 (read-only dashboard, separate terminal).
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
param(
  [string]$Run = "bp2_riverochs_1",   # S3 run-subdir; keep it for resume-across-preemption
  [int]$Iters = 62000,                # --iters backstop (knee <= 60k; convergence stop may fire first,
                                      #   but never before -ConvMinIters)
  [int]$NDeals = 200,                 # --n-deals (the proven prod value)
  [int]$NumBuckets = 160,             # MUST equal the EMD blob's K (ctor-enforced)
  [string]$EmdFrozen = "data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn",
  [string]$RiverOchs = "data/abstraction/river_ochs_160.rochs",  # blueprint-v2 river OCHS table
  [string]$DepthEdges = "",           # non-depth blueprint (depth randomization deliberately deferred)
  [int]$CheckpointEvery = 500,
  [int]$GateEvery = 0,                # --gate-every (fullhist arms: pre-registered gate/measurement
                                      #   cadence; rolling checkpoints every -CheckpointEvery are
                                      #   resume-only, pruned to newest 3 + all gate-cadence ones.
                                      #   0 = omit = legacy: every checkpoint is a gate)
  [int]$CkptVolumeGb = 0,             # --ckpt-volume-gb (root gp3 size for local checkpoint staging;
                                      #   0 = omit = launcher default: 400 GB on fullhist/M-RR arms)
  # --- LEVERS ---
  [switch]$PreflopHistKey,            # blueprint-v3: pass --preflop-hist-key (POKERBOT_PREFLOP_HIST_KEY=1
                                      #   on the box; the aliasing fix). Serve MUST then use the same env
                                      #   (ship_blueprint --preflop-hist-key). Use via bp3_prefhist_babysitter.ps1.
  [switch]$FullHistKey,               # blueprint-v4 Stage-0: pass --fullhist-key (POKERBOT_FULLHIST_KEY=1 on
                                      #   the box; full action-history key on EVERY street). Pair with -Mrr.
                                      #   Serve/eval MUST use the same env. Use via bpv4_stage0_babysitter.ps1.
  [switch]$Mrr,                       # blueprint-v4 Stage-0: pass --mrr (POKERBOT_MRR=1; M-RR action-thinning
                                      #   that makes fullhist convergent, ~10^31 -> ~413M rows). Probe/eval
                                      #   MUST also set POKERBOT_MRR=1 (train/probe abstraction-skew guard).
  [switch]$WorkSteal,                 # pass --work-steal (+33% deals/s @64c measured; pass it)
  [switch]$Dcfr,                      # NOT recommended (CFR+ > DCFR at iter16000 for 6-max)
  [double]$DcfrPruneThreshold = -300000.0,
  [double]$DcfrExploreFrac = 0.05,
  # --- blueprint-v4 v3 trainer schedule (artifacts/blueprint_v4_amendment_v3.md; all off => the
  # ---  rendered bootstrap is byte-identical, sha-compared). Use via bpv4_v3_babysitter.ps1.
  [switch]$V3Lcfr,                    # --v3-lcfr (POKERBOT_LCFR=1: signed regrets + LCFR discount +
                                      #   unweighted averaging). Mutually exclusive with -Dcfr.
  [switch]$V3SignedRegrets,           # --v3-signed-regrets (POKERBOT_SIGNED_REGRETS=1: arm-2 signed
                                      #   regrets + floor + hard pruning, NO LCFR discount, w=t
                                      #   averaging kept). Mutually exclusive with -V3Lcfr / -Dcfr.
  [int]$V3LcfrD = 250,                # --v3-lcfr-d (discount interval, iterations)
  [int]$V3LcfrW = 10000,              # --v3-lcfr-w (discount window, iterations)
  [string]$V3PruneC = "",             # --v3-prune-c (hard prune threshold, bb; "" = pruning off)
  [string]$V3PruneStart = "",         # --v3-prune-start ("" = W/2 default inside the trainer)
  [double]$V3PruneExplore = 0.05,     # --v3-prune-explore (Pluribus 5%)
  [string]$V3RegretFloor = "",        # --v3-regret-floor ("" = C*31/30 default inside the trainer)
  [switch]$V3SnapshotAvg,             # --v3-snapshot-avg (no postflop strat_sum + NLSNAP01 snapshots)
  [string]$V3SnapStart = "",          # --v3-snap-start ("" = 8% of --iters)
  [int]$V3CovHands = 2048,            # --v3-cov-hands (COVERAGE line cadence; 0 disables)
  [switch]$V3Regstats,                # --v3-regstats (REGSTATS percentile lines)
  [long]$RowBudget = 0,               # --row-budget (POKERBOT_ROW_BUDGET G8 RAM backstop; 0 = omit =
                                      #   driver 400M default + byte-identical bootstrap)
  # --- box / market ---
  [string]$InstanceType = "",          # "" = AUTO-PICK cheapest spot-available type via spotpick
  [int]$Vcpus = 128,
  [string]$Arch = "x86_64",
  [switch]$ReuseCode,                 # pass --reuse-code on EVERY launch incl. #1 (2026-07-06 incident:
                                      #   an account-side pipeline reacts to code-tarball uploads and
                                      #   plants an untrusted .so in the build cache; no upload = no
                                      #   trigger. Requires the S3 tarball to already match the tree.)
  [string]$CrateSrc = "tools/battery_native",  # the LIVE crate carries river_ochs.rs (the frozen copy at
                                               # runs/bp_crate_frozen does NOT); freeze a new copy first if
                                               # concurrent crate edits are expected during the run
  [switch]$Spotpick,
  [switch]$OnDemand,
  # --- storm + wall guards (verbatim from the proven babysitter) ---
  [int]$MaxLaunch = 12,
  [int]$MaxNoCkptRelaunches = 3,      # CHURN-ABORT (2026-07-05, user directive after 5 boxes burned
                                      #   with zero ckpts): abort if this many launches IN A ROW die
                                      #   without a single NEW checkpoint landing in S3 (marker-based
                                      #   iters). Every further relaunch in a capacity storm is burn.
  [int]$MinRelaunchGapSec = 600,
  [int]$BootWaitSec = 300,
  [int]$PollSec = 180,
  [int]$MaxHours = 12,
  [string]$Region = "eu-central-1",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr",
  # --- convergence-based stop ---
  [switch]$NoConvergeStop,
  [double]$ConvThreshold = 0.020,
  [int]$ConvK = 3,
  [int]$ConvMinIters = 40000,         # NEW: convergence stop may not fire before this (postflop-coverage
                                      # floor; the 18k premature-stop fix)
  [int]$ConvNumBuckets = 160,
  [string]$ConvDepthEdges = ""
)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..")   # repo root (this script lives in scripts\blueprint2\)

if (-not $Run) {
  $Run = "bp2_{0}" -f ([int][double]::Parse((Get-Date -UFormat %s)))
  Write-Host "WARNING: no -Run given; generated '$Run'. Relaunch-on-preemption resumes THIS subdir." -ForegroundColor Yellow
}
if ($EmdFrozen -and -not (Test-Path $EmdFrozen)) {
  Write-Host "FATAL: -EmdFrozen '$EmdFrozen' not found." -ForegroundColor Red
  exit 1
}
if ($RiverOchs -and -not (Test-Path $RiverOchs)) {
  Write-Host "FATAL: -RiverOchs '$RiverOchs' not found. Run scripts\blueprint2\fit_river_ochs.py first." -ForegroundColor Red
  exit 1
}
if ($RiverOchs -and -not (Test-Path "$RiverOchs.sha256")) {
  Write-Host "FATAL: '$RiverOchs.sha256' sidecar missing (on-box verify + ship gate need it)." -ForegroundColor Red
  exit 1
}

$RUN_PREFIX = "$Prefix/$Run"
$LATEST     = "$RUN_PREFIX/checkpoints/latest.nlcfr"
$MAXSEC     = $MaxHours * 3600
$start      = Get-Date
$launches   = 0
$lastLaunch = [datetime]::MinValue
$doneAtLastLaunch = [int64]-1   # iters at the moment of the previous launch (churn-abort baseline)
$noCkptRelaunches = 0           # consecutive relaunches with no new checkpoint since the prior launch
$tracked    = ""
$prevBox    = ""
$convergedStop = $false
$ConvStateFile = Join-Path $env:TEMP ("conv_stop_{0}.json" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
$lastConvIters = -1

# AUTO-PICK cheapest spot-available type via the shared spotpick lib when -InstanceType was not given.
if (-not $InstanceType) {
  $SpotpickPy = "spotpick.py"
  $picked = ""
  if (Test-Path $SpotpickPy) {
    try { $picked = (& python $SpotpickPy best --vcpus $Vcpus --arch $Arch 2>$null) } catch { $picked = "" }
    if ($picked) { $picked = ("$picked").Trim() }
  }
  if ($picked -match '^[a-z][a-z0-9-]*\.[a-z0-9]+$') {
    $InstanceType = $picked
    Write-Host "AUTO-PICK: spotpick best $Vcpus-vCPU $Arch -> $InstanceType" -ForegroundColor Green
  } else {
    $InstanceType = "c6a.32xlarge"
    Write-Host "AUTO-PICK fallback (spotpick unavailable): $InstanceType" -ForegroundColor Yellow
  }
  $Spotpick = $true
}

function Get-LiveBoxId {
  $r = (aws ec2 describe-instances --region $Region `
        --filters "Name=tag:Run,Values=$Run" "Name=instance-state-name,Values=running,pending" `
        --query "Reservations[].Instances[].InstanceId" --output text 2>$null)
  if ($r) { return (($r -split "\s+") | Where-Object { $_ -like "i-*" } | Select-Object -First 1) }
  return ""
}
function Get-StateById([string]$id) {
  if (-not $id) { return "" }
  $s = (aws ec2 describe-instances --region $Region --instance-ids $id `
        --query "Reservations[].Instances[].State.Name" --output text 2>$null)
  if ($s) { return $s.Trim() } else { return "" }
}
function Get-LatestItersDone {
  # NEW (arm-2 spot-resilience ops, 2026-07-05): prefer the tiny iter_XXXXXX.nlcfr.ok markers the
  # detached uploader writes AFTER each complete upload -- iters parsed from the KEY NAME, nothing
  # downloaded. The old path downloaded the ENTIRE latest.nlcfr (up to ~58 GB on fullhist arms!)
  # every poll just to read 16 header bytes. Markers also mean progress is only ever counted for
  # DURABLE checkpoints (a partial upload has no marker).
  $keys = (aws s3api list-objects-v2 --bucket $Bucket --prefix "$RUN_PREFIX/checkpoints/" `
           --query "Contents[].Key" --output text --region $Region 2>$null)
  if ($keys) {
    $best = [int64]-1
    foreach ($k in ($keys -split "\s+")) {
      if ($k -match "iter_(\d+)\.nlcfr\.ok$") {
        $v = [int64]$Matches[1]
        if ($v -gt $best) { $best = $v }
      }
    }
    if ($best -ge 0) { return $best }
  }
  # Legacy runs (no markers): read ONLY the 16-byte NLCFR001 header of latest.nlcfr via a ranged GET.
  $tmp = Join-Path $env:TEMP "bp2_latest_probe.hdr"
  aws s3api get-object --bucket $Bucket --key $LATEST --range "bytes=0-15" $tmp --region $Region 2>$null | Out-Null
  if (-not (Test-Path $tmp)) { return -1 }
  try {
    $buf = [System.IO.File]::ReadAllBytes($tmp)
    if ($buf.Length -lt 16) { return -1 }
    if ([System.Text.Encoding]::ASCII.GetString($buf, 0, 8) -ne "NLCFR001") { return -1 }
    return [int64][System.BitConverter]::ToUInt64($buf, 8)
  } catch { return -1 } finally { Remove-Item $tmp -ErrorAction SilentlyContinue }
}
function Stop-Run {
  $id = Get-LiveBoxId
  if (-not $id -and $tracked) { $st = Get-StateById $tracked; if ($st -eq "running" -or $st -eq "pending") { $id = $tracked } }
  if ($id) { aws ec2 terminate-instances --instance-ids $id --region $Region | Out-Null; Write-Host "terminated: $id" }
}
function Test-Converged {
  $argsList = @(
    "scripts\blueprint_convergence_stop.py",
    "--run-prefix", $RUN_PREFIX, "--bucket", $Bucket, "--region", $Region,
    "--threshold", $ConvThreshold, "--k", $ConvK,
    "--state-file", $ConvStateFile, "--num-buckets", $ConvNumBuckets
  )
  if ($ConvDepthEdges) { $argsList += @("--depth-edges"); $argsList += ($ConvDepthEdges -split ",") }
  $out = (& python @argsList 2>&1)
  $txt = ($out | Out-String)
  if ($txt -match "STATUS=([A-Z]+)") { $status = $Matches[1] } else { $status = "ERROR" }
  ($txt -split "`r?`n") | Where-Object { $_ -match "STATUS=" } | ForEach-Object { Write-Host "    conv: $_" }
  return $status
}
function Invoke-Launch {
  $argsList = @(
    "scripts\launch_native_cfr_training_spot.py",
    "--run-subdir", $Run, "--iters", $Iters, "--n-deals", $NDeals,
    "--num-buckets", $NumBuckets, "--checkpoint-every", $CheckpointEvery,
    "--max-hours", ($MaxHours + 1),
    "--instance-type", $InstanceType, "--arch", $Arch
  )
  if ($GateEvery -gt 0)    { $argsList += @("--gate-every", $GateEvery) }
  if ($CkptVolumeGb -gt 0) { $argsList += @("--ckpt-volume-gb", $CkptVolumeGb) }
  if ($EmdFrozen)  { $argsList += @("--emd-frozen", $EmdFrozen) }
  if ($RiverOchs)  { $argsList += @("--river-ochs", $RiverOchs) }
  if ($DepthEdges) { $argsList += @("--depth-edges", $DepthEdges) }
  if ($PreflopHistKey) { $argsList += "--preflop-hist-key" }
  if ($FullHistKey) { $argsList += "--fullhist-key" }
  if ($Mrr)        { $argsList += "--mrr" }
  if ($WorkSteal)  { $argsList += "--work-steal" }
  if ($Dcfr)       { $argsList += @("--dcfr", "--dcfr-prune-threshold", $DcfrPruneThreshold,
                                    "--dcfr-explore-frac", $DcfrExploreFrac) }
  if ($V3Lcfr) {
    $argsList += @("--v3-lcfr", "--v3-lcfr-d", $V3LcfrD, "--v3-lcfr-w", $V3LcfrW,
                   "--v3-prune-explore", $V3PruneExplore)
    if ($V3PruneC)      { $argsList += @("--v3-prune-c", $V3PruneC) }
    if ($V3PruneStart)  { $argsList += @("--v3-prune-start", $V3PruneStart) }
    if ($V3RegretFloor) { $argsList += @("--v3-regret-floor", $V3RegretFloor) }
  }
  if ($V3SignedRegrets) {
    # arm 2 (artifacts/blueprint_v4_arm2_amendment.md): signed regrets + floor + hard pruning
    # WITHOUT the LCFR discount; w=t averaging kept. Mutually exclusive with -V3Lcfr (the
    # launcher p.error()s if both are passed through).
    $argsList += @("--v3-signed-regrets", "--v3-prune-explore", $V3PruneExplore)
    if ($V3PruneC)      { $argsList += @("--v3-prune-c", $V3PruneC) }
    if ($V3PruneStart)  { $argsList += @("--v3-prune-start", $V3PruneStart) }
    if ($V3RegretFloor) { $argsList += @("--v3-regret-floor", $V3RegretFloor) }
  }
  if ($V3SnapshotAvg) {
    $argsList += "--v3-snapshot-avg"
    if ($V3SnapStart) { $argsList += @("--v3-snap-start", $V3SnapStart) }
  }
  if ($V3Lcfr -or $V3SignedRegrets -or $V3SnapshotAvg) {
    $argsList += @("--v3-cov-hands", $V3CovHands)
    if ($V3Regstats) { $argsList += "--v3-regstats" }
  }
  if ($RowBudget -gt 0) { $argsList += @("--row-budget", $RowBudget) }
  if ($Spotpick)   { $argsList += "--spotpick" }
  if ($OnDemand)   { $argsList += "--on-demand" }
  if ($CrateSrc)   { $argsList += @("--crate-src", $CrateSrc) }
  if ($ReuseCode -or $launches -ge 1) { $argsList += "--reuse-code" }
  Write-Host "  launching: python $($argsList -join ' ')"
  $out = (& python @argsList 2>&1)
  $out | ForEach-Object { Write-Host "    $_" }
  $txt = ($out | Out-String)
  if ($txt -match "(?i)launched\s+(i-[0-9a-f]+)") { return $Matches[1] }
  return ""
}

$capH = [int]($MAXSEC / 3600)
$leverTxt = (@(if ($PreflopHistKey) { "preflop_hist_key" }; if ($FullHistKey) { "fullhist_key" }; if ($Mrr) { "mrr" }; if ($WorkSteal) { "work_steal" }; if ($Dcfr) { "dcfr" }; if ($V3Lcfr) { "v3_lcfr(D=$V3LcfrD,W=$V3LcfrW,C=$V3PruneC)" }; if ($V3SignedRegrets) { "v3_signed_regrets(C=$V3PruneC,start=$V3PruneStart)" }; if ($V3SnapshotAvg) { "v3_snapshot_avg" }) -join "+"); if (-not $leverTxt) { $leverTxt = "NONE (baseline)" }
Write-Host "=== blueprint-v2 babysitter: $Run ===" -ForegroundColor Cyan
Write-Host "    target=$Iters it  n_deals=$NDeals  flop/turn=EMD K$NumBuckets  river=OCHS ($([System.IO.Path]::GetFileName($RiverOchs)))"
Write-Host "    LEVERS=$leverTxt" -ForegroundColor Green
Write-Host "    caps: ${capH}h wall / $MaxLaunch launches / >=${MinRelaunchGapSec}s between launches / churn-abort after $MaxNoCkptRelaunches no-ckpt relaunches"
Write-Host "    convergence-stop=$(-not $NoConvergeStop) (threshold=$ConvThreshold, K=$ConvK, MIN-ITERS FLOOR=$ConvMinIters)"
Write-Host "    resume: a relaunch reuses --run-subdir '$Run' so the new box loads s3://$Bucket/$LATEST"

while ($true) {
  $el   = [int]((Get-Date) - $start).TotalSeconds
  $done = Get-LatestItersDone
  $live = Get-LiveBoxId
  $draining = ""
  if (-not $live -and $tracked) {
    $st = Get-StateById $tracked
    if ($st -eq "running" -or $st -eq "pending") { $live = $tracked }
    elseif ($st -eq "shutting-down" -or $st -eq "stopping") {
      $draining = $tracked
      Write-Host "    tracked box $tracked is '$st' (still holds quota; will not relaunch until terminated)"
    }
    elseif ($st) { Write-Host "    tracked box $tracked is '$st'"; $tracked = "" }
  }
  $doneTxt = if ($done -ge 0) { "$done/$Iters" } else { "no-ckpt-yet" }
  $boxTxt  = if ($live) { $live } else { "(none)" }
  Write-Host "[t+${el}s] iters=$doneTxt  live=$boxTxt  launches=$launches"
  if ($prevBox -and $live -and $live -ne $prevBox) {
    Write-Host "          BOX CHANGED: $prevBox -> $live (spot reclaim + relaunch / resumed from S3)" -ForegroundColor Yellow
  }
  if ($live) { $prevBox = $live; $tracked = $live }

  # 1) target reached -> done (ground truth = the checkpoint's iters_done, NOT box presence)
  if ($done -ge $Iters) { Write-Host "DONE: latest checkpoint reached $done/$Iters iters (iters BACKSTOP)."; Stop-Run; break }
  # 2) wall cap (BACKSTOP)
  if ($el -ge $MAXSEC) { Write-Host "STOP: wall-clock cap (${capH}h) reached (wall BACKSTOP)."; Stop-Run; break }
  # 2b) CONVERGENCE STOP with the blueprint-v2 MIN-ITERS FLOOR: the bp_emd_k160_1 lesson — the aggregate
  # strategy-delta plateaus once the (dominant) preflop mass converges while postflop rows are still
  # undertrained, so the stop may not fire before $ConvMinIters.
  if (-not $NoConvergeStop -and $done -ge $ConvMinIters -and $done -gt $lastConvIters) {
    $lastConvIters = $done
    $cstat = Test-Converged
    if ($cstat -eq "CONVERGED") {
      Write-Host "DONE: convergence stop -> avg strategy plateaued (delta <= $ConvThreshold for $ConvK ckpts) at $done iters (>= $ConvMinIters floor)." -ForegroundColor Green
      $convergedStop = $true
      Stop-Run; break
    }
  } elseif (-not $NoConvergeStop -and $done -gt 0 -and $done -lt $ConvMinIters -and $done -gt $lastConvIters) {
    $lastConvIters = $done
    Write-Host "    conv: below min-iters floor ($done < $ConvMinIters) - not testing convergence yet"
  }
  # 3) a box is live and not done -> wait
  if ($live) { Start-Sleep -Seconds $PollSec; continue }

  # 3b) EXACT-QUOTA GUARD: a previous box still terminating holds the spot vCPU quota; wait before relaunch.
  if ($draining) {
    Write-Host "  previous box $draining still terminating (holds quota); waiting ${PollSec}s before relaunch"
    Start-Sleep -Seconds $PollSec
    continue
  }

  # 4) NO live box and target NOT reached -> preemption (or never-launched). Relaunch, with storm guards.
  if ($launches -ge $MaxLaunch) { Write-Host "STOP: max launches ($MaxLaunch) reached (storm guard)."; break }
  # 4a) CHURN-ABORT: the box that just died saved NO new checkpoint since its own launch -> streak++.
  # A streak of $MaxNoCkptRelaunches means capacity is being reclaimed faster than the checkpoint
  # cadence; every further relaunch is pure burn. Abort loudly instead (S3 state is preserved).
  if ($launches -gt 0) {
    if ($done -le $doneAtLastLaunch) {
      $noCkptRelaunches++
      Write-Host "  churn guard: box died with NO new checkpoint (iters still $doneTxt); no-progress streak = $noCkptRelaunches/$MaxNoCkptRelaunches" -ForegroundColor Yellow
      if ($noCkptRelaunches -ge $MaxNoCkptRelaunches) {
        Write-Host "ABORT: $MaxNoCkptRelaunches consecutive launches died without saving a single new checkpoint (spot capacity storm). NOT relaunching -- switch instance type/region (spotpick chain) or -OnDemand, then rerun this babysitter to resume from S3." -ForegroundColor Red
        Stop-Run
        break
      }
    } else {
      $noCkptRelaunches = 0
    }
  }
  $sinceLast = ((Get-Date) - $lastLaunch).TotalSeconds
  if ($launches -gt 0 -and $sinceLast -lt $MinRelaunchGapSec) {
    $wait = [int]($MinRelaunchGapSec - $sinceLast)
    Write-Host "  relaunch gap guard: only ${sinceLast}s since last launch (< ${MinRelaunchGapSec}s); waiting ${wait}s"
    Start-Sleep -Seconds $wait
    continue
  }
  $reason = if ($done -ge 0) { "preempted at $done/$Iters -> resume from S3" } else { "no box + no ckpt -> initial launch" }
  Write-Host "  no live box; $reason. Launching #$($launches+1)..." -ForegroundColor Cyan
  $doneAtLastLaunch = $done
  $id = Invoke-Launch
  $lastLaunch = Get-Date
  $launches++
  if ($id) {
    $tracked = $id
    Write-Host "  tracking $id; boot grace ${BootWaitSec}s before next poll (storm guard)"
    Start-Sleep -Seconds $BootWaitSec
  } else {
    Write-Host "  launch did not report an instance id (capacity/quota/launcher error) - backing off ${MinRelaunchGapSec}s" -ForegroundColor Yellow
    Start-Sleep -Seconds $MinRelaunchGapSec
  }
}

$final = Get-LatestItersDone
Write-Host ""
Write-Host "=== BABYSITTER EXIT after ${el}s, $launches launches. latest iters_done=$final / $Iters ==="
if ($convergedStop) {
  Write-Host "Run COMPLETE via CONVERGENCE STOP at $final iters (>= $ConvMinIters floor)." -ForegroundColor Green
} elseif ($final -ge $Iters) {
  Write-Host "Run COMPLETE (iters backstop)." -ForegroundColor Green
} else {
  Write-Host "Run NOT complete (hit a cap or stall). The S3 latest checkpoint is preserved for a manual resume:" -ForegroundColor Yellow
  $resume = "python scripts\launch_native_cfr_training_spot.py --run-subdir $Run --iters $Iters --n-deals $NDeals --num-buckets $NumBuckets --checkpoint-every $CheckpointEvery"
  if ($GateEvery -gt 0) { $resume += " --gate-every $GateEvery" }
  if ($CkptVolumeGb -gt 0) { $resume += " --ckpt-volume-gb $CkptVolumeGb" }
  if ($EmdFrozen) { $resume += " --emd-frozen $EmdFrozen" }
  if ($RiverOchs) { $resume += " --river-ochs $RiverOchs" }
  if ($PreflopHistKey) { $resume += " --preflop-hist-key" }
  if ($FullHistKey) { $resume += " --fullhist-key" }
  if ($Mrr) { $resume += " --mrr" }
  if ($WorkSteal) { $resume += " --work-steal" }
  if ($V3Lcfr) { $resume += " --v3-lcfr --v3-lcfr-d $V3LcfrD --v3-lcfr-w $V3LcfrW"
    if ($V3PruneC) { $resume += " --v3-prune-c $V3PruneC" }
    if ($V3PruneStart) { $resume += " --v3-prune-start $V3PruneStart" } }
  if ($V3SnapshotAvg) { $resume += " --v3-snapshot-avg"
    if ($V3SnapStart) { $resume += " --v3-snap-start $V3SnapStart" } }
  if ($V3Lcfr -or $V3SnapshotAvg) { $resume += " --v3-cov-hands $V3CovHands"
    if ($V3Regstats) { $resume += " --v3-regstats" } }
  Write-Host "  $resume"
}
if ($final -ge 0 -and ($convergedStop -or $final -ge $Iters)) {
  Write-Host "Download the final checkpoint, then follow README_LAUNCH.md 'AFTER THE RUN':" -ForegroundColor Green
  $finalKey = "$RUN_PREFIX/checkpoints/iter_{0:d6}.nlcfr" -f $final
  Write-Host "  aws s3 cp s3://$Bucket/$finalKey runs\native_cfr\$Run.nlcfr --region $Region"
  Write-Host "  (legacy runs without .ok markers keep the old pointer: s3://$Bucket/$LATEST)"
}
