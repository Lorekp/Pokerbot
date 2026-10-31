# Autonomous babysitter for the BLUEPRINT-V2 river-OCHS retrain (bp2_riverochs_1).
#
# Fork of the proven scripts/blueprint_run_babysitter.ps1 (same resume-from-S3, relaunch-storm guards,
# spot-quota guard, spotpick auto-pick, convergence stop, on-box dead-man cost caps) with the two
# blueprint-v2 changes:
#   * -RiverOchs <table>  : passes --river-ochs to the launcher so the trainer keys the RIVER with the
#     OCHS nearest-centroid bucket (design: artifacts/blueprint2_design.md, Change A).
#   * -ConvMinIters 40000 : the convergence stop may NOT fire before this many iters (Change B -- the
#     bp_emd_k160_1 run stopped prematurely at 18k because the aggregate strategy-delta plateaus as soon
#     as the preflop mass converges, while postflop rows are still undertrained).
#
# 2026-07-06 reclaim-latency overhaul (minimise PC-side dead time in the reclaim->restart loop):
#   1. INSTANT RECLAIM DETECTION: each poll also fetches <run-prefix>/reclaim_notice.json. When its
#      instance_id matches the currently-tracked box we immediately enter FAST-POLL mode (30s intervals
#      instead of $PollSec) so the quota-drain guard clears and a relaunch fires with minimal lag.
#      Staleness guard = instance_id match; the notice object is never deleted by this script.
#   2. PROGRESS-GATED RELAUNCH GAP: if the box that just died banked at least one NEW marker (iters
#      advanced since its launch), the gap before relaunch is reduced to 60s. The full $MinRelaunchGapSec
#      (600s) is kept only for zero-progress deaths -- the same boxes the churn-abort guard targets.
#   3. DEFAULT $PollSec 180 -> 90: S3 list + describe-instances at 90s cadence is negligible cost.
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
  [switch]$PresumKey,                 # blueprint-v4 arm-6: pass --presum-key (POKERBOT_PRESUM_KEY=1; postflop
                                      #   preflop-summary key). REFINES fullhist -- use WITH -FullHistKey and
                                      #   -Mrr. Serve/eval MUST use the same env.
  [string]$PreKeep = "",              # arm-3: pass --pre-keep (POKERBOT_PRE_KEEP=<val>; Pluribus-grade
                                      #   preflop FIRST-raise/open bet-bucket keep list, e.g.
                                      #   "2,3,4,5,6"). "" = omit = legacy {4,5,6} = byte-identical
                                      #   bootstrap (amendment sec 24). Serve/eval MUST set
                                      #   POKERBOT_PRE_KEEP to the same value.
  [string]$PreRrKeep = "",            # arm-3: pass --pre-rr-keep (POKERBOT_PRE_RR_KEEP=<val>; preflop
                                      #   3-bet+ re-raise bet-bucket keep list). "" = omit = legacy
                                      #   re-raise menu = byte-identical bootstrap. Serve/eval MUST set
                                      #   POKERBOT_PRE_RR_KEEP to the same value.
  [string]$FlopRrKeep = "",           # arm-3: pass --flop-rr-keep (POKERBOT_FLOP_RR_KEEP=<val>; flop
                                      #   re-raise keep list under M-RR -- which buckets the M-RR
                                      #   re-raise thinning retains on the FLOP; turn/river keep the
                                      #   original thinning). "" = omit = original M-RR on all streets.
                                      #   Serve/eval MUST set POKERBOT_FLOP_RR_KEEP to the same value.
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
  [switch]$DeltaCkpt,                 # --delta-ckpt (R2 delta checkpoints: dirty-row NLDLT001 sidecars
                                      #   between full bases at gate iters; resume-identical, spot lever)
  [int]$CkptSecs = 0,                 # --ckpt-secs (HYBRID rolling-ckpt TIME trigger on the delta path;
                                      #   0 = omit = launcher default 600 s)
  [double]$CkptMaxDirtyGb = 0,        # --ckpt-max-dirty-gb (HYBRID rolling-ckpt SIZE trigger, estimated
                                      #   dirty GB; 0 = omit = launcher default 48 GB)
  [int]$CkptMinGapSecs = 0,           # --ckpt-min-gap-secs (HYBRID rolling-ckpt anti-thrash FLOOR, min
                                      #   seconds between rolling checkpoints; 0 = omit = launcher
                                      #   default 300 s)
  # --- box / market ---
  [string]$InstanceType = "",          # "" = AUTO-PICK cheapest spot-available type via spotpick
  [int]$Vcpus = 128,
  [string]$Arch = "x86_64",
  [switch]$ReuseCode,                 # pass --reuse-code on EVERY launch incl. #1 (2026-07-06 incident:
                                      #   an account-side pipeline reacts to code-tarball uploads and
                                      #   plants an untrusted .so in the build cache; no upload = no
                                      #   trigger. Requires the S3 tarball to already match the tree.)
  [switch]$NoCacheFetch,              # pass --no-cache-fetch on EVERY launch (2026-07-06 incident: the
                                      #   account-side pipeline plants an untrusted .so under the correct
                                      #   cache key AFTER each code-tarball upload, and the on-box cache
                                      #   verify is self-referential, so every box must BUILD from the
                                      #   verified tarball -- never fetch the cache. Pairs with -ReuseCode.)
  [string]$CrateSrc = "tools/battery_native",  # the LIVE crate carries river_ochs.rs (the frozen copy at
                                               # runs/bp_crate_frozen does NOT); freeze a new copy first if
                                               # concurrent crate edits are expected during the run
  [switch]$Spotpick,
  [switch]$OnDemand,
  # --- TWO-PHASE MIGRATION (2026-07-07): run the FAST 128-vCPU box until RAM headroom is consumed,
  #     then switch to the big-RAM X box. Enabled iff -Phase1InstanceType is set; otherwise single-
  #     phase (uses -InstanceType, byte-identical control flow). The on-box driver plants
  #     <run>/migrate_request.json when its RAM/row watermark fires; this babysitter, in phase 1,
  #     sees the flag, writes a DURABLE <run>/migrated_to_phase2.json marker to S3, switches
  #     $InstanceType to -Phase2InstanceType, deletes the flag (re-arm), and relaunches (resume from
  #     S3). The phase is re-derived from the S3 marker on every babysitter restart. ---
  [string]$Phase1InstanceType = "",   # FAST box (e.g. r6a.32xlarge 128vCPU/1TB). "" = single-phase.
  [string]$Phase2InstanceType = "",   # big-RAM X box (e.g. x2iedn.16xlarge 64vCPU/2TB) used post-migrate
  [string]$MigrateMinAvailGb = "",    # --migrate-min-avail-gb on PHASE 1 only (RAM headroom watermark, GiB)
  [string]$MigrateMaxRows = "",       # --migrate-max-rows on PHASE 1 only (deterministic backstop rows)
  [string]$Phase1ExcludeFamilies = "",# SPOTPICK_EXCLUDE_FAMILIES for phase 1 (MUST exclude X families so
                                      #   the 128-vCPU fallback chain can't eat the X quota reserved for
                                      #   phase 2). "" = leave the caller's env untouched. (Phase 2 is
                                      #   single-type -- no exclude needed, there is no 2TB alternate.)
  # --- storm + wall guards (verbatim from the proven babysitter) ---
  [int]$MaxLaunch = 12,
  [int]$MaxNoCkptRelaunches = 3,      # CHURN-ABORT (2026-07-05, user directive after 5 boxes burned
                                      #   with zero ckpts): abort if this many launches IN A ROW die
                                      #   without a single NEW checkpoint landing in S3 (marker-based
                                      #   iters). Every further relaunch in a capacity storm is burn.
  [int]$MinRelaunchGapSec = 600,
  [int]$BootWaitSec = 300,
  [int]$PollSec = 90,                 # 2026-07-06: reduced from 180; S3+describe at 90s is negligible
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
# 2026-07-06: fast-poll mode flag (set when reclaim notice detected for tracked box)
$fastPollActive = $false

# --- TWO-PHASE MIGRATION state ------------------------------------------------------------------
$twoPhase          = [bool]$Phase1InstanceType
$MIGRATE_FLAG_KEY  = "$RUN_PREFIX/migrate_request.json"       # on-box driver plants this at its watermark
$PHASE2_MARKER_KEY = "$RUN_PREFIX/migrated_to_phase2.json"    # DURABLE truth: run has moved to the X box
$phase             = 0   # 0 = single-phase; 1 = fast box; 2 = migrated to X box

# Small S3 object helpers (head/put/delete).
# Test-S3Object returns a STATE STRING "present" / "absent" / "unknown" (NOT a bare bool):
#   * "present" -- head-object succeeded (exit 0).
#   * "absent"  -- head-object failed with a GENUINE 404 / "Not Found" / NoSuchKey (the object
#                  really does not exist).
#   * "unknown" -- an AMBIGUOUS non-zero (throttle, expired credential, DNS, timeout, ...). The
#                  old code mapped ALL non-zero to "absent", so a transient blip at (re)start
#                  mis-derived PHASE 1 and launched a 1TB box against a >1TB post-migration table
#                  -> OOM loop + a false "spot capacity storm" abort. We now RETRY a few times with
#                  a short backoff, and only if it STILL cannot be resolved do we return "unknown"
#                  (fail-closed -- see the phase-derivation block).
function Test-S3Object([string]$key) {
  $attempts = 3
  for ($i = 1; $i -le $attempts; $i++) {
    $err = & aws s3api head-object --bucket $Bucket --key $key --region $Region 2>&1
    $code = $LASTEXITCODE
    if ($code -eq 0) { return "present" }
    $errTxt = ($err | Out-String)
    # A genuine "the object does not exist" answer -- definitively ABSENT. The AWS CLI reports a
    # missing key as a 404 with "Not Found" (head-object has no body so no NoSuchKey XML, but the
    # SDK error text still carries these tokens); match all three forms defensively.
    if ($errTxt -match '(?i)\b404\b|Not\s*Found|NoSuchKey') { return "absent" }
    # Ambiguous non-zero (throttle / expired creds / DNS / timeout). Retry with a short backoff.
    if ($i -lt $attempts) { Start-Sleep -Seconds (2 * $i) }
  }
  return "unknown"
}
function Write-S3Marker([string]$key, [string]$json) {
  $tmp = Join-Path $env:TEMP ("bp2_marker_{0}.json" -f ([guid]::NewGuid().ToString('N')))
  Set-Content -Path $tmp -Value $json -Encoding ascii
  aws s3 cp $tmp "s3://$Bucket/$key" --region $Region 2>$null | Out-Null
  $ok = ($LASTEXITCODE -eq 0)
  Remove-Item $tmp -ErrorAction SilentlyContinue
  return $ok
}
function Remove-S3Object([string]$key) {
  aws s3api delete-object --bucket $Bucket --key $key --region $Region 2>$null | Out-Null
}

if ($twoPhase) {
  # Derive the phase from the DURABLE S3 marker so a babysitter restart resumes the right box (never
  # relaunches a 1TB box that would instantly re-migrate). Marker present => already on the X box.
  $markerState = Test-S3Object $PHASE2_MARKER_KEY
  # FAIL CLOSED on an ambiguous head-object (throttle / expired creds / DNS at (re)start). We choose
  # to default to PHASE 2 rather than abort: the 2TB X box safely resumes ANY table (pre- OR
  # post-migration), so treating "unknown" as phase 2 can NEVER OOM. The only cost of a wrong guess
  # here is running the slower/pricier X box for a pre-migration table -- a benign cost overrun, not a
  # data-loss or crash. The DANGEROUS mistake -- launching a 1TB phase-1 box against a >1TB
  # post-migration table (guaranteed OOM loop + false capacity-storm abort) -- is exactly what the old
  # "any non-zero => absent" behaviour did, and is what this fail-closed default prevents.
  if ($markerState -eq "unknown") {
    Write-Host "TWO-PHASE: could not determine phase from $PHASE2_MARKER_KEY (transient S3/credential error after retries) -> FAILING CLOSED to PHASE 2 (the 2TB X box safely resumes any table; a 1TB phase-1 box could OOM a post-migration table)." -ForegroundColor Yellow
    $phase = 2
  } elseif ($markerState -eq "present") {
    $phase = 2
  } else {
    $phase = 1
  }
  if ($phase -eq 2) {
    $InstanceType = $Phase2InstanceType
    # Phase 2 is SINGLE-TYPE (no spotpick fallback): the only 2TB box at 64-vCPU X quota is
    # x2iedn.16xlarge; any same-vCPU fallback would be a <2TB box that OOMs at the row-heavy tail.
    $Spotpick = $false
    Write-Host "TWO-PHASE: resuming in PHASE 2 on $InstanceType (single-type, no fallback)" -ForegroundColor Cyan
  } else {
    $InstanceType = $Phase1InstanceType
    # Phase 1 keeps spotpick for capacity resilience across the 1TB r-family (the exclude list keeps
    # the 128-vCPU X boxes OUT so the fallback chain can't eat the X quota reserved for phase 2).
    $Spotpick = $true
    Write-Host "TWO-PHASE: PHASE 1 on $InstanceType (fast, spotpick fallback); migrate watermark avail<${MigrateMinAvailGb}GiB / rows>=$MigrateMaxRows -> switch to $Phase2InstanceType" -ForegroundColor Cyan
  }
}

# Set SPOTPICK_EXCLUDE_FAMILIES for the CURRENT phase (only when a per-phase list was given; else the
# caller's env is left untouched -- backward compatible with the single-phase wrappers).
function Set-PhaseExclude {
  if (-not $twoPhase) { return }
  # Only phase 1 uses the spotpick fallback chain; phase 2 is single-type (no exclude needed).
  if ($phase -eq 1 -and $Phase1ExcludeFamilies) { $env:SPOTPICK_EXCLUDE_FAMILIES = $Phase1ExcludeFamilies }
}
Set-PhaseExclude

# AUTO-PICK cheapest spot-available type via the shared spotpick lib when -InstanceType was not given.
# SKIPPED in two-phase mode (the phase owns the instance type).
if (-not $twoPhase -and -not $InstanceType) {
  $SpotpickPy = $env:SPOTPICK_SCRIPT
  $picked = ""
  if ($SpotpickPy -and (Test-Path -LiteralPath $SpotpickPy)) {
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
      if ($k -match "iter_(\d+)\.(nlcfr|nldlt)\.ok$") {
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
# 2026-07-06: check whether S3 holds a reclaim notice for the currently-tracked box.
# Returns the instance_id from the notice if it matches $trackedId, else "".
# Uses s3 cp to a temp file (single small JSON object; head-object doesn't give body).
# Staleness guard: only returns a match when notice instance_id == $trackedId.
# This script never deletes the notice (a dying box may rewrite it).
function Get-ReclaimNoticeId([string]$trackedId) {
  if (-not $trackedId) { return "" }
  $noticeKey = "$RUN_PREFIX/reclaim_notice.json"
  $tmp = Join-Path $env:TEMP "bp2_reclaim_notice.json"
  aws s3 cp "s3://$Bucket/$noticeKey" $tmp --region $Region 2>$null | Out-Null
  if (-not (Test-Path $tmp)) { return "" }
  try {
    $raw = Get-Content $tmp -Raw -ErrorAction Stop
    if ($raw -match '"instance_id"\s*:\s*"(i-[0-9a-f]+)"') {
      $noticeId = $Matches[1]
      if ($noticeId -eq $trackedId) { return $noticeId }
    }
  } catch { }
  finally { Remove-Item $tmp -ErrorAction SilentlyContinue }
  return ""
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
  if ($PresumKey)  { $argsList += "--presum-key" }
  if ($PreKeep)    { $argsList += @("--pre-keep", $PreKeep) }
  if ($PreRrKeep)  { $argsList += @("--pre-rr-keep", $PreRrKeep) }
  if ($FlopRrKeep) { $argsList += @("--flop-rr-keep", $FlopRrKeep) }
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
  # PHASE-MIGRATION watermark: pass the migrate triggers ONLY in phase 1 (the fast box). Omitting them
  # in phase 2 leaves MIGRATE_* at 0 on the X box, so it never re-migrates on resume.
  if ($twoPhase -and $phase -eq 1) {
    if ($MigrateMinAvailGb) { $argsList += @("--migrate-min-avail-gb", $MigrateMinAvailGb) }
    if ($MigrateMaxRows)    { $argsList += @("--migrate-max-rows", $MigrateMaxRows) }
  }
  if ($DeltaCkpt)  { $argsList += "--delta-ckpt" }
  if ($CkptSecs -gt 0)       { $argsList += @("--ckpt-secs", $CkptSecs) }
  if ($CkptMaxDirtyGb -gt 0) { $argsList += @("--ckpt-max-dirty-gb", $CkptMaxDirtyGb) }
  if ($CkptMinGapSecs -gt 0) { $argsList += @("--ckpt-min-gap-secs", $CkptMinGapSecs) }
  if ($Spotpick)   { $argsList += "--spotpick" }
  if ($OnDemand)   { $argsList += "--on-demand" }
  if ($CrateSrc)   { $argsList += @("--crate-src", $CrateSrc) }
  if ($ReuseCode -or $launches -ge 1) { $argsList += "--reuse-code" }
  if ($NoCacheFetch) { $argsList += "--no-cache-fetch" }
  Write-Host "  launching: python $($argsList -join ' ')"
  $out = (& python @argsList 2>&1)
  $out | ForEach-Object { Write-Host "    $_" }
  $txt = ($out | Out-String)
  if ($txt -match "(?i)launched\s+(i-[0-9a-f]+)") { return $Matches[1] }
  return ""
}

$capH = [int]($MAXSEC / 3600)
$leverTxt = (@(if ($PreflopHistKey) { "preflop_hist_key" }; if ($FullHistKey) { "fullhist_key" }; if ($Mrr) { "mrr" }; if ($PresumKey) { "presum_key" }; if ($WorkSteal) { "work_steal" }; if ($Dcfr) { "dcfr" }; if ($V3Lcfr) { "v3_lcfr(D=$V3LcfrD,W=$V3LcfrW,C=$V3PruneC)" }; if ($V3SignedRegrets) { "v3_signed_regrets(C=$V3PruneC,start=$V3PruneStart)" }; if ($V3SnapshotAvg) { "v3_snapshot_avg" }) -join "+"); if (-not $leverTxt) { $leverTxt = "NONE (baseline)" }
Write-Host "=== blueprint-v2 babysitter: $Run ===" -ForegroundColor Cyan
Write-Host "    target=$Iters it  n_deals=$NDeals  flop/turn=EMD K$NumBuckets  river=OCHS ($([System.IO.Path]::GetFileName($RiverOchs)))"
Write-Host "    LEVERS=$leverTxt" -ForegroundColor Green
Write-Host "    caps: ${capH}h wall / $MaxLaunch launches / >=${MinRelaunchGapSec}s between launches / churn-abort after $MaxNoCkptRelaunches no-ckpt relaunches"
if ($twoPhase) {
  Write-Host "    TWO-PHASE: P1=$Phase1InstanceType (fast) --[migrate avail<${MigrateMinAvailGb}GiB / rows>=$MigrateMaxRows]--> P2=$Phase2InstanceType (big-RAM X);  now in PHASE $phase" -ForegroundColor Cyan
} else {
  Write-Host "    single-phase: InstanceType=$InstanceType" -ForegroundColor Cyan
}
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
  # 2b) CONVERGENCE STOP with the blueprint-v2 MIN-ITERS FLOOR: the bp_emd_k160_1 lesson -- the aggregate
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
  # 3) a box is live and not done -> wait (with reclaim fast-poll detection)
  if ($live) {
    # 2026-07-06: check for on-box reclaim notice for the currently tracked box.
    # If found, switch to FAST-POLL (30s) so we clear the quota-drain guard ASAP.
    $reclaimId = Get-ReclaimNoticeId $live
    if ($reclaimId) {
      if (-not $fastPollActive) {
        Write-Host "  RECLAIM NOTICE detected for $live -> switching to FAST-POLL (30s)" -ForegroundColor Yellow
        $fastPollActive = $true
      }
      Start-Sleep -Seconds 30
    } else {
      if ($fastPollActive) {
        Write-Host "  fast-poll: no reclaim notice for current box; returning to normal poll" -ForegroundColor Gray
        $fastPollActive = $false
      }
      Start-Sleep -Seconds $PollSec
    }
    continue
  }

  # Reset fast-poll flag once box is gone
  $fastPollActive = $false

  # 3b) EXACT-QUOTA GUARD: a previous box still terminating holds the spot vCPU quota; wait before relaunch.
  if ($draining) {
    Write-Host "  previous box $draining still terminating (holds quota); waiting ${PollSec}s before relaunch"
    Start-Sleep -Seconds $PollSec
    continue
  }

  # 4) NO live box and target NOT reached -> preemption (or never-launched). Relaunch, with storm guards.
  if ($launches -ge $MaxLaunch) { Write-Host "STOP: max launches ($MaxLaunch) reached (storm guard)."; break }
  # A pending phase-1->2 migration must NOT count against the churn streak: a box that fired its RAM
  # watermark and planted migrate_request.json did its job (it deliberately handed off), so its death
  # with "no new checkpoint since launch" is a HAND-OFF, not a capacity-storm churn death. Resolve the
  # flag ONCE per cycle (tri-state present/absent/unknown) and reuse it in both 4a and the migrate
  # handler below. On "unknown" we do NOT treat it as a pending migration (fail closed to the normal
  # churn accounting -- a spurious skip could mask a real storm), but the head-object is retried inside
  # Test-S3Object so a transient blip rarely reaches here as "unknown".
  $migratePending = $false
  if ($twoPhase -and $phase -eq 1) {
    $migratePending = ((Test-S3Object $MIGRATE_FLAG_KEY) -eq "present")
  }
  # 4a) CHURN-ABORT: the box that just died saved NO new checkpoint since its own launch -> streak++.
  # A streak of $MaxNoCkptRelaunches means capacity is being reclaimed faster than the checkpoint
  # cadence; every further relaunch is pure burn. Abort loudly instead (S3 state is preserved).
  # NOTE: emergency-flush delta markers from the reclaim notice handler count as progress (correct --
  # a box that banked its dying delta DID make progress, so $done > $doneAtLastLaunch and streak resets).
  if ($launches -gt 0) {
    if ($migratePending) {
      # A pending migration is not a churn death -- mirror the success-path reset so a hand-off box
      # can never contribute to (or trip) the capacity-storm abort.
      $noCkptRelaunches = 0
      Write-Host "  churn guard: migrate_request.json pending (phase-1 RAM watermark fired) -- treating as a hand-off, NOT a churn death; no-progress streak reset to 0" -ForegroundColor Gray
    } elseif ($done -le $doneAtLastLaunch) {
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
  # 4b) 2026-07-06 PROGRESS-GATED RELAUNCH GAP:
  # If the box that just died made measurable progress (iters advanced), use a 60s gap (it was a clean
  # preemption and the box was productive -- the churn-abort streak also reset above). Only apply the
  # full $MinRelaunchGapSec (600s) for zero-progress deaths, which are the actual churn-abort cases.
  $effectiveGapSec = $MinRelaunchGapSec
  if ($launches -gt 0 -and $done -gt $doneAtLastLaunch) {
    $effectiveGapSec = 60
  }
  $sinceLast = ((Get-Date) - $lastLaunch).TotalSeconds
  if ($launches -gt 0 -and $sinceLast -lt $effectiveGapSec) {
    $wait = [int]($effectiveGapSec - $sinceLast)
    $gapNote = if ($effectiveGapSec -eq 60) { "progress-gated 60s gap" } else { "${MinRelaunchGapSec}s gap (no-progress)" }
    Write-Host "  relaunch gap guard ($gapNote): only ${sinceLast}s since last launch; waiting ${wait}s"
    Start-Sleep -Seconds $wait
    continue
  }
  # 2026-07-07 TERMINAL-STOP FLAG: a ROW_BUDGET_STOP is indistinguishable from a dry spot death
  # here (box resumes, instantly self-stops, banks nothing), so the overnight loop relaunched
  # at-budget boxes for ~8 cycles. The driver now plants <run>/row_budget_stop.json; HARD-STOP on
  # it -- a config change (raise -RowBudget) + deleting the flag object is the only way forward.
  # 2026-07-07 GATE-KILL STOP FLAG: the pre-launch monitor (or an operator) plants
  # <run>/gate_kill_stop.json when a pre-registered kill-gate (K0/K1a/K1) returns KILL. HARD-STOP on it
  # exactly like the row-budget flag -- a KILLed arm must not be relaunched/funded further. Re-arm by
  # DELETING the flag object (only after a design change; a KILL is normally terminal per sec 24.3).
  $gateStop = (aws s3api head-object --bucket $Bucket --key "$RUN_PREFIX/gate_kill_stop.json" --region $Region 2>$null)
  if ($gateStop) {
    Write-Host "HARD-STOP: $RUN_PREFIX/gate_kill_stop.json present -- a pre-registered KILL-GATE failed. The arm is KILLED (limp-trap / shape pathology per amendment sec 24.3); NOT relaunching. DELETE the flag only after a design change." -ForegroundColor Red
    break
  }
  $stopFlag = (aws s3api head-object --bucket $Bucket --key "$RUN_PREFIX/row_budget_stop.json" --region $Region 2>$null)
  if ($stopFlag) {
    Write-Host "HARD-STOP: $RUN_PREFIX/row_budget_stop.json present -- the run hit its ROW_BUDGET (G8). Raise -RowBudget (and the RAM floor if needed), DELETE the flag object, then relaunch. NOT relaunching." -ForegroundColor Red
    break
  }
  # 2026-07-07 PHASE-MIGRATION: in phase 1, a migrate_request.json means the fast box consumed its RAM
  # headroom. Switch to the big-RAM X box: write the DURABLE phase-2 marker FIRST (so a crash mid-switch
  # still resumes on X), THEN delete the flag (re-arm). Idempotent -- a re-entered switch just re-writes
  # the marker. The migrating box made progress (churn streak already reset above). $migratePending was
  # resolved above (tri-state -> present only).
  if ($migratePending) {
    Write-Host "PHASE-MIGRATION: $RUN_PREFIX/migrate_request.json present at $done iters -- switching PHASE 1 ($InstanceType) -> PHASE 2 ($Phase2InstanceType, big-RAM X box)." -ForegroundColor Magenta
    if (Write-S3Marker $PHASE2_MARKER_KEY ("{{`"from`":`"{0}`",`"to`":`"{1}`",`"iters`":{2}}}" -f $InstanceType, $Phase2InstanceType, $done)) {
      $phase = 2
      $InstanceType = $Phase2InstanceType
      $Spotpick = $false   # phase 2 is single-type (no <2TB fallback that would OOM the tail)
      Set-PhaseExclude
      Remove-S3Object $MIGRATE_FLAG_KEY
      $noCkptRelaunches = 0   # a phase switch is not a churn death
      Write-Host "  now in PHASE 2 on $InstanceType (single-type); migrate triggers OFF; resume from S3." -ForegroundColor Magenta
    } else {
      # MARKER-WRITE FAILURE: do NOT fall through to a phase-1 relaunch -- that box would instantly
      # re-migrate with no progress, churning until a false capacity-storm abort. RETRY the switch
      # (no box spawned) after a short backoff; the migrate flag is still present so next iteration
      # re-enters this handler. This cycle also did not spawn a box, so the churn streak is untouched.
      Write-Host "  WARNING: failed to write $PHASE2_MARKER_KEY (transient S3 error) -- NOT relaunching phase-1 (would re-migrate with no progress). Retrying the switch in ${PollSec}s." -ForegroundColor Yellow
      Start-Sleep -Seconds $PollSec
      continue
    }
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
  # This echo MIRRORS Invoke-Launch flag-for-flag (accurate to the CURRENT phase) so a hand-copied
  # resume never silently drops a correctness knob -- omitting --pre-*-keep/--flop-rr-keep would run the
  # legacy {4,5,6} menu (train/serve abstraction skew), omitting --row-budget would revert to the 400M
  # default (an instant terminal ROW_BUDGET stop), and omitting --fullhist-key/--mrr would train a
  # totally different key space. If you edit Invoke-Launch, edit this block too.
  $resume = "python scripts\launch_native_cfr_training_spot.py --run-subdir $Run --iters $Iters --n-deals $NDeals --num-buckets $NumBuckets --checkpoint-every $CheckpointEvery --max-hours $($MaxHours + 1) --instance-type $InstanceType --arch $Arch"
  if ($GateEvery -gt 0) { $resume += " --gate-every $GateEvery" }
  if ($CkptVolumeGb -gt 0) { $resume += " --ckpt-volume-gb $CkptVolumeGb" }
  if ($EmdFrozen) { $resume += " --emd-frozen $EmdFrozen" }
  if ($RiverOchs) { $resume += " --river-ochs $RiverOchs" }
  if ($DepthEdges) { $resume += " --depth-edges $DepthEdges" }
  if ($PreflopHistKey) { $resume += " --preflop-hist-key" }
  if ($FullHistKey) { $resume += " --fullhist-key" }
  if ($Mrr) { $resume += " --mrr" }
  if ($PresumKey) { $resume += " --presum-key" }
  if ($PreKeep) { $resume += " --pre-keep $PreKeep" }
  if ($PreRrKeep) { $resume += " --pre-rr-keep $PreRrKeep" }
  if ($FlopRrKeep) { $resume += " --flop-rr-keep $FlopRrKeep" }
  if ($WorkSteal) { $resume += " --work-steal" }
  if ($Dcfr) { $resume += " --dcfr --dcfr-prune-threshold $DcfrPruneThreshold --dcfr-explore-frac $DcfrExploreFrac" }
  if ($V3Lcfr) { $resume += " --v3-lcfr --v3-lcfr-d $V3LcfrD --v3-lcfr-w $V3LcfrW --v3-prune-explore $V3PruneExplore"
    if ($V3PruneC) { $resume += " --v3-prune-c $V3PruneC" }
    if ($V3PruneStart) { $resume += " --v3-prune-start $V3PruneStart" }
    if ($V3RegretFloor) { $resume += " --v3-regret-floor $V3RegretFloor" } }
  if ($V3SignedRegrets) { $resume += " --v3-signed-regrets --v3-prune-explore $V3PruneExplore"
    if ($V3PruneC) { $resume += " --v3-prune-c $V3PruneC" }
    if ($V3PruneStart) { $resume += " --v3-prune-start $V3PruneStart" }
    if ($V3RegretFloor) { $resume += " --v3-regret-floor $V3RegretFloor" } }
  if ($V3SnapshotAvg) { $resume += " --v3-snapshot-avg"
    if ($V3SnapStart) { $resume += " --v3-snap-start $V3SnapStart" } }
  if ($V3Lcfr -or $V3SignedRegrets -or $V3SnapshotAvg) { $resume += " --v3-cov-hands $V3CovHands"
    if ($V3Regstats) { $resume += " --v3-regstats" } }
  if ($RowBudget -gt 0) { $resume += " --row-budget $RowBudget" }
  # Migrate triggers are PHASE-1 ONLY (like Invoke-Launch): on the phase-2 X box they stay OFF so it
  # never re-migrates on resume.
  if ($twoPhase -and $phase -eq 1) {
    if ($MigrateMinAvailGb) { $resume += " --migrate-min-avail-gb $MigrateMinAvailGb" }
    if ($MigrateMaxRows) { $resume += " --migrate-max-rows $MigrateMaxRows" }
  }
  if ($DeltaCkpt) { $resume += " --delta-ckpt" }
  if ($CkptSecs -gt 0) { $resume += " --ckpt-secs $CkptSecs" }
  if ($CkptMaxDirtyGb -gt 0) { $resume += " --ckpt-max-dirty-gb $CkptMaxDirtyGb" }
  if ($CkptMinGapSecs -gt 0) { $resume += " --ckpt-min-gap-secs $CkptMinGapSecs" }
  if ($Spotpick) { $resume += " --spotpick" }
  if ($OnDemand) { $resume += " --on-demand" }
  if ($CrateSrc) { $resume += " --crate-src $CrateSrc" }
  $resume += " --reuse-code"   # a manual resume always reuses the existing S3 tarball
  if ($NoCacheFetch) { $resume += " --no-cache-fetch" }
  Write-Host "  $resume"
}
if ($final -ge 0 -and ($convergedStop -or $final -ge $Iters)) {
  Write-Host "Download the final checkpoint, then follow README_LAUNCH.md 'AFTER THE RUN':" -ForegroundColor Green
  $finalKey = "$RUN_PREFIX/checkpoints/iter_{0:d6}.nlcfr" -f $final
  Write-Host "  aws s3 cp s3://$Bucket/$finalKey runs\native_cfr\$Run.nlcfr --region $Region"
  Write-Host "  (legacy runs without .ok markers keep the old pointer: s3://$Bucket/$LATEST)"
}
