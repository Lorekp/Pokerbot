# Autonomous babysitter for the 5-PHASE EMD BLUEPRINT PIPELINE: ONE launch -> spot box(es) that, across
# self-healing cloud-init boots, run BUILD -> PRECOMPUTE -> TRAIN(->converge) -> SHIP -> EVAL and hand back
# a VERIFIED-BETTER blueprint (staged, not auto-activated unless told to).
#
#   PHASE 1 BUILD/PRECOMPUTE : build the EMD crate; ensure the frozen (hole,board)->bucket table (S3 cache
#                              by table-id: download if present, else precompute n_runouts=200 + upload).
#   PHASE 2 TRAIN            : train the EMD blueprint to convergence (the blueprint_convergence_stop plateau).
#   PHASE 4 SHIP             : convert the converged .nlcfr -> serve snapshot, pass the serve-skew gate
#                              (ship_blueprint.py --emd-frozen), STAGE it (versioned snapshot to S3 +
#                              checkpoints/serve_blueprints) -- does NOT flip the live serve pointer.
#   PHASE 5 EVAL             : harsh per-seat LBR of the NEW EMD blueprint AND iter62000 on the SAME seeds
#                              (CRN) + the always-fold anchor; emit a verdict JSON {emd_better, anchor_ok,
#                              per-seat blind deltas, ...} to S3.
#
# RESUME-AT-FIRST-INCOMPLETE (spot-resilient, single launch): every phase writes an S3 completion MARKER
# under {run}/markers/ (table_cached / CONVERGED / SHIPPED / EVALED). On a spot-kill relaunch the box's
# cloud-init re-runs from the top but SKIPS any phase whose marker exists, and THIS babysitter detects the
# markers too, so it relaunches into the FIRST INCOMPLETE phase. A kill during ship/eval RESUMES at ship/
# eval and NEVER re-trains (the converged checkpoint + CONVERGED marker already exist). When the babysitter
# itself decides convergence (the strategy-delta plateau stop), it WRITES the CONVERGED marker before
# relaunching, so the resumed box jumps straight to SHIP.
#
# V2 ADDITIONS vs. emd_pipeline_babysitter.ps1:
#   - AUTO-ACTIVATE GATE: when -AutoActivate is set, instead of the legacy two-boolean short-circuit
#     the v2 gate enforces a PLURIBUS-GRADE checklist (K == K_MAX, anchor in band, both blinds >= min gain,
#     blind-mean >= min, global regression <= max, sha256 stamped, sign-correct verified). Any failure
#     blocks the activation and prints a per-condition reason. The legacy emd_better / anchor_ok short-circuit
#     is intentionally NOT used because the k500 incident (sign-flip + SB regression) would have falsely
#     passed it.
#   - Per-seat delta rendering is delegated to scripts/_emd_verdict_sign.py render -- single source of truth.
#   - SHA256 + K-cap + sign-correct status lines on every refresh; failure to verify shows a red WARNING
#     before the gate decision.
#
# ACTIVATION (default OFF -- safe): by default the pipeline STAGES + EVALS + reports the per-seat verdict and
# STOPS, leaving the user to activate manually after seeing the per-seat blind LBR.
#
# SINGLE LAUNCH (this is the one command the user runs):
#   powershell -ExecutionPolicy Bypass -File scripts\emd_pipeline_babysitter_v2.ps1 -Run run_emd_bp1
#
# Modeled on native_cfr_babysitter.ps1 (spotpick auto-pick, exact-quota relaunch/drain guard, storm
# guards, convergence-stop) -- same machinery + the SHIP/EVAL lifecycle + marker-based resume.
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252; non-ASCII bytes break strings).
param(
  [string]$Run = "",                  # S3 run-subdir; REQUIRED for resume-across-preemption (auto-gen warns)
  [int]$Iters = 250000,               # --iters CONVERGENCE BACKSTOP (the convergence stop ends it earlier)
  [int]$NDeals = 200,                 # --n-deals
  [int]$NumBuckets = 200,             # --num-buckets (MUST equal the centroids' K = the frozen table's K)
  [string]$Centroids = "data/abstraction/emd_buckets_200.json",  # EMD centroids (the frozen table source)
  [int]$NRunouts = 200,               # production precompute board completions/spot (0 = use centroids')
  [string]$DepthEdges = "",           # --depth-edges ("" = non-depth blueprint)
  [int]$CheckpointEvery = 2000,       # --checkpoint-every (preemption-bounded loss)
  [string]$InstanceType = "",          # "" = AUTO-PICK cheapest spot-available 128-vCPU type via spotpick
  [int]$Vcpus = 128,
  [string]$Arch = "x86_64",
  [string]$CrateSrc = "runs/bp_crate_emd/battery_native",  # ISOLATED EMD crate (uploaded as tools/battery_native)
  [int]$TrainProfileEvery = 0,       # --train-profile-every; 0 disables TRAIN_PROFILE_JSON telemetry
  [switch]$TrainFastMerge,            # --train-fast-merge; statistical-equivalence only, default OFF
  [switch]$Spotpick,                  # capacity-diverse fallback chain (auto-on when auto-picking)
  [switch]$OnDemand,                  # no spot (no preemption, pricier)
  [int]$MaxLaunch = 10,               # total relaunch cap (storm guard)
  [int]$MinRelaunchGapSec = 600,      # min wall seconds between launches (storm guard)
  [int]$QuotaDrainSec = 600,           # accepted for wrapper compatibility; quota drain is polled by state
  [int]$BootWaitSec = 300,            # wait after a launch before polling (boot grace)
  [int]$PollSec = 180,                # state poll interval while a box is live
  [int]$MaxHours = 40,                # wall-clock cap for the babysitter
  [string]$Region = "eu-central-1",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr",
  # --- convergence-based stop (EMD-appropriate threshold; a richer abstraction floors slightly higher) ---
  [switch]$NoConvergeStop,            # disable convergence stop; run to fixed -Iters / -MaxHours only
  [switch]$ConvergeLogOnly,           # compute/print convergence status, but do NOT stop early
  [double]$ConvThreshold = 0.020,     # EMD mean-L1 plateau floor (~0.020-0.022; vs 0.018 for NB=200 made-hand)
  [int]$ConvK = 3,                    # consecutive below-threshold checkpoints required to declare CONVERGED
  [string]$ConvDepthEdges = "",       # stop-helper ctor depth-edges (irrelevant to load); CSV like "50,100"
  # --- PHASE 4 SHIP + PHASE 5 EVAL ---
  [string]$ShipName = "emd_bp1",      # version tag for the staged serve snapshot
  [string]$Iter62000SnapshotS3 = "s3://video-processing-6789/pokerbot/prod6000/native_cfr/bp6max_pluribus/serve/iter62000_snapshot.pkl",
  [int]$EvalSeeds = 12,               # CRN seeds the LBR pools over (both blueprints share them)
  [int]$EvalHands = 2000,             # hands per seed (Python LBR is GIL-bound; lean on the CRN per-seat delta)
  # --- eval-based marginal-improvement stop (K200 marathon) ---
  [switch]$EvalConvergeStop,           # stop when fixed-CRN activation-margin gains become marginal
  [switch]$EvalConvergeLogOnly,        # compute/print eval convergence status, but do NOT stop early
  [int]$EvalConvergeMinIters = 300000, # never eval-stop before this floor (avoid noisy early plateaus)
  [int]$EvalConvergeEveryIters = 50000,# evaluate at most once per this many new training iterations
  [int]$EvalConvergeSeeds = 24,        # cheaper than final eval; baseline arm is cached by the helper
  [int]$EvalConvergeHands = 4000,
  [double]$EvalConvergeMinGain = 3.0,  # bb/100 activation-margin improvement needed to reset marginal streak
  [int]$EvalConvergeK = 3,             # consecutive marginal eval probes required to declare done
  [int]$EvalConvergeMinEvals = 4,      # require enough probes before stop can fire
  [int]$MaxShipEvalLaunch = 6,        # relaunch cap for the SHIP/EVAL lifecycle (separate from -MaxLaunch)
  [switch]$StopAfterShip,              # Step-3-only: stop after staged serve artifact; do NOT enter eval
  # --- ACTIVATION (default OFF; safe -- stage + report only) ---
  [switch]$AutoActivate               # ON = run the v2 pluribus-grade auto-activate gate (see gate block)
)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

# ============================================================================================
# V2 GATE CONSTANTS -- pluribus-grade auto-activate acceptance checklist.
# These are the SPEC's acceptance thresholds for any blueprint the babysitter would flip live.
# Do NOT lower K_MAX below 200 (the K=500 incident showed K=500 overfit the bucketer and produced
# the real -140 SB regression -- the k500 verdict is intentionally INELIGIBLE under this gate).
# ============================================================================================
$KMax         = 200
$ConvThresh   = 0.020
$ConvK        = 3
$AnchorLo     = 110.0
$AnchorHi     = 145.0
$TargetAnchor = 125.0
$SBMin        = 15.0
$BBMin        = 15.0
$BlindMeanMin = 15.0
$GlobalRegMax = 10.0
$NBuckets     = 800

if (-not $Run) {
  $Run = "run_emd_{0}" -f ([int][double]::Parse((Get-Date -UFormat %s)))
  Write-Host "WARNING: no -Run given; generated '$Run'. Relaunch-on-preemption resumes THIS subdir." -ForegroundColor Yellow
}

$RUN_PREFIX = "$Prefix/$Run"
$LATEST     = "$RUN_PREFIX/checkpoints/latest.nlcfr"
$MARK_PREFIX = "$RUN_PREFIX/markers"
$M_TABLE     = "$MARK_PREFIX/table_cached"
$M_CONVERGED = "$MARK_PREFIX/CONVERGED"
$M_SHIPPED   = "$MARK_PREFIX/SHIPPED"
$M_EVALED    = "$MARK_PREFIX/EVALED"
$VERDICT_KEY = "$RUN_PREFIX/eval/verdict.json"
$MAXSEC     = $MaxHours * 3600
$start      = Get-Date
$launches   = 0
$lastLaunch = [datetime]::MinValue
$tracked    = ""
$prevBox    = ""
$convergedStop = $false
$ConvStateFile = Join-Path $env:TEMP ("emd_v2_conv_stop_{0}.json" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
$EvalConvStateFile = Join-Path $env:TEMP ("emd_v2_eval_conv_stop_{0}.json" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
$EvalConvJsonFile = Join-Path $env:TEMP ("emd_v2_eval_conv_stop_{0}_latest.json" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
$EvalConvWorkDir = Join-Path $env:TEMP ("emd_v2_eval_conv_work_{0}" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
$lastConvIters = -1
$lastEvalConvIters = -1

function Test-Marker([string]$key) {
  # True if the S3 marker object exists. Use GetObject-style probing because EC2/local roles may
  # have object read/write without ListBucket, which makes `aws s3 ls` lie for exact marker keys.
  $tmp = Join-Path $env:TEMP ("emd_v2_marker_probe_{0}.txt" -f ([guid]::NewGuid().ToString('N')))
  aws s3 cp "s3://$Bucket/$key" $tmp --region $Region --no-progress 2>$null | Out-Null
  $ok = Test-Path $tmp
  Remove-Item $tmp -ErrorAction SilentlyContinue
  return $ok
}
function Write-Marker([string]$key, [string]$note) {
  # Idempotent marker write (small text object). Used by the babysitter when IT decides convergence.
  $tmp = Join-Path $env:TEMP ("emd_v2_marker_{0}.txt" -f ([guid]::NewGuid().ToString('N')))
  Set-Content -Path $tmp -Value ("{0} {1}" -f (Get-Date -Format o), $note) -Encoding Ascii
  aws s3 cp $tmp "s3://$Bucket/$key" --region $Region --no-progress 2>$null | Out-Null
  Remove-Item $tmp -ErrorAction SilentlyContinue
}
function Get-Verdict {
  # Download + parse the eval verdict JSON. Returns $null if absent/unparseable.
  # The tmp path is preserved (NOT deleted) so the downstream sign-correct printer
  # (scripts/_emd_verdict_sign.py render) can re-read the same JSON instead of
  # the buggy literal 'improvement' formatting.
  $tmp = Join-Path $env:TEMP ("emd_v2_verdict_{0}.json" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
  aws s3 cp "s3://$Bucket/$VERDICT_KEY" $tmp --region $Region --no-progress 2>$null | Out-Null
  if (-not (Test-Path $tmp)) { return $null }
  try { $v = (Get-Content $tmp -Raw | ConvertFrom-Json) } catch { $v = $null }
  return $v
}
function Format-Bytes([int64]$bytes) {
  if ($bytes -lt 0) { return "(none)" }
  if ($bytes -ge 1GB) { return ("{0:N2}GB" -f ($bytes / 1GB)) }
  if ($bytes -ge 1MB) { return ("{0:N1}MB" -f ($bytes / 1MB)) }
  if ($bytes -ge 1KB) { return ("{0:N1}KB" -f ($bytes / 1KB)) }
  return ("{0}B" -f $bytes)
}
function Get-S3Size([string]$key) {
  $out = (aws s3api head-object --bucket $Bucket --key $key --region $Region `
    --query ContentLength --output text 2>$null)
  if ($out -and "$out".Trim() -match '^\d+$') { return [int64]"$out".Trim() }
  return [int64]-1
}
function Get-FrozenSummary {
  $ls = aws s3 ls "s3://$Bucket/$RUN_PREFIX/frozen/" --region $Region 2>$null
  $best = [int64]-1
  if ($ls) {
    foreach ($line in $ls) {
      $cols = ($line -split "\s+")
      if ($cols.Count -ge 3 -and $cols[2] -match '^\d+$') {
        $sz = [int64]$cols[2]
        if ($sz -gt $best) { $best = $sz }
      }
    }
  }
  return $best
}
function Get-ConsoleProbe([string]$id) {
  if (-not $id) { return $null }
  $q = "{Len:length(Output),BuildStart:contains(Output, 'PHASE=build START'),BuildDone:contains(Output, 'PHASE=build DONE'),PreStart:contains(Output, 'PHASE=precompute TABLE_ID'),CacheHit:contains(Output, 'PHASE=precompute CACHE_HIT'),CacheMiss:contains(Output, 'PHASE=precompute CACHE_MISS'),PreDone:contains(Output, 'PHASE=precompute DONE'),TrainSkip:contains(Output, 'PHASE=train SKIP'),TrainStart:contains(Output, 'PHASE=train START'),ShipStart:contains(Output, 'PHASE=ship START'),ShipDownload:contains(Output, 'PHASE=ship DOWNLOAD_CKPT'),ShipRun:contains(Output, 'PHASE=ship RUN_SHIP_BLUEPRINT'),Convert:contains(Output, '[1/3] CONVERT'),Validate:contains(Output, '[2/3] VALIDATE'),Install:contains(Output, '[3/3] INSTALL'),ShipDone:contains(Output, 'PHASE=ship DONE'),EvalStart:contains(Output, 'PHASE=eval START'),EvalEmd:contains(Output, 'PHASE=eval EMD_START'),EvalBase:contains(Output, 'PHASE=eval ITER62000_START'),EvalCombine:contains(Output, 'PHASE=eval COMBINE_START'),EvalDone:contains(Output, 'PHASE=eval DONE'),EvalProgress:contains(Output, 'PROGRESS seed'),EvalHandProgress:contains(Output, 'HAND_PROGRESS'),EvalLogSync:contains(Output, 'PHASE=eval LOG_SYNC'),Traceback:contains(Output, 'Traceback'),S3DownloadFail:contains(Output, 'S3_DOWNLOAD_FAIL'),S3UploadFail:contains(Output, 'S3_UPLOAD_FAIL'),S3PutFail:contains(Output, 'S3_PUT-TEXT_FAIL'),S3Any:contains(Output, 'S3_'),AwsError:contains(Output, 'aws: [ERROR]')}"
  $json = (aws ec2 get-console-output --instance-id $id --region $Region --latest --query $q --output json 2>$null | Out-String)
  if (-not $json.Trim()) { return $null }
  try { return ($json | ConvertFrom-Json) } catch { return $null }
}
function Probe-Has($probe, [string]$name) {
  return ($probe -and $probe.PSObject.Properties[$name] -and [bool]$probe.PSObject.Properties[$name].Value)
}
function Format-Probe($probe) {
  if (-not $probe) { return "(console unavailable)" }
  $parts = @()
  if (Probe-Has $probe "BuildStart") { $parts += "build" }
  if (Probe-Has $probe "BuildDone") { $parts += "build-done" }
  if (Probe-Has $probe "CacheHit") { $parts += "cache-hit" }
  if (Probe-Has $probe "CacheMiss") { $parts += "cache-miss" }
  if (Probe-Has $probe "PreDone") { $parts += "precompute-done" }
  if (Probe-Has $probe "TrainSkip") { $parts += "train-skip" }
  if ((Probe-Has $probe "TrainStart") -and -not (Probe-Has $probe "TrainSkip")) { $parts += "train-start" }
  if (Probe-Has $probe "ShipStart") { $parts += "ship-start" }
  if (Probe-Has $probe "Convert") { $parts += "convert" }
  if (Probe-Has $probe "Validate") { $parts += "validate" }
  if (Probe-Has $probe "Install") { $parts += "install" }
  if (Probe-Has $probe "ShipDone") { $parts += "ship-done" }
  if (Probe-Has $probe "EvalStart") { $parts += "eval-start" }
  if (Probe-Has $probe "EvalEmd") { $parts += "eval-emd" }
  if (Probe-Has $probe "EvalBase") { $parts += "eval-iter62000" }
  if (Probe-Has $probe "EvalCombine") { $parts += "eval-combine" }
  if (Probe-Has $probe "EvalDone") { $parts += "eval-done" }
  if (Probe-Has $probe "EvalHandProgress") { $parts += "eval-hand-progress" }
  if (Probe-Has $probe "EvalLogSync") { $parts += "eval-log-sync" }
  if (Probe-Has $probe "Traceback") { $parts += "TRACEBACK" }
  if ((Probe-Has $probe "S3DownloadFail") -and (Probe-Has $probe "CacheMiss")) { $parts += "cache-download-miss" }
  elseif (Probe-Has $probe "S3DownloadFail") { $parts += "S3-DOWNLOAD-FAIL" }
  if (Probe-Has $probe "S3UploadFail") { $parts += "S3-UPLOAD-FAIL" }
  if (Probe-Has $probe "S3PutFail") { $parts += "S3-MARKER-FAIL" }
  if (Probe-Has $probe "AwsError") { $parts += "AWSCLI-ERROR" }
  if (-not $parts.Count) { $parts += "no phase milestone yet" }
  return ("console_len={0} {1}" -f $probe.Len, ($parts -join " -> "))
}

# AUTO-PICK the cheapest spot-available 128-vCPU box via the shared spotpick lib (same as native_cfr).
if (-not $InstanceType) {
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
  # The trainer's ground truth: parse iters_done out of the latest.nlcfr header (magic "NLCFR001"(8) +
  # iters_done u64 LE(8)). -1 = no checkpoint yet (box still booting / building / precomputing / first ckpt).
  $tmp = Join-Path $env:TEMP "emd_v2_latest_probe.nlcfr"
  aws s3 cp "s3://$Bucket/$LATEST" $tmp --region $Region --no-progress 2>$null | Out-Null
  if (-not (Test-Path $tmp)) { return -1 }
  try {
    $fs = [System.IO.File]::OpenRead($tmp)
    try {
      $buf = New-Object byte[] 16
      if ($fs.Read($buf, 0, 16) -lt 16) { return -1 }
      if ([System.Text.Encoding]::ASCII.GetString($buf, 0, 8) -ne "NLCFR001") { return -1 }
      return [int64][System.BitConverter]::ToUInt64($buf, 8)
    } finally { $fs.Close() }
  } catch { return -1 } finally { Remove-Item $tmp -ErrorAction SilentlyContinue }
}
function Stop-Run {
  $id = Get-LiveBoxId
  if (-not $id -and $tracked) { $st = Get-StateById $tracked; if ($st -eq "running" -or $st -eq "pending") { $id = $tracked } }
  if ($id) { aws ec2 terminate-instances --instance-ids $id --region $Region | Out-Null; Write-Host "terminated: $id" }
}
function Test-Converged {
  # Diff the two latest versioned checkpoints (mean-L1 avg-strategy delta over shared infosets); persist the
  # below-threshold streak so the K-consecutive rule holds across rounds. Abstraction-agnostic (the blob
  # carries its own tables), so it works unchanged for the EMD table. Returns CONVERGED|CONTINUING|UNKNOWN|ERROR.
  $argsList = @(
    "scripts\blueprint_convergence_stop.py",
    "--run-prefix", $RUN_PREFIX, "--bucket", $Bucket, "--region", $Region,
    "--threshold", $ConvThresh, "--k", $ConvK,
    "--state-file", $ConvStateFile, "--num-buckets", $NumBuckets
  )
  if ($ConvDepthEdges) { $argsList += @("--depth-edges"); $argsList += ($ConvDepthEdges -split ",") }
  $out = (& python @argsList 2>&1)
  $txt = ($out | Out-String)
  if ($txt -match "STATUS=([A-Z]+)") { $status = $Matches[1] } else { $status = "ERROR" }
  ($txt -split "`r?`n") | Where-Object { $_ -match "STATUS=" } | ForEach-Object { Write-Host "    conv: $_" }
  return $status
}
function Test-EvalConverged {
  # Run the native CRN activation-margin probe. It caches the iter62000 baseline arm and only evaluates
  # the latest EMD checkpoint after EvalConvergeEveryIters. Returns CONVERGED|CONTINUING|UNKNOWN|ERROR.
  $argsList = @(
    "scripts\emd_eval_convergence_stop.py",
    "--run-prefix", $RUN_PREFIX, "--bucket", $Bucket, "--region", $Region,
    "--num-buckets", $NumBuckets,
    "--iter62000-snapshot-s3", $Iter62000SnapshotS3,
    "--seeds", $EvalConvergeSeeds, "--hands", $EvalConvergeHands,
    "--min-iters", $EvalConvergeMinIters,
    "--eval-every-iters", $EvalConvergeEveryIters,
    "--min-evals", $EvalConvergeMinEvals,
    "--min-improvement-bb", $EvalConvergeMinGain,
    "--k", $EvalConvergeK,
    "--state-file", $EvalConvStateFile,
    "--json-out", $EvalConvJsonFile,
    "--workdir", $EvalConvWorkDir,
    "--sb-min", $SBMin,
    "--bb-min", $BBMin,
    "--blind-mean-min", $BlindMeanMin,
    "--global-reg-max", $GlobalRegMax
  )
  $out = (& python @argsList 2>&1)
  $txt = ($out | Out-String)
  if ($txt -match "STATUS=([A-Z]+)") { $status = $Matches[1] } else { $status = "ERROR" }
  ($txt -split "`r?`n") | Where-Object { $_ -match "STATUS=" } | ForEach-Object { Write-Host "    eval-conv: $_" }
  return $status
}
function Invoke-Launch {
  param([bool]$ForceReuseCode = $false, [bool]$NoReuseCode = $false)
  $argsList = @(
    "scripts\launch_emd_pipeline_spot.py",
    "--run-subdir", $Run, "--iters", $Iters, "--n-deals", $NDeals,
    "--num-buckets", $NumBuckets, "--checkpoint-every", $CheckpointEvery,
    "--centroids", $Centroids, "--n-runouts", $NRunouts,
    "--instance-type", $InstanceType, "--arch", $Arch,
    "--ship-name", $ShipName, "--iter62000-snapshot-s3", $Iter62000SnapshotS3,
    "--eval-seeds", $EvalSeeds, "--eval-hands", $EvalHands
  )
  if ($DepthEdges) { $argsList += @("--depth-edges", $DepthEdges) }
  if ($Spotpick)   { $argsList += "--spotpick" }
  if ($OnDemand)   { $argsList += "--on-demand" }
  if ($CrateSrc)   { $argsList += @("--crate-src", $CrateSrc) }
  $argsList += @("--train-profile-every", $TrainProfileEvery)
  if ($TrainFastMerge) { $argsList += "--train-fast-merge" }
  if ($StopAfterShip) { $argsList += "--stop-after-ship" }
  # FIRST launch uploads the code tarball (crate + centroids + ship/eval py). Later train relaunches reuse it
  # only after the frozen table/checkpoint exists; early build/precompute failures must be able to re-upload a
  # fixed tarball. Ship/eval relaunches can force a fresh tarball so evaluator fixes are picked up while the
  # S3 markers still keep the resumed box from retraining or rebuilding the table.
  $reuseExistingCode = $ForceReuseCode
  if (-not $NoReuseCode -and -not $reuseExistingCode -and $launches -ge 1) {
    $reuseExistingCode = (Test-Marker $M_TABLE) -or ((Get-LatestItersDone) -ge 0)
  }
  if ($NoReuseCode) { Write-Host "  code upload: forced fresh tarball (no --reuse-code)" -ForegroundColor DarkCyan }
  if ($reuseExistingCode) { $argsList += "--reuse-code" }
  Write-Host "  launching: python $($argsList -join ' ')"
  $out = (& python @argsList 2>&1)
  $out | ForEach-Object { Write-Host "    $_" }
  $txt = ($out | Out-String)
  if ($txt -match "(?i)launched\s+(i-[0-9a-f]+)") { return $Matches[1] }
  return ""
}

# ============================================================================================
# V2 GATE: pluribus-grade auto-activate decision. Replaces the legacy two-boolean
# short-circuit. ALL conditions must hold for the gate to open. Each failing condition is logged
# with a specific reason so the user can fix the underlying problem and re-evaluate.
# ============================================================================================
function Test-AutoActivateGate([object]$v) {
  if (-not $v) {
    Write-Warning "GATE: no verdict JSON loaded"
    return $false
  }
  $tmp = Join-Path $env:TEMP ("emd_v2_gate_{0}.json" -f ([guid]::NewGuid().ToString('N')))
  try {
    $v | ConvertTo-Json -Depth 32 | Set-Content -Path $tmp -Encoding Ascii
    $out = (& python scripts/_emd_auto_activate_gate.py check --verdict $tmp 2>&1)
    $out | ForEach-Object { Write-Host "GATE: $_" }
    return ($LASTEXITCODE -eq 0)
  } finally {
    Remove-Item $tmp -ErrorAction SilentlyContinue
  }
}

$capH = [int]($MAXSEC / 3600)
Write-Host "=== EMD-pipeline babysitter V2: $Run  target=$Iters it, n_deals=$NDeals, buckets=$NumBuckets ==="
Write-Host "    pipeline: build -> precompute -> TRAIN(converge) -> SHIP(stage) -> EVAL(verdict)"
Write-Host "    caps: ${capH}h wall / $MaxLaunch train launches / $MaxShipEvalLaunch ship-eval launches / >=${MinRelaunchGapSec}s between"
$fanoutPerVcpu = [double]$NDeals / [Math]::Max(1, $Vcpus)
Write-Host ("    cpu-fanout: n_deals={0} / vcpus={1} -> {2:0.0} deal tasks/vCPU/iter" -f $NDeals, $Vcpus, $fanoutPerVcpu)
if (($Vcpus -ge 64) -and ($NDeals -lt ($Vcpus * 4))) {
  Write-Host ("    cpu-fanout WARNING: this underfeeds a large Rayon pool; prefer n_deals >= {0} or use scripts\pluribus_blueprint_run.ps1, which scales the marathon budget automatically." -f ($Vcpus * 4)) -ForegroundColor Yellow
}
if ($NoConvergeStop) {
  $convMode = "OFF (fixed-budget run; no convergence probes)"
} elseif ($ConvergeLogOnly) {
  $convMode = "LOG-ONLY threshold=$ConvThresh K=$ConvK (never stops early)"
} else {
  $convMode = "STOP threshold=$ConvThresh K=$ConvK (EMD floors slightly higher than the NB=200 0.018)"
}
Write-Host "    conv: $convMode"
if ($EvalConvergeStop -or $EvalConvergeLogOnly) {
  $evalConvMode = if ($EvalConvergeLogOnly) { "LOG-ONLY" } else { "STOP" }
  Write-Host ("    eval-conv: {0} min_iters={1} every={2} seeds={3} hands={4} min_gain={5:0.0}bb K={6} min_evals={7}" -f `
    $evalConvMode, $EvalConvergeMinIters, $EvalConvergeEveryIters, $EvalConvergeSeeds, $EvalConvergeHands, `
    $EvalConvergeMinGain, $EvalConvergeK, $EvalConvergeMinEvals)
}
Write-Host "    resume: markers table_cached/CONVERGED/SHIPPED/EVALED -> relaunch into the FIRST incomplete phase"
Write-Host "    train-profile: every=$TrainProfileEvery fast_merge=$([bool]$TrainFastMerge) (fast_merge requires separate verifier approval)"
# PowerShell can't use `if` as an expression inside a `-f` format string; assign the string first.
if ($AutoActivate) {
  $activateStr = "AUTO (V2 pluribus-grade gate: K=$KMax / anchor in [$AnchorLo,$AnchorHi] / SB>=${SBMin} / BB>=$BBMin / blind-mean>=$BlindMeanMin / global-regression<=$GlobalRegMax / sha256 / sign-correct)"
} else {
  $activateStr = 'OFF (stage + report; user activates manually)'
}
Write-Host ("    activate: {0}" -f $activateStr)

# ===== RESUME SHORT-CIRCUIT: if a prior run already CONVERGED, skip the whole TRAIN loop and jump to the
# SHIP/EVAL lifecycle (a kill during ship/eval must NOT re-train). =====
if (Test-Marker $M_CONVERGED) {
  Write-Host "RESUME: CONVERGED marker already present -> skipping TRAIN; entering SHIP/EVAL lifecycle." -ForegroundColor Green
  $convergedStop = $true
} else {

while ($true) {
  $el   = [int]((Get-Date) - $start).TotalSeconds
  $done = Get-LatestItersDone           # -1 = no ckpt yet (booting/building/precomputing/first ckpt)
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
  $doneTxt = if ($done -ge 0) { "$done/$Iters" } else { "no-ckpt-yet (build/precompute/first-ckpt)" }
  $boxTxt  = if ($live) { $live } else { "(none)" }
  Write-Host "[t+${el}s] iters=$doneTxt  live=$boxTxt  launches=$launches"
  if ($prevBox -and $live -and $live -ne $prevBox) {
    Write-Host "          BOX CHANGED: $prevBox -> $live (spot reclaim + relaunch / resumed from S3)" -ForegroundColor Yellow
  }
  if ($live) { $prevBox = $live; $tracked = $live }

  # 0) the box (or a prior boot) already reached SHIP/EVAL on its own -> CONVERGED marker exists. Leave the
  # train loop WITHOUT terminating (the box continues ship/eval on-box, or a resumed box will). The ship/eval
  # lifecycle below polls for EVALED. This fires when the box finished training naturally and rolled on.
  if (Test-Marker $M_CONVERGED) {
    Write-Host "TRAIN complete: CONVERGED marker present (box rolling on to SHIP/EVAL on-box)." -ForegroundColor Green
    $convergedStop = $true; break
  }
  # 1) iters target reached -> the box will finish train then CONTINUE to ship+eval on THIS boot. Do NOT
  # terminate it; just hand off to the ship/eval lifecycle loop (it polls for the markers the box writes).
  if ($done -ge $Iters) {
    Write-Host "DONE: latest checkpoint reached $done/$Iters iters (iters BACKSTOP); box continues to SHIP/EVAL on-box."
    $convergedStop = $true; break
  }
  # 2) wall cap (BACKSTOP) -- covers BOTH precompute and train. Hard stop (kill the box; do NOT ship a
  # non-converged blueprint automatically).
  if ($el -ge $MAXSEC) { Write-Host "STOP: wall-clock cap (${capH}h) reached (wall BACKSTOP)."; Stop-Run; break }
  # 2b) CONVERGENCE PROBE/STOP: when a NEW checkpoint has landed, diff the two latest.
  # In normal mode the babysitter DECIDES convergence here -> it WRITES the CONVERGED marker (so the resumed
  # box's cloud-init skips train and jumps to SHIP), TERMINATES the mid-train box, and breaks to ship/eval.
  # In -ConvergeLogOnly mode this is telemetry only. This is useful for expensive EMD runs where a strategy-
  # delta plateau can mean "the coarse abstraction stopped moving" rather than "the blueprint is strong".
  if (-not $NoConvergeStop -and $done -gt 0 -and $done -gt $lastConvIters) {
    $lastConvIters = $done
    $cstat = Test-Converged
    if ($cstat -eq "CONVERGED") {
      if ($ConvergeLogOnly) {
        Write-Host "CONV: plateau detected at $done iters, but -ConvergeLogOnly is set -> continuing to fixed target." -ForegroundColor Yellow
      } else {
      Write-Host "DONE: convergence stop -> avg strategy plateaued (delta <= $ConvThresh for $ConvK ckpts) at $done iters." -ForegroundColor Green
      Write-Marker $M_CONVERGED "babysitter_convergence_stop_at_$done"
      Write-Host "  wrote CONVERGED marker; terminating the mid-train box -> resume at SHIP/EVAL." -ForegroundColor Green
      $convergedStop = $true
      Stop-Run; break
      }
    }
  }
  # 2c) EVAL-MARGINAL PROBE/STOP: periodically score the latest checkpoint against iter62000 using the
  # same CRN per-seat blind verdict surface as activation. This is the K200 marathon stop: when activation
  # margin stops improving materially for EvalConvergeK probes, ship/eval the best-recent plateau instead
  # of burning the fixed iters budget. The final verified-better gate still runs after shipping.
  if (($EvalConvergeStop -or $EvalConvergeLogOnly) -and $done -gt 0 -and $done -gt $lastEvalConvIters) {
    $lastEvalConvIters = $done
    $estat = Test-EvalConverged
    if ($estat -eq "CONVERGED") {
      if ($EvalConvergeLogOnly) {
        Write-Host "EVAL-CONV: marginal-improvement plateau detected at $done iters, but -EvalConvergeLogOnly is set -> continuing." -ForegroundColor Yellow
      } else {
        Write-Host "DONE: eval convergence stop -> activation-margin improvements are marginal for $EvalConvergeK probes at $done iters." -ForegroundColor Green
        Write-Marker $M_CONVERGED "babysitter_eval_marginal_stop_at_$done"
        Write-Host "  wrote CONVERGED marker; terminating the mid-train box -> resume at SHIP/EVAL." -ForegroundColor Green
        $convergedStop = $true
        Stop-Run; break
      }
    }
  }
  # 3) a box is live and not done -> just wait (it is building / precomputing / training)
  if ($live) { Start-Sleep -Seconds $PollSec; continue }

  # 3b) EXACT-QUOTA GUARD: the previous box is still terminating (still counts against the spot vCPU quota).
  if ($draining) {
    Write-Host "  previous box $draining still terminating (holds quota); waiting ${PollSec}s before relaunch"
    Start-Sleep -Seconds $PollSec
    continue
  }

  # 4) NO live box and target NOT reached -> preemption (or never-launched). Relaunch, with storm guards.
  if ($launches -ge $MaxLaunch) { Write-Host "STOP: max launches ($MaxLaunch) reached (storm guard)."; break }
  $sinceLast = ((Get-Date) - $lastLaunch).TotalSeconds
  if ($launches -gt 0 -and $sinceLast -lt $MinRelaunchGapSec) {
    $wait = [int]($MinRelaunchGapSec - $sinceLast)
    Write-Host "  relaunch gap guard: only ${sinceLast}s since last launch (< ${MinRelaunchGapSec}s); waiting ${wait}s"
    Start-Sleep -Seconds $wait
    continue
  }
  $reason = if ($done -ge 0) { "preempted at $done/$Iters -> resume from S3 (table re-downloaded, no recompute)" } else { "no box + no ckpt -> initial launch" }
  Write-Host "  no live box; $reason. Launching #$($launches+1)..." -ForegroundColor Cyan
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

}  # end: else (TRAIN loop) -- entered only when CONVERGED was not already present at start.

$final = Get-LatestItersDone
Write-Host ""
Write-Host "=== TRAIN phase exited after ${el}s, $launches launches. latest iters_done=$final / $Iters ==="

# ============================================================================================
# SHIP/EVAL LIFECYCLE: drive PHASE 4 (SHIP) + PHASE 5 (EVAL) to completion (the EVALED marker).
# Only entered when training CONVERGED (the convergence stop, the iters backstop, or a resumed run that
# already had the CONVERGED marker). A wall-cap or storm-guard abort leaves $convergedStop false -> we do
# NOT ship a non-converged blueprint; we just report the manual-resume hint and exit.
# ============================================================================================
if (-not $convergedStop) {
  Write-Host "Run NOT converged (hit the wall cap or a storm/launch guard). NOT shipping. The S3 latest" -ForegroundColor Yellow
  Write-Host "checkpoint + cached frozen table are preserved for a manual resume:" -ForegroundColor Yellow
  Write-Host "  powershell -File scripts\emd_pipeline_babysitter_v2.ps1 -Run $Run   # resumes at the first incomplete phase"
  return
}

Write-Host ""
if ($StopAfterShip) {
  Write-Host "=== SHIP lifecycle: driving staged serve artifact only (-StopAfterShip; eval disabled) ===" -ForegroundColor Cyan
} else {
  Write-Host "=== SHIP/EVAL lifecycle: driving stage + verified-better eval to the EVALED marker ===" -ForegroundColor Cyan
}
$seLaunches  = 0
$seLastLaunch = [datetime]::MinValue
$seStart = Get-Date
$seMaxSec = [Math]::Max(0, $MAXSEC - $el)   # remaining wall budget carries into ship/eval

while (-not (Test-Marker $M_EVALED)) {
  $seEl = [int]((Get-Date) - $seStart).TotalSeconds
  $live = Get-LiveBoxId
  $draining = ""
  if (-not $live -and $tracked) {
    $st = Get-StateById $tracked
    if ($st -eq "running" -or $st -eq "pending") { $live = $tracked }
    elseif ($st -eq "shutting-down" -or $st -eq "stopping") { $draining = $tracked }
    elseif ($st) { $tracked = "" }
  }
  $tableCached = Test-Marker $M_TABLE
  $convergedMarker = Test-Marker $M_CONVERGED
  $shipped = Test-Marker $M_SHIPPED
  $evaled = Test-Marker $M_EVALED
  $phaseTxt = if ($evaled) { "EVALED" } elseif ($shipped -and $StopAfterShip) { "SHIPPED(STOP)" } elseif ($shipped) { "SHIPPED->EVAL" } else { "SHIP" }
  $frozenBytes = Get-FrozenSummary
  $serveBytes = Get-S3Size "$RUN_PREFIX/serve/$ShipName.pkl"
  $verdictBytes = Get-S3Size $VERDICT_KEY
  Write-Host "[ship/eval t+${seEl}s] phase=$phaseTxt  live=$(if ($live) { $live } else { '(none)' })  seLaunches=$seLaunches"
  Write-Host ("    s3: table={0} converged={1} shipped={2} evaled={3} | frozen={4} serve={5} verdict={6}" -f `
    $tableCached, $convergedMarker, $shipped, $evaled, (Format-Bytes $frozenBytes), (Format-Bytes $serveBytes), (Format-Bytes $verdictBytes)) -ForegroundColor DarkCyan
  if ($live) {
    $probe = Get-ConsoleProbe $live
    $probeTxt = Format-Probe $probe
    $s3Fatal = (Probe-Has $probe "S3UploadFail") -or (Probe-Has $probe "S3PutFail") -or ((Probe-Has $probe "S3DownloadFail") -and -not (Probe-Has $probe "CacheMiss"))
    $probeColor = if ((Probe-Has $probe "Traceback") -or $s3Fatal -or (Probe-Has $probe "AwsError") -or ((Probe-Has $probe "TrainStart") -and -not (Probe-Has $probe "TrainSkip"))) { "Red" } else { "DarkGray" }
    Write-Host "    live: $probeTxt" -ForegroundColor $probeColor
  }

  if ($evaled) { break }
  if ($StopAfterShip -and $shipped) {
    Write-Host "STOP-AFTER-SHIP: staged serve artifact is present; not launching/running eval." -ForegroundColor Green
    Write-Host "  staged: s3://$Bucket/$RUN_PREFIX/serve/$ShipName.pkl"
    return
  }

  # wall backstop (shared budget): stop launching but leave any live box to finish.
  if ($seMaxSec -gt 0 -and $seEl -ge $seMaxSec -and -not $live) {
    Write-Host "STOP: wall budget exhausted in ship/eval with no live box." -ForegroundColor Yellow; break
  }
  # a box is live (running ship/eval on-box) -> just wait.
  if ($live) { Start-Sleep -Seconds $PollSec; continue }
  # previous box still terminating (holds quota) -> wait.
  if ($draining) { Write-Host "  prev box $draining terminating; waiting ${PollSec}s"; Start-Sleep -Seconds $PollSec; continue }

  # no live box + not EVALED -> (re)launch a box that RESUMES at the first incomplete phase (SHIP or EVAL).
  # It re-downloads the cached frozen table + the converged checkpoint and runs only the unfinished phases.
  # Re-upload the code tarball here so eval-resume fixes (for example partial-result persistence) can be
  # applied after a spot reclaim. Markers keep the resumed box from retraining or reshipping unnecessarily.
  if ($seLaunches -ge $MaxShipEvalLaunch) {
    Write-Host "STOP: max ship/eval launches ($MaxShipEvalLaunch) reached (storm guard); EVALED not reached." -ForegroundColor Yellow; break
  }
  $sinceLast = ((Get-Date) - $seLastLaunch).TotalSeconds
  if ($seLaunches -gt 0 -and $sinceLast -lt $MinRelaunchGapSec) {
    $wait = [int]($MinRelaunchGapSec - $sinceLast)
    Write-Host "  ship/eval relaunch gap guard: ${sinceLast}s < ${MinRelaunchGapSec}s; waiting ${wait}s"
    Start-Sleep -Seconds $wait; continue
  }
  Write-Host "  no live box; launching a SHIP/EVAL-resume box #$($seLaunches+1) (fresh-code; resume at first incomplete phase)..." -ForegroundColor Cyan
  $id = Invoke-Launch -NoReuseCode $true
  $seLastLaunch = Get-Date
  $seLaunches++
  if ($id) {
    $tracked = $id
    Write-Host "  tracking $id; boot grace ${BootWaitSec}s before next poll"
    Start-Sleep -Seconds $BootWaitSec
  } else {
    Write-Host "  ship/eval launch reported no instance id - backing off ${MinRelaunchGapSec}s" -ForegroundColor Yellow
    Start-Sleep -Seconds $MinRelaunchGapSec
  }
}

# ============================================================================================
# VERDICT REPORT + (optional) AUTO-ACTIVATE
# ============================================================================================
Write-Host ""
Write-Host "=== EMD-PIPELINE BABYSITTER V2: pipeline complete (SHIPPED=$(Test-Marker $M_SHIPPED) EVALED=$(Test-Marker $M_EVALED)) ===" -ForegroundColor Cyan
$verdict = Get-Verdict
if (-not $verdict) {
  Write-Host "No verdict JSON yet at s3://$Bucket/$VERDICT_KEY (eval did not complete in budget)." -ForegroundColor Yellow
  Write-Host "The converged checkpoint + staged snapshot (if SHIPPED) are preserved; re-run the babysitter to resume EVAL:" -ForegroundColor Yellow
  Write-Host "  powershell -File scripts\emd_pipeline_babysitter_v2.ps1 -Run $Run"
  return
}

$emdBetter = [bool]$verdict.emd_better
$anchorOk  = [bool]$verdict.anchor_ok
Write-Host "VERDICT (per-seat blind LBR, CRN; emd vs iter62000):" -ForegroundColor Cyan
Write-Host ("  anchor: emd={0:+0.0} iter62000={1:+0.0}  (target ~+{2:0}) -> anchor_ok={3}" -f `
  [double]$verdict.emd_anchor_bb_per_100, [double]$verdict.iter62000_anchor_bb_per_100, `
  [double]$verdict.anchor_target, $anchorOk) -ForegroundColor $(if ($anchorOk) { 'Green' } else { 'Red' })
# Sign-correct per-seat rendering delegated to scripts/_emd_verdict_sign.py
# (single source of truth -- prevents the literal 'improvement' sign-flip bug
# that previously labeled the k500 SB regression as an improvement).
$verdictTmp = Join-Path $env:TEMP ("emd_v2_verdict_{0}.json" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
$rendered = & python scripts/_emd_verdict_sign.py render --verdict $verdictTmp 2>$null
if ($rendered) {
  foreach ($ln in ($rendered -split "`n")) { if ($ln) { Write-Host ("  " + $ln) } }
} else {
  # Fallback only if the helper failed to run; do NOT inline a literal label.
  Write-Host "  (verdict render helper unavailable; see $verdictTmp)" -ForegroundColor Yellow
}
Write-Host ("  mean blind gain={0:+0.0}  global: iter62000={1:+0.0} -> emd={2:+0.0}" -f `
  [double]$verdict.mean_blind_gain, [double]$verdict.iter62000_pooled_bb_per_100, [double]$verdict.emd_pooled_bb_per_100)
Write-Host ("  ===> emd_better = {0}" -f $emdBetter) -ForegroundColor $(if ($emdBetter) { 'Green' } else { 'Yellow' })

if (-not $AutoActivate) {
  Write-Host ""
  Write-Host "STAGED (not activated). -AutoActivate is OFF (default). Review the per-seat LBR above, then" -ForegroundColor Yellow
  Write-Host "activate manually when satisfied:" -ForegroundColor Yellow
  Write-Host "  python scripts\ship_blueprint.py --checkpoint checkpoints\serve_blueprints\$ShipName.pkl --emd-frozen <frozen> --num-buckets $NumBuckets --activate"
  Write-Host "  (the staged snapshot is also at s3://$Bucket/$RUN_PREFIX/serve/$ShipName.pkl)"
  return
}

# -AutoActivate ON (V2): the gate is the pluribus-grade checklist, NOT the legacy emd_better/anchor_ok
# short-circuit. This is the explicit fix for the k500 sign-flip bug class (where emd_better=True on a
# verdict that actually had a real -140 SB regression because of the literal 'improvement' label).
Write-Host ""
Write-Host "AUTO-ACTIVATE (V2 pluribus-grade gate):" -ForegroundColor Cyan
$gateOk = Test-AutoActivateGate $verdict
if ($gateOk) {
  Write-Host "GATE: ALL CONDITIONS PASS -> flipping the live serve pointer." -ForegroundColor Green
  # Fetch the exact frozen table stamped by the verdict (needed for the EMD ship env) + the staged snapshot.
  $frozenS3 = [string]$verdict.emd_frozen_s3
  $serveS3 = if ($verdict.PSObject.Properties["serve_blueprint_s3"] -and [string]$verdict.serve_blueprint_s3) {
    [string]$verdict.serve_blueprint_s3
  } else {
    "s3://$Bucket/$RUN_PREFIX/serve/$ShipName.pkl"
  }
  $frozenTmp = Join-Path $env:TEMP ("emd_v2_frozen_{0}.emdfrzn" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
  $snapTmp = Join-Path $env:TEMP ("emd_v2_staged_{0}.pkl" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
  aws s3 cp $serveS3 $snapTmp --region $Region --no-progress 2>$null | Out-Null
  aws s3 cp $frozenS3 $frozenTmp --region $Region --no-progress 2>$null | Out-Null
  aws s3 cp "$frozenS3.sha256" "$frozenTmp.sha256" --region $Region --no-progress 2>$null | Out-Null
  if ((Test-Path $snapTmp) -and (Test-Path $frozenTmp)) {
    $aargs = @("scripts\ship_blueprint.py", "--checkpoint", $snapTmp, "--emd-frozen", $frozenTmp,
               "--num-buckets", $NumBuckets, "--name", $ShipName, "--activate")
    Write-Host "  python $($aargs -join ' ')"
    $out = (& python @aargs 2>&1); $out | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -eq 0) {
      Write-Host "ACTIVATED: the live serve pointer now serves the EMD blueprint '$ShipName'." -ForegroundColor Green
      Write-Host "  revert with: python scripts\ship_blueprint.py --revert"
    } else {
      Write-Host "ACTIVATION FAILED (ship_blueprint.py rc=$LASTEXITCODE -- the serve-skew gate refused). Staying STAGED." -ForegroundColor Red
    }
  } else {
    Write-Host "ACTIVATION SKIPPED: could not fetch the staged snapshot and/or frozen table from S3. Staying STAGED." -ForegroundColor Red
  }
  Remove-Item $snapTmp, $frozenTmp -ErrorAction SilentlyContinue
} else {
  Write-Host ""
  Write-Host "AUTO-ACTIVATE BLOCKED. See warnings above. Staying STAGED; iter62000 remains live." -ForegroundColor Red
}
