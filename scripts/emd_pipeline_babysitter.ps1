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
# ACTIVATION (default OFF -- safe): by default the pipeline STAGES + EVALS + reports the per-seat verdict and
# STOPS, leaving the user to activate manually after seeing the per-seat blind LBR. With -AutoActivate the
# babysitter NEVER auto-activates. The legacy two-boolean auto-activate
# path is disabled; use emd_pipeline_babysitter_v2.ps1 for the Python-gated path.
#
# SINGLE LAUNCH (this is the one command the user runs):
#   powershell -ExecutionPolicy Bypass -File scripts\emd_pipeline_babysitter.ps1 -Run run_emd_bp1
#   # everything else has EMD-appropriate defaults; add -AutoActivate to auto-flip on a passing verdict.
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
  [switch]$Spotpick,                  # capacity-diverse fallback chain (auto-on when auto-picking)
  [switch]$OnDemand,                  # no spot (no preemption, pricier)
  [int]$MaxLaunch = 10,               # total relaunch cap (storm guard)
  [int]$MinRelaunchGapSec = 600,      # min wall seconds between launches (storm guard)
  [int]$BootWaitSec = 300,            # wait after a launch before polling (boot grace)
  [int]$PollSec = 180,                # state poll interval while a box is live
  [int]$MaxHours = 28,                # overall wall cap (BACKSTOP; covers precompute + train; conv stops earlier)
  [string]$Region = "eu-central-1",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr",
  # --- convergence-based stop (EMD-appropriate threshold; a richer abstraction floors slightly higher) ---
  [switch]$NoConvergeStop,            # disable convergence stop; run to fixed -Iters / -MaxHours only
  [double]$ConvThreshold = 0.020,     # EMD mean-L1 plateau floor (~0.020-0.022; vs 0.018 for NB=200 made-hand)
  [int]$ConvK = 3,                    # consecutive below-threshold checkpoints required to declare CONVERGED
  [string]$ConvDepthEdges = "",       # stop-helper ctor depth-edges (irrelevant to load); CSV like "50,100"
  # --- PHASE 4 SHIP + PHASE 5 EVAL ---
  [string]$ShipName = "emd_bp1",      # version tag for the staged serve snapshot
  [string]$Iter62000SnapshotS3 = "s3://video-processing-6789/pokerbot/prod6000/native_cfr/bp6max_pluribus/serve/iter62000_snapshot.pkl",
  [int]$EvalSeeds = 12,               # CRN seeds the LBR pools over (both blueprints share them)
  [int]$EvalHands = 2000,             # hands per seed (Python LBR is GIL-bound; lean on the CRN per-seat delta)
  [int]$MaxShipEvalLaunch = 6,        # relaunch cap for the SHIP/EVAL lifecycle (separate from -MaxLaunch)
  # --- ACTIVATION (default OFF; safe -- stage + report only) ---
  [switch]$AutoActivate               # DISABLED in this legacy script; use emd_pipeline_babysitter_v2.ps1
)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

if ($AutoActivate) {
  Write-Host "ERROR: -AutoActivate is disabled in legacy emd_pipeline_babysitter.ps1." -ForegroundColor Red
  Write-Host "Use scripts\emd_pipeline_babysitter_v2.ps1, which delegates activation to scripts/_emd_auto_activate_gate.py." -ForegroundColor Red
  exit 2
}

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
$ConvStateFile = Join-Path $env:TEMP ("emd_conv_stop_{0}.json" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
$lastConvIters = -1

function Test-Marker([string]$key) {
  # True if the S3 marker object exists. Use GetObject-style probing because EC2/local roles may
  # have object read/write without ListBucket, which makes `aws s3 ls` lie for exact marker keys.
  $tmp = Join-Path $env:TEMP ("emd_marker_probe_{0}.txt" -f ([guid]::NewGuid().ToString('N')))
  aws s3 cp "s3://$Bucket/$key" $tmp --region $Region --no-progress 2>$null | Out-Null
  $ok = Test-Path $tmp
  Remove-Item $tmp -ErrorAction SilentlyContinue
  return $ok
}
function Write-Marker([string]$key, [string]$note) {
  # Idempotent marker write (small text object). Used by the babysitter when IT decides convergence.
  $tmp = Join-Path $env:TEMP ("emd_marker_{0}.txt" -f ([guid]::NewGuid().ToString('N')))
  Set-Content -Path $tmp -Value ("{0} {1}" -f (Get-Date -Format o), $note) -Encoding Ascii
  aws s3 cp $tmp "s3://$Bucket/$key" --region $Region --no-progress 2>$null | Out-Null
  Remove-Item $tmp -ErrorAction SilentlyContinue
}
function Get-Verdict {
  # Download + parse the eval verdict JSON. Returns $null if absent/unparseable.
  # The tmp path is preserved (NOT deleted) so downstream sign-correct printers
  # (scripts/_emd_verdict_sign.py render) can re-read the same JSON instead of
  # the buggy literal 'improvement' formatting. The script's end-of-run cleanup
  # best-effort removes it via the existing $verdictTmp tracking variable.
  $tmp = Join-Path $env:TEMP ("emd_verdict_{0}.json" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
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
  $tmp = Join-Path $env:TEMP "emd_latest_probe.nlcfr"
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
    "--threshold", $ConvThreshold, "--k", $ConvK,
    "--state-file", $ConvStateFile, "--num-buckets", $NumBuckets
  )
  if ($ConvDepthEdges) { $argsList += @("--depth-edges"); $argsList += ($ConvDepthEdges -split ",") }
  $out = (& python @argsList 2>&1)
  $txt = ($out | Out-String)
  if ($txt -match "STATUS=([A-Z]+)") { $status = $Matches[1] } else { $status = "ERROR" }
  ($txt -split "`r?`n") | Where-Object { $_ -match "STATUS=" } | ForEach-Object { Write-Host "    conv: $_" }
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

$capH = [int]($MAXSEC / 3600)
Write-Host "=== EMD-pipeline babysitter: $Run  target=$Iters it, n_deals=$NDeals, buckets=$NumBuckets ==="
Write-Host "    pipeline: build -> precompute -> TRAIN(converge) -> SHIP(stage) -> EVAL(verdict)"
Write-Host "    caps: ${capH}h wall / $MaxLaunch train launches / $MaxShipEvalLaunch ship-eval launches / >=${MinRelaunchGapSec}s between"
Write-Host "    conv: threshold=$ConvThreshold K=$ConvK (EMD floors slightly higher than the NB=200 0.018)"
Write-Host "    resume: markers table_cached/CONVERGED/SHIPPED/EVALED -> relaunch into the FIRST incomplete phase"
Write-Host "    activate: OFF (legacy auto-activate disabled; use emd_pipeline_babysitter_v2.ps1 for Python-gated activation)"

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
  # 2b) CONVERGENCE STOP: when a NEW checkpoint has landed, diff the two latest; stop early if plateaued.
  # The babysitter DECIDES convergence here -> it WRITES the CONVERGED marker (so the resumed box's cloud-
  # init skips train and jumps to SHIP), TERMINATES the mid-train box, and breaks to the ship/eval lifecycle.
  if (-not $NoConvergeStop -and $done -gt 0 -and $done -gt $lastConvIters) {
    $lastConvIters = $done
    $cstat = Test-Converged
    if ($cstat -eq "CONVERGED") {
      Write-Host "DONE: convergence stop -> avg strategy plateaued (delta <= $ConvThreshold for $ConvK ckpts) at $done iters." -ForegroundColor Green
      Write-Marker $M_CONVERGED "babysitter_convergence_stop_at_$done"
      Write-Host "  wrote CONVERGED marker; terminating the mid-train box -> resume at SHIP/EVAL." -ForegroundColor Green
      $convergedStop = $true
      Stop-Run; break
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
  Write-Host "  powershell -File scripts\emd_pipeline_babysitter.ps1 -Run $Run   # resumes at the first incomplete phase"
  return
}

Write-Host ""
Write-Host "=== SHIP/EVAL lifecycle: driving stage + verified-better eval to the EVALED marker ===" -ForegroundColor Cyan
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
  $phaseTxt = if ($evaled) { "EVALED" } elseif ($shipped) { "SHIPPED->EVAL" } else { "SHIP" }
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
# VERDICT REPORT + (optional) ACTIVATION
# ============================================================================================
Write-Host ""
Write-Host "=== EMD-PIPELINE BABYSITTER: pipeline complete (SHIPPED=$(Test-Marker $M_SHIPPED) EVALED=$(Test-Marker $M_EVALED)) ===" -ForegroundColor Cyan
$verdict = Get-Verdict
if (-not $verdict) {
  Write-Host "No verdict JSON yet at s3://$Bucket/$VERDICT_KEY (eval did not complete in budget)." -ForegroundColor Yellow
  Write-Host "The converged checkpoint + staged snapshot (if SHIPPED) are preserved; re-run the babysitter to resume EVAL:" -ForegroundColor Yellow
  Write-Host "  powershell -File scripts\emd_pipeline_babysitter.ps1 -Run $Run"
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
$verdictTmp = Join-Path $env:TEMP ("emd_verdict_{0}.json" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
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

Write-Host ""
Write-Host "STAGED (not activated). Legacy auto-activate is disabled in this script." -ForegroundColor Yellow
Write-Host "Use scripts\emd_pipeline_babysitter_v2.ps1 for Python-gated activation, or activate manually after review:" -ForegroundColor Yellow
Write-Host "  python scripts\ship_blueprint.py --checkpoint checkpoints\serve_blueprints\$ShipName.pkl --emd-frozen <exact-frozen> --num-buckets $NumBuckets --activate"
Write-Host "  (the staged snapshot is also at s3://$Bucket/$RUN_PREFIX/serve/$ShipName.pkl)"
return
