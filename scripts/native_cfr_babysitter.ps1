# Autonomous babysitter for the NATIVE Rust CFR+ 6-max cash blueprint trainer
# (scripts/launch_native_cfr_training_spot.py, Workstream F).
#
# WHY: the launcher has an IN-BOX spot-interrupt handler (it checkpoints to S3 on the reclaim notice),
# but nothing RELAUNCHES a replacement box after the spot instance dies. A large/long run on a 128-vCPU
# spot box WILL get preempted; this script polls the instance state + the S3 latest checkpoint and, when
# the box is gone BEFORE the run hit its --iters target, relaunches with the SAME --run-subdir so the new
# box RESUMES from the S3 latest checkpoint (driver does load_bytes(latest) on boot).
#
# Run from your terminal (your AWS creds; the launcher uses boto3 with the same default profile):
#   powershell -ExecutionPolicy Bypass -File scripts\native_cfr_babysitter.ps1 -Run run_bp1 -Iters 20000 -NDeals 400
#   # add -Spotpick to let a capacity error fall down the spotpick chain instead of hard-failing.
#
# RELAUNCH-STORM PROTECTION (the cloud-eval babysitters had a relaunch-storm bug; this avoids it):
#   * MaxLaunch hard cap on total relaunches.
#   * MinRelaunchGapSec minimum wall time between two launches (so a box that dies during boot does not
#     trigger a tight relaunch loop).
#   * After a launch we WAIT for boot (BootWaitSec) before the next state poll, and we VERIFY the box we
#     launched by INSTANCE ID (not only the laggy tag filter), exactly like rnr_validation_babysitter.ps1.
#   * "done" is decided by the S3 LATEST checkpoint's iters_done >= target (the trainer's ground truth),
#     NOT by a box being absent (an absent box mid-run == preemption -> relaunch, not done).
#
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252; non-ASCII bytes break strings).
param(
  [string]$Run = "",                  # S3 run-subdir; REQUIRED for resume-across-preemption (auto-gen warns)
  [int]$Iters = 20000,                # --iters target (must match what you intend the run to reach)
  [int]$NDeals = 400,                 # --n-deals
  [int]$NumBuckets = 200,             # --num-buckets  (MUST equal serve's 200; see report)
  [string]$DepthEdges = "",           # --depth-edges  ("" = non-depth blueprint, the deployed default)
  [int]$CheckpointEvery = 500,        # --checkpoint-every
  [string]$InstanceType = "",          # "" = AUTO-PICK cheapest spot-available type via spotpick (below)
  [int]$Vcpus = 128,                   # auto-pick target vCPU count (128 = c6*.32xlarge class)
  [string]$Arch = "x86_64",
  [string]$CrateSrc = "runs/bp_crate_frozen/battery_native",  # FROZEN crate copy uploaded as tools/battery_native
                                                              # (isolates this run from concurrent live-crate edits)
  [switch]$Spotpick,                  # pass --spotpick to the launcher (capacity-diverse fallback chain)
  [switch]$OnDemand,                  # pass --on-demand (no spot; no preemption, but pricier)
  [int]$MaxLaunch = 10,               # total relaunch cap (storm guard)
  [int]$MinRelaunchGapSec = 600,      # min wall seconds between launches (storm guard)
  [int]$BootWaitSec = 300,            # wait after a launch before polling (boot grace, storm guard)
  [int]$PollSec = 180,                # state poll interval while a box is live
  [int]$MaxHours = 48,                # overall wall-clock cap (BACKSTOP; convergence stops earlier)
  [string]$Region = "eu-central-1",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr",
  # --- convergence-based stop (efficiency: stop AT CONVERGENCE, not at the fixed --iters/wall cap) ---
  # After each NEW checkpoint, scripts/blueprint_convergence_stop.py diffs the two latest versioned
  # checkpoints (mean-L1 average-strategy delta over shared infosets) and reports CONVERGED once the
  # delta has stayed <= -ConvThreshold for -ConvK consecutive checkpoints. We then TERMINATE + stop.
  # The -Iters and -MaxHours caps remain as BACKSTOPS. Set -NoConvergeStop to disable (fixed-iters mode).
  [switch]$NoConvergeStop,            # disable convergence stop; run to fixed -Iters / -MaxHours only
  [double]$ConvThreshold = 0.018,     # mean-L1 plateau floor. iter62000 (NB=200) plateaued at 0.018.
                                      # A RICHER abstraction floors slightly HIGHER (more thinly-visited
                                      # rows move more per visit): for the NB=600 better-blueprint run use
                                      # ~0.020-0.022, OR run a few -NoConvergeStop ckpts first, read the
                                      # delta where its per-step drop flattens to ~3-4%, and set that.
                                      # Too-tight only delays the stop (iters/wall backstops still fire);
                                      # the K-consecutive rule guards against a too-loose premature stop.
  [int]$ConvK = 3,                    # consecutive below-threshold checkpoints required to declare CONVERGED
  [int]$ConvNumBuckets = 200,         # stop-helper NlheCFRTrainer ctor arg (IRRELEVANT to load; set 600 for NB=600)
  [string]$ConvDepthEdges = ""        # ctor depth-edges for the stop helper (irrelevant to load); CSV like "50,100"
)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

if (-not $Run) {
  $Run = "run_bp_{0}" -f ([int][double]::Parse((Get-Date -UFormat %s)))
  Write-Host "WARNING: no -Run given; generated '$Run'. Relaunch-on-preemption resumes THIS subdir." -ForegroundColor Yellow
}

$RUN_PREFIX = "$Prefix/$Run"
$LATEST     = "$RUN_PREFIX/checkpoints/latest.nlcfr"
$MAXSEC     = $MaxHours * 3600
$start      = Get-Date
$launches   = 0
$lastLaunch = [datetime]::MinValue
$tracked    = ""        # instance id of the box we last launched (verified by id, not the laggy tag filter)
$prevBox    = ""
$convergedStop = $false  # set true when we stop early on a convergence plateau (vs an iters/wall backstop)
# convergence-stop state: a per-run JSON streak file (next to TEMP) + the last iters we convergence-checked,
# so we only RE-CHECK when a NEW checkpoint has landed (the helper also de-dups internally on the pair id).
$ConvStateFile = Join-Path $env:TEMP ("conv_stop_{0}.json" -f ($Run -replace '[^A-Za-z0-9_]', '_'))
$lastConvIters = -1

# AUTO-PICK the best (cheapest spot-available) instance type via the shared spotpick lib when
# -InstanceType was not given. spotpick ranks by spot $/vCPU over spot-available types, so this lands on
# the cheapest $Vcpus-vCPU box (currently c6a.32xlarge) and turns ON the capacity-diverse fallback chain
# (-Spotpick) so a sold-out pool advances to the next type instead of hard-failing.
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
    $InstanceType = "c6a.32xlarge"   # spotpick unavailable -> cheapest known 128-vCPU as a safe default
    Write-Host "AUTO-PICK fallback (spotpick unavailable): $InstanceType" -ForegroundColor Yellow
  }
  $Spotpick = $true                  # always enable the capacity-diverse fallback chain when auto-picking
}

function Get-LiveBoxId {
  # Tag filter for our run; returns the first running/pending instance id, or "".
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
  # The trainer's ground truth: parse iters_done out of the latest.nlcfr blob header.
  # Format (trainer.rs::serialize): magic "NLCFR001" (8 bytes) + iters_done u64 LE (8 bytes) + maps.
  $tmp = Join-Path $env:TEMP "native_cfr_latest_probe.nlcfr"
  aws s3 cp "s3://$Bucket/$LATEST" $tmp --region $Region --no-progress 2>$null | Out-Null
  if (-not (Test-Path $tmp)) { return -1 }   # no checkpoint yet (fresh box still booting/first ckpt)
  try {
    $fs = [System.IO.File]::OpenRead($tmp)
    try {
      $buf = New-Object byte[] 16
      $n = $fs.Read($buf, 0, 16)
      if ($n -lt 16) { return -1 }
      $magic = [System.Text.Encoding]::ASCII.GetString($buf, 0, 8)
      if ($magic -ne "NLCFR001") { return -1 }
      $iters = [System.BitConverter]::ToUInt64($buf, 8)   # host is little-endian; blob is LE
      return [int64]$iters
    } finally { $fs.Close() }
  } catch { return -1 } finally { Remove-Item $tmp -ErrorAction SilentlyContinue }
}
function Stop-Run {
  $id = Get-LiveBoxId
  if (-not $id -and $tracked) { $st = Get-StateById $tracked; if ($st -eq "running" -or $st -eq "pending") { $id = $tracked } }
  if ($id) { aws ec2 terminate-instances --instance-ids $id --region $Region | Out-Null; Write-Host "terminated: $id" }
}
function Test-Converged {
  # Run the convergence-stop helper against THIS run's S3 prefix. It diffs the two latest versioned
  # checkpoints (mean-L1 avg-strategy delta over shared infosets) and persists the below-threshold
  # streak in $ConvStateFile so the K-consecutive rule holds across babysitter rounds. Returns:
  #   "CONVERGED" | "CONTINUING" | "UNKNOWN" | "ERROR".
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
  # Echo the helper's machine-readable STATUS line for the run log (delta/streak/infosets visible).
  ($txt -split "`r?`n") | Where-Object { $_ -match "STATUS=" } | ForEach-Object { Write-Host "    conv: $_" }
  return $status
}
function Invoke-Launch {
  $argsList = @(
    "scripts\launch_native_cfr_training_spot.py",
    "--run-subdir", $Run, "--iters", $Iters, "--n-deals", $NDeals,
    "--num-buckets", $NumBuckets, "--checkpoint-every", $CheckpointEvery,
    "--instance-type", $InstanceType, "--arch", $Arch
  )
  if ($DepthEdges) { $argsList += @("--depth-edges", $DepthEdges) }
  if ($Spotpick)   { $argsList += "--spotpick" }
  if ($OnDemand)   { $argsList += "--on-demand" }
  if ($CrateSrc)   { $argsList += @("--crate-src", $CrateSrc) }  # upload from the FROZEN crate copy
  # FIRST launch ($launches=0) uploads the code tarball; RELAUNCHES reuse it so a locally-changing working
  # tree (e.g. a concurrent search-opt editing the crate) can't re-upload mid-edit and break a resumed box.
  if ($launches -ge 1) { $argsList += "--reuse-code" }
  Write-Host "  launching: python $($argsList -join ' ')"
  $out = (& python @argsList 2>&1)
  $out | ForEach-Object { Write-Host "    $_" }
  $txt = ($out | Out-String)
  # The native launcher prints "Launched i-xxxx as <type> (...)". (The eval-suite launcher prints
  # "LAUNCHED i-xxxx"; accept either so a copy/paste of the wrong launcher is still tracked.)
  if ($txt -match "(?i)launched\s+(i-[0-9a-f]+)") { return $Matches[1] }
  return ""
}

$capH = [int]($MAXSEC / 3600)
Write-Host "=== native-CFR babysitter: $Run  target=$Iters it, n_deals=$NDeals, buckets=$NumBuckets, depth='$DepthEdges' ==="
Write-Host "    caps: ${capH}h wall / $MaxLaunch launches / >=${MinRelaunchGapSec}s between launches"
Write-Host "    resume: a relaunch reuses --run-subdir '$Run' so the new box loads s3://$Bucket/$LATEST"

while ($true) {
  $el   = [int]((Get-Date) - $start).TotalSeconds
  $done = Get-LatestItersDone           # -1 = no ckpt yet; else iters_done in the S3 latest blob
  $live = Get-LiveBoxId
  $draining = ""   # a previous box still holding the spot vCPU quota (shutting-down/stopping)
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
  # 2b) CONVERGENCE STOP: when a NEW checkpoint has landed, diff the two latest versioned checkpoints;
  # if the avg-strategy delta has plateaued (<= ConvThreshold for ConvK consecutive checkpoints), STOP
  # early (terminate the box) instead of burning the rest of the iters/wall budget. Needs >= 2 ckpts.
  if (-not $NoConvergeStop -and $done -gt 0 -and $done -gt $lastConvIters) {
    $lastConvIters = $done
    $cstat = Test-Converged
    if ($cstat -eq "CONVERGED") {
      Write-Host "DONE: convergence stop -> avg strategy plateaued (delta <= $ConvThreshold for $ConvK ckpts) at $done iters. Saved the rest of the budget." -ForegroundColor Green
      $convergedStop = $true
      Stop-Run; break
    }
  }
  # 3) a box is live and not done -> just wait
  if ($live) { Start-Sleep -Seconds $PollSec; continue }

  # 3b) EXACT-QUOTA GUARD: the previous box is still terminating (shutting-down STILL counts against the
  # spot vCPU quota), so launching a new 128-vCPU box on a 128 limit would exceed it and fail. Wait for the
  # old box to fully terminate; the next loop relaunches into the freed quota. (Binds when quota == one
  # box's vCPUs, e.g. the just-acquired 128.)
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
  $reason = if ($done -ge 0) { "preempted at $done/$Iters -> resume from S3" } else { "no box + no ckpt -> initial launch" }
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

$final = Get-LatestItersDone
Write-Host ""
Write-Host "=== BABYSITTER EXIT after ${el}s, $launches launches. latest iters_done=$final / $Iters ==="
if ($convergedStop) {
  Write-Host "Run COMPLETE via CONVERGENCE STOP at $final iters (avg strategy plateaued; iters/wall caps NOT hit)." -ForegroundColor Green
  Write-Host "Snapshot the blueprint from the latest checkpoint:" -ForegroundColor Green
  Write-Host "  aws s3 cp s3://$Bucket/$LATEST runs\native_cfr\$Run.nlcfr --region $Region"
  Write-Host "  (then load via battery_native deserialize + NlheCFRTrainer.snapshot() -> TabularBlueprintAdapter)"
} elseif ($final -ge $Iters) {
  Write-Host "Run COMPLETE (iters backstop). Snapshot the blueprint from the latest checkpoint:" -ForegroundColor Green
  Write-Host "  aws s3 cp s3://$Bucket/$LATEST runs\native_cfr\$Run.nlcfr --region $Region"
  Write-Host "  (then load via battery_native deserialize + NlheCFRTrainer.snapshot() -> TabularBlueprintAdapter)"
} else {
  Write-Host "Run NOT complete (hit a cap or stall). The S3 latest checkpoint is preserved for a manual resume:" -ForegroundColor Yellow
  Write-Host "  python scripts\launch_native_cfr_training_spot.py --run-subdir $Run --iters $Iters --n-deals $NDeals --num-buckets $NumBuckets"
}
