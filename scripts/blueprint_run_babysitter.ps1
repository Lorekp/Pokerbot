# Autonomous babysitter for the FULL native CFR+ 6-max blueprint run WITH the throughput/convergence levers
# (scripts/launch_native_cfr_training_spot.py). Supersedes native_cfr_babysitter.ps1 by passing the
# verified levers the old one couldn't: --work-steal (+33% deals/s @64c, MEASURED), --dcfr (signed-regret
# Discounted-CFR + Regret-Based Pruning, ~2x fewer iters on Leduc), and --emd-frozen (the K160 production
# abstraction). All the proven machinery is kept: resume-from-S3-on-preemption, relaunch-storm guards,
# spot-quota guard, spotpick auto-pick, and the convergence STOP (stop at the avg-strategy plateau, not at
# the fixed --iters/wall cap).
#
# Run (your AWS creds; same boto3 default profile the launcher uses):
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint_run_babysitter.ps1 -Run bp_dcfr_1 -WorkSteal
#   # add -Dcfr once the 6-max DCFR cert confirms it; -NoConvergeStop to run to the fixed -Iters instead.
#
# Pairs with watch_blueprint_run.ps1 (read-only dashboard). ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
param(
  [string]$Run = "",                  # S3 run-subdir; REQUIRED for resume-across-preemption (auto-gen warns)
  [int]$Iters = 62000,                # --iters target (iter62000 = the LBR-unexploitable "superhuman" point;
                                      #   the convergence stop usually fires first ~iter48-50k)
  [int]$NDeals = 200,                 # --n-deals (the proven prod value; work_steal helps MORE at low n_deals)
  [int]$NumBuckets = 160,             # --num-buckets  (MUST equal the --emd-frozen table's K; K160 blob => 160)
  [string]$EmdFrozen = "data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn",  # "" = made-hand path
  [string]$DepthEdges = "",           # --depth-edges  ("" = non-depth blueprint, the deployed default)
  [int]$CheckpointEvery = 500,        # --checkpoint-every
  # --- LEVERS (the reason this script exists) ---
  [switch]$WorkSteal,                 # pass --work-steal (atomic deal-counter scheduler; +33% @64c measured)
  [switch]$Dcfr,                      # pass --dcfr (signed-regret DCFR + RBP pruning; gate on the 6-max cert)
  [double]$DcfrPruneThreshold = -300000.0,  # --dcfr-prune-threshold (RBP floor, bb regret units)
  [double]$DcfrExploreFrac = 0.05,    # --dcfr-explore-frac (fraction of iters that explore all actions)
  # --- box / market ---
  [string]$InstanceType = "",          # "" = AUTO-PICK cheapest spot-available type via spotpick
  [int]$Vcpus = 128,                   # auto-pick target vCPU count (128 = c6*.32xlarge class; RAYON pins to 64 phys)
  [string]$Arch = "x86_64",
  [string]$CrateSrc = "tools/battery_native",  # crate copy uploaded as tools/battery_native (point at a FROZEN
                                               # copy to isolate this run from concurrent live-crate edits)
  [switch]$Spotpick,                  # pass --spotpick (capacity-diverse fallback chain)
  [switch]$OnDemand,                  # pass --on-demand (no spot; no preemption, pricier)
  # --- storm + wall guards ---
  [int]$MaxLaunch = 12,               # total relaunch cap (storm guard)
  [int]$MinRelaunchGapSec = 600,      # min wall seconds between launches (storm guard)
  [int]$BootWaitSec = 300,            # wait after a launch before polling (boot grace, storm guard)
  [int]$PollSec = 180,                # state poll interval while a box is live
  [int]$MaxHours = 12,                # overall wall-clock cap (BACKSTOP; convergence stops earlier ~1.5-2h)
  [string]$Region = "eu-central-1",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr",
  # --- convergence-based stop (stop AT CONVERGENCE, not at the fixed --iters/wall cap) ---
  [switch]$NoConvergeStop,            # disable convergence stop; run to fixed -Iters / -MaxHours only
  [double]$ConvThreshold = 0.020,     # mean-L1 plateau floor. NB=200 made-hand plateaued at 0.018; a richer
                                      # EMD abstraction floors slightly higher -> 0.020 default. Read a few
                                      # -NoConvergeStop ckpts and set it where the per-step delta drop flattens.
  [int]$ConvK = 3,                    # consecutive below-threshold checkpoints required to declare CONVERGED
  [int]$ConvNumBuckets = 160,         # stop-helper ctor arg (irrelevant to load; match -NumBuckets for clarity)
  [string]$ConvDepthEdges = ""        # ctor depth-edges for the stop helper (irrelevant to load); CSV like "50,100"
)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

if (-not $Run) {
  $Run = "bp_{0}" -f ([int][double]::Parse((Get-Date -UFormat %s)))
  Write-Host "WARNING: no -Run given; generated '$Run'. Relaunch-on-preemption resumes THIS subdir." -ForegroundColor Yellow
}
if ($EmdFrozen -and -not (Test-Path $EmdFrozen)) {
  Write-Host "FATAL: -EmdFrozen '$EmdFrozen' not found. Pass the correct blob, or '' for the made-hand path." -ForegroundColor Red
  exit 1
}

$RUN_PREFIX = "$Prefix/$Run"
$LATEST     = "$RUN_PREFIX/checkpoints/latest.nlcfr"
$MAXSEC     = $MaxHours * 3600
$start      = Get-Date
$launches   = 0
$lastLaunch = [datetime]::MinValue
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
  # The trainer's ground truth: iters_done from the latest.nlcfr header (magic "NLCFR001"(8) + u64 LE(8)).
  $tmp = Join-Path $env:TEMP "bp_latest_probe.nlcfr"
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
    "--instance-type", $InstanceType, "--arch", $Arch
  )
  if ($EmdFrozen)  { $argsList += @("--emd-frozen", $EmdFrozen) }
  if ($DepthEdges) { $argsList += @("--depth-edges", $DepthEdges) }
  if ($WorkSteal)  { $argsList += "--work-steal" }
  if ($Dcfr)       { $argsList += @("--dcfr", "--dcfr-prune-threshold", $DcfrPruneThreshold,
                                    "--dcfr-explore-frac", $DcfrExploreFrac) }
  if ($Spotpick)   { $argsList += "--spotpick" }
  if ($OnDemand)   { $argsList += "--on-demand" }
  if ($CrateSrc)   { $argsList += @("--crate-src", $CrateSrc) }
  # FIRST launch uploads the code tarball (+ the 152MB EMD blob); RELAUNCHES reuse it so a locally-changing
  # working tree can't re-upload mid-edit and break a resumed box.
  if ($launches -ge 1) { $argsList += "--reuse-code" }
  Write-Host "  launching: python $($argsList -join ' ')"
  $out = (& python @argsList 2>&1)
  $out | ForEach-Object { Write-Host "    $_" }
  $txt = ($out | Out-String)
  if ($txt -match "(?i)launched\s+(i-[0-9a-f]+)") { return $Matches[1] }
  return ""
}

$capH = [int]($MAXSEC / 3600)
$leverTxt = (@(if ($WorkSteal) { "work_steal" }; if ($Dcfr) { "dcfr" }) -join "+"); if (-not $leverTxt) { $leverTxt = "NONE (baseline)" }
$absTxt = if ($EmdFrozen) { "EMD K$NumBuckets ($([System.IO.Path]::GetFileName($EmdFrozen)))" } else { "made-hand K$NumBuckets" }
Write-Host "=== blueprint babysitter: $Run ===" -ForegroundColor Cyan
Write-Host "    target=$Iters it  n_deals=$NDeals  abstraction=$absTxt  depth='$DepthEdges'"
Write-Host "    LEVERS=$leverTxt" -ForegroundColor Green
Write-Host "    caps: ${capH}h wall / $MaxLaunch launches / >=${MinRelaunchGapSec}s between launches"
Write-Host "    convergence-stop=$(-not $NoConvergeStop) (threshold=$ConvThreshold, K=$ConvK)"
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
  # 2b) CONVERGENCE STOP: on a NEW checkpoint, diff the two latest; STOP early if the avg-strategy delta
  # has plateaued (<= ConvThreshold for ConvK consecutive checkpoints). Needs >= 2 ckpts.
  if (-not $NoConvergeStop -and $done -gt 0 -and $done -gt $lastConvIters) {
    $lastConvIters = $done
    $cstat = Test-Converged
    if ($cstat -eq "CONVERGED") {
      Write-Host "DONE: convergence stop -> avg strategy plateaued (delta <= $ConvThreshold for $ConvK ckpts) at $done iters. Saved the rest of the budget." -ForegroundColor Green
      $convergedStop = $true
      Stop-Run; break
    }
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
} elseif ($final -ge $Iters) {
  Write-Host "Run COMPLETE (iters backstop)." -ForegroundColor Green
} else {
  Write-Host "Run NOT complete (hit a cap or stall). The S3 latest checkpoint is preserved for a manual resume:" -ForegroundColor Yellow
  $resume = "python scripts\launch_native_cfr_training_spot.py --run-subdir $Run --iters $Iters --n-deals $NDeals --num-buckets $NumBuckets"
  if ($EmdFrozen) { $resume += " --emd-frozen $EmdFrozen" }
  if ($WorkSteal) { $resume += " --work-steal" }
  if ($Dcfr)      { $resume += " --dcfr" }
  Write-Host "  $resume"
}
if ($final -ge 0 -and ($convergedStop -or $final -ge $Iters)) {
  Write-Host "Snapshot the blueprint from the latest checkpoint:" -ForegroundColor Green
  Write-Host "  aws s3 cp s3://$Bucket/$LATEST runs\native_cfr\$Run.nlcfr --region $Region"
}
