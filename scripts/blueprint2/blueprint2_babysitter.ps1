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
  # --- LEVERS ---
  [switch]$WorkSteal,                 # pass --work-steal (+33% deals/s @64c measured; pass it)
  [switch]$Dcfr,                      # NOT recommended (CFR+ > DCFR at iter16000 for 6-max)
  [double]$DcfrPruneThreshold = -300000.0,
  [double]$DcfrExploreFrac = 0.05,
  # --- box / market ---
  [string]$InstanceType = "",          # "" = AUTO-PICK cheapest spot-available type via spotpick
  [int]$Vcpus = 128,
  [string]$Arch = "x86_64",
  [string]$CrateSrc = "tools/battery_native",  # the LIVE crate carries river_ochs.rs (the frozen copy at
                                               # runs/bp_crate_frozen does NOT); freeze a new copy first if
                                               # concurrent crate edits are expected during the run
  [switch]$Spotpick,
  [switch]$OnDemand,
  # --- storm + wall guards (verbatim from the proven babysitter) ---
  [int]$MaxLaunch = 12,
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
  $tmp = Join-Path $env:TEMP "bp2_latest_probe.nlcfr"
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
    "--max-hours", ($MaxHours + 1),
    "--instance-type", $InstanceType, "--arch", $Arch
  )
  if ($EmdFrozen)  { $argsList += @("--emd-frozen", $EmdFrozen) }
  if ($RiverOchs)  { $argsList += @("--river-ochs", $RiverOchs) }
  if ($DepthEdges) { $argsList += @("--depth-edges", $DepthEdges) }
  if ($WorkSteal)  { $argsList += "--work-steal" }
  if ($Dcfr)       { $argsList += @("--dcfr", "--dcfr-prune-threshold", $DcfrPruneThreshold,
                                    "--dcfr-explore-frac", $DcfrExploreFrac) }
  if ($Spotpick)   { $argsList += "--spotpick" }
  if ($OnDemand)   { $argsList += "--on-demand" }
  if ($CrateSrc)   { $argsList += @("--crate-src", $CrateSrc) }
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
Write-Host "=== blueprint-v2 babysitter: $Run ===" -ForegroundColor Cyan
Write-Host "    target=$Iters it  n_deals=$NDeals  flop/turn=EMD K$NumBuckets  river=OCHS ($([System.IO.Path]::GetFileName($RiverOchs)))"
Write-Host "    LEVERS=$leverTxt" -ForegroundColor Green
Write-Host "    caps: ${capH}h wall / $MaxLaunch launches / >=${MinRelaunchGapSec}s between launches"
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
  Write-Host "Run COMPLETE via CONVERGENCE STOP at $final iters (>= $ConvMinIters floor)." -ForegroundColor Green
} elseif ($final -ge $Iters) {
  Write-Host "Run COMPLETE (iters backstop)." -ForegroundColor Green
} else {
  Write-Host "Run NOT complete (hit a cap or stall). The S3 latest checkpoint is preserved for a manual resume:" -ForegroundColor Yellow
  $resume = "python scripts\launch_native_cfr_training_spot.py --run-subdir $Run --iters $Iters --n-deals $NDeals --num-buckets $NumBuckets"
  if ($EmdFrozen) { $resume += " --emd-frozen $EmdFrozen" }
  if ($RiverOchs) { $resume += " --river-ochs $RiverOchs" }
  if ($WorkSteal) { $resume += " --work-steal" }
  Write-Host "  $resume"
}
if ($final -ge 0 -and ($convergedStop -or $final -ge $Iters)) {
  Write-Host "Download the final checkpoint, then follow README_LAUNCH.md 'AFTER THE RUN':" -ForegroundColor Green
  Write-Host "  aws s3 cp s3://$Bucket/$LATEST runs\native_cfr\$Run.nlcfr --region $Region"
}
