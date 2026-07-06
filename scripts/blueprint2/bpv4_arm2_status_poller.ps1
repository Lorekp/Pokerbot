# Read-only rich-status sidecar for the BLUEPRINT-V4 arm-2 retry run (bpv4_h200_arm2).
# Runs ALONGSIDE the existing dumb retry supervisor (scripts\blueprint2\bpv4_arm2_retry.ps1)
# and its child babysitter -- it launches nothing, terminates nothing, and never mutates S3/EC2.
# Every poll it appends ONE structured status line (+ occasional EVENT lines) to the rich pane
# artifacts\_arm2_status.log .
#   NOTE: the legacy pane artifacts\_arm2_retry.log is held under an EXCLUSIVE write lock by the
#   still-running old retry supervisor (its stdout redirection) for that process's whole lifetime,
#   so a read-only sidecar CANNOT append there without killing that process. Rich status therefore
#   goes to _arm2_status.log; watch it with:  Get-Content artifacts\_arm2_status.log -Wait -Tail 40
#   If the old retry loop is ever stopped, point -OutLog back at _arm2_retry.log for one pane.
#
# Data sources (ALL list/get/describe only):
#   * aws s3api list-objects-v2  -> checkpoint marker keys + LastModified + Size (pace, blob GB, ETA)
#   * aws s3 cp logs/train.log   -> grep REGSTATS/COVERAGE at gate checkpoints (K0 band print)
#   * aws ec2 describe-instances -> live box id/type/state/age + reclaim lifetimes per pool
#   * local Get-CimInstance      -> is the babysitter child process still alive?
#
# Deployment = SIDECAR (observability without touching the fragile spot lifecycle). See report.
# Stop: create artifacts\_arm2_poller.stop . Hard wall: -MaxHours (default 26, outlives the 24h retry).
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252). No AWS creds are ever printed.
#
# Run (repo root, separate terminal):
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\bpv4_arm2_status_poller.ps1
param(
  [int]$IntervalSec = 240,          # ~4 min status cadence
  [int]$MaxHours    = 26,
  [string]$Run      = "bpv4_h200_arm2",
  [string]$Bucket   = "video-processing-6789",
  [string]$Region   = "eu-central-1",
  [double]$SpotUsdPerHr = 1.9,      # ~32xl spot; est only
  [int]$StallMin    = 25,           # box alive but no new marker for > this => stall alarm
  [string]$OutLog   = "artifacts\_arm2_status.log"  # rich pane; _arm2_retry.log is lock-held by old loop
)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..")

$Prefix   = "pokerbot/prod6000/native_cfr/$Run"
$CkptPfx  = "$Prefix/checkpoints/"
$LogUrl   = "s3://$Bucket/$Prefix/logs/train.log"
$StopFile = "artifacts\_arm2_poller.stop"
$StateFile= "artifacts\_arm2_poller_state.json"
$TmpLog   = "artifacts\_arm2_poller_trainlog.tmp"

# Pre-registered gate ladder (amendment sec 11): name -> iter.
$Gates = [ordered]@{ "K0@5000"=5000; "K1@20000"=20000; "K2@30000"=30000 }
$K0Lo = 234; $K0Hi = 2344   # REGSTATS neg_p999 pass band (bb) at K0.
$Target = 40000

function Write-Line([string]$prefix, [string]$msg) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $txt = "[" + $prefix + " " + $ts + "] " + $msg
  for ($try=0; $try -lt 5; $try++) {
    try { Add-Content -Path $OutLog -Value $txt -Encoding ascii -ErrorAction Stop; return }
    catch { Start-Sleep -Milliseconds 400 }
  }
}
function WL([string]$msg)   { Write-Line "poll"  $msg }
function WEVT([string]$msg) { Write-Line "EVENT" $msg }

# Persisted state so a poller restart does not re-fire events.
$state = @{
  firedGates      = @()      # gate names already announced
  lastMarkerIter  = -1
  lastMarkerEpoch = 0        # unix seconds of last marker LastModified we announced
  knownBoxes      = @{}      # id -> @{ type=; launchEpoch=; state= }
  deadLifetimes   = @{}      # familyKey -> array of minutes (most recent last)
  cumInstEpoch    = @{}      # id -> last-seen epoch (for spend accumulation of running boxes)
  cumHoursDead    = 0.0      # accumulated instance-hours of boxes we saw then lost
}
if (Test-Path $StateFile) {
  try {
    $loaded = Get-Content $StateFile -Raw | ConvertFrom-Json
    if ($loaded) {
      if ($loaded.firedGates)     { $state.firedGates     = @($loaded.firedGates) }
      if ($loaded.lastMarkerIter) { $state.lastMarkerIter = [int64]$loaded.lastMarkerIter }
      if ($loaded.lastMarkerEpoch){ $state.lastMarkerEpoch= [int64]$loaded.lastMarkerEpoch }
      if ($loaded.cumHoursDead)   { $state.cumHoursDead   = [double]$loaded.cumHoursDead }
      # knownBoxes / deadLifetimes / cumInstEpoch rehydrate as hashtables
      if ($loaded.knownBoxes) { foreach ($p in $loaded.knownBoxes.PSObject.Properties) { $state.knownBoxes[$p.Name] = @{ type=$p.Value.type; launchEpoch=[int64]$p.Value.launchEpoch; state=$p.Value.state } } }
      if ($loaded.deadLifetimes) { foreach ($p in $loaded.deadLifetimes.PSObject.Properties) { $state.deadLifetimes[$p.Name] = @($p.Value) } }
      if ($loaded.cumInstEpoch) { foreach ($p in $loaded.cumInstEpoch.PSObject.Properties) { $state.cumInstEpoch[$p.Name] = [int64]$p.Value } }
    }
  } catch { }
}
function Save-State {
  try { ($state | ConvertTo-Json -Depth 6) | Set-Content -Path $StateFile -Encoding ascii } catch { }
}

function Now-Epoch { [int64]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds()) }
function Iso-Epoch([string]$iso) {
  try { return [int64]([DateTimeOffset]::Parse($iso).ToUnixTimeSeconds()) } catch { return 0 }
}
function Family([string]$itype) {
  # c6id.32xlarge -> c6id ; group reclaim lifetimes by pool family.
  if ($itype -match "^([a-z0-9]+)\.") { return $Matches[1] } else { return $itype }
}

# Query checkpoint markers: returns @{ iter=; epoch=; blobGB=; count=; pace=; recent=@(...) }
function Get-CkptInfo {
  $raw = aws s3api list-objects-v2 --bucket $Bucket --prefix $CkptPfx `
           --query "Contents[].{K:Key,S:Size,M:LastModified}" --output json --region $Region 2>$null
  $info = @{ iter=[int64]-1; epoch=0; blobGB=0.0; count=0; pace=0.0; markers=@() }
  if (-not $raw) { return $info }
  $items = $null
  try { $items = $raw | ConvertFrom-Json } catch { return $info }
  if (-not $items) { return $info }
  $markers = @()   # @{ iter=; epoch= } for each .ok marker
  $bigBlobGB = 0.0
  foreach ($o in $items) {
    if ($o.K -match "iter_(\d+)\.(nlcfr|nldlt)\.ok$") {
      $markers += @{ iter=[int64]$Matches[1]; epoch=(Iso-Epoch $o.M) }
    } elseif ($o.K -match "iter_(\d+)\.nlcfr$") {
      $gb = [double]$o.S / 1GB
      if ($gb -gt $bigBlobGB) { $bigBlobGB = $gb }
    }
  }
  if ($markers.Count -eq 0) { return $info }
  $markers = @($markers | Sort-Object { $_.iter })
  $last = $markers[-1]
  $info.iter   = $last.iter
  $info.epoch  = $last.epoch
  $info.blobGB = [math]::Round($bigBlobGB,1)
  $info.count  = $markers.Count
  $info.markers= $markers
  # pace iters/hr from last few markers (spanning up to 4 intervals, key-names + LastModified only)
  if ($markers.Count -ge 2) {
    $n = [Math]::Min(5, $markers.Count)
    $a = $markers[$markers.Count - $n]
    $b = $last
    $dIter = $b.iter - $a.iter
    $dSec  = $b.epoch - $a.epoch
    if ($dSec -gt 0 -and $dIter -gt 0) { $info.pace = [math]::Round($dIter * 3600.0 / $dSec, 1) }
  }
  return $info
}

# EC2 boxes for this run. Returns array of @{ id=; type=; state=; launchEpoch= }
function Get-Boxes {
  $raw = aws ec2 describe-instances --region $Region --filters "Name=tag:Run,Values=$Run" `
           --query "Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,State:State.Name,Launch:LaunchTime}" `
           --output json 2>$null
  $out = @()
  if (-not $raw) { return $out }
  $items = $null
  try { $items = $raw | ConvertFrom-Json } catch { return $out }
  if (-not $items) { return $out }
  foreach ($b in $items) {
    $out += @{ id=$b.Id; type=$b.Type; state=$b.State; launchEpoch=(Iso-Epoch $b.Launch) }
  }
  return $out
}

# Is the babysitter child process still alive?
function Babysitter-Alive {
  try {
    $procs = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction Stop
    foreach ($p in $procs) {
      if ($p.CommandLine -and ($p.CommandLine -match "bpv4_arm2_babysitter\.ps1" -or $p.CommandLine -match "blueprint2_babysitter\.ps1")) { return $true }
    }
  } catch { }
  return $false
}

# Fetch train.log and grep the REGSTATS neg_p999 for K0 announcement. Returns string or $null.
function Get-Regstats {
  try {
    aws s3 cp $LogUrl $TmpLog --region $Region 2>$null | Out-Null
    if (-not (Test-Path $TmpLog)) { return $null }
    $lines = Get-Content $TmpLog -ErrorAction SilentlyContinue
    $hit = $lines | Select-String -Pattern "REGSTATS" | Select-Object -Last 1
    if ($hit) { return $hit.Line.Trim() }
  } catch { }
  return $null
}

WEVT "sidecar status poller ONLINE (read-only) run=$Run region=$Region interval=${IntervalSec}s; supervising existing retry loop + child babysitter WITHOUT touching lifecycle."

# One-time bootstrap: seed reclaim-cadence history from boxes ALREADY terminated before we started,
# so the pool-lifetime signal is populated on the very first status line (we cannot time a
# transition we never observed; use launch->now as a lower-bound estimate, marked ~).
if (($state.deadLifetimes.Keys.Count -eq 0) -and ($state.knownBoxes.Keys.Count -eq 0)) {
  $bootBoxes = Get-Boxes
  $bootNow = Now-Epoch
  $seed = @($bootBoxes | Where-Object { $_.state -eq "terminated" -or $_.state -eq "shutting-down" } | Sort-Object { $_.launchEpoch })
  foreach ($b in $seed) {
    $fam = Family $b.type
    $lifeMin = [math]::Round(($bootNow - $b.launchEpoch)/60.0)
    if (-not $state.deadLifetimes.ContainsKey($fam)) { $state.deadLifetimes[$fam] = @() }
    $state.deadLifetimes[$fam] = @($state.deadLifetimes[$fam]) + $lifeMin
    $state.knownBoxes[$b.id] = @{ type=$b.type; launchEpoch=$b.launchEpoch; state=$b.state }
  }
  if ($seed.Count -gt 0) {
    $famList = @()
    foreach ($k in $state.deadLifetimes.Keys) { $famList += ($k + ":" + (@($state.deadLifetimes[$k]) -join "/")) }
    WEVT ("bootstrap: seeded reclaim history (launch->now lower bounds, ~) for " + $seed.Count + " pre-terminated boxes -> " + ($famList -join "  "))
  }
}

$deadline = (Get-Date).AddHours($MaxHours)
$cycleGuess = 0

while ((Get-Date) -lt $deadline) {
  if (Test-Path $StopFile) { WEVT "poller stop sentinel -> exit"; break }

  $nowEp = Now-Epoch
  $ck    = Get-CkptInfo
  $boxes = Get-Boxes
  $baby  = Babysitter-Alive

  # ---- reconcile boxes: detect launches, reclaims, accumulate spend ----
  $liveBox = $null
  $seenNow = @{}
  foreach ($b in $boxes) {
    $seenNow[$b.id] = $true
    if ($b.state -eq "running" -or $b.state -eq "pending") { $liveBox = $b }
    if (-not $state.knownBoxes.ContainsKey($b.id)) {
      # newly observed box
      $state.knownBoxes[$b.id] = @{ type=$b.type; launchEpoch=$b.launchEpoch; state=$b.state }
      if ($b.state -eq "running" -or $b.state -eq "pending") {
        WEVT ("box LAUNCHED " + $b.id + " (" + $b.type + ") state=" + $b.state)
        $cycleGuess++
      }
    } else {
      $prev = $state.knownBoxes[$b.id].state
      if ($prev -ne $b.state) {
        if (($b.state -eq "shutting-down" -or $b.state -eq "terminated") -and ($prev -eq "running" -or $prev -eq "pending" -or $prev -eq "shutting-down")) {
          if ($prev -ne "shutting-down") {
            $lifeMin = [math]::Round(($nowEp - $b.launchEpoch)/60.0)
            $fam = Family $b.type
            if (-not $state.deadLifetimes.ContainsKey($fam)) { $state.deadLifetimes[$fam] = @() }
            $state.deadLifetimes[$fam] = @($state.deadLifetimes[$fam]) + $lifeMin
            $state.cumHoursDead += ($nowEp - $b.launchEpoch)/3600.0
            WEVT ("box RECLAIMED " + $b.id + " (" + $b.type + ") lifetime=" + $lifeMin + "min -> pool " + $fam + " now needs a relaunch")
          }
        }
        $state.knownBoxes[$b.id].state = $b.state
      }
    }
  }

  # ---- checkpoint / marker events ----
  if ($ck.iter -ge 0 -and $ck.iter -gt $state.lastMarkerIter) {
    $sincePrev = ""
    if ($state.lastMarkerEpoch -gt 0 -and $ck.epoch -gt 0) {
      $mm = [math]::Round(($ck.epoch - $state.lastMarkerEpoch)/60.0,1)
      $sincePrev = " (+" + $mm + "min since prev marker)"
    }
    WEVT ("checkpoint LANDED iter=" + $ck.iter + " blob=" + $ck.blobGB + "GB" + $sincePrev)
    $state.lastMarkerIter  = $ck.iter
    $state.lastMarkerEpoch = $ck.epoch

    # gate-checkpoint reached?
    foreach ($gname in $Gates.Keys) {
      $giter = $Gates[$gname]
      if ($ck.iter -ge $giter -and ($state.firedGates -notcontains $gname)) {
        $state.firedGates = @($state.firedGates) + $gname
        if ($gname -eq "K0@5000") {
          $rs = Get-Regstats
          if ($rs) {
            WEVT ("GATE " + $gname + " REACHED. REGSTATS: " + $rs + "  [K0 pass band neg_p999 in [" + $K0Lo + "," + $K0Hi + "] bb]")
          } else {
            WEVT ("GATE " + $gname + " REACHED (iter " + $ck.iter + "). REGSTATS not yet in train.log; band=[" + $K0Lo + "," + $K0Hi + "] bb -- check logs/train.log.")
          }
        } else {
          WEVT ("GATE " + $gname + " REACHED (iter " + $ck.iter + "). Run the pre-registered kill-gate for " + $gname + ".")
        }
      }
    }
  }

  # ---- stall alarm: box alive but no fresh marker for > StallMin ----
  $stallNote = ""
  if ($liveBox -and $state.lastMarkerEpoch -gt 0) {
    $ageMarkerMin = [math]::Round(($nowEp - $state.lastMarkerEpoch)/60.0)
    $boxAgeMin    = [math]::Round(($nowEp - $liveBox.launchEpoch)/60.0)
    # only alarm once box is old enough to have produced a marker (first ckpt ~ many min in)
    if ($ageMarkerMin -gt $StallMin -and $boxAgeMin -gt $StallMin) {
      $stallNote = " STALL?"
      WEVT ("STALL ALARM: box " + $liveBox.id + " alive " + $boxAgeMin + "min but no new marker for " + $ageMarkerMin + "min (> " + $StallMin + "). Check: aws ec2 get-console-output --instance-id " + $liveBox.id + " --region " + $Region + " --latest ; and logs/train.log tail for a crash/OOM/row-budget abort.")
    }
  }

  # ---- ETA math ----
  $etaGate = "n/a"; $eta40k = "n/a"
  if ($ck.pace -gt 0 -and $ck.iter -ge 0) {
    # next unreached gate
    $nextG = $null; $nextGIter = 0
    foreach ($gname in $Gates.Keys) { if ($ck.iter -lt $Gates[$gname]) { $nextG = $gname; $nextGIter = $Gates[$gname]; break } }
    if ($nextG) {
      $hrsG = [math]::Round(($nextGIter - $ck.iter) / $ck.pace, 1)
      $etaGate = $nextG + " in ~" + $hrsG + "h"
    } else { $etaGate = "all K-gates passed" }
    if ($ck.iter -lt $Target) {
      $hrs40 = [math]::Round(($Target - $ck.iter) / $ck.pace, 1)
      $eta40k = "~" + $hrs40 + "h"
    } else { $eta40k = "DONE" }
  }

  # ---- spend estimate (running boxes' live age + accumulated dead hours) ----
  $liveHours = 0.0
  if ($liveBox) { $liveHours = ($nowEp - $liveBox.launchEpoch)/3600.0 }
  $spendUsd = [math]::Round(($state.cumHoursDead + $liveHours) * $SpotUsdPerHr, 2)

  # ---- reclaim cadence for the live pool (main strategic signal) ----
  $reclaimNote = ""
  if ($liveBox) {
    $fam = Family $liveBox.type
    if ($state.deadLifetimes.ContainsKey($fam) -and @($state.deadLifetimes[$fam]).Count -gt 0) {
      $lt = @($state.deadLifetimes[$fam])
      $last3 = $lt[[Math]::Max(0,$lt.Count-3)..($lt.Count-1)]
      $reclaimNote = "; pool " + $fam + " reclaims (last3 min): " + ($last3 -join "/")
    }
  }

  # ---- box descriptor ----
  $boxStr = "(none)"
  if ($liveBox) {
    $ageMin = [math]::Round(($nowEp - $liveBox.launchEpoch)/60.0)
    $boxStr = $liveBox.id + " " + $liveBox.type + " " + $liveBox.state + " age=" + $ageMin + "min"
  }
  $ckAgeMin = "?"
  if ($ck.epoch -gt 0) { $ckAgeMin = [string][math]::Round(($nowEp - $ck.epoch)/60.0) }

  $cool = ""
  if (-not $liveBox -and $baby) { $cool = " | supervisor idle (no live box: reclaim gap or cooldown)" }

  $line = "cyc~" + $cycleGuess + " | baby=" + $(if($baby){"alive"}else{"DOWN"}) +
          " | box: " + $boxStr +
          " | ckpt: iter=" + $ck.iter + " age=" + $ckAgeMin + "min blob=" + $ck.blobGB + "GB (" + $ck.count + " markers)" +
          " | pace=" + $ck.pace + " it/hr" +
          " | ETA " + $etaGate + " ; 40k " + $eta40k +
          " | spend~$" + $spendUsd + " (est)" +
          $reclaimNote + $cool + $stallNote
  WL $line

  if (-not $baby -and $liveBox) {
    WEVT "NOTE: no babysitter process found but a box is live -- retry supervisor may have exited/errored while a box runs. Verify scripts\blueprint2\bpv4_arm2_retry.ps1 (pid file artifacts\_arm2_retry.pid)."
  }

  Save-State
  # sleep in 30s slices so the stop sentinel is honored promptly
  $slept = 0
  while ($slept -lt $IntervalSec) {
    if (Test-Path $StopFile) { break }
    Start-Sleep -Seconds 30; $slept += 30
  }
}
WEVT "sidecar status poller OFFLINE at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
