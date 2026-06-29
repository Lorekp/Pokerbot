# At-a-glance dashboard for the 5-PHASE EMD BLUEPRINT PIPELINE run. READ-ONLY.
# V2 = the post-Phase-2 dashboard + pluribus-grade auto-activate gate panels.
#
#   powershell -ExecutionPolicy Bypass -File scripts\watch_emd_pipeline_v2.ps1 -Run run_emd_bp1 -Iters 250000
#   powershell -ExecutionPolicy Bypass -File scripts\watch_emd_pipeline_v2.ps1 -Run run_emd_k500_pluribus_20260625 -Iters 320000 -NumBuckets 500 -Centroids data\abstraction\emd_buckets_500.json -ShipName emd_k500_pluribus
#
# V2 ADDITIONS vs. watch_emd_pipeline.ps1:
#   - PRE-LAUNCH SANITY CHECKS (BEFORE the main while loop):
#       (1) K-cap: if -Centroids is provided and the centroids' K > 200, exit 2 with a red FATAL.
#           K=500 is the known overfit (real SB -140 regression) and is BANNED for production auto-activate.
#       (2) SHA256 sidecar: if a frozen blob path is inferable, the .sha256 sidecar MUST exist. If missing,
#           exit 2 with a red FATAL directing the user to re-run precompute_emd_frozen_lookup.py.
#   - SHA256 status line (green when stamped, red WARNING when missing on every refresh cycle).
#   - K-cap status line (green when K<=200, red WARNING when K=500).
#   - Auto-activate gate status (when EVALED): "PASS" green or "FAIL: <reasons>" red.
#   - Verdict rendered via scripts/_emd_verdict_sign.py render (single source of truth).
#
# Shows the CURRENT PHASE (BUILD/PRECOMPUTE/TRAIN/SHIP/EVAL) + per-phase progress, the markers, and -- once
# EVAL finishes -- the verified-better VERDICT (per-seat blind LBR emd vs iter62000 + emd_better) and the
# ACTIVATION status.
# ASCII ONLY (PS 5.1 cp1252).
param(
  [string]$Run = "run_emd_bp1",
  [int]$Iters = 250000,
  [int]$NDeals = 0,                   # optional: lets TRAIN display sampled-deals/sec, not only raw it/s
  [int]$NumBuckets = 0,               # optional: validates against the centroids K and displays config
  [string]$Centroids = "",            # optional: local EMD centroids JSON; enables exact table-id/frozen lookup
  [int]$NRunouts = 200,               # table-id component; normally matches the centroids' n_runouts
  [int]$Refresh = 45,
  [double]$SpotPrice = 1.88,           # fallback spot $/hr if spotpick is unavailable
  [int]$StallMin = 25,                 # alarm if no new checkpoint AND no precompute heartbeat in this many min
  [double]$ExpectTableMB = 160,        # rough expected frozen-table size (for the precompute % bar)
  [string]$ShipName = "emd_bp1",       # staged serve snapshot name under {run}/serve/
  [switch]$StopAfterShip,              # Step-3-only: SHIPPED is terminal; eval is not expected
  [string]$Region = "eu-central-1",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr"
)
$RUN_PREFIX  = "$Prefix/$Run"
$CKPT_PREFIX = "$RUN_PREFIX/checkpoints/"
$LATEST      = "$RUN_PREFIX/checkpoints/latest.nlcfr"
$CLOUDLOG    = "$RUN_PREFIX/logs/cloud-init-output.log"  # only lands at box exit; live phase comes from console
$PRECOMPLOG  = "$RUN_PREFIX/logs/precompute.log"
$TRAINLOG    = "$RUN_PREFIX/logs/train.log"
$TRAINSTATUS = "$RUN_PREFIX/progress/train_status.txt"
$SHIPLOG     = "$RUN_PREFIX/logs/ship.log"
$EVALEMDLOG  = "$RUN_PREFIX/logs/eval_emd.log"
$EVALBASELOG = "$RUN_PREFIX/logs/eval_iter62000.log"
$EVALCOMBINELOG = "$RUN_PREFIX/logs/eval_combine.log"
$EVALEMDPART = "$RUN_PREFIX/eval/emd_partial.json"
$EVALBASEPART = "$RUN_PREFIX/eval/iter62000_partial.json"
$SHIPART_KEY = "$RUN_PREFIX/serve/$ShipName.pkl"
$VERDICT_KEY = "$RUN_PREFIX/eval/verdict.json"
$MARK_PREFIX = "$RUN_PREFIX/markers"
$M_TABLE     = "$MARK_PREFIX/table_cached"
$M_CONVERGED = "$MARK_PREFIX/CONVERGED"
$M_SHIPPED   = "$MARK_PREFIX/SHIPPED"
$M_EVALED    = "$MARK_PREFIX/EVALED"

# ============================================================================================
# V2 GATE CONSTANTS (must match scripts/emd_pipeline_babysitter_v2.ps1)
# ============================================================================================
$KMax         = 200
$AnchorLo     = 110.0
$AnchorHi     = 145.0
$TargetAnchor = 125.0
$SBMin        = 15.0
$BBMin        = 15.0
$BlindMeanMin = 15.0
$GlobalRegMax = 10.0

$t0 = Get-Date
$lastProgressKey = -1
$lastHeartTime = Get-Date

function Bar([double]$frac, [int]$width = 40) {
  if ($frac -lt 0) { $frac = 0 }; if ($frac -gt 1) { $frac = 1 }
  $d = [int]([math]::Round($frac * $width))
  return ("#" * $d) + ("-" * ($width - $d))
}
function Get-LatestItersDone {
  $tmp = Join-Path $env:TEMP "emd_watch_v2_probe.nlcfr"
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
function Get-S3Text([string]$key) {
  # Fetch a (possibly-not-yet-existing) S3 text object to stdout; "" if absent.
  $txt = (aws s3 cp "s3://$Bucket/$key" - --region $Region 2>$null | Out-String)
  return $txt
}
function Get-S3ObjectMeta([string]$key) {
  $meta = [ordered]@{ Exists = $false; Bytes = [int64]-1; LastModified = $null; AgeSec = -1 }
  $json = (aws s3api head-object --bucket $Bucket --key $key --region $Region `
    --query "{ContentLength:ContentLength,LastModified:LastModified}" --output json 2>$null | Out-String)
  if (-not $json.Trim()) { return [pscustomobject]$meta }
  try {
    $d = $json | ConvertFrom-Json
    $meta.Exists = $true
    $meta.Bytes = [int64]$d.ContentLength
    $meta.LastModified = [datetimeoffset]::Parse([string]$d.LastModified).UtcDateTime
    $meta.AgeSec = [int](((Get-Date).ToUniversalTime() - $meta.LastModified).TotalSeconds)
  } catch {}
  return [pscustomobject]$meta
}
function Fmt-Age([object]$meta) {
  if (-not $meta -or -not $meta.Exists -or -not $meta.LastModified) { return "(missing)" }
  $sec = [Math]::Max(0, [int]$meta.AgeSec)
  if ($sec -lt 120) { return ("{0}s old" -f $sec) }
  if ($sec -lt 7200) { return ("{0}m old" -f [int]($sec / 60)) }
  return ("{0:N1}h old" -f ($sec / 3600.0))
}
function Get-TrainProgressRecord([string]$source, [string]$txt, [object]$meta) {
  if (-not $txt) { return $null }
  $pat = "(TRAIN_HEARTBEAT|PROGRESS)\s+(?:ts=\S+\s+)?iters_done=(\d+)/(\d+)\s+infosets=(\d+).*?rate=([0-9.]+)\s*it/s(?:.*?eta=([+\-0-9.]+)s)?"
  $m = [regex]::Matches($txt, $pat)
  if (-not $m.Count) { return $null }
  $x = $m[$m.Count - 1]
  $eta = -1.0
  if ($x.Groups[6].Success) { $eta = [double]$x.Groups[6].Value }
  return [pscustomobject]@{
    Source = $source
    Kind = $x.Groups[1].Value
    Iters = [int64]$x.Groups[2].Value
    Target = [int64]$x.Groups[3].Value
    Infosets = [int64]$x.Groups[4].Value
    Rate = [double]$x.Groups[5].Value
    EtaSec = $eta
    Meta = $meta
  }
}
function Fmt-ProgressRecord([object]$rec) {
  if (-not $rec) { return "(none)" }
  $age = if ($rec.Meta) { Fmt-Age $rec.Meta } else { "console age unknown" }
  return ("{0} {1} @ {2:N3} it/s ({3})" -f $rec.Kind, $rec.Iters, $rec.Rate, $age)
}
function Test-MetaNewer([object]$a, [object]$b, [int]$slopSec = 5) {
  if (-not $a -or -not $a.Exists -or -not $a.LastModified) { return $false }
  if (-not $b -or -not $b.Exists -or -not $b.LastModified) { return $true }
  return ($a.LastModified -gt $b.LastModified.AddSeconds($slopSec))
}
function Get-EC2CpuUtilization([string]$iid) {
  if (-not $iid) { return $null }
  $end = (Get-Date).ToUniversalTime()
  $start = $end.AddMinutes(-20)
  $json = (aws cloudwatch get-metric-statistics --region $Region --namespace AWS/EC2 `
    --metric-name CPUUtilization --dimensions "Name=InstanceId,Value=$iid" `
    --statistics Average Maximum --period 300 --start-time $($start.ToString("o")) --end-time $($end.ToString("o")) `
    --output json 2>$null | Out-String)
  if (-not $json.Trim()) { return $null }
  try {
    $d = @((($json | ConvertFrom-Json).Datapoints))
    if (-not $d.Count) { return $null }
    $latest = $d | Sort-Object -Property Timestamp | Select-Object -Last 1
    $ts = [datetimeoffset]::Parse([string]$latest.Timestamp).UtcDateTime
    return [pscustomobject]@{
      Average = [double]$latest.Average
      Maximum = [double]$latest.Maximum
      Timestamp = $ts
      AgeSec = [int](($end - $ts).TotalSeconds)
      PeriodSec = 300
    }
  } catch { return $null }
}
function Fmt-Bytes([int64]$bytes) {
  if ($bytes -lt 0) { return "(none)" }
  if ($bytes -ge 1GB) { return ("{0:N2} GB" -f ($bytes / 1GB)) }
  if ($bytes -ge 1MB) { return ("{0:N1} MB" -f ($bytes / 1MB)) }
  if ($bytes -ge 1KB) { return ("{0:N1} KB" -f ($bytes / 1KB)) }
  return ("{0} B" -f $bytes)
}
function Get-S3Size([string]$key) {
  $out = (aws s3api head-object --bucket $Bucket --key $key --region $Region `
    --query ContentLength --output text 2>$null)
  if ($out -and "$out".Trim() -match '^\d+$') { return [int64]"$out".Trim() }
  return [int64]-1
}
function Get-FileSha256Hex([string]$path) {
  try {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
      $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $path))
      $hash = $sha.ComputeHash($bytes)
      return (($hash | ForEach-Object { $_.ToString("x2") }) -join "")
    } finally {
      $sha.Dispose()
    }
  } catch { return "" }
}
function Get-EmdTableId([string]$path, [int]$nBins, [int]$nRunouts) {
  try {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
      $raw = [System.IO.File]::ReadAllBytes((Resolve-Path $path))
      $suffix = [System.Text.Encoding]::UTF8.GetBytes("|n_bins=$nBins|n_runouts=$nRunouts")
      $all = New-Object byte[] ($raw.Length + $suffix.Length)
      [Array]::Copy($raw, 0, $all, 0, $raw.Length)
      [Array]::Copy($suffix, 0, $all, $raw.Length, $suffix.Length)
      $hash = $sha.ComputeHash($all)
      return ((($hash | ForEach-Object { $_.ToString("x2") }) -join "").Substring(0, 16))
    } finally {
      $sha.Dispose()
    }
  } catch { return "" }
}
function Get-CentroidInfo([string]$path) {
  $info = [ordered]@{
    Path = $path; Exists = $false; Kind = ""; K = 0; NBins = 0; NRunouts = 0;
    Sha16 = ""; TableId = ""; Warning = ""
  }
  if (-not $path) { return [pscustomobject]$info }
  if (-not (Test-Path $path)) {
    $info.Warning = "centroids file not found"
    return [pscustomobject]$info
  }
  $info.Exists = $true
  try {
    $d = (Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json)
    $cent = @($d.centroids)
    $info.Kind = "$($d.kind)"
    $info.K = $cent.Count
    $info.NBins = [int]$d.n_bins
    $info.NRunouts = [int]$d.n_runouts
    $info.Sha16 = (Get-FileSha256Hex $path).Substring(0, 16)
    $runoutsForTable = if ($NRunouts -gt 0) { $NRunouts } else { $info.NRunouts }
    $info.TableId = Get-EmdTableId $path $info.NBins $runoutsForTable
    if ($NumBuckets -gt 0 -and $info.K -ne $NumBuckets) {
      $info.Warning = "NumBuckets=$NumBuckets but centroids K=$($info.K)"
    } elseif ($info.NRunouts -gt 0 -and $NRunouts -gt 0 -and $info.NRunouts -ne $NRunouts) {
      $info.Warning = "NRunouts=$NRunouts but centroids n_runouts=$($info.NRunouts)"
    }
  } catch {
    $info.Warning = "could not parse centroids JSON"
  }
  return [pscustomobject]$info
}
function Get-FrozenArtifact {
  if ($CENTROID_INFO -and $CENTROID_INFO.TableId) {
    $name = "$($CENTROID_INFO.TableId).emdfrzn"
    $key = "$RUN_PREFIX/frozen/$name"
    $sz = Get-S3Size $key
    if ($sz -ge 0) {
      return [pscustomobject]@{ Bytes = $sz; Name = $name; Exact = $true; TableId = $CENTROID_INFO.TableId }
    }
  }
  $ls = aws s3 ls "s3://$Bucket/$RUN_PREFIX/frozen/" --region $Region 2>$null
  $best = [int64]-1
  $bestName = ""
  if ($ls) {
    foreach ($line in $ls) {
      $cols = ($line -split "\s+")
      if ($cols.Count -ge 4 -and $cols[2] -match '^\d+$') {
        $sz = [int64]$cols[2]
        if ($sz -gt $best) { $best = $sz; $bestName = $cols[3] }
      }
    }
  }
  return [pscustomobject]@{ Bytes = $best; Name = $bestName; Exact = $false; TableId = "" }
}
function Get-ConsoleProbe([string]$iid) {
  if (-not $iid) { return $null }
  $q = "{Len:length(Output),BuildStart:contains(Output, 'PHASE=build START'),BuildDone:contains(Output, 'PHASE=build DONE'),PreStart:contains(Output, 'PHASE=precompute TABLE_ID'),CacheHit:contains(Output, 'PHASE=precompute CACHE_HIT'),CacheMiss:contains(Output, 'PHASE=precompute CACHE_MISS'),PreDone:contains(Output, 'PHASE=precompute DONE'),TrainSkip:contains(Output, 'PHASE=train SKIP'),TrainStart:contains(Output, 'PHASE=train START'),TrainProgress:contains(Output, 'PROGRESS iters_done='),ShipStart:contains(Output, 'PHASE=ship START'),ShipDownload:contains(Output, 'PHASE=ship DOWNLOAD_CKPT'),ShipRun:contains(Output, 'PHASE=ship RUN_SHIP_BLUEPRINT'),Convert:contains(Output, '[1/3] CONVERT'),Validate:contains(Output, '[2/3] VALIDATE'),Install:contains(Output, '[3/3] INSTALL'),ShipDone:contains(Output, 'PHASE=ship DONE'),EvalStart:contains(Output, 'PHASE=eval START'),EvalEmd:contains(Output, 'PHASE=eval EMD_START'),EvalBase:contains(Output, 'PHASE=eval ITER62000_START'),EvalCombine:contains(Output, 'PHASE=eval COMBINE_START'),EvalDone:contains(Output, 'PHASE=eval DONE'),EvalProgress:contains(Output, 'PROGRESS seed'),EvalHandProgress:contains(Output, 'HAND_PROGRESS'),EvalLogSync:contains(Output, 'PHASE=eval LOG_SYNC'),Traceback:contains(Output, 'Traceback'),S3DownloadFail:contains(Output, 'S3_DOWNLOAD_FAIL'),S3UploadFail:contains(Output, 'S3_UPLOAD_FAIL'),S3PutFail:contains(Output, 'S3_PUT-TEXT_FAIL'),S3Any:contains(Output, 'S3_'),AwsError:contains(Output, 'aws: [ERROR]')}"
  $json = (aws ec2 get-console-output --instance-id $iid --region $Region --latest --query $q --output json 2>$null | Out-String)
  if (-not $json.Trim()) { return $null }
  try { return ($json | ConvertFrom-Json) } catch { return $null }
}

# ============================================================================================
# V2 AUTO-ACTIVATE GATE: same pluribus-grade checklist as the babysitter. Returns a hashtable
# with Pass=$true|$false and Reasons=@("reason1","reason2",...).
# ============================================================================================
function Test-AutoActivateGate([object]$v) {
  $r = [ordered]@{ Pass = $true; Reasons = @() }
  if (-not $v) { $r.Pass = $false; $r.Reasons += "verdict JSON missing"; return $r }
  $tmp = Join-Path $env:TEMP ("emd_watch_gate_{0}.json" -f ([guid]::NewGuid().ToString('N')))
  try {
    $v | ConvertTo-Json -Depth 32 | Set-Content -Path $tmp -Encoding Ascii
    $out = (& python scripts/_emd_auto_activate_gate.py check --verdict $tmp 2>&1)
    $txt = ($out | Out-String).Trim()
    if ($LASTEXITCODE -eq 0) {
      $r.Pass = $true
      $r.Reasons = @()
    } else {
      $r.Pass = $false
      if ($txt -match '^FAIL:\s*(.*)$') {
        $r.Reasons = @($Matches[1] -split ';\s*')
      } elseif ($txt) {
        $r.Reasons = @($txt)
      } else {
        $r.Reasons = @("python gate failed with no output")
      }
    }
  } finally {
    Remove-Item $tmp -ErrorAction SilentlyContinue
  }
  return $r
}

# ============================================================================================
# V2 PRE-LAUNCH SANITY CHECKS (run BEFORE the main while loop). On FATAL we exit 2.
# (1) K-cap: centroids' K must be <= $KMax. K=500 is BANNED (k500 incident: -140 SB regression).
# (2) SHA256 sidecar: for the inferable frozen blob path, the .sha256 sidecar MUST exist.
# ============================================================================================
$CENTROID_INFO = Get-CentroidInfo $Centroids
$preLaunchFails = @()
if ($Centroids -and $CENTROID_INFO.Exists -and $CENTROID_INFO.K -gt $KMax) {
  $preLaunchFails += ("FATAL: K=$($CENTROID_INFO.K) exceeds production cap $KMax. K=500 is banned.")
}
# SHA256 sidecar: if a TableId is computable AND the S3 frozen blob exists, the .sha256 sidecar must too.
$preLaunchFrozenName = ""
$preLaunchFrozenKey = ""
if ($CENTROID_INFO -and $CENTROID_INFO.TableId) {
  $preLaunchFrozenName = "$($CENTROID_INFO.TableId).emdfrzn"
  $preLaunchFrozenKey  = "$RUN_PREFIX/frozen/$preLaunchFrozenName"
  $preLaunchFrozenBytes = Get-S3Size $preLaunchFrozenKey
  if ($preLaunchFrozenBytes -ge 0) {
    $preLaunchShaKey = "$preLaunchFrozenKey.sha256"
    $preLaunchShaBytes = Get-S3Size $preLaunchShaKey
    if ($preLaunchShaBytes -lt 0) {
      $preLaunchFails += ("FATAL: SHA256 sidecar missing for $preLaunchFrozenName. Re-run precompute_emd_frozen_lookup.py.")
    }
  }
}
if ($preLaunchFails.Count -gt 0) {
  foreach ($f in $preLaunchFails) { Write-Host $f -ForegroundColor Red }
  Write-Host "PRE-LAUNCH SANITY FAILED. Exiting 2." -ForegroundColor Red
  exit 2
}

while ($true) {
  # --- live box ---
  $boxRaw = aws ec2 describe-instances --region $Region `
    --filters "Name=tag:Run,Values=$Run" "Name=instance-state-name,Values=running,pending" `
    --query "Reservations[].Instances[].[InstanceId,State.Name,InstanceType,LaunchTime]" --output text 2>$null
  $boxLine = "(no live box - launching, reclaimed, or finished)"
  $instanceId = ""
  $upMin = 0
  if ($boxRaw) {
    $f = ($boxRaw -split "\s+")
    $instanceId = $f[0]
    if ($boxRaw -match "(\d{4}-\d{2}-\d{2}T[\d:.]+[+\-]\d{2}:?\d{2})") {
      try { $upMin = [int](((Get-Date).ToUniversalTime() - [datetimeoffset]::Parse($Matches[1]).UtcDateTime).TotalMinutes) } catch {}
    }
    $boxLine = ("{0}  {1}  {2}  up {3}m" -f $f[0], $f[1], $f[2], $upMin)
  }
  $instType  = if ($boxRaw) { $f[2] } else { "" }
  $livePrice = Get-SpotpickPrice $instType
  $cpu = if ($instanceId -match '^i-') { Get-EC2CpuUtilization $instanceId } else { $null }

  # --- checkpoint listing (the train heartbeat) ---
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

  # --- frozen-table object (presence/size => precompute progress / cache hit) ---
  $frozenArtifact = Get-FrozenArtifact
  $frozenMB = if ($frozenArtifact.Bytes -ge 0) { [double]$frozenArtifact.Bytes / 1e6 } else { -1.0 }
  $effectiveShipName = $ShipName
  $shippedMarkerTxt = Get-S3Text $M_SHIPPED
  if ($shippedMarkerTxt -match "staged=([A-Za-z0-9_.-]+)") {
    $effectiveShipName = $Matches[1]
  }
  $SHIPART_KEY = "$RUN_PREFIX/serve/$effectiveShipName.pkl"
  $serveBytes = Get-S3Size $SHIPART_KEY
  $verdictBytes = Get-S3Size $VERDICT_KEY
  $shipLogBytes = Get-S3Size $SHIPLOG
  $evalEmdLogBytes = Get-S3Size $EVALEMDLOG
  $evalBaseLogBytes = Get-S3Size $EVALBASELOG
  $evalCombineLogBytes = Get-S3Size $EVALCOMBINELOG
  $evalEmdPartBytes = Get-S3Size $EVALEMDPART
  $evalBasePartBytes = Get-S3Size $EVALBASEPART

  # --- SHA256 sidecar probe (live, every refresh) ---
  $sha256Key = ""
  $sha256Bytes = [int64]-1
  $sha256Stamped = $false
  if ($frozenArtifact -and $frozenArtifact.Name) {
    $sha256Key = "$RUN_PREFIX/frozen/$($frozenArtifact.Name).sha256"
    $sha256Bytes = Get-S3Size $sha256Key
    $sha256Stamped = ($sha256Bytes -ge 0)
  }

  # --- authoritative checkpoint iters_done from latest.nlcfr ---
  $latestMeta = Get-S3ObjectMeta $LATEST
  $trainStatusMeta = Get-S3ObjectMeta $TRAINSTATUS
  $trainLogMeta = Get-S3ObjectMeta $TRAINLOG
  $ckptDone = Get-LatestItersDone
  if ($ckptDone -lt 0) { $ckptDone = $maxCkptIter }
  $itersDone = $ckptDone

  # --- markers (authoritative phase completion; use object reads, not ListBucket-dependent ls) ---
  function _HasMarker([string]$key) {
    $tmp = Join-Path $env:TEMP ("emd_watch_v2_marker_{0}.txt" -f ([guid]::NewGuid().ToString('N')))
    aws s3 cp "s3://$Bucket/$key" $tmp --region $Region --no-progress 2>$null | Out-Null
    $ok = Test-Path $tmp
    Remove-Item $tmp -ErrorAction SilentlyContinue
    return $ok
  }
  $mkTable     = _HasMarker $M_TABLE
  $mkConverged = _HasMarker $M_CONVERGED
  $mkShipped   = _HasMarker $M_SHIPPED
  $mkEvaled    = _HasMarker $M_EVALED
  # Phase 1 is definitely complete if any checkpoint/train/ship evidence exists, even when the
  # optional table_cached marker was never written by an older on-box script.
  $precompComplete = $mkTable -or $mkConverged -or $mkShipped -or $mkEvaled -or ($itersDone -ge 0) -or ($maxCkptIter -ge 0)

  # --- PHASE inference (live, from console-output tail; logs only land at box exit) ---
  $console = ""
  $probe = $null
  if ($boxRaw -and $f[0] -match '^i-') {
    $probe = Get-ConsoleProbe $f[0]
    $console = (aws ec2 get-console-output --instance-id $f[0] --region $Region --latest --query Output --output text 2>$null | Out-String)
  }
  function _Probe([string]$name) {
    return ($probe -and $probe.PSObject.Properties[$name] -and [bool]$probe.PSObject.Properties[$name].Value)
  }
  # Prefer the most advanced marker observed (console live, else the S3 logs at-exit).
  $precompTxt = Get-S3Text $PRECOMPLOG
  $trainTxt   = Get-S3Text $TRAINLOG
  $trainStatusTxt = Get-S3Text $TRAINSTATUS
  $shipTxt    = Get-S3Text $SHIPLOG
  $evalEmdTxt = Get-S3Text $EVALEMDLOG
  $evalBaseTxt = Get-S3Text $EVALBASELOG
  $evalCombineTxt = Get-S3Text $EVALCOMBINELOG
  $allTxt     = $console + "`n" + $precompTxt + "`n" + $trainTxt + "`n" + $trainStatusTxt + "`n" + $shipTxt + "`n" + $evalEmdTxt + "`n" + $evalBaseTxt + "`n" + $evalCombineTxt

  # Train heartbeat/status is fresher than checkpoint cadence. Use it for the displayed iteration and
  # stall key whenever it is newer than latest.nlcfr or has advanced beyond the latest checkpoint.
  $trainStatusRec = Get-TrainProgressRecord "heartbeat-s3" $trainStatusTxt $trainStatusMeta
  $consoleTrainRec = Get-TrainProgressRecord "console" $console $null
  $trainLogRec = Get-TrainProgressRecord "train.log-s3" $trainTxt $trainLogMeta
  $trainSource = if ($itersDone -ge 0) { "checkpoint" } else { "none" }
  $trainSourceRec = $null
  if ($trainStatusRec -and ($itersDone -lt 0 -or $trainStatusRec.Iters -gt $itersDone -or (($trainStatusRec.Iters -ge $itersDone) -and (Test-MetaNewer $trainStatusMeta $latestMeta)))) {
    $itersDone = $trainStatusRec.Iters
    $trainSource = "heartbeat-s3"
    $trainSourceRec = $trainStatusRec
  } elseif ($consoleTrainRec -and ($itersDone -lt 0 -or $consoleTrainRec.Iters -gt $itersDone)) {
    $itersDone = $consoleTrainRec.Iters
    $trainSource = "console"
    $trainSourceRec = $consoleTrainRec
  } elseif ($trainLogRec -and ($itersDone -lt 0 -or $trainLogRec.Iters -gt $itersDone)) {
    $itersDone = $trainLogRec.Iters
    $trainSource = "train.log-s3"
    $trainSourceRec = $trainLogRec
  }
  $rate = 0.0; $infosets = -1; $logRate = $false; $etaSec = -1
  $rateSource = "est"
  $rateRec = if ($trainSourceRec) { $trainSourceRec } elseif ($trainStatusRec) { $trainStatusRec } elseif ($consoleTrainRec) { $consoleTrainRec } else { $trainLogRec }
  if ($rateRec) {
    $rate = $rateRec.Rate
    $infosets = $rateRec.Infosets
    $logRate = $true
    $rateSource = $rateRec.Source
    if ($rateRec.EtaSec -ge 0) { $etaSec = $rateRec.EtaSec }
  }
  $trainLogStale = ($trainLogMeta.Exists -and $trainStatusMeta.Exists -and $trainStatusMeta.LastModified -and $trainLogMeta.LastModified -and ($trainLogMeta.LastModified -lt $trainStatusMeta.LastModified.AddMinutes(-2)))
  if ($trainStatusRec -or $consoleTrainRec -or $trainLogRec) { $precompComplete = $true }

  # Live console wins over inferred marker phase: a broken resume can retrain after CONVERGED, and the
  # dashboard should show that truth instead of optimistically labeling it SHIP.
  $phase = "?"
  if ($mkEvaled)                                         { $phase = "eval-done" }
  elseif ((_Probe "EvalStart") -or $allTxt -match "PHASE=eval START") { $phase = "eval" }
  elseif ($mkShipped -and $StopAfterShip)                { $phase = "ship-done" }
  elseif ($mkShipped)                                    { $phase = "eval" }    # shipped -> eval running
  elseif ((_Probe "ShipStart") -or $allTxt -match "PHASE=ship START") { $phase = "ship" }
  elseif (((_Probe "TrainStart") -and -not (_Probe "TrainSkip")) -or ($allTxt -match "PHASE=train START" -and $allTxt -notmatch "PHASE=train SKIP") -or ($trainStatusRec -and -not $mkConverged)) { $phase = "train" }
  elseif ($mkConverged)                                  { $phase = "ship" }    # converged -> ship expected/next
  elseif ($itersDone -ge 0)                              { $phase = "train" }
  elseif ($allTxt -match "PHASE=train")                 { $phase = "train" }
  elseif ((_Probe "PreStart") -or (_Probe "PreDone") -or $allTxt -match "PHASE=precompute (CACHE_HIT|CACHE_MISS|UPLOADED|DONE)|PRECOMPUTE_HEARTBEAT|PRECOMPUTE start") { $phase = "precompute" }
  elseif ((_Probe "BuildStart") -or $allTxt -match "PHASE=build") { $phase = "build" }
  elseif ($boxRaw)                                       { $phase = "boot" }

  # precompute detail
  $cacheHit = ($allTxt -match "PHASE=precompute CACHE_HIT")
  $precompRate = $null; $precompSpots = $null
  if ($allTxt -match "PRECOMPUTE done spots=(\d+).*rate=(\d+)") { $precompSpots = [int]$Matches[1]; $precompRate = [int]$Matches[2] }
  # latest heartbeat out_bytes (growth = liveness)
  $hbBytes = -1
  $mhb = [regex]::Matches($allTxt, "PRECOMPUTE_HEARTBEAT.*out_bytes=(\d+)")
  if ($mhb.Count) { $hbBytes = [int64]$mhb[$mhb.Count - 1].Groups[1].Value }

  # train detail (rate/infosets/eta from TRAIN_HEARTBEAT/PROGRESS lines)
  if (-not $logRate -and $phase -eq "train" -and $upMin -gt 0 -and $itersDone -gt 0) { $rate = $itersDone / ($upMin * 60.0) }
  $etaStr = "--"
  if ($phase -eq "train" -and $rate -gt 1e-9 -and $itersDone -ge 0 -and $itersDone -lt $Iters) {
    $etaSec = ($Iters - $itersDone) / $rate
    if ($etaSec -lt 5400) { $etaStr = ("~{0}m" -f [int]($etaSec / 60)) } else { $etaStr = ("~{0:N1}h" -f ($etaSec / 3600.0)) }
  } elseif ($itersDone -ge $Iters) { $etaStr = "DONE" }
  $ecc = if ($etaSec -ge 0) { ($etaSec / 3600.0) * $livePrice } else { -1 }
  $spent = ($upMin / 60.0) * $livePrice

  # --- stall: track the most-advanced "progress unit" for the active phase ---
  $shipUnit = 0
  if ($mkShipped -or $serveBytes -ge 0 -or (_Probe "ShipDone")) { $shipUnit = 6 }
  elseif ((_Probe "Install")) { $shipUnit = 5 }
  elseif ((_Probe "Validate")) { $shipUnit = 4 }
  elseif ((_Probe "Convert")) { $shipUnit = 3 }
  elseif ((_Probe "ShipRun") -or (_Probe "ShipDownload") -or (_Probe "ShipStart")) { $shipUnit = 2 }
  elseif ((_Probe "TrainSkip") -or (_Probe "PreDone") -or $mkConverged) { $shipUnit = 1 }

  $evalUnit = 0
  if ($mkEvaled -or $verdictBytes -ge 0 -or (_Probe "EvalDone")) { $evalUnit = 6 }
  elseif ((_Probe "EvalCombine")) { $evalUnit = 5 }
  elseif ((_Probe "EvalBase")) { $evalUnit = 4 }
  elseif ((_Probe "EvalEmd") -or (_Probe "EvalProgress") -or (_Probe "EvalHandProgress") -or (_Probe "EvalLogSync")) { $evalUnit = 3 }
  elseif ((_Probe "EvalStart") -or $mkShipped) { $evalUnit = 2 }
  elseif ($serveBytes -ge 0) { $evalUnit = 1 }

  $evalHandUnit = 0
  $mhp = [regex]::Matches($allTxt, "\[(anchor|emd|iter62000)\] HAND_PROGRESS seed=\d+ seed_idx=(\d+)/(\d+) hand=(\d+)/(\d+)[^\r\n]*")
  if ($mhp.Count) {
    $hm = $mhp[$mhp.Count - 1]
    $label = $hm.Groups[1].Value
    $stageBase = if ($label -eq "iter62000") { 400000000 } elseif ($label -eq "emd") { 200000000 } else { 100000000 }
    $seedIdx = [int]$hm.Groups[2].Value
    $hand = [int]$hm.Groups[4].Value
    $handTotal = [int]$hm.Groups[5].Value
    $evalHandUnit = $stageBase + (($seedIdx - 1) * [Math]::Max($handTotal, 1)) + $hand
  }
  $evalLogBytesTotal = [int64]0
  foreach ($b in @($evalEmdLogBytes, $evalBaseLogBytes, $evalCombineLogBytes)) { if ($b -gt 0) { $evalLogBytesTotal += $b } }

  $evalStageUnit = ([int64]$evalUnit * 1000000000)
  $evalSubProgress = [Math]::Max([int64]$evalHandUnit, [int64]$evalLogBytesTotal)
  $rawProgress = if ($phase -eq "train") { [int64]([Math]::Max($itersDone, $maxCkptIter)) } elseif ($phase -eq "precompute") { [int64]([Math]::Max($hbBytes, 0)) } elseif ($phase -eq "ship") { [int64]$shipUnit } elseif ($phase -eq "eval") { [int64]($evalStageUnit + $evalSubProgress) } else { [int64]0 }
  $phaseRank = switch ($phase) { "boot" { 1 } "build" { 2 } "precompute" { 3 } "train" { 4 } "ship" { 5 } "ship-done" { 6 } "eval" { 7 } "eval-done" { 8 } default { 0 } }
  $progressKey = ([int64]$phaseRank * 1000000000000) + $rawProgress
  if ($progressKey -gt $lastProgressKey) { $lastProgressKey = $progressKey; $lastHeartTime = Get-Date }
  $stallMinObs = [int]((Get-Date) - $lastHeartTime).TotalMinutes
  $stalled = ($rawProgress -gt 0 -and $stallMinObs -ge $StallMin)

  Clear-Host
  Write-Host ("=== EMD pipeline V2: {0}   {1}   (watching {2}m) ===" -f $Run, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), [int]((Get-Date) - $t0).TotalMinutes) -ForegroundColor Cyan
  $boxColor = if ($boxRaw -match "running") { "Green" } elseif ($boxRaw) { "Yellow" } else { "DarkGray" }
  Write-Host ("box:    {0}" -f $boxLine) -ForegroundColor $boxColor
  if ($instanceId -match '^i-') {
    if ($cpu) {
      $cpuAge = if ($cpu.AgeSec -lt 120) { ("{0}s old" -f $cpu.AgeSec) } else { ("{0}m old" -f [int]($cpu.AgeSec / 60)) }
      Write-Host ("cpu:    {0:N1}% avg / {1:N1}% max (CloudWatch {2}s, {3})" -f $cpu.Average, $cpu.Maximum, $cpu.PeriodSec, $cpuAge) -ForegroundColor DarkCyan
    } else {
      Write-Host "cpu:    CloudWatch CPUUtilization pending/unavailable (read-only probe)" -ForegroundColor DarkGray
    }
  }

  $phaseColor = switch ($phase) { "train" { "Green" } "precompute" { "Cyan" } "build" { "Yellow" } "boot" { "Yellow" } "ship" { "Magenta" } "ship-done" { "Green" } "eval" { "Magenta" } "eval-done" { "Green" } default { "DarkGray" } }
  Write-Host ("PHASE:  {0}" -f $phase.ToUpper()) -ForegroundColor $phaseColor

  # 5-phase marker strip (DONE green / pending gray) so resume-at-first-incomplete is visible at a glance.
  function _Mk([bool]$b) { if ($b) { "[x]" } else { "[ ]" } }
  Write-Host ("phases: 1.BUILD/PRECOMP {0}  2.TRAIN {1}  4.SHIP {2}  5.EVAL {3}" -f `
    (_Mk $precompComplete), (_Mk $mkConverged), (_Mk $mkShipped), (_Mk $mkEvaled)) -ForegroundColor DarkCyan

  # === V2 PANEL: K-CAP STATUS LINE (green if K=200, red WARNING if K=500) ==================
  if ($Centroids -and $CENTROID_INFO.Exists) {
    $kColor = if ($CENTROID_INFO.K -gt $KMax) { "Red" } else { "Green" }
    $kMsg = "K-CAP: K=$($CENTROID_INFO.K) (cap=$KMax)"
    if ($CENTROID_INFO.K -gt $KMax) { $kMsg += "  WARNING: K=$($CENTROID_INFO.K) is BANNED for auto-activate" }
    Write-Host $kMsg -ForegroundColor $kColor
  }

  # === V2 PANEL: SHA256 STATUS LINE (green if stamped, red WARNING if missing) ============
  if ($frozenArtifact -and $frozenArtifact.Name) {
    $sColor = if ($sha256Stamped) { "Green" } else { "Red" }
    $sTxt = "SHA256: $($frozenArtifact.Name)"
    if ($sha256Stamped) {
      $sTxt += ("  sidecar present ({0})" -f (Fmt-Bytes $sha256Bytes))
    } else {
      $sTxt += "  WARNING: .sha256 sidecar missing (re-run precompute_emd_frozen_lookup.py)"
    }
    Write-Host $sTxt -ForegroundColor $sColor
  }

  if ($Centroids) {
    $cfgColor = if ($CENTROID_INFO.Warning) { "Red" } elseif ($CENTROID_INFO.Exists) { "DarkCyan" } else { "Yellow" }
    $tableTxt = if ($CENTROID_INFO.TableId) { $CENTROID_INFO.TableId } else { "(unknown)" }
    $shaTxt = if ($CENTROID_INFO.Sha16) { $CENTROID_INFO.Sha16 } else { "(unknown)" }
    Write-Host ("config:  buckets={0} centroids={1} kind={2} K={3} bins={4} runouts={5} sha16={6} table_id={7}" -f `
      $(if ($NumBuckets -gt 0) { $NumBuckets } else { "(auto)" }), $Centroids, $CENTROID_INFO.Kind, `
      $CENTROID_INFO.K, $CENTROID_INFO.NBins, $(if ($NRunouts -gt 0) { $NRunouts } else { $CENTROID_INFO.NRunouts }), `
      $shaTxt, $tableTxt) -ForegroundColor $cfgColor
    if ($CENTROID_INFO.Warning) { Write-Host ("         WARNING: {0}" -f $CENTROID_INFO.Warning) -ForegroundColor Red }
  }
  Write-Host ("markers: table={0} converged={1} shipped={2} evaled={3}" -f `
    $mkTable, $mkConverged, $mkShipped, $mkEvaled) -ForegroundColor DarkGray
  $frozenTxt = if ($frozenMB -ge 0) { ("{0:N1} MB" -f $frozenMB) } else { "(none)" }
  $frozenNameTxt = if ($frozenArtifact.Name) { $frozenArtifact.Name } else { "(none)" }
  $frozenModeTxt = if ($frozenArtifact.Exact) { "exact" } else { "listed" }
  Write-Host ("artifacts: frozen={0} {1} ({2})  serve/{3}.pkl={4}  verdict={5}  logs(ship/emd/base/combine)={6}/{7}/{8}/{9}" -f `
    $frozenTxt, $frozenNameTxt, $frozenModeTxt, $effectiveShipName, (Fmt-Bytes $serveBytes), (Fmt-Bytes $verdictBytes), `
    (Fmt-Bytes $shipLogBytes), (Fmt-Bytes $evalEmdLogBytes), (Fmt-Bytes $evalBaseLogBytes), (Fmt-Bytes $evalCombineLogBytes)) -ForegroundColor DarkCyan
  if ($phase -eq "eval" -or $phase -eq "eval-done") {
    Write-Host ("partials: emd={0}  iter62000={1}" -f (Fmt-Bytes $evalEmdPartBytes), (Fmt-Bytes $evalBasePartBytes)) -ForegroundColor DarkCyan
  }
  if ($phase -eq "ship-done") {
    Write-Host "STEP 3 COMPLETE: staged serve artifact present; -StopAfterShip means eval is intentionally not expected." -ForegroundColor Green
  }
  if ($probe) {
    $milestones = @()
    if (_Probe "BuildStart") { $milestones += "build" }
    if (_Probe "BuildDone") { $milestones += "build-done" }
    if (_Probe "CacheHit") { $milestones += "cache-hit" }
    if (_Probe "CacheMiss") { $milestones += "cache-miss" }
    if (_Probe "PreDone") { $milestones += "precompute-done" }
    if (_Probe "TrainSkip") { $milestones += "train-skip" }
    if ((_Probe "TrainStart") -and -not (_Probe "TrainSkip")) {
      if ($mkConverged) { $milestones += "TRAIN-RESTART" } else { $milestones += "train-start" }
    }
    if (_Probe "ShipStart") { $milestones += "ship-start" }
    if (_Probe "Convert") { $milestones += "convert" }
    if (_Probe "Validate") { $milestones += "validate" }
    if (_Probe "Install") { $milestones += "install" }
    if (_Probe "ShipDone") { $milestones += "ship-done" }
    if (_Probe "EvalStart") { $milestones += "eval-start" }
    if (_Probe "EvalEmd") { $milestones += "eval-emd" }
    if (_Probe "EvalBase") { $milestones += "eval-iter62000" }
    if (_Probe "EvalCombine") { $milestones += "eval-combine" }
    if (_Probe "EvalDone") { $milestones += "eval-done" }
    if (_Probe "EvalHandProgress") { $milestones += "eval-hand-progress" }
    if (_Probe "EvalLogSync") { $milestones += "eval-log-sync" }
    if (_Probe "Traceback") { $milestones += "TRACEBACK" }
    $s3Fatal = (_Probe "S3UploadFail") -or (_Probe "S3PutFail") -or ((_Probe "S3DownloadFail") -and -not (_Probe "CacheMiss"))
    if ((_Probe "S3DownloadFail") -and (_Probe "CacheMiss")) { $milestones += "cache-download-miss" }
    elseif (_Probe "S3DownloadFail") { $milestones += "S3-DOWNLOAD-FAIL" }
    if (_Probe "S3UploadFail") { $milestones += "S3-UPLOAD-FAIL" }
    if (_Probe "S3PutFail") { $milestones += "S3-MARKER-FAIL" }
    if (_Probe "AwsError") { $milestones += "AWSCLI-ERROR" }
    $milestoneTxt = if ($milestones.Count) { ($milestones -join " -> ") } else { "(console up; no phase milestone yet)" }
    Write-Host ("live:    console_len={0}  {1}" -f $probe.Len, $milestoneTxt) -ForegroundColor $(if ((_Probe "Traceback") -or $s3Fatal -or (_Probe "AwsError") -or ((_Probe "TrainStart") -and -not (_Probe "TrainSkip") -and $mkConverged)) { "Red" } else { "DarkGray" })
  }

  if ($phase -eq "build" -or $phase -eq "boot") {
    Write-Host "        building the EMD crate (cargo --release --features python) / booting..." -ForegroundColor DarkGray
  }
  if ($phase -eq "precompute") {
    if ($cacheHit) {
      $tidTxt = if ($CENTROID_INFO.TableId) { "; table_id=$($CENTROID_INFO.TableId)" } else { "" }
      Write-Host ("        CACHE HIT - cached frozen table downloaded ({0:N1} MB{1}); recompute skipped" -f $frozenMB, $tidTxt) -ForegroundColor Green
    } elseif ($precompSpots) {
      $tidTxt = if ($CENTROID_INFO.TableId) { " table_id=$($CENTROID_INFO.TableId)" } else { "" }
      Write-Host ("        precompute DONE: {0:N0} canonical spots @ {1:N0} spot/s{2}" -f $precompSpots, $precompRate, $tidTxt) -ForegroundColor Green
    } else {
      $pfrac = if ($ExpectTableMB -gt 0 -and $hbBytes -ge 0) { ($hbBytes / 1e6) / $ExpectTableMB } else { 0 }
      $mb = if ($hbBytes -ge 0) { $hbBytes / 1e6 } else { 0 }
      Write-Host ("        enumerating ~16.5M canonical flop+turn spots  [{0}] {1:P0}  (~{2:N1}/{3:N0} MB written)" -f (Bar $pfrac), $pfrac, $mb, $ExpectTableMB) -ForegroundColor Cyan
      $tidTxt = if ($CENTROID_INFO.TableId) { " table_id=$($CENTROID_INFO.TableId);" } else { "" }
      Write-Host "        (${tidTxt}the table is written at the END of the rayon enumeration; out_bytes jumps near completion)" -ForegroundColor DarkGray
    }
  }
  if ($phase -eq "train") {
    $frac = if ($Iters -gt 0 -and $itersDone -ge 0) { [double]$itersDone / $Iters } else { 0 }
    $idTxt = if ($itersDone -ge 0) { "$itersDone" } else { "?" }
    $barColor = if ($frac -ge 0.99) { "Green" } elseif ($frac -ge 0.5) { "Cyan" } else { "Yellow" }
    Write-Host ("iters:  {0}/{1}  [{2}] {3:P1}" -f $idTxt, $Iters, (Bar $frac), $frac) -ForegroundColor $barColor
    $infTxt = if ($infosets -ge 0) { "{0:N0}" -f $infosets } else { "(pending log)" }
    $rateTxt = if ($logRate) { "{0:N3} it/s ({1})" -f $rate, $rateSource } else { "{0:N3} it/s (est)" -f $rate }
    Write-Host ("rate:   {0}   infosets: {1}   ETA: {2}" -f $rateTxt, $infTxt, $etaStr) -ForegroundColor Yellow
    if ($NDeals -gt 0 -and $rate -gt 0) {
      $dealsPerSec = $rate * [double]$NDeals
      $seatTraversalsPerSec = $dealsPerSec * 6.0
      Write-Host ("work:   n_deals={0} -> {1:N0} sampled deals/s  (~{2:N0} seat traversals/s)" -f $NDeals, $dealsPerSec, $seatTraversalsPerSec) -ForegroundColor Yellow
    }
    Write-Host ("ckpts:  {0} versioned (max iter {1}; latest {2})   |   frozen table {3:N1} MB" -f $nCkpt, $maxCkptIter, (Fmt-Age $latestMeta), $frozenMB) -ForegroundColor DarkCyan
    $sourceColor = if ($trainSource -eq "checkpoint") { "DarkCyan" } elseif ($trainSource -eq "heartbeat-s3") { "Green" } elseif ($trainSource -eq "console") { "Yellow" } else { "DarkGray" }
    Write-Host ("source: iters={0}; heartbeat={1}; console={2}; train.log={3}" -f `
      $trainSource, (Fmt-ProgressRecord $trainStatusRec), (Fmt-ProgressRecord $consoleTrainRec), (Fmt-ProgressRecord $trainLogRec)) -ForegroundColor $sourceColor
    if (-not $trainStatusRec) {
      Write-Host ("        progress/train_status.txt missing or unparseable at s3://$Bucket/$TRAINSTATUS") -ForegroundColor Yellow
    }
    if ($trainLogStale) {
      Write-Host ("        stale train.log: S3 log is {0}; heartbeat object is {1}; live TRAIN uses progress/train_status.txt/console" -f (Fmt-Age $trainLogMeta), (Fmt-Age $trainStatusMeta)) -ForegroundColor Yellow
    } elseif (-not $trainLogMeta.Exists) {
      Write-Host "        train.log not uploaded yet; live TRAIN progress comes from heartbeat/console until log sync or exit" -ForegroundColor DarkGray
    }
    $eccTxt = if ($ecc -ge 0) { ("~`${0:N2}" -f $ecc) } else { "--" }
    Write-Host ("cost:   spent ~`${0:N2} this box   |   ECC {1} to finish   (@ `${2:N3}/hr {3})" -f $spent, $eccTxt, $livePrice, $instType) -ForegroundColor Magenta
  } else {
    $eccTxt = if ($ecc -ge 0) { ("~`${0:N2}" -f $ecc) } else { "--" }
    Write-Host ("cost:   spent ~`${0:N2} this box   (@ `${1:N3}/hr {2})" -f $spent, $livePrice, $instType) -ForegroundColor Magenta
  }

  # --- SHIP detail (the serve-skew gate) ---
  if ($phase -eq "ship") {
    $shipStep = "waiting for SHIP/EVAL resume box (CONVERGED set, SHIPPED absent)"
    if ($boxRaw) { $shipStep = "resume box booting/building before ship" }
    if ((_Probe "TrainStart") -and -not (_Probe "TrainSkip") -and $mkConverged) { $shipStep = "WARNING: box restarted TRAIN despite CONVERGED marker" }
    elseif ((_Probe "ShipDone") -or $mkShipped -or $serveBytes -ge 0) { $shipStep = "SHIP DONE - staged snapshot present" }
    elseif ((_Probe "Install") -or $allTxt -match "\[3/3\] INSTALL") { $shipStep = "[3/3] INSTALL (staging serve snapshot)" }
    elseif ((_Probe "Validate") -or $allTxt -match "\[2/3\] VALIDATE") { $shipStep = "[2/3] VALIDATE (serve-skew gate)" }
    elseif ((_Probe "Convert") -or $allTxt -match "\[1/3\] CONVERT") { $shipStep = "[1/3] CONVERT (.nlcfr -> snapshot)" }
    elseif ((_Probe "ShipRun") -or $allTxt -match "PHASE=ship RUN_SHIP_BLUEPRINT") { $shipStep = "starting ship_blueprint.py" }
    elseif ((_Probe "ShipDownload") -or $allTxt -match "PHASE=ship DOWNLOAD_CKPT") { $shipStep = "downloading converged latest.nlcfr" }
    elseif ((_Probe "ShipStart") -or $allTxt -match "PHASE=ship START") { $shipStep = "ship phase entered" }
    elseif (_Probe "TrainSkip") { $shipStep = "resume accepted: TRAIN skipped; entering ship next" }
    elseif (_Probe "PreDone") { $shipStep = "resume boot: precompute/cache ready; checking markers" }
    elseif (_Probe "CacheHit") { $shipStep = "resume boot: frozen cache hit; marker checks next" }
    elseif (_Probe "CacheMiss") { $shipStep = "resume boot: frozen cache miss; rebuilding table" }
    elseif (_Probe "BuildDone") { $shipStep = "resume boot: native crate built; loading frozen table" }
    if ($allTxt -match "VALIDATION FAILED")    { $shipStep = "VALIDATION FAILED (serve-skew gate refused)" }
    Write-Host ("ship:   $shipStep") -ForegroundColor Magenta
    if ($mkShipped) { Write-Host "        SHIPPED: staged snapshot (NOT activated) -> proceeding to EVAL" -ForegroundColor Green }
    if ($serveBytes -ge 0) { Write-Host ("        staged artifact: s3://$Bucket/$SHIPART_KEY ({0})" -f (Fmt-Bytes $serveBytes)) -ForegroundColor Green }
  }

  # --- EVAL detail (per-seat LBR of emd + iter62000 on CRN seeds + the anchor) ---
  if ($phase -eq "eval" -or $phase -eq "eval-done") {
    $evalStep = if ($mkShipped) { "waiting for / running verified-better eval" } else { "waiting for SHIPPED marker before eval" }
    if ((_Probe "EvalDone") -or $mkEvaled -or $verdictBytes -ge 0) { $evalStep = "EVAL DONE - verdict present" }
    elseif ((_Probe "EvalCombine")) { $evalStep = "combining EMD + iter62000 CRN results" }
    elseif ((_Probe "EvalBase")) { $evalStep = "running iter62000 baseline LBR arm" }
    elseif ((_Probe "EvalEmd") -or (_Probe "EvalProgress")) { $evalStep = "running EMD LBR arm" }
    elseif ((_Probe "EvalStart")) { $evalStep = "eval phase entered" }
    Write-Host ("eval:   $evalStep") -ForegroundColor Magenta
    # anchor line (printed loudly by the eval driver)
    $anchorLine = ""
    $ma = [regex]::Matches($allTxt, "ANCHOR CALIBRATION: always-fold = ([+\-0-9.]+) bb/100[^\r\n]*")
    if ($ma.Count) { $anchorLine = $ma[$ma.Count - 1].Value }
    if ($anchorLine) { Write-Host ("        $anchorLine") -ForegroundColor $(if ($anchorLine -match "OK") { "Green" } elseif ($anchorLine -match "OFF|ABORT") { "Red" } else { "Yellow" }) }
    # latest per-seed progress line (emd or iter62000 arm)
    $mp = [regex]::Matches($allTxt, "PROGRESS seed \d+ \((\d+)/(\d+), (\d+)%\)[^\r\n]*")
    if ($mp.Count) { Write-Host ("        " + $mp[$mp.Count - 1].Value.Trim()) -ForegroundColor Cyan }
    # latest in-seed heartbeat (pure-Python LBR can spend a long time before completing a whole seed)
    $mhp2 = [regex]::Matches($allTxt, "\[(anchor|emd|iter62000)\] HAND_PROGRESS[^\r\n]*")
    if ($mhp2.Count) { Write-Host ("        " + $mhp2[$mhp2.Count - 1].Value.Trim()) -ForegroundColor Cyan }
    if ($verdictBytes -ge 0) { Write-Host ("        verdict: s3://$Bucket/$VERDICT_KEY ({0})" -f (Fmt-Bytes $verdictBytes)) -ForegroundColor Green }
  }

  if ($stalled) {
    Write-Host ("STALL:  no progress in {0}m (>= {1}m alarm) - box hung, reclaimed mid-chunk, or a long uninterrupted enumeration" -f $stallMinObs, $StallMin) -ForegroundColor Red
  } elseif ($rawProgress -gt 0) {
    Write-Host ("alive:  last progress {0}m ago (alarm at {1}m)" -f $stallMinObs, $StallMin) -ForegroundColor DarkGray
  }

  # --- VERDICT (once EVAL is done) + activation status ---
  if ($mkEvaled) {
    $vTxt = Get-S3Text $VERDICT_KEY
    Write-Host ""
    if ($vTxt -and $vTxt.Trim()) {
      $v = $null; try { $v = ($vTxt | ConvertFrom-Json) } catch { $v = $null }
      if ($v) {
        $emdBetter = [bool]$v.emd_better; $anchorOk = [bool]$v.anchor_ok
        Write-Host "=== VERIFIED-BETTER VERDICT (per-seat blind LBR, CRN) ===" -ForegroundColor Cyan
        Write-Host ("  anchor: emd={0:+0.0} iter62000={1:+0.0} (target ~+{2:0}) -> anchor_ok={3}" -f `
          [double]$v.emd_anchor_bb_per_100, [double]$v.iter62000_anchor_bb_per_100, [double]$v.anchor_target, $anchorOk) `
          -ForegroundColor $(if ($anchorOk) { "Green" } else { "Red" })
        # Sign-correct per-seat rendering delegated to scripts/_emd_verdict_sign.py
        # (single source of truth -- prevents the literal 'improved' sign-flip bug
        # that previously labeled the k500 SB regression as an improvement).
        $tmpPath = Join-Path $env:TEMP ("emd_watch_v2_verdict_{0}.json" -f ([guid]::NewGuid().ToString('N')))
        try {
          [System.IO.File]::WriteAllText($tmpPath, $vTxt, [System.Text.Encoding]::ASCII)
          $rendered = & python scripts/_emd_verdict_sign.py render --verdict $tmpPath 2>$null
          if ($rendered) {
            foreach ($ln in ($rendered -split "`n")) { if ($ln) { Write-Host ("  " + $ln) } }
          } else {
            Write-Host "  (verdict render helper unavailable; verdict JSON at $tmpPath)" -ForegroundColor Yellow
          }
        } finally {
          Remove-Item $tmpPath -ErrorAction SilentlyContinue
        }
        Write-Host ("  global: iter62000={0:+0.0} -> emd={1:+0.0}   mean-blind-gain={2:+0.0}" -f `
          [double]$v.iter62000_pooled_bb_per_100, [double]$v.emd_pooled_bb_per_100, [double]$v.mean_blind_gain) -ForegroundColor DarkCyan
        Write-Host ("  ===> emd_better = {0}" -f $emdBetter) -ForegroundColor $(if ($emdBetter) { "Green" } else { "Yellow" })

        # === V2 PANEL: AUTO-ACTIVATE GATE STATUS (PASS green / FAIL: <reasons> red) ==================
        $gate = Test-AutoActivateGate $v
        if ($gate.Pass) {
          Write-Host "  auto-activate gate: PASS (all pluribus-grade conditions hold)" -ForegroundColor Green
        } else {
          Write-Host ("  auto-activate gate: FAIL: {0}" -f (($gate.Reasons) -join "; ")) -ForegroundColor Red
        }

        # Activation status is derived only from the Python auto-activate gate.
        if ($gate.Pass) {
          Write-Host "  activation: GATE PASS. Auto-flips only if the babysitter ran with -AutoActivate;" -ForegroundColor Green
          Write-Host "              otherwise STAGED -- activate manually via ship_blueprint.py --activate." -ForegroundColor Green
        } else {
          Write-Host "  activation: GATE FAIL -> STAYS STAGED; iter62000 remains live." -ForegroundColor Yellow
        }
      } else {
        Write-Host "VERDICT JSON present but unparseable yet (still uploading)." -ForegroundColor Yellow
      }
    } else {
      Write-Host "EVALED marker set but verdict JSON not fetched yet." -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "PIPELINE COMPLETE (EVALED). Box self-terminates; the babysitter reports + (opt) activates." -ForegroundColor Green
    break
  }
  Write-Host ""
  Write-Host ("(refresh {0}s; Ctrl-C to stop)" -f $Refresh)
  Start-Sleep -Seconds $Refresh
}
