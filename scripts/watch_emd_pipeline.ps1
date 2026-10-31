# At-a-glance dashboard for the 5-PHASE EMD BLUEPRINT PIPELINE run. READ-ONLY.
#   powershell -ExecutionPolicy Bypass -File scripts\watch_emd_pipeline.ps1 -Run run_emd_bp1 -Iters 250000
#   powershell -ExecutionPolicy Bypass -File scripts\watch_emd_pipeline.ps1 -Run run_emd_bp1 -Refresh 30 -StallMin 25
#   powershell -ExecutionPolicy Bypass -File scripts\watch_emd_pipeline.ps1 -Run run_emd_k500_pluribus_20260625 -Iters 320000 -NumBuckets 500 -Centroids data\abstraction\emd_buckets_500.json -ShipName emd_k500_pluribus
#
# Shows the CURRENT PHASE (BUILD/PRECOMPUTE/TRAIN/SHIP/EVAL) + per-phase progress, the markers, and -- once
# EVAL finishes -- the verified-better VERDICT (per-seat blind LBR emd vs iter62000 + emd_better) and the
# ACTIVATION status. Sourced from the S3 markers + logs + the live checkpoint listing + the box console:
#   build      : crate compile (PHASE=build START seen, no precompute marker yet).
#   precompute : enumerating ~16.5M canonical flop+turn spots (CACHE_HIT instant; else heartbeat % bar).
#   train      : iters_done/target (% bar), it/s, infosets, ETA, convergence status, ECC ($ to finish).
#   ship       : converting + serve-skew gate (ship_blueprint.py [1/3][2/3][3/3]); SHIPPED marker on pass.
#   eval       : per-seat LBR of emd + iter62000 on CRN seeds + the always-fold anchor; verdict JSON.
# Live box + uptime + spotpick hourly price + STALL alarm throughout. Colorized like watch_native_cfr.ps1.
# ASCII ONLY (PS 5.1 cp1252).
param(
  [string]$Run = "run_emd_bp1",
  [int]$Iters = 250000,
  [int]$NumBuckets = 0,               # optional: validates against the centroids K and displays config
  [string]$Centroids = "",            # optional: local EMD centroids JSON; enables exact table-id/frozen lookup
  [int]$NRunouts = 200,               # table-id component; normally matches the centroids' n_runouts
  [int]$Refresh = 45,
  [double]$SpotPrice = 1.88,           # fallback spot $/hr if spotpick is unavailable
  [int]$StallMin = 25,                 # alarm if no new checkpoint AND no precompute heartbeat in this many min
  [double]$ExpectTableMB = 160,        # rough expected frozen-table size (for the precompute % bar)
  [string]$ShipName = "emd_bp1",       # staged serve snapshot name under {run}/serve/
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
$t0 = Get-Date
$lastProgressKey = -1
$lastHeartTime = Get-Date

function Bar([double]$frac, [int]$width = 40) {
  if ($frac -lt 0) { $frac = 0 }; if ($frac -gt 1) { $frac = 1 }
  $d = [int]([math]::Round($frac * $width))
  return ("#" * $d) + ("-" * ($width - $d))
}
function Get-LatestItersDone {
  $tmp = Join-Path $env:TEMP "emd_watch_probe.nlcfr"
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
    $out = (& python -c "import os,sys;root=os.environ.get('SPOTPICK_ROOT');sys.path.insert(0,root) if root else None;import spotpick;print({i.name:i.spot for i in spotpick.load()}.get('$it',''))" 2>$null)
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
      return [pscustomobject]@{ Bytes = $sz; Name = $name; Exact = $true }
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
  return [pscustomobject]@{ Bytes = $best; Name = $bestName; Exact = $false }
}
function Get-ConsoleProbe([string]$iid) {
  if (-not $iid) { return $null }
  $q = "{Len:length(Output),BuildStart:contains(Output, 'PHASE=build START'),BuildDone:contains(Output, 'PHASE=build DONE'),PreStart:contains(Output, 'PHASE=precompute TABLE_ID'),CacheHit:contains(Output, 'PHASE=precompute CACHE_HIT'),CacheMiss:contains(Output, 'PHASE=precompute CACHE_MISS'),PreDone:contains(Output, 'PHASE=precompute DONE'),TrainSkip:contains(Output, 'PHASE=train SKIP'),TrainStart:contains(Output, 'PHASE=train START'),TrainProgress:contains(Output, 'PROGRESS iters_done='),ShipStart:contains(Output, 'PHASE=ship START'),ShipDownload:contains(Output, 'PHASE=ship DOWNLOAD_CKPT'),ShipRun:contains(Output, 'PHASE=ship RUN_SHIP_BLUEPRINT'),Convert:contains(Output, '[1/3] CONVERT'),Validate:contains(Output, '[2/3] VALIDATE'),Install:contains(Output, '[3/3] INSTALL'),ShipDone:contains(Output, 'PHASE=ship DONE'),EvalStart:contains(Output, 'PHASE=eval START'),EvalEmd:contains(Output, 'PHASE=eval EMD_START'),EvalBase:contains(Output, 'PHASE=eval ITER62000_START'),EvalCombine:contains(Output, 'PHASE=eval COMBINE_START'),EvalDone:contains(Output, 'PHASE=eval DONE'),EvalProgress:contains(Output, 'PROGRESS seed'),EvalHandProgress:contains(Output, 'HAND_PROGRESS'),EvalLogSync:contains(Output, 'PHASE=eval LOG_SYNC'),Traceback:contains(Output, 'Traceback'),S3DownloadFail:contains(Output, 'S3_DOWNLOAD_FAIL'),S3UploadFail:contains(Output, 'S3_UPLOAD_FAIL'),S3PutFail:contains(Output, 'S3_PUT-TEXT_FAIL'),S3Any:contains(Output, 'S3_'),AwsError:contains(Output, 'aws: [ERROR]')}"
  $json = (aws ec2 get-console-output --instance-id $iid --region $Region --latest --query $q --output json 2>$null | Out-String)
  if (-not $json.Trim()) { return $null }
  try { return ($json | ConvertFrom-Json) } catch { return $null }
}

$CENTROID_INFO = Get-CentroidInfo $Centroids

while ($true) {
  # --- live box ---
  $boxRaw = aws ec2 describe-instances --region $Region `
    --filters "Name=tag:Run,Values=$Run" "Name=instance-state-name,Values=running,pending" `
    --query "Reservations[].Instances[].[InstanceId,State.Name,InstanceType,LaunchTime]" --output text 2>$null
  $boxLine = "(no live box - launching, reclaimed, or finished)"
  $upMin = 0
  if ($boxRaw) {
    $f = ($boxRaw -split "\s+")
    if ($boxRaw -match "(\d{4}-\d{2}-\d{2}T[\d:.]+[+\-]\d{2}:?\d{2})") {
      try { $upMin = [int](((Get-Date).ToUniversalTime() - [datetimeoffset]::Parse($Matches[1]).UtcDateTime).TotalMinutes) } catch {}
    }
    $boxLine = ("{0}  {1}  {2}  up {3}m" -f $f[0], $f[1], $f[2], $upMin)
  }
  $instType  = if ($boxRaw) { $f[2] } else { "" }
  $livePrice = Get-SpotpickPrice $instType

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
  $serveBytes = Get-S3Size $SHIPART_KEY
  $verdictBytes = Get-S3Size $VERDICT_KEY
  $shipLogBytes = Get-S3Size $SHIPLOG
  $evalEmdLogBytes = Get-S3Size $EVALEMDLOG
  $evalBaseLogBytes = Get-S3Size $EVALBASELOG
  $evalCombineLogBytes = Get-S3Size $EVALCOMBINELOG
  $evalEmdPartBytes = Get-S3Size $EVALEMDPART
  $evalBasePartBytes = Get-S3Size $EVALBASEPART

  # --- authoritative iters_done from latest.nlcfr ---
  $itersDone = Get-LatestItersDone
  if ($itersDone -lt 0) { $itersDone = $maxCkptIter }

  # --- markers (authoritative phase completion; use object reads, not ListBucket-dependent ls) ---
  function _HasMarker([string]$key) {
    $tmp = Join-Path $env:TEMP ("emd_watch_marker_{0}.txt" -f ([guid]::NewGuid().ToString('N')))
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
  $shipTxt    = Get-S3Text $SHIPLOG
  $evalEmdTxt = Get-S3Text $EVALEMDLOG
  $evalBaseTxt = Get-S3Text $EVALBASELOG
  $evalCombineTxt = Get-S3Text $EVALCOMBINELOG
  $allTxt     = $console + "`n" + $precompTxt + "`n" + $trainTxt + "`n" + $shipTxt + "`n" + $evalEmdTxt + "`n" + $evalBaseTxt + "`n" + $evalCombineTxt

  # Live console wins over inferred marker phase: a broken resume can retrain after CONVERGED, and the
  # dashboard should show that truth instead of optimistically labeling it SHIP.
  $phase = "?"
  if ($mkEvaled)                                         { $phase = "eval-done" }
  elseif ((_Probe "EvalStart") -or $allTxt -match "PHASE=eval START") { $phase = "eval" }
  elseif ($mkShipped)                                    { $phase = "eval" }    # shipped -> eval running
  elseif ((_Probe "ShipStart") -or $allTxt -match "PHASE=ship START") { $phase = "ship" }
  elseif (((_Probe "TrainStart") -and -not (_Probe "TrainSkip")) -or ($allTxt -match "PHASE=train START" -and $allTxt -notmatch "PHASE=train SKIP")) { $phase = "train" }
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

  # train detail (rate/infosets/eta from PROGRESS lines)
  $rate = 0.0; $infosets = -1; $logRate = $false; $etaSec = -1
  if ($trainTxt -or $console) {
    $src = $trainTxt + "`n" + $console
    $mRate = [regex]::Matches($src, "rate=([0-9.]+)\s*it/s")
    if ($mRate.Count) { $rate = [double]$mRate[$mRate.Count - 1].Groups[1].Value; $logRate = $true }
    $mInf = [regex]::Matches($src, "infosets=(\d+)")
    if ($mInf.Count) { $infosets = [int]$mInf[$mInf.Count - 1].Groups[1].Value }
  }
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
  $rawProgress = if ($phase -eq "train") { [int64]$maxCkptIter } elseif ($phase -eq "precompute") { [int64]([Math]::Max($hbBytes, 0)) } elseif ($phase -eq "ship") { [int64]$shipUnit } elseif ($phase -eq "eval") { [int64]($evalStageUnit + $evalSubProgress) } else { [int64]0 }
  $phaseRank = switch ($phase) { "boot" { 1 } "build" { 2 } "precompute" { 3 } "train" { 4 } "ship" { 5 } "eval" { 6 } "eval-done" { 7 } default { 0 } }
  $progressKey = ([int64]$phaseRank * 1000000000000) + $rawProgress
  if ($progressKey -gt $lastProgressKey) { $lastProgressKey = $progressKey; $lastHeartTime = Get-Date }
  $stallMinObs = [int]((Get-Date) - $lastHeartTime).TotalMinutes
  $stalled = ($rawProgress -gt 0 -and $stallMinObs -ge $StallMin)

  Clear-Host
  Write-Host ("=== EMD pipeline: {0}   {1}   (watching {2}m) ===" -f $Run, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), [int]((Get-Date) - $t0).TotalMinutes) -ForegroundColor Cyan
  $boxColor = if ($boxRaw -match "running") { "Green" } elseif ($boxRaw) { "Yellow" } else { "DarkGray" }
  Write-Host ("box:    {0}" -f $boxLine) -ForegroundColor $boxColor

  $phaseColor = switch ($phase) { "train" { "Green" } "precompute" { "Cyan" } "build" { "Yellow" } "boot" { "Yellow" } "ship" { "Magenta" } "eval" { "Magenta" } "eval-done" { "Green" } default { "DarkGray" } }
  Write-Host ("PHASE:  {0}" -f $phase.ToUpper()) -ForegroundColor $phaseColor

  # 5-phase marker strip (DONE green / pending gray) so resume-at-first-incomplete is visible at a glance.
  function _Mk([bool]$b) { if ($b) { "[x]" } else { "[ ]" } }
  Write-Host ("phases: 1.BUILD/PRECOMP {0}  2.TRAIN {1}  4.SHIP {2}  5.EVAL {3}" -f `
    (_Mk $precompComplete), (_Mk $mkConverged), (_Mk $mkShipped), (_Mk $mkEvaled)) -ForegroundColor DarkCyan
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
    $frozenTxt, $frozenNameTxt, $frozenModeTxt, $ShipName, (Fmt-Bytes $serveBytes), (Fmt-Bytes $verdictBytes), `
    (Fmt-Bytes $shipLogBytes), (Fmt-Bytes $evalEmdLogBytes), (Fmt-Bytes $evalBaseLogBytes), (Fmt-Bytes $evalCombineLogBytes)) -ForegroundColor DarkCyan
  if ($phase -eq "eval" -or $phase -eq "eval-done") {
    Write-Host ("partials: emd={0}  iter62000={1}" -f (Fmt-Bytes $evalEmdPartBytes), (Fmt-Bytes $evalBasePartBytes)) -ForegroundColor DarkCyan
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
    $rateTxt = if ($logRate) { "{0:N3} it/s (log)" -f $rate } else { "{0:N3} it/s (est)" -f $rate }
    Write-Host ("rate:   {0}   infosets: {1}   ETA: {2}" -f $rateTxt, $infTxt, $etaStr) -ForegroundColor Yellow
    Write-Host ("ckpts:  {0} versioned (max iter {1})   |   frozen table {2:N1} MB" -f $nCkpt, $maxCkptIter, $frozenMB) -ForegroundColor DarkCyan
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
        $tmpPath = Join-Path $env:TEMP ("emd_watch_verdict_{0}.json" -f ([guid]::NewGuid().ToString('N')))
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
        Write-Host "  activation: use watch_emd_pipeline_v2.ps1 / _emd_auto_activate_gate.py for PASS/FAIL." -ForegroundColor Yellow
        Write-Host "              This legacy watcher does not declare activation eligibility from the old two-boolean shortcut." -ForegroundColor Yellow
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
