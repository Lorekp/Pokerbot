# At-a-glance dashboard for the NATIVE Rust CFR+ 6-max blueprint training run. READ-ONLY.
#   powershell -ExecutionPolicy Bypass -File scripts\watch_native_cfr.ps1 -Run run_bp1 -Iters 20000
#   powershell -ExecutionPolicy Bypass -File scripts\watch_native_cfr.ps1 -Run run_bp1 -Iters 20000 -Refresh 30 -SpotPrice 1.2 -StallMin 20
#
# Shows: the live spot box + uptime, current iters_done/target (% bar), it/s rate, n_infosets, ETA to
# target, rough $ spent (uptime_h * spot $/hr), and a STALL alarm if no NEW checkpoint in -StallMin min.
# Sources: the S3 train.log (.../<Run>/logs/train.log, written at run end) AND the live checkpoint listing
# (.../<Run>/checkpoints/iter_NNNNNN.nlcfr, written every --checkpoint-every iters DURING the run) — the
# checkpoints are the live heartbeat (the log only lands when the box exits), so progress is visible in
# real time even though the trainer is one long-running process.
#
# Pairs with native_cfr_babysitter.ps1 (which launches/relaunches). ASCII ONLY (PS 5.1 cp1252).
param(
  [string]$Run = "run_bp1",
  [int]$Iters = 20000,                 # the --iters target (for % + ETA)
  [int]$Refresh = 45,
  [double]$SpotPrice = 1.88,           # approx spot $/hr (c6a.32xlarge auto-pick ~ $1.88; adjust per box)
  [int]$StallMin = 20,                 # alarm if no new checkpoint in this many minutes
  [string]$Region = "eu-central-1",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr"
)
$RUN_PREFIX = "$Prefix/$Run"
$CKPT_PREFIX = "$RUN_PREFIX/checkpoints/"
$LATEST      = "$RUN_PREFIX/checkpoints/latest.nlcfr"
$LOGKEY      = "$RUN_PREFIX/logs/train.log"
$t0 = Get-Date
$lastCkptIter = -1
$lastCkptChangeTime = Get-Date

function Bar([double]$frac, [int]$width = 40) {
  if ($frac -lt 0) { $frac = 0 }; if ($frac -gt 1) { $frac = 1 }
  $d = [int]([math]::Round($frac * $width))
  return ("#" * $d) + ("-" * ($width - $d))
}
function Get-LatestItersDone {
  # iters_done from the latest.nlcfr header: magic "NLCFR001"(8) + iters_done u64 LE(8). (trainer.rs)
  $tmp = Join-Path $env:TEMP "native_cfr_watch_probe.nlcfr"
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
  # spot $/hr for instance type $it from the shared spotpick catalog (the SAME source the babysitter
  # auto-picks from), cached per type. Falls back to -SpotPrice if spotpick/python is unavailable.
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
  $livePrice = Get-SpotpickPrice $instType   # spot $/hr of the LIVE box's type (spotpick catalog); -SpotPrice fallback

  # --- checkpoint listing (the live heartbeat) ---
  $ls = aws s3 ls "s3://$Bucket/$CKPT_PREFIX" --region $Region 2>$null
  $ckptIters = @()
  $lastCkptTimeUtc = $null
  if ($ls) {
    foreach ($line in $ls) {
      $name = ($line -split "\s+")[-1]
      if ($name -match "^iter_(\d+)\.nlcfr$") { $ckptIters += [int]$Matches[1] }
      # capture the S3 LastModified of the most recent line (date time at cols 0..1)
      if ($line -match "^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})") {
        try { $lm = [datetime]::ParseExact("$($Matches[1]) $($Matches[2])", "yyyy-MM-dd HH:mm:ss", $null); if (-not $lastCkptTimeUtc -or $lm -gt $lastCkptTimeUtc) { $lastCkptTimeUtc = $lm } } catch {}
      }
    }
  }
  $maxCkptIter = if ($ckptIters.Count) { ($ckptIters | Measure-Object -Maximum).Maximum } else { -1 }
  $nCkpt = $ckptIters.Count

  # --- authoritative iters_done from latest.nlcfr (covers the in-flight chunk past the last versioned ckpt) ---
  $itersDone = Get-LatestItersDone
  if ($itersDone -lt 0) { $itersDone = $maxCkptIter }    # fall back to the newest versioned checkpoint
  $frac = if ($Iters -gt 0 -and $itersDone -ge 0) { [double]$itersDone / $Iters } else { 0 }

  # --- stall detection: has the max checkpoint iter advanced since we last saw it? ---
  if ($maxCkptIter -gt $lastCkptIter) { $lastCkptIter = $maxCkptIter; $lastCkptChangeTime = Get-Date }
  $stallMinObs = [int]((Get-Date) - $lastCkptChangeTime).TotalMinutes
  $stalled = ($maxCkptIter -ge 0 -and $stallMinObs -ge $StallMin)

  # --- rate + ETA from the log (PROGRESS lines: 'rate=<x> it/s', 'infosets=<n>'); else from uptime ---
  $rate = 0.0; $infosets = -1; $logRate = $false
  $logTxt = (aws s3 cp "s3://$Bucket/$LOGKEY" - --region $Region 2>$null | Out-String)
  if ($logTxt) {
    $mRate = [regex]::Matches($logTxt, "rate=([0-9.]+)\s*it/s")
    if ($mRate.Count) { $rate = [double]$mRate[$mRate.Count - 1].Groups[1].Value; $logRate = $true }
    $mInf = [regex]::Matches($logTxt, "infosets=(\d+)")
    if ($mInf.Count) { $infosets = [int]$mInf[$mInf.Count - 1].Groups[1].Value }
  }
  # If the log hasn't landed yet (box still running), estimate rate from uptime + iters.
  if (-not $logRate -and $upMin -gt 0 -and $itersDone -gt 0) { $rate = $itersDone / ($upMin * 60.0) }
  $etaStr = "--"; $etaSec = -1
  if ($rate -gt 1e-9 -and $itersDone -ge 0 -and $itersDone -lt $Iters) {
    $etaSec = ($Iters - $itersDone) / $rate
    if ($etaSec -lt 5400) { $etaStr = ("~{0}m" -f [int]($etaSec / 60)) }
    else { $etaStr = ("~{0:N1}h" -f ($etaSec / 3600.0)) }
  } elseif ($itersDone -ge $Iters) { $etaStr = "DONE" }
  # ECC = estimated cost to COMPLETION = remaining time x the LIVE box's spot $/hr (spotpick)
  $ecc = if ($etaSec -ge 0) { ($etaSec / 3600.0) * $livePrice } else { -1 }

  # --- rough $ spent (this box's uptime only; relaunches reset uptime so this is per-box, not cumulative) ---
  $spent = ($upMin / 60.0) * $livePrice

  Clear-Host
  Write-Host ("=== native-CFR train: {0}   {1}   (watching {2}m) ===" -f $Run, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), [int]((Get-Date) - $t0).TotalMinutes) -ForegroundColor Cyan
  $boxColor = if ($boxRaw -match "running") { "Green" } elseif ($boxRaw) { "Yellow" } else { "DarkGray" }
  Write-Host ("box:    {0}" -f $boxLine) -ForegroundColor $boxColor
  $idTxt = if ($itersDone -ge 0) { "$itersDone" } else { "?" }
  $barColor = if ($frac -ge 0.99) { "Green" } elseif ($frac -ge 0.5) { "Cyan" } else { "Yellow" }
  Write-Host ("iters:  {0}/{1}  [{2}] {3:P1}" -f $idTxt, $Iters, (Bar $frac), $frac) -ForegroundColor $barColor
  $infTxt = if ($infosets -ge 0) { "{0:N0}" -f $infosets } else { "(in log only)" }
  $rateTxt = if ($logRate) { "{0:N3} it/s (log)" -f $rate } else { "{0:N3} it/s (est)" -f $rate }
  Write-Host ("rate:   {0}   infosets: {1}   ETA: {2}" -f $rateTxt, $infTxt, $etaStr) -ForegroundColor Yellow
  Write-Host ("ckpts:  {0} versioned (max iter {1})" -f $nCkpt, $maxCkptIter) -ForegroundColor DarkCyan
  $eccTxt = if ($ecc -ge 0) { ("~`${0:N2}" -f $ecc) } else { "--" }
  Write-Host ("cost:   spent ~`${0:N2} this box   |   ECC {1} to finish   (@ `${2:N3}/hr {3})" -f $spent, $eccTxt, $livePrice, $instType) -ForegroundColor Magenta
  if ($maxCkptIter -ge 0) {
    if ($stalled) {
      Write-Host ("STALL:  no new checkpoint in {0}m (>= {1}m alarm) - box hung, reclaimed mid-chunk, or first chunk still running" -f $stallMinObs, $StallMin) -ForegroundColor Red
    } else {
      Write-Host ("ckpt:   last advanced {0}m ago (alarm at {1}m)" -f $stallMinObs, $StallMin) -ForegroundColor DarkGray
    }
  }
  if ($itersDone -ge $Iters -and $itersDone -ge 0) {
    Write-Host ""
    Write-Host "RUN COMPLETE (latest checkpoint reached target). The babysitter terminates the box." -ForegroundColor Green
    break
  }
  Write-Host ""
  Write-Host ("(refresh {0}s; Ctrl-C to stop)" -f $Refresh)
  Start-Sleep -Seconds $Refresh
}
