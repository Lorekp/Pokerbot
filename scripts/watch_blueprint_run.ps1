# At-a-glance dashboard for the FULL native CFR+ 6-max blueprint run (lever-enabled). READ-ONLY.
#   powershell -ExecutionPolicy Bypass -File scripts\watch_blueprint_run.ps1 -Run bp_dcfr_1 -Iters 62000
#   powershell -ExecutionPolicy Bypass -File scripts\watch_blueprint_run.ps1 -Run bp_dcfr_1 -Refresh 30 -StallMin 20
#
# Shows: live spot box + uptime, iters_done/target (% bar), it/s rate, n_infosets, ETA, rough $ spent +
# ECC (estimated cost to completion), a STALL alarm, AND the active LEVERS (work_steal/dcfr) + abstraction
# (EMD stamp) parsed from the box's train log. Sources: the live checkpoint listing (heartbeat, every
# --checkpoint-every iters) + the latest.nlcfr header (authoritative iters_done) + the train.log (rate,
# infosets, levers). Pairs with blueprint_run_babysitter.ps1. ASCII ONLY (PS 5.1 cp1252).
param(
  [string]$Run = "bp_dcfr_1",
  [int]$Iters = 62000,                 # the --iters target (for % + ETA)
  [int]$Refresh = 45,
  [double]$SpotPrice = 1.88,           # approx spot $/hr fallback (c6a.32xlarge ~ $1.88; spotpick overrides)
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
  $tmp = Join-Path $env:TEMP "bp_watch_probe.nlcfr"
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

  # --- checkpoint listing (the live heartbeat) ---
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

  $itersDone = Get-LatestItersDone
  if ($itersDone -lt 0) { $itersDone = $maxCkptIter }
  $frac = if ($Iters -gt 0 -and $itersDone -ge 0) { [double]$itersDone / $Iters } else { 0 }

  # --- stall detection ---
  if ($maxCkptIter -gt $lastCkptIter) { $lastCkptIter = $maxCkptIter; $lastCkptChangeTime = Get-Date }
  $stallMinObs = [int]((Get-Date) - $lastCkptChangeTime).TotalMinutes
  $stalled = ($maxCkptIter -ge 0 -and $stallMinObs -ge $StallMin)

  # --- rate + infosets + LEVERS from the train log (lands at run end; PROGRESS lines during run if tee'd) ---
  $rate = 0.0; $infosets = -1; $logRate = $false; $leverLine = ""; $emdLine = ""
  $logTxt = (aws s3 cp "s3://$Bucket/$LOGKEY" - --region $Region 2>$null | Out-String)
  if ($logTxt) {
    $mRate = [regex]::Matches($logTxt, "rate=([0-9.]+)\s*it/s")
    if ($mRate.Count) { $rate = [double]$mRate[$mRate.Count - 1].Groups[1].Value; $logRate = $true }
    $mInf = [regex]::Matches($logTxt, "infosets=(\d+)")
    if ($mInf.Count) { $infosets = [int]$mInf[$mInf.Count - 1].Groups[1].Value }
    $mLev = [regex]::Match($logTxt, "LEVERS work_steal=\S+ dcfr=\S+[^\r\n]*")
    if ($mLev.Success) { $leverLine = $mLev.Value.Trim() }
    $mEmd = [regex]::Match($logTxt, "EMD frozen lookup ACTIVE:[^\r\n]*")
    if ($mEmd.Success) { $emdLine = ($mEmd.Value -replace "EMD frozen lookup ACTIVE:\s*", "EMD ").Trim() }
  }
  if (-not $logRate -and $upMin -gt 0 -and $itersDone -gt 0) { $rate = $itersDone / ($upMin * 60.0) }
  $etaStr = "--"; $etaSec = -1
  if ($rate -gt 1e-9 -and $itersDone -ge 0 -and $itersDone -lt $Iters) {
    $etaSec = ($Iters - $itersDone) / $rate
    if ($etaSec -lt 5400) { $etaStr = ("~{0}m" -f [int]($etaSec / 60)) } else { $etaStr = ("~{0:N1}h" -f ($etaSec / 3600.0)) }
  } elseif ($itersDone -ge $Iters) { $etaStr = "DONE" }
  $ecc = if ($etaSec -ge 0) { ($etaSec / 3600.0) * $livePrice } else { -1 }
  $spent = ($upMin / 60.0) * $livePrice

  Clear-Host
  Write-Host ("=== blueprint train: {0}   {1}   (watching {2}m) ===" -f $Run, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), [int]((Get-Date) - $t0).TotalMinutes) -ForegroundColor Cyan
  $boxColor = if ($boxRaw -match "running") { "Green" } elseif ($boxRaw) { "Yellow" } else { "DarkGray" }
  Write-Host ("box:    {0}" -f $boxLine) -ForegroundColor $boxColor
  if ($leverLine) { Write-Host ("levers: {0}   {1}" -f $leverLine, $emdLine) -ForegroundColor Green }
  elseif ($emdLine) { Write-Host ("abstr:  {0}" -f $emdLine) -ForegroundColor Green }
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
