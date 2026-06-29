# Sequential 5000-iteration cloud A/B for EMD trainer CPU fanout.
#
# Default is DRY-RUN. Add -Launch to spend cloud money.
# The two runs are intentionally sequential:
#   1. old/default fanout: n_deals=200
#   2. high-fanout shape: n_deals=1024
#
# Both runs use the same current code, same K160/R400 frozen lookup, same checkpoint cadence,
# TRAIN_PROFILE_JSON telemetry, STOP_AFTER_SHIP, and CloudWatch CPU summary.
param(
  [string]$RunBase = "",
  [int]$Iters = 5000,
  [int]$OldNDeals = 200,
  [int]$NewNDeals = 1024,
  [int]$NumBuckets = 160,
  [string]$Centroids = "data/abstraction/emd_buckets_160_r400.json",
  [string]$Frozen = "data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn",
  [int]$NRunouts = 400,
  [int]$CheckpointEvery = 1000,
  [int]$TrainProfileEvery = 50,
  [int]$Vcpus = 128,
  [int]$MaxHours = 4,
  [int]$MaxLaunch = 4,
  [int]$MinRelaunchGapSec = 600,
  [int]$QuotaDrainSec = 600,
  [int]$CpuLagWaitSec = 420,
  [switch]$OnDemand,
  [switch]$Launch,
  [string]$Region = "eu-central-1",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not $RunBase) {
  $RunBase = "cpu_fanout_ab_" + (Get-Date -Format "yyyyMMdd_HHmm")
}

$OldRun = "${RunBase}_old_n${OldNDeals}"
$NewRun = "${RunBase}_new_n${NewNDeals}"
$ArtifactJson = "artifacts/${RunBase}_summary.json"
$ArtifactMd = "artifacts/${RunBase}_summary.md"

function Format-Cmd([string[]]$ArgsList) {
  return "powershell " + ($ArgsList -join " ")
}

function Require-Path([string]$Path) {
  if (-not (Test-Path $Path)) {
    throw "Missing required path: $Path"
  }
}

function Get-TableId([string]$CentroidsPath, [int]$Runouts) {
  $out = (& python -c "from pathlib import Path; import sys; sys.path.insert(0,'scripts'); import launch_emd_pipeline_spot as l; print(l.table_id(Path(sys.argv[1]),20,int(sys.argv[2])))" $CentroidsPath $Runouts 2>&1)
  if ($LASTEXITCODE -ne 0) {
    throw "table_id computation failed: $out"
  }
  return ("$out").Trim()
}

function Ensure-ShaSidecar([string]$FrozenPath) {
  $shaPath = "$FrozenPath.sha256"
  if (Test-Path $shaPath) { return $shaPath }
  Write-Host "SHA sidecar missing; writing $shaPath" -ForegroundColor Yellow
  $out = (& python -c "import hashlib, pathlib, sys; p=pathlib.Path(sys.argv[1]); q=pathlib.Path(sys.argv[2]); q.write_text(hashlib.sha256(p.read_bytes()).hexdigest() + '\n', encoding='ascii')" $FrozenPath $shaPath 2>&1)
  if ($LASTEXITCODE -ne 0) {
    throw "failed to write SHA sidecar: $out"
  }
  return $shaPath
}

Require-Path $Centroids
Require-Path $Frozen
$FrozenSha = Ensure-ShaSidecar $Frozen
$TableId = Get-TableId $Centroids $NRunouts

function Prepare-FrozenCache([string]$Run) {
  $frozenS3 = "s3://$Bucket/$Prefix/$Run/frozen/$TableId.emdfrzn"
  $shaS3 = "$frozenS3.sha256"
  if (-not $Launch) {
    Write-Host "Would pre-upload frozen cache:" -ForegroundColor DarkCyan
    Write-Host "  aws s3 cp $Frozen $frozenS3 --region $Region --no-progress"
    Write-Host "  aws s3 cp $FrozenSha $shaS3 --region $Region --no-progress"
    return
  }
  Write-Host "Pre-uploading frozen cache for $Run -> $frozenS3" -ForegroundColor Cyan
  aws s3 cp $Frozen $frozenS3 --region $Region --no-progress | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "frozen upload failed for $Run" }
  aws s3 cp $FrozenSha $shaS3 --region $Region --no-progress | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "frozen SHA upload failed for $Run" }
}

function Build-BabysitterArgs([string]$Run, [int]$NDeals, [string]$ShipName) {
  $args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\emd_pipeline_babysitter_v2.ps1",
    "-Run", $Run,
    "-Iters", "$Iters",
    "-NDeals", "$NDeals",
    "-NumBuckets", "$NumBuckets",
    "-Centroids", $Centroids,
    "-NRunouts", "$NRunouts",
    "-CheckpointEvery", "$CheckpointEvery",
    "-MaxHours", "$MaxHours",
    "-CrateSrc", "tools\battery_native",
    "-TrainProfileEvery", "$TrainProfileEvery",
    "-NoConvergeStop",
    "-ShipName", $ShipName,
    "-EvalSeeds", "1",
    "-EvalHands", "1",
    "-MaxLaunch", "$MaxLaunch",
    "-MinRelaunchGapSec", "$MinRelaunchGapSec",
    "-QuotaDrainSec", "$QuotaDrainSec",
    "-Vcpus", "$Vcpus",
    "-Spotpick",
    "-StopAfterShip"
  )
  if ($OnDemand) { $args += "-OnDemand" }
  return $args
}

function Build-WatcherArgs([string]$Run, [int]$NDeals, [string]$ShipName) {
  return @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\watch_emd_pipeline_v2.ps1",
    "-Run", $Run,
    "-Iters", "$Iters",
    "-NDeals", "$NDeals",
    "-NumBuckets", "$NumBuckets",
    "-Centroids", $Centroids,
    "-NRunouts", "$NRunouts",
    "-ShipName", $ShipName,
    "-ExpectTableMB", "153",
    "-Refresh", "60",
    "-StopAfterShip"
  )
}

function Get-S3TextOrEmpty([string]$Key) {
  return (aws s3 cp "s3://$Bucket/$Key" - --region $Region 2>$null | Out-String)
}

function Get-TrainSummary([string]$Run, [int]$NDeals) {
  $runPrefix = "$Prefix/$Run"
  $txt = Get-S3TextOrEmpty "$runPrefix/logs/train.log"
  $source = "logs/train.log"
  if (-not $txt.Trim()) {
    $txt = Get-S3TextOrEmpty "$runPrefix/progress/train_status.txt"
    $source = "progress/train_status.txt"
  }

  $progressRegex = "(TRAIN_HEARTBEAT|PROGRESS)\s+iters_done=(\d+)/(\d+)\s+infosets=(\d+)\s+elapsed=([0-9.]+)s\s+rate=([0-9.]+)\s+it/s\s+eta=([+\-0-9.]+)s"
  $matches = [regex]::Matches($txt, $progressRegex)
  $last = if ($matches.Count) { $matches[$matches.Count - 1] } else { $null }
  $rate = if ($last) { [double]$last.Groups[6].Value } else { 0.0 }
  $sampledDealsPerSec = $rate * [double]$NDeals
  $profileLines = @($txt -split "`r?`n" | Where-Object { $_ -like "TRAIN_PROFILE_JSON *" })

  $lastProfile = $null
  if ($profileLines.Count -gt 0) {
    try {
      $lastProfile = $profileLines[$profileLines.Count - 1].Substring("TRAIN_PROFILE_JSON ".Length) | ConvertFrom-Json
    } catch {
      $lastProfile = $null
    }
  }

  return [pscustomobject]@{
    Source = $source
    ProgressRows = $matches.Count
    ProfileRows = $profileLines.Count
    ItersDone = if ($last) { [int64]$last.Groups[2].Value } else { -1 }
    ItersTarget = if ($last) { [int64]$last.Groups[3].Value } else { $Iters }
    Infosets = if ($last) { [int64]$last.Groups[4].Value } else { -1 }
    ElapsedSec = if ($last) { [double]$last.Groups[5].Value } else { -1.0 }
    RateItersPerSec = $rate
    SampledDealsPerSec = $sampledDealsPerSec
    SeatTraversalsPerSec = $sampledDealsPerSec * 6.0
    LastProfile = $lastProfile
  }
}

function Get-RunInstances([string]$Run) {
  $json = (aws ec2 describe-instances --region $Region `
    --filters "Name=tag:Run,Values=$Run" `
    --query "Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime,State.Name]" --output json 2>$null | Out-String)
  if (-not $json.Trim()) { return @() }
  $rows = @()
  foreach ($row in ($json | ConvertFrom-Json)) {
    $rows += ,$row
  }
  return $rows
}

function Get-InstanceCpuSummary([object]$Row) {
  $iid = [string]$Row[0]
  $instanceType = [string]$Row[1]
  $launchTimeRaw = [string]$Row[2]
  $state = [string]$Row[3]
  $launchTime = [datetimeoffset]::Parse($launchTimeRaw).UtcDateTime
  $start = $launchTime.AddMinutes(-2)
  $end = (Get-Date).ToUniversalTime()
  $json = (aws cloudwatch get-metric-statistics --region $Region --namespace AWS/EC2 `
    --metric-name CPUUtilization --dimensions "Name=InstanceId,Value=$iid" `
    --statistics Average Maximum --period 300 `
    --start-time $($start.ToString("o")) --end-time $($end.ToString("o")) `
    --output json 2>$null | Out-String)
  if (-not $json.Trim()) {
    return [pscustomobject]@{ InstanceId = $iid; InstanceType = $instanceType; State = $state; LaunchTime = $launchTimeRaw; Samples = 0; AverageCpuPct = $null; MaxCpuPct = $null; LatestTimestamp = $null }
  }
  $dps = @((($json | ConvertFrom-Json).Datapoints))
  if (-not $dps.Count) {
    return [pscustomobject]@{ InstanceId = $iid; InstanceType = $instanceType; State = $state; LaunchTime = $launchTimeRaw; Samples = 0; AverageCpuPct = $null; MaxCpuPct = $null; LatestTimestamp = $null }
  }
  $avg = ($dps | Measure-Object -Property Average -Average).Average
  $max = ($dps | Measure-Object -Property Maximum -Maximum).Maximum
  $latest = $dps | Sort-Object -Property Timestamp | Select-Object -Last 1
  return [pscustomobject]@{
    InstanceId = $iid
    InstanceType = $instanceType
    State = $state
    LaunchTime = $launchTimeRaw
    Samples = $dps.Count
    AverageCpuPct = [double]$avg
    MaxCpuPct = [double]$max
    LatestTimestamp = [string]$latest.Timestamp
  }
}

function Get-RunSummary([string]$Label, [string]$Run, [int]$NDeals) {
  $train = Get-TrainSummary $Run $NDeals
  $instances = @(Get-RunInstances $Run)
  $cpuRows = @()
  foreach ($row in $instances) {
    $cpuRows += (Get-InstanceCpuSummary $row)
  }
  $cpuWithSamples = @($cpuRows | Where-Object { $_.Samples -gt 0 })
  $avgCpu = if ($cpuWithSamples.Count) { ($cpuWithSamples | Measure-Object -Property AverageCpuPct -Average).Average } else { $null }
  $maxCpu = if ($cpuWithSamples.Count) { ($cpuWithSamples | Measure-Object -Property MaxCpuPct -Maximum).Maximum } else { $null }
  return [pscustomobject]@{
    Label = $Label
    Run = $Run
    NDeals = $NDeals
    Iters = $Iters
    TrainProfileEvery = $TrainProfileEvery
    CheckpointEvery = $CheckpointEvery
    Train = $train
    Cpu = [pscustomobject]@{
      AverageCpuPctAcrossInstances = $avgCpu
      MaxCpuPctAcrossInstances = $maxCpu
      Instances = $cpuRows
    }
  }
}

function Write-SummaryArtifacts([object[]]$Summaries) {
  $Summaries | ConvertTo-Json -Depth 24 | Set-Content -Path $ArtifactJson -Encoding Ascii
  $old = $Summaries | Where-Object { $_.Label -eq "old-default-n200" } | Select-Object -First 1
  $new = $Summaries | Where-Object { $_.Label -eq "new-fanout-n1024" } | Select-Object -First 1
  $speedup = ""
  if ($old -and $new -and $old.Train.SampledDealsPerSec -gt 0) {
    $speedup = "{0:N2}x sampled-deals/sec" -f ($new.Train.SampledDealsPerSec / $old.Train.SampledDealsPerSec)
  } else {
    $speedup = "n/a"
  }
  $lines = @(
    "# EMD CPU Fanout A/B",
    "",
    ("- run base: {0}" -f $RunBase),
    ("- table id: {0}" -f $TableId),
    ("- centroids: {0}" -f $Centroids),
    ("- frozen: {0}" -f $Frozen),
    "- speedup: $speedup",
    "",
    "| label | run | n_deals | it/s | sampled deals/s | seat trav/s | avg CPU | max CPU | profile rows |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|"
  )
  foreach ($s in $Summaries) {
    $avgCpu = if ($null -ne $s.Cpu.AverageCpuPctAcrossInstances) { "{0:N1}%" -f $s.Cpu.AverageCpuPctAcrossInstances } else { "n/a" }
    $maxCpu = if ($null -ne $s.Cpu.MaxCpuPctAcrossInstances) { "{0:N1}%" -f $s.Cpu.MaxCpuPctAcrossInstances } else { "n/a" }
    $lines += ("| {0} | `{1}` | {2} | {3:N3} | {4:N0} | {5:N0} | {6} | {7} | {8} |" -f `
      $s.Label, $s.Run, $s.NDeals, $s.Train.RateItersPerSec, $s.Train.SampledDealsPerSec, `
      $s.Train.SeatTraversalsPerSec, $avgCpu, $maxCpu, $s.Train.ProfileRows)
  }
  $lines | Set-Content -Path $ArtifactMd -Encoding Ascii
}

function Invoke-ProbeRun([string]$Label, [string]$Run, [int]$NDeals) {
  $shipName = "emd_cpu_${Label}"
  $shipName = $shipName -replace "[^A-Za-z0-9_.-]", "_"
  $args = Build-BabysitterArgs $Run $NDeals $shipName
  $watchArgs = Build-WatcherArgs $Run $NDeals $shipName
  Write-Host ""
  Write-Host "=== ${Label}: $Run ===" -ForegroundColor Cyan
  Write-Host ("fanout: n_deals={0}, vcpus={1}, {2:N1} deal tasks/vCPU/iter" -f $NDeals, $Vcpus, ([double]$NDeals / [Math]::Max(1, $Vcpus))) -ForegroundColor Cyan
  Prepare-FrozenCache $Run
  Write-Host "Babysitter:"
  Write-Host (Format-Cmd $args)
  Write-Host "Watcher:"
  Write-Host (Format-Cmd $watchArgs)
  if (-not $Launch) { return $null }

  & powershell @args
  if ($LASTEXITCODE -ne 0) {
    throw "$Label babysitter failed with exit code $LASTEXITCODE"
  }
  if ($CpuLagWaitSec -gt 0) {
    Write-Host "Waiting ${CpuLagWaitSec}s for CloudWatch CPU datapoints to settle..." -ForegroundColor DarkCyan
    Start-Sleep -Seconds $CpuLagWaitSec
  }
  $summary = Get-RunSummary $Label $Run $NDeals
  Write-Host ("SUMMARY {0}: {1:N3} it/s, {2:N0} sampled deals/s, avg CPU={3}, max CPU={4}, profile_rows={5}" -f `
    $Label, $summary.Train.RateItersPerSec, $summary.Train.SampledDealsPerSec, `
    $(if ($null -ne $summary.Cpu.AverageCpuPctAcrossInstances) { "{0:N1}%" -f $summary.Cpu.AverageCpuPctAcrossInstances } else { "n/a" }), `
    $(if ($null -ne $summary.Cpu.MaxCpuPctAcrossInstances) { "{0:N1}%" -f $summary.Cpu.MaxCpuPctAcrossInstances } else { "n/a" }), `
    $summary.Train.ProfileRows) -ForegroundColor Green
  return $summary
}

Write-Host "=== EMD CPU fanout A/B setup ===" -ForegroundColor Cyan
Write-Host "RunBase: $RunBase"
Write-Host "TableId: $TableId"
Write-Host "Old first: $OldRun (n_deals=$OldNDeals)"
Write-Host "New second: $NewRun (n_deals=$NewNDeals)"
Write-Host "Iters: $Iters  CheckpointEvery: $CheckpointEvery  TrainProfileEvery: $TrainProfileEvery"
Write-Host "Mode: $(if ($Launch) { 'LAUNCH' } else { 'DRY-RUN' })"

$summaries = @()
$oldSummary = Invoke-ProbeRun "old-default-n200" $OldRun $OldNDeals
if ($oldSummary) { $summaries += $oldSummary }
$newSummary = Invoke-ProbeRun "new-fanout-n1024" $NewRun $NewNDeals
if ($newSummary) { $summaries += $newSummary }

if ($Launch) {
  Write-SummaryArtifacts $summaries
  Write-Host ""
  Write-Host "Wrote:"
  Write-Host "  $ArtifactJson"
  Write-Host "  $ArtifactMd"
} else {
  Write-Host ""
  Write-Host "DRY RUN only. Re-run with -Launch to start the two sequential EC2 probes." -ForegroundColor Yellow
}
