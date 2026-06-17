# Hourly monitor for homeserver exploitability AB shard build.
# Runs poll_ab_shards.sh on homeserver via SSH; appends to local log; emits loop tick.
$ErrorActionPreference = "Continue"
$Root = "."
$RemoteRoot = "."
$LocalLog = Join-Path $Root "logs\ab_shards_monitor.jsonl"
$IntervalSec = 3600

function Get-ShardStatus {
    $raw = ssh -o ConnectTimeout=30 -o BatchMode=yes monitor "bash $RemoteRoot/scripts/poll_ab_shards.sh" 2>&1
    return ($raw | Out-String).Trim()
}

function Write-MonitorLine {
    param([string]$Json, [string]$Note = "")
    $dir = Split-Path $LocalLog -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $entry = if ($Note) { "$Json # $Note" } else { $Json }
    Add-Content -Path $LocalLog -Value $entry -Encoding utf8
    Write-Host $entry
}

# Initial check (immediate)
$status = Get-ShardStatus
Write-MonitorLine -Json $status -Note "initial"

if ($status -match '"status":"done"') {
    Write-Host "AGENT_LOOP_TICK_ab_shards DONE"
    exit 0
}
if ($status -match '"status":"(failed|stale)"') {
    Write-Host "AGENT_LOOP_TICK_ab_shards FAILED"
    exit 2
}

# Hourly loop
while ($true) {
    Start-Sleep -Seconds $IntervalSec
    $status = Get-ShardStatus
    Write-MonitorLine -Json $status
    if ($status -match '"status":"done"') {
        Write-Host "AGENT_LOOP_TICK_ab_shards DONE"
        break
    }
    if ($status -match '"status":"(failed|stale)"') {
        Write-Host "AGENT_LOOP_TICK_ab_shards FAILED"
        break
    }
    $tick = 'AGENT_LOOP_TICK_ab_shards {"prompt":"Check homeserver AB shard build progress; when done SCP parquets and run --kl-only locally"}'
    Write-Host $tick
}
