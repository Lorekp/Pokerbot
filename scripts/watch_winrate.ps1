<#
.SYNOPSIS
    Live at-a-glance dashboard for an overnight_winrate.ps1 run.

.DESCRIPTION
    Re-reads the run's pooled_summary.json + shard checkpoints every few seconds and redraws a compact
    dashboard: per-field win-rate + 95% CI + a WIN/LOSE/~0 verdict, sample size (decks / hero-hands /
    shards), throughput since the watcher started, a freshness/liveness indicator (time since the last
    checkpoint), and an estimate of how many more decks/how long to reach a target CI half-width.
    Read-only; safe to start/stop anytime, independent of the run itself.

.EXAMPLE
    .\scripts\watch_winrate.ps1
    .\scripts\watch_winrate.ps1 -OutDir runs\overnight_winrate -IntervalSec 3 -TargetHalfWidth 3.0
#>
[CmdletBinding()]
param(
    [string]$OutDir = "runs\overnight_winrate",
    [int]$IntervalSec = 5,
    [double]$TargetHalfWidth = 5.0,   # bb/100; ETA is to shrink each field's 95% half-width to this
    [int]$RecentLines = 6
)

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
# Format numbers with '.' decimals (consistent with the Python POOL lines) regardless of OS locale.
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture

$summaryJson = Join-Path $OutDir "pooled_summary.json"
$shardDir    = Join-Path $OutDir "shards"
$mainLog     = Join-Path $OutDir "run.log"

$watchStart   = Get-Date
$startShards  = $null
$startDecks   = $null
$lastGood     = $null

function Fmt([double]$v, [int]$w = 8) {
    # Format with an explicit sign (positive;negative;zero), then pad to width. Building the width into
    # the format token ("{0,$w:...}") would make PowerShell read "$w:" as a drive reference and fail.
    return ("{0:+0.0;-0.0;0.0}" -f $v).PadLeft($w)
}

while ($true) {
    $now = Get-Date

    # Load the rolled-up checkpoint (keep the last good copy if it is mid-write).
    $summary = $null
    if (Test-Path $summaryJson) {
        try { $summary = Get-Content $summaryJson -Raw | ConvertFrom-Json } catch { $summary = $lastGood }
    }
    if ($summary) { $lastGood = $summary } else { $summary = $lastGood }

    # Disk-level shard count + last checkpoint freshness.
    $shardFiles = @(Get-ChildItem -Path $shardDir -Filter "*.json" -ErrorAction SilentlyContinue)
    $shardsOnDisk = $shardFiles.Count
    $lastCkptAge = $null
    if ($shardsOnDisk -gt 0) {
        $newest = ($shardFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
        $lastCkptAge = [int]([TimeSpan]($now - $newest)).TotalSeconds
    }

    # Totals across fields (for throughput).
    $totalShards = 0; $totalDecks = 0
    if ($summary) {
        foreach ($p in $summary.PSObject.Properties) {
            $totalShards += [int]$p.Value.shards
            $totalDecks  += [int]$p.Value.decks
        }
    }
    if ($null -eq $startShards) { $startShards = $totalShards; $startDecks = $totalDecks }
    $elapsed = ($now - $watchStart)
    $elapsedMin = [Math]::Max($elapsed.TotalMinutes, 1e-6)
    $shPerMin = ($totalShards - $startShards) / $elapsedMin
    $dkPerMin = ($totalDecks - $startDecks) / $elapsedMin
    $avgDecksPerShard = if ($totalShards -gt 0) { [double]$totalDecks / $totalShards } else { 0 }

    # ---- draw ----
    try { Clear-Host } catch { }   # no-op when stdout is redirected (no real console)
    $live = if ($null -eq $lastCkptAge) { "no checkpoints yet" }
            elseif ($lastCkptAge -le ($IntervalSec * 4 + 30)) { "LIVE (last checkpoint ${lastCkptAge}s ago)" }
            else { "STALLED? (last checkpoint ${lastCkptAge}s ago)" }
    $pyN = @(Get-Process python -ErrorAction SilentlyContinue).Count
    Write-Host ("=== WIN-RATE WATCHER ===  {0}   {1}   refresh {2}s" -f $OutDir, $now.ToString("HH:mm:ss"), $IntervalSec)
    Write-Host ("watching {0:hh\:mm\:ss}   shards on disk: {1}   python procs: {2}   {3}" -f $elapsed, $shardsOnDisk, $pyN, $live)
    Write-Host ""

    if (-not $summary -or $summary.PSObject.Properties.Count -eq 0) {
        Write-Host "  (waiting for the first pooled checkpoint...)"
    } else {
        $hdr = "{0,-26} {1,10} {2,-20} {3,11} {4,8} {5,9} {6,7} {7,6}" -f `
               "FIELD","BASE","95% CI","HALFWIDTH","EXPLOIT","DECKS","HANDS","SHARDS"
        Write-Host $hdr
        Write-Host ("-" * $hdr.Length)
        foreach ($p in ($summary.PSObject.Properties | Sort-Object Name)) {
            $v = $p.Value
            $lo = [double]$v.off_base_ci95[0]; $hi = [double]$v.off_base_ci95[1]
            $verdict = if ($lo -gt 0) { "[WIN+]" } elseif ($hi -lt 0) { "[LOSE-]" } else { "[~0]" }
            $ci = ("[{0},{1}]" -f (Fmt $lo 7), (Fmt $hi 7))
            Write-Host ("{0,-26} {1} {2,-20} +/-{3,6:0.0} {4} {5,9} {6,7} {7,6}  {8}" -f `
                $p.Name,
                (Fmt ([double]$v.off_base_bb100) 9),
                $ci,
                [double]$v.off_base_halfwidth95,
                (Fmt ([double]$v.on_exploit_bb100) 7),
                [int]$v.decks, [int]$v.hero_hands, [int]$v.shards,
                $verdict)
        }
        Write-Host ""
        Write-Host ("throughput: {0,5:0.0} shards/min  (~{1:0} decks/min)   avg {2:0} decks/shard" -f `
            $shPerMin, $dkPerMin, $avgDecksPerShard)

        # Per-field ETA to the target half-width (SE ~ 1/sqrt(decks): need ~ n*(h/t)^2 decks).
        if ($dkPerMin -gt 0 -and $TargetHalfWidth -gt 0) {
            foreach ($p in ($summary.PSObject.Properties | Sort-Object Name)) {
                $v = $p.Value
                $h = [double]$v.off_base_halfwidth95; $n = [int]$v.decks
                if ($h -le $TargetHalfWidth -and $h -gt 0) {
                    Write-Host ("  {0,-26} already within +/-{1:0.0} bb/100" -f $p.Name, $TargetHalfWidth)
                } elseif ($h -gt 0 -and $n -gt 0) {
                    $needDecks = [Math]::Ceiling($n * [Math]::Pow($h / $TargetHalfWidth, 2)) - $n
                    $etaMin = $needDecks / $dkPerMin
                    Write-Host ("  {0,-26} ~{1} more decks for +/-{2:0.0} bb/100  (~{3:0} min at current rate)" -f `
                        $p.Name, $needDecks, $TargetHalfWidth, $etaMin)
                }
            }
        }
    }

    # Recent log tail.
    if (Test-Path $mainLog) {
        Write-Host ""
        Write-Host "recent:"
        Get-Content $mainLog -Tail $RecentLines | ForEach-Object { Write-Host ("  " + $_) }
    }
    Write-Host ""
    Write-Host "(Ctrl-C to stop the watcher; the run keeps going)"

    Start-Sleep -Seconds $IntervalSec
}
