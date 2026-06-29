# cpu_util_sample.ps1 — per-second CPU utilization sampler.
#
# Spawns a child process (the cargo/runner) and records every 1 second:
#   - total CPU% (Processor(_Total)\% Processor Time)
#   - per-process CPU% (child only)
#   - number of threads in the child
#   - wall time
#
# Writes a CSV to the output path and prints the final summary line.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\cpu_util_sample.ps1 `
#     -CommandLine "cmd /c 'cargo run --release --example bench_trainer --release -- 200 200 ./artifacts/flamegraph/blob.bin'" `
#     -OutputCsv ./artifacts/flamegraph/cpu_timeline.csv `
#     -SampleSec 1 `
#     -TimeoutSec 1200
#
# Pass -NoTracking to skip per-process sampling (just total CPU%).

param(
    [Parameter(Mandatory=$true)][string]$CommandLine,
    [Parameter(Mandatory=$true)][string]$OutputCsv,
    [int]$SampleSec = 1,
    [int]$TimeoutSec = 1800,
    [switch]$NoTracking
)

$ErrorActionPreference = 'Stop'
$outDir = Split-Path $OutputCsv -Parent
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }

Write-Host "=== cpu_util_sample.ps1 ==="
Write-Host "Command: $CommandLine"
Write-Host "Output:  $OutputCsv"
Write-Host "Sample:  every ${SampleSec}s, timeout ${TimeoutSec}s"

# --- launch child process (synchronous via cmd /c) ---
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "cmd.exe"
$psi.Arguments = "/c `"$CommandLine`""
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true

$proc = [System.Diagnostics.Process]::Start($psi)
$startTime = Get-Date
Write-Host "Started PID $($proc.Id) at $startTime"

# --- writer thread ---
$csvPath = $OutputCsv
$writer = [System.IO.StreamWriter]::new($csvPath, $false, [System.Text.Encoding]::UTF8)
$writer.WriteLine("timestamp,elapsed_sec,total_cpu_pct,process_cpu_pct,threads")
$writer.Flush()

$totalCpuCounter = New-Object System.Diagnostics.PerformanceCounter('Processor', '% Processor Time', '_Total')
$totalCpuCounter.NextValue() | Out-Null  # prime

# --- per-process tracking: walk child + descendants every sample ---
function Get-ProcessTreeStats($rootPid) {
    try {
        $root = Get-Process -Id $rootPid -ErrorAction SilentlyContinue
        if ($null -eq $root) { return $null }
        $pids = New-Object 'System.Collections.Generic.HashSet[int]'
        $stack = New-Object 'System.Collections.Generic.Stack[int]'
        $stack.Push($rootPid)
        while ($stack.Count -gt 0) {
            $pid = $stack.Pop()
            if ($pids.Contains($pid)) { continue }
            $pids.Add($pid) | Out-Null
            try {
                $p = Get-CimInstance Win32_Process -Filter "ParentProcessId=$pid" -ErrorAction SilentlyContinue
                foreach ($c in $p) { $stack.Push([int]$c.ProcessId) }
            } catch {}
        }
        $totalCpu = 0.0
        $totalThreads = 0
        foreach ($pid in $pids) {
            $p = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($null -eq $p) { continue }
            $totalCpu += [double]$p.TotalProcessorTime.TotalSeconds
            $totalThreads += $p.Threads.Count
        }
        # convert cumulative CPU seconds to instantaneous %: (delta_seconds / elapsed) * 100 / n_cores
        $cpuSec = $totalCpu
        $cores = [Environment]::ProcessorCount
        # caller divides by elapsed to get %
        return [pscustomobject]@{ CpuSec = $cpuSec; Threads = $totalThreads; Cores = $cores }
    } catch { return $null }
}

$prevStats = $null
$prevTime = $null
$totalStartCpu = $totalCpuCounter.NextValue()  # already primed
$sampleCount = 0
$rows = New-Object 'System.Collections.Generic.List[object]'

while (-not $proc.HasExited) {
    Start-Sleep -Seconds $SampleSec
    $now = Get-Date
    $elapsed = ($now - $startTime).TotalSeconds

    $totalPct = $totalCpuCounter.NextValue()
    $procPct = ''
    $threads = ''
    if (-not $NoTracking) {
        $stats = Get-ProcessTreeStats $proc.Id
        if ($null -ne $stats) {
            if ($null -ne $prevStats) {
                $deltaCpu = $stats.CpuSec - $prevStats.CpuSec
                $deltaTime = ($now - $prevTime).TotalSeconds
                $procPct = [math]::Round(($deltaCpu / $deltaTime) * 100.0 / $stats.Cores, 2)
            }
            $threads = $stats.Threads
            $prevStats = $stats
            $prevTime = $now
        }
    }

    $row = [pscustomobject]@{
        ts = $now.ToString('HH:mm:ss')
        elapsed = [math]::Round($elapsed, 1)
        total_cpu_pct = [math]::Round($totalPct, 2)
        process_cpu_pct = $procPct
        threads = $threads
    }
    $rows.Add($row) | Out-Null
    $writer.WriteLine("$($row.ts),$($row.elapsed),$($row.total_cpu_pct),$($row.process_cpu_pct),$($row.threads)")
    $writer.Flush()
    $sampleCount++

    if ($elapsed -gt $TimeoutSec) {
        Write-Host "TIMEOUT after ${TimeoutSec}s — killing child"
        try { $proc.Kill($true) } catch {}
        break
    }
}

# --- drain stdout/stderr ---
$stdout = $proc.StandardOutput.ReadToEnd()
$stderr = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()
$endTime = Get-Date
$wallSec = ($endTime - $startTime).TotalSeconds
$exitCode = $proc.ExitCode

$writer.WriteLine("# exit_code=$exitCode wall_sec=$([math]::Round($wallSec,2)) samples=$sampleCount")
$writer.Close()

# --- summary ---
$validPct = $rows | ForEach-Object { $_.total_cpu_pct } | Where-Object { $_ -gt 0 }
$avgCpu = if ($validPct) { [math]::Round(($validPct | Measure-Object -Average).Average, 2) } else { 0 }
$maxCpu = if ($validPct) { [math]::Round(($validPct | Measure-Object -Maximum).Maximum, 2) } else { 0 }

Write-Host ""
Write-Host "=== SUMMARY ==="
Write-Host ("exit_code   : {0}" -f $exitCode)
Write-Host ("wall_sec    : {0}" -f ([math]::Round($wallSec, 2)))
Write-Host ("samples     : {0}" -f $sampleCount)
$avgLabel = ("avg CPU  : {0} (total, across cores)" -f $avgCpu)
$maxLabel = ("max CPU  : {0}" -f $maxCpu)
Write-Host $avgLabel
Write-Host $maxLabel
if ($maxCpu -lt 95) {
    $failMsg = ("GATE FAIL : max CPU {0} < 95 percent -> trainer is under-utilized" -f $maxCpu)
    Write-Host $failMsg -ForegroundColor Yellow
} else {
    $passMsg = ("GATE PASS : max CPU {0} >= 95 percent -> trainer is well-utilized" -f $maxCpu)
    Write-Host $passMsg -ForegroundColor Green
}
Write-Host ""
Write-Host "--- stdout ---"
Write-Host $stdout
Write-Host "--- stderr ---"
Write-Host $stderr
