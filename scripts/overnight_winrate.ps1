<#
.SYNOPSIS
    Overnight, run-until-killed seat-rotated absolute win-rate (bb/100) measurement.

.DESCRIPTION
    Fans thread-pinned eval shards across the whole CPU (one single-threaded subprocess per core, so N
    processes fill N cores without oversubscribing). Each shard is an INDEPENDENT seat-rotated duplicate
    eval over a disjoint deck-seed range and writes a durable per-shard JSON checkpoint. After every
    round the pooler aggregates ALL shards so far into a running bb/100 + 95% CI (pooled_summary.json)
    whose interval tightens overnight. Loops until you stop it (Ctrl-C); completed shards persist, so a
    later run resumes and keeps accumulating precision. The native Rust CFR fast path is enabled by
    default (statistically equivalent, ~1.75x/process); pass -NoNative to force the pure-Python loop.

.EXAMPLE
    pwsh -File scripts\overnight_winrate.ps1
    pwsh -File scripts\overnight_winrate.ps1 -Fields real_recreational_table -DecksPerShard 80 -Workers 10

.NOTES
    Resume-safe: re-run with the same -OutDir to keep adding shards to the same pool.
    Stop with Ctrl-C; the running workers are terminated and a final pool is written.
#>
[CmdletBinding()]
param(
    [string]$Snapshot = "checkpoints/bp6max_lossless2_iter200.pkl",
    [string[]]$Fields = @("real_recreational_table", "real_solid_table", "real_pro_table"),
    [int]$DecksPerShard = 50,
    [int]$Iters = 25,
    [int]$Workers = 0,                       # 0 => ProcessorCount - 2 (leave headroom; never oversubscribe)
    [string]$OutDir = "runs\overnight_winrate",
    [string]$Python = "python",
    [switch]$NoNative,
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($Workers -le 0) { $Workers = [Environment]::ProcessorCount - 2 }
if ($Workers -lt 1) { $Workers = 1 }

# Accept either an array (-Fields a,b) or a single comma-joined string (-Fields "a,b").
$fieldList = @($Fields | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($fieldList.Count -eq 0) { throw "no fields specified" }

$shardDir = Join-Path $OutDir "shards"
$logDir   = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force -Path $shardDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir   | Out-Null
$mainLog     = Join-Path $OutDir "run.log"
$summaryJson = Join-Path $OutDir "pooled_summary.json"
$evalScript  = Join-Path $root "scripts\eval_exploitation_battery.py"
$poolScript  = Join-Path $root "scripts\pool_rotated_checkpoints.py"

function Log([string]$msg) {
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $mainLog -Value $line -Encoding utf8
}

# Single-thread numpy/BLAS per worker + the native + lossless flags. Children inherit this env.
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:POKERBOT_PREFLOP_LOSSLESS = "1"
$env:POKERBOT_SEARCH_PREFLOP = "1"
if ($NoNative) {
    $env:POKERBOT_NATIVE_SUBGAME = "0"; $env:POKERBOT_NATIVE_ROLLOUT = "0"
} else {
    $env:POKERBOT_NATIVE_SUBGAME = "1"; $env:POKERBOT_NATIVE_ROLLOUT = "1"
}

# Run a native exe synchronously with file redirects (avoids the PS 5.1 2>&1 ErrorRecord trap) and
# streams its output into the main log under a tag.
function Invoke-Logged([string]$file, [string[]]$arguments, [string]$tag) {
    # NOTE: use -PassThru + .WaitForExit() (NOT Start-Process -Wait). With redirected output, -Wait
    # blocks until the stream handles close -- if the child spawns grandchildren that inherit those
    # handles (e.g. cargo -> rustc/link) it can hang forever even after the child exits. WaitForExit()
    # waits for the process itself.
    $o = Join-Path $OutDir ".proc.out"
    $e = Join-Path $OutDir ".proc.err"
    $p = Start-Process -FilePath $file -ArgumentList $arguments -WorkingDirectory $root -NoNewWindow `
            -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
    $p.WaitForExit()
    if (Test-Path $o) { Get-Content $o | ForEach-Object { if ($_ -ne "") { Log "$tag $_" } } }
    if ((Test-Path $e) -and (Get-Item $e).Length -gt 0) {
        Get-Content $e | ForEach-Object { if ($_ -ne "") { Log "$tag-ERR $_" } }
    }
    return $p.ExitCode
}

Log "=== overnight win-rate run starting ==="
Log "snapshot   = $Snapshot"
Log "fields     = $($fieldList -join ',')"
Log "decks/shard= $DecksPerShard   iters= $Iters   workers= $Workers   native= $(-not $NoNative)"
Log "cpu cores  = $([Environment]::ProcessorCount)   outdir= $OutDir"

if (-not (Test-Path $Snapshot)) { Log "WARN snapshot not found at $Snapshot (shards will fail until it exists)" }

# Build the native extension once (best-effort; Python fallback is correct, just slower).
if (-not $NoNative -and -not $NoBuild) {
    $nativeDir = Join-Path $root "tools\battery_native"
    if (Test-Path (Join-Path $nativeDir "Cargo.toml")) {
        Log "building native extension: cargo build --release --features python"
        $cargo = (Get-Command cargo -ErrorAction SilentlyContinue)
        if ($cargo) {
            $bo = Join-Path $OutDir ".cargo.out"; $be = Join-Path $OutDir ".cargo.err"
            # -PassThru + WaitForExit (NOT -Wait): cargo spawns rustc/link children that inherit the
            # redirected handles, and Start-Process -Wait blocks on those handles forever even after
            # cargo exits. Cap it at 10 min so a genuinely wedged build can't stall the whole night.
            $cp = Start-Process -FilePath $cargo.Source -ArgumentList @("build","--release","--features","python") `
                    -WorkingDirectory $nativeDir -NoNewWindow -PassThru `
                    -RedirectStandardOutput $bo -RedirectStandardError $be
            if ($cp.WaitForExit(600000)) {
                Log "cargo build finished"
            } else {
                Log "WARN cargo build exceeded 10 min; killing and continuing (existing .pyd / Python fallback)"
                try { $cp.Kill() } catch {}
            }
        } else {
            Log "WARN cargo not on PATH; skipping native build (Python fallback)"
        }
    }
}

# Verify native availability (informational). Use the call operator (quotes the -c body as one arg;
# Start-Process would split it on spaces) and drop stderr to $null (safe: not a 2>&1 merge).
try {
    $nv = & $Python -c "from pokerbot.search import native_solver as ns; print('native_subgame_available =', ns.native_subgame_available())" 2>$null
    Log "INIT $nv"
} catch { Log "INIT WARN native check failed: $($_.Exception.Message)" }

# Resume-safe per-field seed counters: next seed = (max seed already on disk for that field) + 1.
function Get-NextSeed([string]$field) {
    $max = -1
    Get-ChildItem -Path $shardDir -Filter "$($field)_s*.json" -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.BaseName -match "_s(\d+)$") { $s = [int]$Matches[1]; if ($s -gt $max) { $max = $s } }
    }
    return ($max + 1)
}
$seed = @{}
foreach ($f in $fieldList) { $seed[$f] = (Get-NextSeed $f) }
Log "resume seeds: $(($fieldList | ForEach-Object { "$_=$($seed[$_])" }) -join '  ')"

# Show the resumed pool (if any) before starting.
[void](Invoke-Logged $Python @($poolScript,"--shards",$shardDir,"--out",$summaryJson) "POOL")

$script:active = @()   # processes launched this round, for clean shutdown on Ctrl-C
$round = 0
try {
    while ($true) {
        $round++
        # One round = exactly $Workers shards (one per core), round-robin across the fields.
        $tasks = @()
        for ($w = 0; $w -lt $Workers; $w++) {
            $f = $fieldList[$w % $fieldList.Count]
            $s = $seed[$f]; $seed[$f] = $s + 1
            $tasks += [pscustomobject]@{ Field = $f; Seed = $s; Out = (Join-Path $shardDir "$($f)_s$($s).json") }
        }
        Log "round $round : launching $($tasks.Count) shards ($DecksPerShard decks x $Iters iters each)"

        $script:active = @()
        foreach ($t in $tasks) {
            $slog = Join-Path $logDir "$($t.Field)_s$($t.Seed).out"
            $serr = Join-Path $logDir "$($t.Field)_s$($t.Seed).err"
            $a = @($evalScript, "--hero","pluribus", "--blueprint-snapshot",$Snapshot,
                   "--field",$t.Field, "--seed","$($t.Seed)", "--hands","$DecksPerShard",
                   "--rotate-seats","--root-mixer","--root-mixer-unopened","--allin-ev",
                   "--iterations","$Iters","--no-aivat-leafq","--out",$t.Out)
            $p = Start-Process -FilePath $Python -ArgumentList $a -WorkingDirectory $root -NoNewWindow `
                    -PassThru -RedirectStandardOutput $slog -RedirectStandardError $serr
            $script:active += [pscustomobject]@{ Proc = $p; Task = $t; Err = $serr }
        }

        $okCount = 0
        foreach ($pp in $script:active) {
            $pp.Proc.WaitForExit()
            # A successful shard writes its checkpoint JSON; treat a missing/empty file as the failure
            # signal (Start-Process -PassThru ExitCode is unreliable in Windows PowerShell 5.1).
            $ok = (Test-Path $pp.Task.Out) -and ((Get-Item $pp.Task.Out).Length -gt 0)
            if ($ok) { $okCount++ }
            else { Log "WARN shard $($pp.Task.Field) s$($pp.Task.Seed) produced no checkpoint (see $($pp.Err))" }
        }
        $script:active = @()
        Log "round $round : $okCount/$($tasks.Count) shards wrote checkpoints"

        # Pool + checkpoint (rewrites pooled_summary.json each round = the rolled-up checkpoint).
        Log "round $round complete; pooled win-rate to date:"
        [void](Invoke-Logged $Python @($poolScript,"--shards",$shardDir,"--out",$summaryJson) "POOL")
    }
}
finally {
    if ($script:active.Count -gt 0) {
        Log "stopping; terminating $($script:active.Count) in-flight shard(s)"
        foreach ($pp in $script:active) {
            try { if (-not $pp.Proc.HasExited) { $pp.Proc.Kill() } } catch {}
        }
    }
    Log "=== run stopped after round $round; final pooled summary -> $summaryJson ==="
    [void](Invoke-Logged $Python @($poolScript,"--shards",$shardDir,"--out",$summaryJson) "POOL")
}
