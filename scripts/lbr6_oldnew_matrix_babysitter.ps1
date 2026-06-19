# Autonomous babysitter for the OLD-vs-NEW 6-max blueprint LBR EXPLOITABILITY MATRIX.
#
# SIX cells. Each is a SEPARATE run-subdir (own S3 prefix, own DONE marker, independently resumable),
# shipping its OWN blueprint pkl + serve env, with its OWN menu / target / hand count:
#
#   FC (the OLD-vs-NEW tradeoff; 25k hands each, fc is low-variance ~+/-41 bb/100):
#     cell1 OLD fc        : dcfrvr (176494, WEVA)        search_preflop ON  LOSSLESS=0  blueprint-search
#     cell2 NEW fc fast   : lossless2 (96553)            search_preflop OFF LOSSLESS=1  blueprint-search
#     cell3 NEW fc robust : (same NEW pkl)               search_preflop ON  LOSSLESS=1  blueprint-search
#   FCPA (NEW blueprint ONLY, per the decision):
#     cell4 NEW fcpa fast : lossless2  search_pf OFF LOSSLESS=1  blueprint-search  25k (coarse ~+/-100)
#     cell5 NEW fcpa robust: lossless2 search_pf ON  LOSSLESS=1  blueprint-search  25k (coarse ~+/-100)
#     cell6 NEW fcpa raw  : lossless2  (NO search)    LOSSLESS=1  blueprint-only    100k (CRN-tight, cheap)
#
# The babysitter walks the cells: for each not-yet-DONE cell with no live box it launches ONE spot box and
# waits; on reclaim it relaunches and the on-box driver resumes from already-uploaded lbr6_*.json. When all
# cells are DONE it pulls each SUMMARY.json and prints a combined verdict table.
#
# Run from YOUR terminal (your AWS creds -- the assistant IAM cannot RunInstances):
#   powershell -ExecutionPolicy Bypass -File scripts\lbr6_oldnew_matrix_babysitter.ps1
# Watch read-only in another terminal:
#   powershell -ExecutionPolicy Bypass -File scripts\watch_lbr6_oldnew_matrix.ps1
#
# ASCII ONLY (PS 5.1 + BOM-less UTF-8 mangles non-ASCII into string-breaking bytes).
param(
  [string]$RunTag   = "lbr6_oldnew1",   # cells become run_<RunTag>_cell1..6
  [int]$Seeds       = 50,               # fc + fcpa-search cells: 50 x 500 = 25k hands
  [int]$Hands       = 500,
  [int]$RawSeeds    = 100,              # fcpa blueprint-only cell: 100 x 1000 = 100k (cheap, no search)
  [int]$RawHands    = 1000,
  [int]$Iters       = 10,               # fixed CFR+ iters per opponent solve (box-speed independent)
  [string]$Leaf     = "uniform",
  [string]$Only     = ""               # optional CSV of cell names to run (e.g. "cell4,cell5"); empty=all
)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

$REGION = "eu-central-1"
$BUCKET = "video-processing-6789"
$BASE   = "pokerbot/prod6000/eval_suite"
$MAXSEC = 16 * 3600     # overall wall cap across all cells
$MAXLAUNCH_PER_CELL = 6
$OLD_BP = Join-Path $env:TEMP "bp6max_dcfrvr_iter200.pkl"
$NEW_BP = Join-Path $env:TEMP "bp6max_lossless2_iter200.pkl"
$OUT    = "runs\lbr6_oldnew_matrix"
New-Item -ItemType Directory -Force $OUT | Out-Null

# --- cell table: name | run | pkl | lossless | search_preflop | menu | target | seeds | hands ------------
$cells = @(
  @{ name="cell1_OLD_fc";          run="run_${RunTag}_cell1"; bp=$OLD_BP; lossless=$false; searchpf=$true;  menu="fc";   target="blueprint-search"; seeds=$Seeds;    hands=$Hands   },
  @{ name="cell2_NEW_fc_fast";     run="run_${RunTag}_cell2"; bp=$NEW_BP; lossless=$true;  searchpf=$false; menu="fc";   target="blueprint-search"; seeds=$Seeds;    hands=$Hands   },
  @{ name="cell3_NEW_fc_robust";   run="run_${RunTag}_cell3"; bp=$NEW_BP; lossless=$true;  searchpf=$true;  menu="fc";   target="blueprint-search"; seeds=$Seeds;    hands=$Hands   },
  @{ name="cell4_NEW_fcpa_fast";   run="run_${RunTag}_cell4"; bp=$NEW_BP; lossless=$true;  searchpf=$false; menu="fcpa"; target="blueprint-search"; seeds=$Seeds;    hands=$Hands   },
  @{ name="cell5_NEW_fcpa_robust"; run="run_${RunTag}_cell5"; bp=$NEW_BP; lossless=$true;  searchpf=$true;  menu="fcpa"; target="blueprint-search"; seeds=$Seeds;    hands=$Hands   },
  @{ name="cell6_NEW_fcpa_raw";    run="run_${RunTag}_cell6"; bp=$NEW_BP; lossless=$true;  searchpf=$false; menu="fcpa"; target="blueprint";        seeds=$RawSeeds; hands=$RawHands }
)
if ($Only) {
  $keep = $Only.Split(",") | ForEach-Object { $_.Trim() }
  $cells = $cells | Where-Object { $keep -contains $_.name -or $keep -contains ($_.name.Split("_")[0]) }
  if (-not $cells) { Write-Host "ABORT: -Only '$Only' matched no cells"; exit 1 }
}
foreach ($c in $cells) {
  if (-not (Test-Path $c.bp)) { Write-Host "ABORT: missing blueprint $($c.bp) for $($c.name)"; exit 1 }
}

$start = Get-Date

function Test-Done([string]$run) {
  aws s3 ls "s3://$BUCKET/$BASE/$run/eval_suite_DONE.marker" --region $REGION 2>$null | Out-Null
  return $?
}
function Get-Boxes([string]$run, [string]$states) {
  $r = (aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$run" "Name=instance-state-name,Values=$states" --query "Reservations[].Instances[].InstanceId" --output text 2>$null)
  if ($r) { return ($r -replace "\s+", " ").Trim() } else { return "" }
}
function Get-StateById([string]$id) {
  if (-not $id) { return "" }
  $s = (aws ec2 describe-instances --region $REGION --instance-ids $id --query "Reservations[].Instances[].State.Name" --output text 2>$null)
  if ($s) { return $s.Trim() } else { return "" }
}
function Stop-Run([string]$run) {
  $ids = (Get-Boxes $run "running,pending,stopping")
  if ($ids) {
    $idList = @($ids.Split(" ") | Where-Object { $_ -like "i-*" })
    if ($idList.Count) { aws ec2 terminate-instances --instance-ids $idList --region $REGION | Out-Null; Write-Host "  terminated: $idList" }
  }
}
function Launch-Cell($c) {
  $ll = if ($c.lossless) { "--preflop-lossless" } else { "" }
  Write-Host "  launching $($c.name): bp=$([System.IO.Path]::GetFileName($c.bp)) menu=$($c.menu) target=$($c.target) searchpf=$($c.searchpf) $($c.seeds)x$($c.hands)"
  $argList = @(
    "scripts\launch_eval_suite_spot.py",
    "--run-subdir", $c.run, "--mode", "lbr6",
    "--seeds", $c.seeds, "--hands", $c.hands,
    "--lbr6-target", $c.target, "--lbr6-single-menu",
    "--lbr6-menu", $c.menu
  )
  # search-only flags: leaf/iters/search-preflop apply ONLY to the blueprint-search target.
  if ($c.target -eq "blueprint-search") {
    $argList += @("--lbr6-search-leaf", $Leaf, "--lbr6-search-iters", $Iters)
    if ($c.searchpf) { $argList += "--lbr6-search-preflop" }
  }
  $argList += @("--local-bp", $c.bp)
  if ($ll) { $argList += $ll }
  $out = (& python @argList 2>&1)
  $out | ForEach-Object { Write-Host "    $_" }
  $txt = ($out | Out-String)
  if ($txt -match "LAUNCHED (i-[0-9a-f]+)") { return $Matches[1] } else { return "" }
}

$capH = [int]($MAXSEC / 3600)
Write-Host "=== OLD-vs-NEW LBR MATRIX babysitter: tag=$RunTag ($($cells.Count) cells, iters=$Iters $Leaf, cap ${capH}h) ==="
foreach ($c in $cells) { Write-Host ("  {0,-22} -> {1}  [{2} {3} pf={4} {5}x{6}]" -f $c.name, $c.run, $c.menu, $c.target, $c.searchpf, $c.seeds, $c.hands) }

# per-cell launch bookkeeping
$tracked  = @{}; foreach ($c in $cells) { $tracked[$c.run] = "" }
$launches = @{}; foreach ($c in $cells) { $launches[$c.run] = 0 }

# One-time cleanup of stale FAIL_*/SUMMARY from a prior failed attempt (keep lbr6_*.json + DONE markers).
foreach ($c in $cells) {
  if (-not (Test-Done $c.run)) {
    aws s3 rm "s3://$BUCKET/$BASE/$($c.run)/results/" --region $REGION --recursive --exclude "*" --include "FAIL_*" --include "SUMMARY.json" 2>$null | Out-Null
  }
}

while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  $allDone = $true
  $statusLine = @()
  foreach ($c in $cells) {
    $run = $c.run
    $done = Test-Done $run
    if (-not $done) { $allDone = $false }
    $live = Get-Boxes $run "running,pending"
    if (-not $live -and $tracked[$run]) {
      $st = Get-StateById $tracked[$run]
      if ($st -eq "running" -or $st -eq "pending") { $live = $tracked[$run] } else { $tracked[$run] = "" }
    }
    $boxId = ""
    if ($live) { $boxId = (@($live.Split(" ") | Where-Object { $_ -like "i-*" }) -join ",") }
    $statusLine += ("{0}=[{1} live='{2}' L{3}]" -f $c.name, $(if ($done) { "DONE" } else { "..." }), $boxId, $launches[$run])

    # Launch logic: a not-done cell with no live box (and budget left) gets ONE box. We launch cells
    # sequentially (one fresh launch per loop) so we never exceed the spot quota with many at once.
    if (-not $done -and -not $live -and $launches[$run] -lt $MAXLAUNCH_PER_CELL -and $el -lt $MAXSEC) {
      Write-Host "[t+${el}s] $($c.name): no live box -> launching #$($launches[$run] + 1)"
      $id = Launch-Cell $c
      if ($id) {
        $tracked[$run] = $id; $launches[$run]++
        Write-Host "  tracking $id; boot+tag grace 300s"
        Start-Sleep -Seconds 300
      } else {
        Write-Host "  launch did not start a box (quota/capacity?) - backing off 180s (not counted)"
        Start-Sleep -Seconds 180
      }
      break   # one launch per loop iteration; re-evaluate all cells next pass
    }
  }
  Write-Host "[t+${el}s] $($statusLine -join '  ')"

  if ($allDone) { Write-Host "ALL CELLS DONE"; break }
  if ($el -ge $MAXSEC) { Write-Host "STOP: time cap reached"; foreach ($c in $cells) { Stop-Run $c.run }; break }
  Start-Sleep -Seconds 180
}

# --- combined verdict table -------------------------------------------------
Write-Host ""
Write-Host "=== OLD-vs-NEW LBR MATRIX RESULTS ==="
$rows = @()
foreach ($c in $cells) {
  $local = Join-Path $OUT "$($c.run)_SUMMARY.json"
  $bb = "n/a"; $cilo = "n/a"; $cihi = "n/a"; $verdict = "NO SUMMARY"
  if (Test-Done $c.run) {
    aws s3 cp "s3://$BUCKET/$BASE/$($c.run)/results/SUMMARY.json" $local --region $REGION --no-progress 2>$null | Out-Null
    if (Test-Path $local) {
      try {
        $j = Get-Content $local -Raw | ConvertFrom-Json
        $m = $j.menus.($c.menu)
        if ($m) { $bb = [math]::Round($m.bb_per_100, 1); $cilo = [math]::Round($m.ci_lo, 1); $cihi = [math]::Round($m.ci_hi, 1) }
        $verdict = $j.verdict
      } catch { $verdict = "PARSE-ERR ($local)" }
    }
  }
  $rows += [pscustomobject]@{ cell = $c.name; menu = $c.menu; bb_per_100 = $bb; ci_lo = $cilo; ci_hi = $cihi; verdict = $verdict }
}
$rows | Format-Table -AutoSize | Out-String | Write-Host
Write-Host "Priors (reference): cell1 OLD fc ~ -83.5; cell2 NEW fc fast ~ +28.8; cell3 NEW fc robust = the unknown."
Write-Host "fcpa cells (4/5) probe the AGGRESSION/bluff leak; cell6 raw is the no-search fcpa baseline (tight)."
Write-Host "More-negative bb/100 = HARDER to exploit (LBR loses more). Positive ci_lo = a best-responder PROVABLY profits."
Write-Host "Per-cell SUMMARYs downloaded to $OUT\."
Write-Host "BABYSITTER EXIT after ${el}s"
