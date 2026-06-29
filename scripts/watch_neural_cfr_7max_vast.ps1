# Read-only dashboard for the vast.ai 7-max neural-CFR RESUME run. NEVER launches/destroys (pair with
# scripts\neural_cfr_7max_vast_babysitter.ps1). Shows: live vast box + uptime, current iteration +
# throughput (ms/iter) + ETA-to-cap, the LBR exploitability trend (chip-LBR vs the same-cell always-
# fold floor per NORMAL cell), $ spent so far vs the $71 cap, and a STALL alarm.
#
#   powershell -ExecutionPolicy Bypass -File scripts\watch_neural_cfr_7max_vast.ps1 -Run run_1700000000
#   ... -Refresh 45 -MaxBudget 71 -Bid 0.33
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252; non-ASCII breaks strings).
param(
  [string]$Run = "",
  [string]$Bucket = "video-processing-6789",
  [string]$Region = "eu-central-1",
  [string]$PrefixBase = "pokerbot/prod6000/neural_cfr/vast_7max",
  [int]$Refresh = 60,
  [double]$MaxBudget = 71.0,
  [double]$Bid = 0.33,            # used for the $-spent estimate (set to your actual bid)
  [int]$StallMin = 30,            # stall alarm: no iter advance for this many minutes
  [string]$Label = ""
)
$ErrorActionPreference = "Continue"
if (-not $Run) { Write-Host "ERROR: -Run <run_subdir> is required."; exit 1 }
if (-not $Label) { $Label = "pokerbot-7max-$Run" }
$PREFIX = "$PrefixBase/$Run"
$t0 = Get-Date
$prevIter = -1
$prevIterTime = Get-Date
$lastAdvance = Get-Date

function Get-VastBox {
  $raw = (vastai show instances --raw 2>$null)
  if (-not $raw) { return $null }
  try { $arr = $raw | ConvertFrom-Json } catch { return $null }
  foreach ($inst in $arr) { if ("$($inst.label)" -eq $Label) { return $inst } }
  return $null
}
function Get-Json([string]$key) {
  $raw = (aws s3 cp "s3://$Bucket/$PREFIX/$key" - --region $Region 2>$null)
  if ($raw) { try { return ($raw | ConvertFrom-Json) } catch { return $null } }
  return $null
}
function Bar([double]$frac, [int]$w = 28) {
  if ($frac -lt 0) { $frac = 0 }; if ($frac -gt 1) { $frac = 1 }
  $f = [int]([math]::Round($frac * $w))
  return ("#" * $f) + ("-" * ($w - $f))
}

while ($true) {
  Clear-Host
  $watchMin = [int]((Get-Date) - $t0).TotalMinutes
  Write-Host ("=== vast 7-max neural-CFR   $Run   {0}   (watching {1}m) ===" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $watchMin)

  # --- box ---
  $box = Get-VastBox
  $upH = 0.0
  if ($box) {
    $status = "$($box.actual_status)"
    $gpu = "$($box.gpu_name)"
    $dph = $box.dph_total
    $startEpoch = $box.start_date
    if ($startEpoch) {
      try { $upH = ((Get-Date).ToUniversalTime() - ([datetimeoffset]::FromUnixTimeSeconds([long]$startEpoch)).UtcDateTime).TotalHours } catch {}
    }
    Write-Host ("box:   id {0}  {1}  {2}  dph `${3}  up {4:N1}h" -f $box.id, $status, $gpu, $dph, $upH)
  } else {
    Write-Host "box:   (no live box with label $Label - preempted / between relaunches / finished)"
  }

  # --- progress / throughput / ETA ---
  $hb = Get-Json "state/heartbeat.json"
  $iter = $null; $total = $null; $ms = $null
  if ($hb) { $iter = $hb.iter; $total = $hb.total; $ms = $hb.ms_per_iter }
  if ($null -ne $iter) {
    if ($iter -ne $prevIter) { $lastAdvance = Get-Date; $prevIter = $iter; $prevIterTime = Get-Date }
    $msTxt = if ($null -ne $ms) { ("{0:N0} ms/iter" -f $ms) } else { "?? ms/iter" }
    $etaTxt = "--"
    if (($null -ne $total) -and ($null -ne $ms) -and ($ms -gt 0)) {
      $remIter = [double]$total - [double]$iter
      $etaSec = $remIter * ([double]$ms / 1000.0)
      if ($etaSec -lt 5400) { $etaTxt = ("~{0}m" -f [int]($etaSec / 60)) }
      else { $etaTxt = ("~{0:N1}h" -f ($etaSec / 3600.0)) }
    }
    $frac = if ($total) { [double]$iter / [double]$total } else { 0 }
    Write-Host ("iter:  {0}/{1}  [{2}]  {3}  ETA-to-cap {4}" -f $iter, $total, (Bar $frac), $msTxt, $etaTxt)
  } else {
    Write-Host "iter:  (no heartbeat yet - box booting / XLA compiling [3-4 min on resume is normal])"
  }

  # --- $ spent vs cap ---
  $spent = $upH * $Bid
  $budgetFrac = if ($MaxBudget -gt 0) { $spent / $MaxBudget } else { 0 }
  Write-Host ("cost:  ~`${0:N2} / `${1:N0}  [{2}]  (@ `${3}/hr bid)" -f $spent, $MaxBudget, (Bar $budgetFrac), $Bid)

  # --- exploitability trend (the run-to-gate metric) ---
  $g = Get-Json "state/tournament_eval_net.json"
  Write-Host ""
  Write-Host "LBR gate (chip-exploit vs same-cell always-fold floor; PASS = below floor):"
  if ($g -and $g.cells) {
    foreach ($ck in @("NORMAL:deep", "NORMAL:shallow", "NORMAL:bubble")) {
      $cell = $g.cells.$ck
      if ($cell -and $cell.pillars.E1_E2_lbr_icm.status -eq "built") {
        $r = $cell.pillars.E1_E2_lbr_icm.result
        $chip = [double]$r.chip_exploit_bb_per_100
        $floor = [double]$r.anchor_chip_exploit_at_cell_bb_per_100
        $lt = [bool]$r.less_exploitable_than_always_fold
        $ratio = if ($floor -ne 0) { $chip / $floor } else { 0 }
        $flag = if ($lt) { "PASS" } else { "HIGH" }
        $col = if ($lt) { "Green" } else { "Yellow" }
        Write-Host ("  {0,-16} chip {1,7:N0}  floor {2,6:N0}  ({3:N2}x)  $-verdict {4,-12} {5}" -f $ck, $chip, $floor, $ratio, "$($r.verdict)", $flag) -ForegroundColor $col
      } else {
        Write-Host ("  {0,-16} (not yet evaluated)" -f $ck)
      }
    }
  } else {
    Write-Host "  (no gate JSON yet - first eval runs after GATE_EVERY_SEC ~90 min)"
  }

  # --- stall + done ---
  $staleMin = [int]((Get-Date) - $lastAdvance).TotalMinutes
  aws s3 ls "s3://$Bucket/$PREFIX/sevenmax_DONE.marker" --region $Region 2>$null | Out-Null
  $isDone = $?
  Write-Host ""
  if ($box -and ($box.actual_status -eq "running") -and ($staleMin -ge $StallMin) -and ($null -ne $iter)) {
    Write-Host ("  STALL ALARM: no iter advance for ${staleMin}m on a running box (hung? OOM-compile? check vastai logs $($box.id))") -ForegroundColor Red
  }
  if ($isDone) {
    Write-Host "DONE: sevenmax_DONE.marker present - run COMPLETE (gate met / budget / iters)." -ForegroundColor Green
    Write-Host "The babysitter downloads runs/vast_7max/$Run/ and prints the final gate verdict."
    break
  }
  Write-Host ("(refresh {0}s; Ctrl-C to stop; read-only)" -f $Refresh)
  Start-Sleep -Seconds $Refresh
}
