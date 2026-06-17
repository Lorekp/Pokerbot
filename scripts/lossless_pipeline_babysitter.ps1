# Autonomous END-TO-END babysitter for the LOSSLESS-PREFLOP-CARDS blueprint (Pluribus-faithful Stage 2).
# ONE command runs the whole pipeline; pair with scripts\watch_lossless_pipeline.ps1 to watch it.
#   powershell -ExecutionPolicy Bypass -File scripts\lossless_pipeline_babysitter.ps1
#
# Phases (resumable; writes runs\lossless_pipeline\STATUS.txt every loop for the watcher):
#   1 TRAIN  - from-scratch DCFR+VR 6-max blueprint, --preflop-lossless (169 canonical CARDS x position,
#              bounded betting). This is SMALLER than the 10k-WEVA blueprint that already fit, so it does
#              NOT overflow the hogwild table (the cards+betting v1 did: SharedHogwildTable full -> all
#              workers died, pinned at iter 17). Relaunched on every spot reclaim; resumes from S3.
#   2 PULL   - download the highest iter_*/sd_cfr_hu_iter_*.pkl snapshot.
#   3 GATE   - local fc preflop-attribution (25k hands) vs the new blueprint+search (search_preflop OFF):
#              reads the PREFLOP bb/100. Today the leak is +174.7; PASS = it collapses to the noise floor.
#   4 VERDICT- writes PASS/REVIEW/FAIL + the ship commands.
# ASCII ONLY (PS 5.1 mangles non-ASCII).
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")   # repo root

# --- config -----------------------------------------------------------------
$REGION   = "eu-central-1"
$RUN      = "run_6max_lossless2"   # FRESH prefix: v1's iter_* are the incompatible cards+betting key space
$BUCKET   = "video-processing-6789"
$PREFIX   = "pokerbot/prod6000/sd_cfr_hu/$RUN"
$CAPACITY = 8388608      # 2^23 headroom (cards-only is ~10x smaller than the 10k-WEVA bp that fit 2^22)
$TARGET   = 200          # stop training once this many checkpoints exist
$MAXSEC   = 16 * 3600    # hard wall-clock cap for the TRAIN phase
$MAXLAUNCH = 12          # never launch more than this many training boxes
$STALL    = 1200         # if a box is alive but iter unchanged this long, recycle it (hung)
$GATE_HANDS = 25000
$PASS_BB  = 25.0         # preflop bb/100 <= this => PASS; >60 => FAIL; between => REVIEW
$OUT      = "runs\lossless_pipeline"
$STATUS   = "$OUT\STATUS.txt"
$GATELOG  = "$OUT\gate.log"
$DONE     = "$OUT\PIPELINE_DONE.txt"
$SNAP     = Join-Path $env:TEMP "bp6max_lossless_iter200.pkl"
New-Item -ItemType Directory -Force $OUT | Out-Null

$start = Get-Date
$launches = 0
$tracked = ""
$lastIter = -1
$lastIterChange = Get-Date

function Write-Status([string]$phase, [string]$line) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  @("run        = $RUN", "phase      = $phase", "elapsed_s  = $el", "launches   = $launches",
    "capacity   = $CAPACITY", "snapshot   = $SNAP", "updated    = $((Get-Date).ToString('u'))",
    "note       = $line") | Set-Content -Encoding ASCII $STATUS
  Write-Host "[t+${el}s][$phase] $line"
}
function Get-LatestIter {
  $out = (aws s3 ls "s3://$BUCKET/$PREFIX/" --recursive --region $REGION 2>$null)
  $m = [regex]::Matches(($out -join "`n"), "sd_cfr_hu_iter_(\d+)\.pkl")
  if ($m.Count -gt 0) { return ($m | ForEach-Object { [int]$_.Groups[1].Value } | Measure-Object -Maximum).Maximum }
  return 0
}
function Get-LatestSnapKey {
  $out = (aws s3 ls "s3://$BUCKET/$PREFIX/" --recursive --region $REGION 2>$null)
  $best = ""; $bi = -1
  foreach ($mm in [regex]::Matches(($out -join "`n"), "(\S*sd_cfr_hu_iter_(\d+)\.pkl)")) {
    $i = [int]$mm.Groups[2].Value
    if ($i -gt $bi) { $bi = $i; $best = $mm.Groups[1].Value }
  }
  return $best
}
function Get-Boxes([string]$states) {
  $r = (aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=$states" --query "Reservations[].Instances[].InstanceId" --output text 2>$null)
  if ($r) { return ($r -replace "\s+", " ").Trim() } else { return "" }
}
function Get-StateById([string]$id) {
  if (-not $id) { return "" }
  $s = (aws ec2 describe-instances --region $REGION --instance-ids $id --query "Reservations[].Instances[].State.Name" --output text 2>$null)
  if ($s) { return $s.Trim() } else { return "" }
}
function Stop-Run {
  $ids = Get-Boxes "running,pending,stopping"
  if ($ids) {
    $idList = @($ids.Split(" ") | Where-Object { $_ -like "i-*" })
    if ($idList.Count) { aws ec2 terminate-instances --instance-ids $idList --region $REGION | Out-Null; Write-Host "terminated: $idList" }
  }
}
function Wait-NoBoxes {
  # Terminate stragglers + wait until the spot count for this Run frees up (fixes MaxSpotInstanceCountExceeded).
  Stop-Run
  for ($i = 0; $i -lt 24; $i++) {
    if (-not (Get-Boxes "running,pending,stopping,shutting-down")) { return }
    Start-Sleep -Seconds 15
  }
}

if (Test-Path $DONE) { Write-Host "Pipeline already DONE (see $DONE). Delete it to re-run."; Get-Content $DONE; exit 0 }

# === PHASE 1: TRAIN =========================================================
while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds
  $iter = Get-LatestIter
  if ($iter -ne $lastIter) { $lastIter = $iter; $lastIterChange = Get-Date }
  $live = Get-Boxes "running,pending"
  if (-not $live -and $tracked) {
    $st = Get-StateById $tracked
    if ($st -eq "running" -or $st -eq "pending") { $live = $tracked } else { $tracked = "" }
  }
  $stalledSec = [int]((Get-Date) - $lastIterChange).TotalSeconds
  Write-Status "1-TRAIN" "iter=$iter/$TARGET live='$live' launches=$launches stalled=${stalledSec}s"

  if ($iter -ge $TARGET) { Write-Host "TRAIN done: iter $iter"; Stop-Run; break }
  if ($el -ge $MAXSEC)   { Write-Status "1-TRAIN" "TIME CAP at iter=$iter; proceeding to gate"; Stop-Run; break }

  if ($live) {
    if ($stalledSec -ge $STALL) {
      Write-Status "1-TRAIN" "STALL: iter stuck ${stalledSec}s with box alive -> recycling box"
      Wait-NoBoxes; $tracked = ""; $lastIterChange = Get-Date
    } else {
      Start-Sleep -Seconds 150
    }
    continue
  }
  if ($launches -ge $MAXLAUNCH) { Write-Status "1-TRAIN" "STOP: max launches ($MAXLAUNCH) hit"; break }

  Wait-NoBoxes   # clear any straggler so the spot quota is free before we launch
  $next = $launches + 1
  Write-Host "  no live box -> launching #$next (resume from S3)"
  $out = (& python scripts\launch_sd_cfr_training_spot.py --instance-type m7g.16xlarge --arch arm64 --workers 56 --iterations 200 --traversals 2000 --deal-sweeps 1 --max-depth 64 --betting-abstraction --postflop-buckets 200 --client-bet-sizing --trainer hogwild --hogwild-capacity $CAPACITY --no-sft --occupancy 6max --cfr-variant dcfr --vr-baseline --preflop-lossless --run-subdir $RUN 2>&1)
  $out | ForEach-Object { Write-Host $_ }
  $txt = ($out | Out-String)
  if ($txt -match "Launched (i-[0-9a-f]+)") {
    $tracked = $Matches[1]; $launches++
    Write-Host "  tracking $tracked; boot+tag grace"
    Start-Sleep -Seconds 300
  } else {
    Write-Host "  launch did not start a box (MaxSpot/quota/capacity?) - backing off 180s (not counted)"
    Start-Sleep -Seconds 180
  }
}

# === PHASE 2: PULL ==========================================================
$key = Get-LatestSnapKey
if (-not $key) { Write-Status "2-PULL" "NO snapshot in S3 - abort"; "FAIL: no snapshot in s3://$BUCKET/$PREFIX" | Set-Content -Encoding ASCII $DONE; exit 1 }
Write-Status "2-PULL" "downloading $key"
aws s3 cp "s3://$BUCKET/$key" "$SNAP" --region $REGION --no-progress
if (-not (Test-Path $SNAP)) { Write-Status "2-PULL" "download FAILED"; "FAIL: snapshot download failed ($key)" | Set-Content -Encoding ASCII $DONE; exit 1 }

# === PHASE 3: GATE (local, decisive) ========================================
$env:POKERBOT_BETTING_ABSTRACTION = "1"
$env:POKERBOT_POSTFLOP_BUCKETS    = "1"
$env:POKERBOT_POSTFLOP_MODE       = "fast"
$env:POKERBOT_POSTFLOP_NUM_BUCKETS= "200"
$env:POKERBOT_PREFLOP_LOSSLESS    = "1"
Write-Status "3-GATE" "fc preflop-attribution, $GATE_HANDS hands (search_preflop OFF) -> $GATELOG"
python scripts\lbr6_attribute.py --hands $GATE_HANDS --seed 0 --blueprint-snapshot "$SNAP" 2>&1 | Tee-Object -FilePath $GATELOG

# === PHASE 4: VERDICT =======================================================
$overall = ""; $preflop = $null
$ov = Select-String -Path $GATELOG -Pattern "overall:\s*([+\-][\d.]+)\s*bb/100" | Select-Object -First 1
if ($ov) { $overall = $ov.Matches[0].Groups[1].Value }
$pf = Select-String -Path $GATELOG -Pattern "^\s*preflop\s" | Select-Object -First 1
if ($pf) { $toks = ($pf.Line.Trim() -split "\s+"); $preflop = [double]($toks[-1]) }

$verdict = "REVIEW"
if ($preflop -ne $null) {
  if ([math]::Abs($preflop) -le $PASS_BB) { $verdict = "PASS" }
  elseif ($preflop -ge 60.0) { $verdict = "FAIL" }
}
$report = @(
  "=== LOSSLESS-PREFLOP-CARDS PIPELINE: $verdict ===",
  "snapshot         : $SNAP",
  "overall fc bb/100: $overall   (deployed baseline was +156)",
  "PREFLOP bb/100   : $preflop   (deployed baseline was +174.7; PASS threshold <= $PASS_BB)",
  "gate log         : $GATELOG",
  "",
  "IF PASS - ship it:",
  "  Copy-Item `"$SNAP`" `"$env:TEMP\bp6max_v2_iter200.pkl`"   # or repoint POKERBOT_BLUEPRINT_SNAPSHOT",
  "  setx POKERBOT_SEARCH_PREFLOP 0   # revert the stopgap: leak gone AND preflop-search latency gone",
  "",
  "IF REVIEW/FAIL - cards-only was not enough; escalate to lossless preflop BETTING:",
  "  retrain with --preflop-lossless --preflop-lossless-betting + a much larger --hogwild-capacity",
  "  (e.g. 67108864) and more iters; gate the same way."
)
$report | Tee-Object -FilePath $DONE
Write-Status "4-DONE-$verdict" "preflop=$preflop overall=$overall"
Write-Host ""
Write-Host "PIPELINE COMPLETE -> $verdict  (full report in $DONE)"
