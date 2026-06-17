# PHASE 0: root-mixer re-test vs FAIR (hand-strength-conditional) bots.
# Arm A = --root-mixer (validated facing-only subset); Arm B = + --root-mixer-unopened.
# Baseline (no mixer) reuses the existing 18-rep eb3000_hs_* results — CRN same-seed comparable.
# 2 arms x 4 archetypes x 18 seeds = 144 runs; queued per-archetype so station resolves first.
$ErrorActionPreference = "Continue"
Set-Location "."
$env:POKERBOT_BETTING_ABSTRACTION = "1"
$env:PYTHONIOENCODING = "utf-8"
$bp = "checkpoints/bp6max_v2_iter200.pkl"
$maxPar = 11
$running = @()
foreach ($arch in @("hs_station", "hs_nit", "hs_lag", "hs_tag")) {
  foreach ($seed in 0..17) {
    foreach ($arm in @("A", "B")) {
      while (@($running | Where-Object { -not $_.HasExited }).Count -ge $maxPar) { Start-Sleep -Seconds 30 }
      $extra = @("--root-mixer")
      if ($arm -eq "B") { $extra += "--root-mixer-unopened" }
      $p = Start-Process -FilePath "python" -ArgumentList (@(
        "scripts/eval_exploitation_battery.py", "--hero", "pluribus", "--hands", "3000",
        "--seed", "$seed", "--iterations", "10", "--allin-ev", "--no-aivat-leafq",
        "--blueprint-snapshot", $bp, "--archetypes", $arch, "--no-gear-changer"
      ) + $extra + @(
        "--out", "runs/eb3000_rm${arm}_${arch}_s${seed}.json"
      )) -RedirectStandardOutput "runs/rm${arm}_${arch}_s${seed}.log" `
        -RedirectStandardError "runs/rm${arm}_${arch}_s${seed}.err" -WindowStyle Hidden -PassThru
      $running += $p
    }
  }
}
$running | Wait-Process
"DONE $(Get-Date)" | Out-File -FilePath "runs/rootmixer_fair_DONE.marker" -Encoding utf8
