# THE FAIR TEST: production solver-internal hero vs HAND-STRENGTH-CONDITIONAL (human-realistic)
# archetypes. 4 archetypes x 6 disjoint seeds x 3000 hands, outside the session tree (restart-proof).
$ErrorActionPreference = "Continue"
Set-Location "."
$env:POKERBOT_BETTING_ABSTRACTION = "1"
$env:PYTHONIOENCODING = "utf-8"
$bp = "checkpoints/bp6max_v2_iter200.pkl"
$maxPar = 11
$running = @()
foreach ($arch in @("hs_station", "hs_nit", "hs_lag", "hs_tag")) {
  foreach ($seed in 0..5) {
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $maxPar) { Start-Sleep -Seconds 30 }
    $p = Start-Process -FilePath "python" -ArgumentList @(
      "scripts/eval_exploitation_battery.py", "--hero", "pluribus", "--hands", "3000",
      "--seed", "$seed", "--iterations", "10", "--allin-ev", "--no-aivat-leafq",
      "--blueprint-snapshot", $bp, "--archetypes", $arch, "--no-gear-changer",
      "--out", "runs/eb3000_hs_${arch}_s${seed}.json"
    ) -RedirectStandardOutput "runs/hs_${arch}_s${seed}.log" `
      -RedirectStandardError "runs/hs_${arch}_s${seed}.err" -WindowStyle Hidden -PassThru
    $running += $p
  }
}
$running | Wait-Process
"DONE $(Get-Date)" | Out-File -FilePath "runs/hs_fair_test_DONE.marker" -Encoding utf8
