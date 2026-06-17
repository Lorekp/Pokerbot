# Station resolution push: 18 disjoint-seed replicas (seeds 6-23) outside the session tree.
$ErrorActionPreference = "Continue"
Set-Location "."
$env:POKERBOT_BETTING_ABSTRACTION = "1"
$env:PYTHONIOENCODING = "utf-8"
$bp = "checkpoints/bp6max_v2_iter200.pkl"
$seeds = 6..23
$maxPar = 11
$running = @()
foreach ($seed in $seeds) {
  while (@($running | Where-Object { -not $_.HasExited }).Count -ge $maxPar) { Start-Sleep -Seconds 30 }
  $p = Start-Process -FilePath "python" -ArgumentList @(
    "scripts/eval_exploitation_battery.py", "--hero", "pluribus", "--hands", "3000",
    "--seed", "$seed", "--iterations", "10", "--allin-ev", "--no-aivat-leafq",
    "--blueprint-snapshot", $bp, "--archetypes", "calling_station", "--no-gear-changer",
    "--out", "runs/eb3000_finalD_calling_station_s${seed}.json"
  ) -RedirectStandardOutput "runs/finalD_calling_station_s${seed}.log" `
    -RedirectStandardError "runs/finalD_calling_station_s${seed}.err" -WindowStyle Hidden -PassThru
  $running += $p
}
$running | Wait-Process
"DONE $(Get-Date)" | Out-File -FilePath "runs/station_replicas_DONE.marker" -Encoding utf8
