# Definitive-table runner: 15 disjoint-seed battery legs, launched OUTSIDE the Claude Code session
# tree (via Task Scheduler) so app updates/restarts cannot kill them. Each leg writes its own log +
# JSON under runs/; this script waits for all to finish so the scheduled task shows "running" until done.
$ErrorActionPreference = "Continue"
Set-Location "."
$env:POKERBOT_BETTING_ABSTRACTION = "1"
$env:PYTHONIOENCODING = "utf-8"
$bp = "checkpoints/bp6max_v2_iter200.pkl"
$procs = @()
foreach ($arch in @("calling_station", "tag", "nit")) {
  foreach ($seed in 1..5) {
    $log = "runs/finalD_${arch}_s${seed}.log"
    $p = Start-Process -FilePath "python" -ArgumentList @(
      "scripts/eval_exploitation_battery.py", "--hero", "pluribus", "--hands", "3000",
      "--seed", "$seed", "--iterations", "10", "--allin-ev", "--no-aivat-leafq",
      "--blueprint-snapshot", $bp, "--archetypes", $arch, "--no-gear-changer",
      "--out", "runs/eb3000_finalD_${arch}_s${seed}.json"
    ) -RedirectStandardOutput $log -RedirectStandardError "runs/finalD_${arch}_s${seed}.err" `
      -WindowStyle Hidden -PassThru
    $procs += $p
  }
}
$procs | Wait-Process
"DONE $(Get-Date)" | Out-File -FilePath "runs/finalD_ALL_DONE.marker" -Encoding utf8
