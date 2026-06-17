# G4 adaptation sweep wrapper — the python driver itself waits for the Phase-0 DONE marker,
# then runs 27 configs x 6 seeds with the heuristic hero (restart-proof: completed outputs skipped).
$ErrorActionPreference = "Continue"
Set-Location "."
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force "runs/sweep" | Out-Null
python scripts/run_adaptation_sweep.py --par 8 *> runs/sweep/driver.log
