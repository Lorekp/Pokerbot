# G3 prior A/B wrapper — the python driver waits for the sweep marker, then runs old-vs-fitted
# prior on the same 18 seeds with the heuristic hero (restart-proof: completed outputs skipped).
$ErrorActionPreference = "Continue"
Set-Location "."
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force "runs/prior_ab" | Out-Null
python scripts/run_prior_ab.py --par 8 *> runs/prior_ab/driver.log
