# Launch the supervised overnight 7-MAX (tournament) neural-CFR GPU run so it SURVIVES the launching
# terminal closing. ASCII only. Mirrors start_hu_overnight.ps1 (the proven HU launcher).
#
# Why this exists: `wsl -d Debian -- bash -lc 'self-detaching-script'` does NOT work for a long job -- when
# that wsl.exe returns, WSL tears the session down and reaps the setsid/nohup child (the 2026-06-23 resume
# died exactly this way, writing nothing: no train log, no GPU load, babysitter PID dead). The fix: keep ONE
# hidden, detached wsl.exe alive running the babysitter FOREGROUND (the _run_7max_supervisor_fg.sh wrapper
# sets _7MAX_BABYSAT=1 to skip the script's self-detach). That wsl.exe blocks on the supervisor for the whole
# run, so the WSL VM stays up; Start-Process detaches it from this shell so closing this window is fine. It
# dies only on machine sleep / reboot / logoff -- keep the box awake.
#
# NOTE: the 7-max XLA graph needs the full WSL memory (12GB OOM-kills the compile -- see .wslconfig history);
# keep .wslconfig memory at 22GB while this runs.
#
#   powershell -ExecutionPolicy Bypass -File scripts\neural_cfr\start_7max_overnight.ps1
$ErrorActionPreference = "Stop"
# Every argument is space-free so Start-Process cannot mis-split them.
$wrapper = "./scripts/neural_cfr/_run_7max_supervisor_fg.sh"
$p = Start-Process -FilePath "wsl" -ArgumentList @("-d","Debian","--","bash",$wrapper) -WindowStyle Hidden -PassThru
Write-Host ("Launched hidden wsl.exe (Windows PID {0}) running the foreground 7-max supervisor." -f $p.Id)
Write-Host "It resumes from seven_v1_full.pkl (iter ~78,300) and recompiles for ~3-4 min"
Write-Host "(no iter line during XLA compile -- that is normal), then steps to 100,000."
Write-Host ""
Write-Host "WATCH:  wsl -d Debian -- bash -lc 'bash ./scripts/neural_cfr/watch_7max_overnight.sh 30'"
Write-Host 'STOP:   wsl -d Debian -- bash -lc ''kill $(cat ./runs/sevenmax_neural/seven_train.pid) $(cat ./runs/sevenmax_neural/babysit.pid)'''
