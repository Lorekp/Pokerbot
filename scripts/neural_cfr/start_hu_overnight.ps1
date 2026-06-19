# Launch the supervised overnight HU neural-CFR run so it SURVIVES the launching terminal closing. ASCII only.
#
# Why this exists: `wsl -d Debian -- bash -lc 'self-detaching-script'` does NOT work for a long job -- when
# that wsl.exe returns, WSL tears the session down and reaps the setsid/nohup child (prior launches died this
# way, writing nothing). The fix: keep ONE hidden, detached wsl.exe alive running the babysitter FOREGROUND
# (the _run_supervisor_fg.sh wrapper sets _HU_BABYSAT=1 to skip the script's self-detach). That wsl.exe blocks
# on the supervisor for the whole run, so the WSL VM stays up; Start-Process detaches it from this shell so
# closing this window is fine. It dies only on machine sleep / reboot / logoff -- keep the box awake overnight.
#
#   powershell -ExecutionPolicy Bypass -File scripts\neural_cfr\start_hu_overnight.ps1
$ErrorActionPreference = "Stop"
# Every argument is space-free so Start-Process cannot mis-split them.
$wrapper = "./scripts/neural_cfr/_run_supervisor_fg.sh"
$p = Start-Process -FilePath "wsl" -ArgumentList @("-d","Debian","--","bash",$wrapper) -WindowStyle Hidden -PassThru
Write-Host ("Launched hidden wsl.exe (Windows PID {0}) running the foreground supervisor." -f $p.Id)
Write-Host "It compiles for ~3-4 min (no iter line during XLA compile -- normal), then steps."
Write-Host ""
Write-Host "WATCH:  wsl -d Debian -- bash -lc 'bash ./scripts/neural_cfr/watch_hu_overnight.sh 30'"
Write-Host 'STOP:   wsl -d Debian -- bash -lc ''kill $(cat ./runs/hu_neural/hu_train.pid) $(cat ./runs/hu_neural/babysit.pid)'''
