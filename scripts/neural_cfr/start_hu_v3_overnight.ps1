# Launch the supervised overnight HU v3 neural-CFR GPU run so it SURVIVES the launching terminal
# closing. ASCII only. Mirrors start_hu_overnight.ps1 (the proven HU launcher).
#
# v3 config (vs v2): FRESH checkpoint stem hu_vrpdcfr_v3.pkl + AVG_STEPS=256 (up from 128, the
# highest-ROI single change -- heavier fit on the SERVED avg-strategy net, targets v2's
# over-aggression leak) + ITERS=600000 (run-to-gate, not a hard target). Same validated defaults
# otherwise (trav=128 batch=1024 hidden=256 lr=1e-4 ckpt-every=25 info_suits=1 cuda) and the same
# tournament HU training distribution (POKERBOT_HU_DEPTH_BB=5,100 ANTE_BB=0.125 ANTE_ZERO_PROB=0.2).
# All v3 env lives in the wrapper (_run_hu_v3_supervisor_fg.sh) so this .ps1 is identical to v2's.
#
# DOES NOT RESUME v2: the AVG_STEPS change makes the avg-strategy net / reservoir buffers
# incompatible with v2, so v3 MUST start clean. The fresh stem guarantees the babysitter sees no
# *_full.pkl and starts from scratch; it then resumes loss-lessly from hu_vrpdcfr_v3_full.pkl on any
# crash/OOM/kill.
#
# Why this exists: `wsl -d Debian -- bash -lc 'self-detaching-script'` does NOT work for a long job --
# when that wsl.exe returns, WSL tears the session down and reaps the setsid/nohup child (prior
# launches died this way, writing nothing: no train log, no GPU load, babysitter PID dead). The fix:
# keep ONE hidden, detached wsl.exe alive running the babysitter FOREGROUND (the
# _run_hu_v3_supervisor_fg.sh wrapper sets _HU_BABYSAT=1 to skip the script's self-detach). That
# wsl.exe blocks on the supervisor for the whole run, so the WSL VM stays up; Start-Process detaches
# it from this shell so closing this window is fine. It dies only on machine sleep / reboot / logoff
# -- keep the box awake.
#
# DO NOT launch while the 7-max job (#39) holds the GPU -- they contend for it. Launch only after
# 7-max frees the GPU.
#
#   powershell -ExecutionPolicy Bypass -File scripts\neural_cfr\start_hu_v3_overnight.ps1
$ErrorActionPreference = "Stop"
# Every argument is space-free so Start-Process cannot mis-split them.
$wrapper = "./scripts/neural_cfr/_run_hu_v3_supervisor_fg.sh"
$p = Start-Process -FilePath "wsl" -ArgumentList @("-d","Debian","--","bash",$wrapper) -WindowStyle Hidden -PassThru
Write-Host ("Launched hidden wsl.exe (Windows PID {0}) running the foreground HU v3 supervisor." -f $p.Id)
Write-Host "Config: FRESH stem hu_vrpdcfr_v3.pkl  AVG_STEPS=256  ITERS=600000  (does NOT resume v2)."
Write-Host "It compiles for ~3-4 min (no iter line during XLA compile -- normal), then steps to 600,000."
Write-Host ""
Write-Host "PID FILES:"
Write-Host "  runs/hu_neural/babysit.pid     (supervisor PID)"
Write-Host "  runs/hu_neural/hu_train.pid    (current training PID)"
Write-Host "  runs/hu_neural/hu_train.curlog (path to the live train log)"
Write-Host ""
Write-Host "WATCH (rich PS dashboard):"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\neural_cfr\watch_hu_v3.ps1 30"
Write-Host "WATCH (bash, via wsl):"
Write-Host "  wsl -d Debian -- bash -lc 'bash ./scripts/neural_cfr/watch_hu_overnight.sh 30'"
Write-Host 'STOP:   wsl -d Debian -- bash -lc ''kill $(cat ./runs/hu_neural/hu_train.pid) $(cat ./runs/hu_neural/babysit.pid)'''
