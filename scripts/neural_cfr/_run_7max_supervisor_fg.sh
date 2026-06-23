#!/usr/bin/env bash
# Tiny wrapper so the Windows launcher (start_7max_overnight.ps1) can pass space-free args. Sets
# _7MAX_BABYSAT=1 so babysit_7max_overnight.sh runs its supervisor loop FOREGROUND (skips its own
# self-detach), keeping the launching wsl.exe alive for the whole run. Mirrors _run_supervisor_fg.sh (HU).
export _7MAX_BABYSAT=1
# RECOVERY 2026-06-23: the 100k run crashed at iter 78,325 and was rebuilt from a partial checkpoint
# (scripts/neural_cfr/_recover_7max_ckpt.py). The babysitter resume math is RELATIVE
# (total_iterations = start + --iterations), so to STOP at the 100k TARGET (not 78.6k+100k=178k) we pass
# the REMAINING count. 100000 - 78625 = 21375. Remove this line for a fresh from-scratch run.
export ITERS=21375
exec bash ./scripts/neural_cfr/babysit_7max_overnight.sh
