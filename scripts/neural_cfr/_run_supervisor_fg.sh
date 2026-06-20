#!/usr/bin/env bash
# Tiny wrapper so the Windows launcher can pass space-free args. Sets _HU_BABYSAT=1 so the babysitter runs
# its supervisor loop FOREGROUND (no self-detach), keeping the launching wsl.exe alive for the whole run.
export _HU_BABYSAT=1
exec bash ./scripts/neural_cfr/babysit_hu_overnight.sh
