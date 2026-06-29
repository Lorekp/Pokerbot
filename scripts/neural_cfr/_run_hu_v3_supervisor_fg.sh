#!/usr/bin/env bash
# Foreground wrapper for the HU v3 neural-CFR run. Lets the Windows launcher
# (start_hu_v3_overnight.ps1) pass space-free args. Sets _HU_BABYSAT=1 so
# babysit_hu_overnight.sh runs its supervisor loop FOREGROUND (skips its own
# self-detach), keeping the launching wsl.exe alive for the whole run.
# Mirrors _run_supervisor_fg.sh (HU v2) + _run_7max_supervisor_fg.sh (env-override pattern).
export _HU_BABYSAT=1
# --- v3 config (the three deltas vs v2 defaults) --------------------------------------------------
# FRESH stem: AVG_STEPS change makes the avg-strategy net / reservoirs incompatible with v2 ->
# must start clean. The babysitter only resumes if ${CKPT%.pkl}_full.pkl exists; for this fresh
# stem it does not on first launch, so v3 starts from scratch (then resumes loss-lessly on crash).
export CKPT=checkpoints/sd_cfr_hu_neural/hu_vrpdcfr_v3.pkl
export AVG_STEPS=256          # up from v2's 128: heavier fit on the SERVED avg-strategy net (the over-aggression leak fix)
export ITERS=600000           # run-to-gate, not a hard target; checkpoints preserve progress; user stops when the gate passes
# --- tournament HU training distribution (unchanged from v2; set explicitly so the wrapper is self-contained) ---
export POKERBOT_HU_DEPTH_BB=5,100
export POKERBOT_HU_ANTE_BB=0.125
export POKERBOT_HU_ANTE_ZERO_PROB=0.2
exec bash ./scripts/neural_cfr/babysit_hu_overnight.sh
