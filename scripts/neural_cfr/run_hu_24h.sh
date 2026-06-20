#!/usr/bin/env bash
# Launch the LARGER heads-up NLHE neural-CFR run (target up to ~24h). ASCII only.
#
# This is a thin wrapper over babysit_hu_overnight.sh that pins a FRESH checkpoint stem. The run MUST
# start clean: the prior checkpoints/sd_cfr_hu_neural/hu_vrpdcfr.pkl was trained BEFORE the chance-node
# traversal fix (its postflop networks learned against a corrupted tree where every street's first
# decision was overwritten) AND at epsilon=0.1. Resuming it would carry both leaks forward, so this
# points at a new stem (hu_vrpdcfr_v2) -- the old run is left intact for comparison.
#
# What you get: the babysitter self-detaches (survives the terminal closing), checkpoints atomically
# every 25 iters, and on any crash/OOM/kill RESUMES from the last good _full checkpoint -- now restoring
# the Adam moments, both reservoirs, and the iteration counter (the durability fixes), so a mid-run
# restart is loss-less. A restart-storm guard aborts a structurally-broken config instead of looping.
#
# Start (one command, from inside WSL with the CUDA venv available):
#   wsl -d Debian -- bash -lc './scripts/neural_cfr/run_hu_24h.sh'
# Watch (refresh every 30s):
#   wsl -d Debian -- bash -lc './scripts/neural_cfr/watch_hu_overnight.sh 30'
# Stop (any time; "run until killed" -- the latest checkpoint is always good and resumable):
#   wsl -d Debian -- bash -lc 'cd .;
#     kill $(cat runs/hu_neural/babysit.pid) 2>/dev/null; kill $(cat runs/hu_neural/hu_train.pid) 2>/dev/null'
#
# Tuning is the validated default (many iters x moderate steps: trav=128, steps t/a/h=16/128/32, lr=1e-4,
# epsilon now defaults to the reference 0.6 in the CLI). Override any inline, e.g.
#   AVG_STEPS=256 .../run_hu_24h.sh        # heavier fit on the SERVED average-strategy net
#   CKPT=checkpoints/sd_cfr_hu_neural/hu_vrpdcfr_v3.pkl .../run_hu_24h.sh   # another fresh stem
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# Fresh post-fix checkpoint stem (override CKPT to resume a specific prior run instead).
export CKPT="${CKPT:-checkpoints/sd_cfr_hu_neural/hu_vrpdcfr_v2.pkl}"

# --- Tournament HU training distribution -----------------------------------------------------------
# HU in a tournament appears ONLY at the very end (final 2), where blinds are high -> SHALLOW effective
# stacks. We train a per-deal log-uniform depth band (shallow-weighted) that also reaches cash depth,
# plus the target structure's ~0.125 BB per-player ante, zeroing a fraction of deals so the SAME net
# also covers no-ante cash HU (it conditions on the encoded stacks + pot). The net is CHIP-EV; ICM is
# applied at SERVE time from the actual stacks + remaining payouts (flat MTT payouts + deal-making make
# a payout-agnostic net the only correct choice). Override any inline.
export POKERBOT_HU_DEPTH_BB="${POKERBOT_HU_DEPTH_BB:-5,100}"      # 5-100bb, log-uniform (shallow-weighted)
export POKERBOT_HU_ANTE_BB="${POKERBOT_HU_ANTE_BB:-0.125}"       # per-player ante (target MTT structure)
export POKERBOT_HU_ANTE_ZERO_PROB="${POKERBOT_HU_ANTE_ZERO_PROB:-0.2}"  # 20% no-ante deals (cash HU)

exec "$HERE/babysit_hu_overnight.sh" "$@"
