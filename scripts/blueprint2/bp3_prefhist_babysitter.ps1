# Autonomous babysitter for the BLUEPRINT-V3 preflop-history-key retrain (bp3_prefhist_1).
#
# Thin wrapper over blueprint2_babysitter.ps1: the bp2_riverochs_1 config VERBATIM (EMD-K160
# flop/turn blob + river-OCHS K160 table + n_deals 200 + 62k backstop + 40k convergence floor +
# work_steal) PLUS the one blueprint-v3 change: --preflop-hist-key, which exports
# POKERBOT_PREFLOP_HIST_KEY=1 on the box so the trainer keys PREFLOP infosets with the extra
# b"H"+[n_limps,n_raises,opener_pos+1] bytes — the causal fix for the preflop betting-state
# ALIASING root cause (artifacts/preflop_mispricing_diagnosis.md, E5 A/B validated).
#
# Run (repo root; leave the terminal open):
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\bp3_prefhist_babysitter.ps1
#
# Mid-run KILL GATE (~10-20k iters, ~$5 sunk): in a second terminal run
#   python scripts\blueprint2\check_prefhist_kill_gates.py
# and terminate the run if it prints KILL (see README_LAUNCH.md, bp3 section).
#
# The resulting checkpoint MUST be served with POKERBOT_PREFLOP_HIST_KEY=1
# (ship_blueprint.py --preflop-hist-key stamps + strictly guards this).
# Any extra blueprint2_babysitter.ps1 parameter can be appended and is forwarded verbatim.
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
& (Join-Path $PSScriptRoot "blueprint2_babysitter.ps1") `
    -Run "bp3_prefhist_1" `
    -PreflopHistKey `
    -WorkSteal `
    @args
