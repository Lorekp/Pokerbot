# Autonomous babysitter for the BLUEPRINT-V4 **ARM 3** run (bpv4_emd160_arm3) -- arm-2b plus
# the Pluribus-grade menu deltas mandated by amendment sec 24:
#   * -PreKeep "2,3,4,5,6"   : POKERBOT_PRE_KEEP=2,3,4,5,6 -- widens the preflop FIRST-raise
#       (open) bet-bucket keep list from the legacy {4,5,6} to {2,3,4,5,6}, opening five
#       open-raise sizes: 2.0bb / 2.12bb / 2.5bb / 3.25bb / 4.0bb (bucket indices per the
#       current preflop grid). The primary axis tested by arm-3 (Pluribus uses 1-14 open
#       sizes; ours had 3).
#   * -PreRrKeep "4,5,6"     : POKERBOT_PRE_RR_KEEP=4,5,6 -- preflop 3-bet+ (re-raise) menu
#       pinned to the LEGACY sizes; only the first-raise menu widens in arm-3.
#   * -FlopRrKeep "1,4"      : POKERBOT_FLOP_RR_KEEP=1,4 -- widens the flop re-raise keep
#       list under M-RR to {1/2 pot, 1x pot, AI}; turn/river keep the original M-RR
#       re-raise thinning.
#
# Design of record : amendment sec 24. -RowBudget 4500000000 FINALIZED per sec 24.2 (2TB
#                    RAM-guard derivation under the sec-16.3 rule; measured x1.25-1.43 row
#                    multiplier vs arm-2b in the paired local A/B; hard 2TB ceiling ~5.26B).
#                    LAUNCH GATED on X-family spot quota + sec 24.4 checklist (2TB REQUIRED --
#                    1TB binds right at K1@20000; do not fall back to r6a for this arm).
# Prior verdicts   : bpv4_emd160_arm2b K1-KILLED @20k -- LCFR exonerated, AA-limp / BB-overfold
#                    reproduced WITHOUT LCFR => structural under FULLHIST+MRR with narrow menu
#                    (artifacts/bpv4_cloud_stage0_package.md). Arm-3 axis = wider preflop.
#
# !!! SERVE/EVAL WARNING: the trained checkpoint MUST be served and evaluated with ALL of:
#   POKERBOT_FULLHIST_KEY=1      (full action-history key on every street)
#   POKERBOT_MRR=1               (M-RR action-thinning)
#   POKERBOT_PRE_KEEP=2,3,4,5,6  (widened preflop first-raise menu; arm-3 addition)
#   POKERBOT_PRE_RR_KEEP=4,5,6   (legacy preflop 3-bet+ menu, pinned; arm-3 addition)
#   POKERBOT_FLOP_RR_KEEP=1,4    (widened flop re-raise keep under M-RR; arm-3 addition)
# A train/serve abstraction skew SILENTLY biases every eval verdict.
#
# Thin wrapper over blueprint2_babysitter.ps1 (proven resume-from-S3 + storm/quota guards).
# All arm-2b flags carried verbatim EXCEPT -CkptVolumeGb bumped 800->1200 (wider preflop menu
# grows base blob; 1200 GB ensures prestage + fresh serialize coexist on the EBS volume).
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
#
# POOL WEATHER (inherited from arm-2b): c6a/c5-family/c6i/c6in/c6id/c7a/m6i/m6id/m6in/m6idn/
#   m6a/m7a/r6i/x1/x1e excluded (256GB-or-under + churn-heavy families; see arm-2b header).
#   x2idn.32xlarge (2TB, <5% interruption band) stays IN as preferred alternate; quota pending.
#
# Run (repo root; leave the terminal open):
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\bpv4_arm3_babysitter.ps1
$env:SPOTPICK_EXCLUDE_FAMILIES = "c6a,c5,c5a,c5ad,c5d,c5n,c6i,c6in,c6id,c7a,m6i,m6id,m6in,m6idn,m6a,m7a,r6i,x1,x1e"
& (Join-Path $PSScriptRoot "blueprint2_babysitter.ps1") `
    -Run "bpv4_emd160_arm3" `
    -FullHistKey `
    -Mrr `
    -PreKeep "2,3,4,5,6" `
    -PreRrKeep "4,5,6" `
    -FlopRrKeep "1,4" `
    -WorkSteal `
    -V3SignedRegrets `
    -V3PruneC "-1000000" `
    -V3PruneStart "5000" `
    -V3SnapshotAvg `
    -V3SnapStart "3200" `
    -V3CovHands 2048 `
    -V3Regstats `
    -RowBudget 4500000000 `
    -Iters 40000 `
    -NDeals 8192 `
    -NumBuckets 160 `
    -EmdFrozen "data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn" `
    -RiverOchs "" `
    -DeltaCkpt `
    -CheckpointEvery 250 `
    -CkptSecs 600 `
    -CkptMaxDirtyGb 48 `
    -CkptMinGapSecs 300 `
    -GateEvery 2500 `
    -CkptVolumeGb 1200 `
    -InstanceType "x2idn.32xlarge" `
    -Spotpick `
    -MaxHours 20 `
    -NoConvergeStop `
    @args
