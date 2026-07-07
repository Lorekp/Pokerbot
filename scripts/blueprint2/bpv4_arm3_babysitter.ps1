# Autonomous babysitter for the BLUEPRINT-V4 **ARM 3** run (bpv4_emd160_arm3) -- arm-2b plus
# the Pluribus-grade menu deltas mandated by amendment sec 24:
#   * -PreKeep "2,3,4,5,6"   : POKERBOT_PRE_KEEP=2,3,4,5,6 -- widens the preflop FIRST-raise
#       (open) bet-bucket keep list from the legacy {4,5,6} to {2,3,4,5,6}, opening five
#       open-raise sizes: 2.0bb / 2.12bb / 2.5bb / 3.25bb / 4.0bb. The primary axis tested by arm-3.
#   * -PreRrKeep "4,5,6"     : POKERBOT_PRE_RR_KEEP=4,5,6 -- preflop 3-bet+ menu pinned LEGACY.
#   * -FlopRrKeep "1,4"      : POKERBOT_FLOP_RR_KEEP=1,4 -- widens the flop re-raise keep under M-RR.
#
# ============================ TWO-PHASE MIGRATION (amendment sec 25, 2026-07-07) ============================
# The 64-vCPU X-family spot quota (approved 2026-07-07) buys x2iedn.16xlarge (64 vCPU / 2TB / $3.72/hr) --
# the ONLY 2TB box at 64 vCPU. The design-of-record x2idn.32xlarge (128 vCPU / 2TB / same $3.72/hr) needs a
# 128-vCPU X quota we do NOT have, so the affordable X box has HALF the compute of the original plan. That
# makes "run the fast 128-vCPU box as long as possible, switch to X only when RAM forces it" a clear win on
# BOTH cost and speed:
#   PHASE 1  r8i.32xlarge  128 vCPU / 1TB / ~$1.9-2.0/hr live spot  -- primary: deepest spot placement
#            score (3-in-all-AZs), best 128-vCPU >=1TB box; 2x compute vs phase 2. r6a.32xlarge
#            (128 vCPU / 1TB / ~$2.87/hr) stays a spotpick FALLBACK (not in the phase-1 exclude list).
#   PHASE 2  x2iedn.16xlarge 64 vCPU / 2TB / $3.72/hr  -- the tail, once the table outgrows 1TB.
# The on-box driver plants <run>/migrate_request.json when MemAvailable drops below -MigrateMinAvailGb OR
# rows reach -MigrateMaxRows (a deterministic backstop against a broken RAM probe -- the 2026-07-06
# incident); this babysitter then writes the DURABLE <run>/migrated_to_phase2.json marker, switches to
# x2iedn.16xlarge, and resumes from S3. Phase is re-derived from the marker on every babysitter restart.
# Migrate watermark 150 GiB / 2.30B rows => migrates at ~2.2-2.3B cum rows (~RSS 850-880 GiB on the 1TB
# box; serialize is STREAMED so its transient is only ~24 B/row ~= 55 GiB, comfortably inside the headroom).
# Phase 1 keeps spotpick over the 1TB r-family; the phase-1 exclude keeps the 128-vCPU X boxes OUT so the
# fallback chain cannot eat the X quota reserved for phase 2. Phase 2 is SINGLE-TYPE (no <2TB fallback that
# would OOM the row-heavy tail).
#
# Design of record : amendment sec 24 (menus/gates) + sec 25 (two-phase). -RowBudget 4500000000 is the
#                    TERMINAL RAM backstop on the 2TB phase-2 box (never fires on phase 1 -- migrate goes
#                    first at ~2.2B). LAUNCH GATED on the sec 24.4 checklist + the 64-vCPU X quota (DONE).
# Prior verdicts   : bpv4_emd160_arm2b K1-KILLED @20k -- LCFR exonerated, AA-limp / BB-overfold reproduced
#                    WITHOUT LCFR => structural under FULLHIST+MRR with narrow menu. Arm-3 axis = wider preflop.
#
# !!! SERVE/EVAL WARNING: the trained checkpoint MUST be served and evaluated with ALL of:
#   POKERBOT_FULLHIST_KEY=1      (full action-history key on every street)
#   POKERBOT_MRR=1               (M-RR action-thinning)
#   POKERBOT_PRE_KEEP=2,3,4,5,6  (widened preflop first-raise menu; arm-3 addition)
#   POKERBOT_PRE_RR_KEEP=4,5,6   (legacy preflop 3-bet+ menu, pinned; arm-3 addition)
#   POKERBOT_FLOP_RR_KEEP=1,4    (widened flop re-raise keep under M-RR; arm-3 addition)
# A train/serve abstraction skew SILENTLY biases every eval verdict. (The MIGRATE_* envs are TRAINING-only
# -- they never touch the served strategy; do NOT set them at serve time.)
#
# Thin wrapper over blueprint2_babysitter.ps1 (proven resume-from-S3 + storm/quota guards + two-phase).
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
#
# Run (repo root; leave the terminal open):
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\bpv4_arm3_babysitter.ps1
& (Join-Path $PSScriptRoot "blueprint2_babysitter.ps1") `
    -Run "bpv4_emd160_arm3" `
    -FullHistKey `
    -Mrr `
    -PreKeep "2,3,4,5,6" `
    -PreRrKeep "4,5,6" `
    -FlopRrKeep "1,4" `
    -WorkSteal `
    -NoCacheFetch `
    -V3SignedRegrets `
    -V3PruneC "-1000000" `
    -V3PruneStart "5000" `
    -V3SnapshotAvg `
    -V3SnapStart "3200" `
    -V3CovHands 2048 `
    -V3Regstats `
    -RowBudget 4500000000 `
    -Phase1InstanceType "r8i.32xlarge" `
    -Phase2InstanceType "x2iedn.16xlarge" `
    -MigrateMinAvailGb "150" `
    -MigrateMaxRows "2300000000" `
    -Phase1ExcludeFamilies "c5,c5a,c5ad,c5d,c5n,c6a,c6i,c6id,c6in,c7a,c8i,c8id,i4i,m6a,m6i,m6id,m6idn,m6in,m7a,m8i,m8id,r6i,trn1,trn1n,x1,x1e,x2idn,x2iedn,x8i" `
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
    -MaxHours 20 `
    -NoConvergeStop `
    @args
