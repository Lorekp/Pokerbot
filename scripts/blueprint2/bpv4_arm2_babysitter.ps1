# Autonomous babysitter for the BLUEPRINT-V4 **ARM 2b** run (bpv4_emd160_arm2b) -- the
# v3-closeout-mandated DECOMPOSITION arm upgraded to production abstraction per the pre-launch
# review: FULLHIST + M-RR keying (parity-validated) + frozen EMD-K160 postflop buckets + SIGNED
# REGRETS + hard negative-regret pruning + snapshot averaging, WITHOUT the LCFR discount (v3 K1
# hard-kill: LCFR improves coverage but destroys preflop shape), CFR+ w=t linear averaging KEPT,
# nd=8192 x 40k iters (traversal-identical to the closeout's 20k x 16384; amendment sec 11).
#
# Design of record : artifacts/blueprint_v4_arm2_amendment.md sec 2 + sec 11 + sec 12
#                    (PRE-REGISTERED gates + numbers, sec 12 committed BEFORE this launch)
# Pre-launch review: artifacts/bpv4_arm2_prelaunch_review.md (VERDICT: LAUNCH WITH CHANGES --
#                    all changes below are the sec-12 deltas)
# Prior verdicts   : artifacts/bpv4_v3_closeout.md, artifacts/bpv4_stage0_verdict.md
#
# Thin wrapper over blueprint2_babysitter.ps1 (proven resume-from-S3 + storm/quota guards).
# Arm-2b config (every number derives from the amendment doc):
#   * -EmdFrozen data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn -NumBuckets 160 (sec 12.1):
#       made-hand is a documented +188 bb/100 structural leak; EMD-K160 is the certified fix and
#       what the serving blueprint trains on. FULLHIST x EMD composition parity-gated:
#       artifacts/_arm2b_fullhist_parity_k160.json (3630 states, 0 mismatches).
#   * -V3SignedRegrets                 : POKERBOT_SIGNED_REGRETS=1 -- signed regrets + deep floor +
#       pruning, NO discount events, w=t averaging kept (sec 2.1).
#   * -V3PruneC -50000 -V3PruneStart 5000 (sec 12.2): C = -64 x stationary neg_p999 (794.3 bb,
#       fresh SIGNED-REGRETS calibration, sec 2) -- sec 11's -30000 was a mis-derivation from v3's
#       LCFR regime and is RETIRED. Floor = C*31/30 = -51,667 (trainer default).
#       K0 REGSTATS sanity (BINDING): at the iter-5000 checkpoint neg_p999 must be in
#       **[390, 3910] bb** (= [C/64/2, C/64*5]); outside => STOP, rescale C to -64 x measured
#       p999 (1 s.f.), relaunch from the 5000 checkpoint.
#   * -Iters 40000 -NDeals 8192 -CheckpointEvery 250 -GateEvery 2500 : sec 11 traversal-identical
#       spec point (327.7M traversals); 250-iter rolling checkpoints (SPOT, sec 13 relaunch: the
#       2026-07-06 perf merge makes the storm math work -- training ~2.9x faster verified locally
#       + serialize pause 212s->~96s at 41GB -- fresh-box time-to-first-marker ~8-12 min vs the
#       measured 20-27 min reclaim windows), gate ladder checkpoints kept forever.
#   * -CkptVolumeGb 400                : root gp3 (700 MiB/s) staging volume for the detached uploader.
#   * -InstanceType c6a.32xlarge -Spotpick (sec 13): 128 vCPU / 256 GB at ~$1.88/h spot; projected
#       ~$20-35 total at the merged pace incl. churn rework. Capacity fallback chain is all
#       128-vCPU/>=256GB types. User directive 2026-07-06: spot, reassessed viable after the
#       profiler (3.2x, byte-identical, golden sha 77bc2aa8 verbatim) + checkpoint (2.2x pause,
#       format-compatible, resume-identity test) merges -- both independently re-verified.
#   * -V3SnapshotAvg -V3SnapStart 3200 : 8% of 40k (v3 rule verbatim, sec 11).
#   * -RowBudget 550000000             : G8 RAM backstop on max(strat,cum); K160 shrinks the space
#       vs the K200 census, so this is looser than registered -- kept verbatim.
#   * -V3CovHands 2048 -V3Regstats     : COVERAGE + REGSTATS at every checkpoint; train.log uploads
#       at every checkpoint (v3 lesson).
#
# Run (repo root; leave the terminal open):
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\bpv4_arm2_babysitter.ps1
#
# Mid-run kill gates (second terminal; amendment sec 6 schedule at the sec-11 x2 re-index,
# thresholds UNCHANGED):
#   K0 @5000  : REGSTATS neg_p999 in [390, 3910] bb (grep REGSTATS from s3 logs/train.log)
#   K1 @20000 : python scripts\blueprint2\bpv4_kill_gate.py --run-subdir bpv4_emd160_arm2b --iter 20000 --trend-hard-iters 30000
#               + COVERAGE line: C1 on-policy postflop miss <= 55% (missed by >10pt => KILL)
#   K2 @30000 : same command --iter 30000 (full level caps + trend clauses HARD from 30k)
#   K3 @>=15000: on-policy postflop miss RISING >5pt vs the checkpoint 10,000 iters prior => KILL
#   C3 @40000 : overall postflop miss <= 35% AND flop-street miss <= 30% (else the sec-12.6
#               extension DECISION goes to the user with the fitted coverage curve -- NOT auto-funded)
#
# AFTER THE RUN: download checkpoints + snapshots, then
#   tools\battery_native\target\release\examples\snap_avg.exe out.nlcfr iter_040000.nlcfr snap_*.nlsnap
# then the sec-12.5 COMPARISON GATE vs checkpoints/serve_blueprint_6max.pkl (CRN-paired 8192-deck
# seat-rotated H2H + pool; PROMOTE only if paired CI excludes 0 in the candidate's favor on BOTH;
# CI-LB <= -15 bb/100 on either => HARD reject). Serve/eval MUST set POKERBOT_FULLHIST_KEY=1 AND
# POKERBOT_MRR=1 (train/probe abstraction-skew guard).
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
#
# POOL WEATHER (2026-07-06 ~14:15): c6a lifetimes collapsed to 9-20 min (5 dry boxes across two
# cycles); c6i measured 27-51 min this morning. m6i stays head (capacity-failed once, retried each
# launch); c6a/c5-family excluded from the ALTS so a capacity fallback lands on c6i/c6in/m6a
# instead of re-rolling the churniest pool.
# RAM FLOOR (2026-07-06 ~16:30, profile run i-0f9aa667065ac1388): the run's RSS is 191 GB at iter
# 1000 AND GROWS; a 256 GB box (c7a landed via fallback) page-thrashed to 0.194 it/s under the
# 60 GB checkpoint serialize+upload churn. ALL 256 GB 128-vCPU families excluded (c6i/c6in/c6id/
# c7a) -- the chain now lands on 512 GB+ types only (m6i head; m6a/m6id/m6in/m7a/r6i alternates).
$env:SPOTPICK_EXCLUDE_FAMILIES = "c6a,c5,c5a,c5ad,c5d,c5n,c6i,c6in,c6id,c7a"
& (Join-Path $PSScriptRoot "blueprint2_babysitter.ps1") `
    -Run "bpv4_emd160_arm2b" `
    -FullHistKey `
    -Mrr `
    -WorkSteal `
    -V3SignedRegrets `
    -V3PruneC "-50000" `
    -V3PruneStart "5000" `
    -V3SnapshotAvg `
    -V3SnapStart "3200" `
    -V3CovHands 2048 `
    -V3Regstats `
    -RowBudget 550000000 `
    -Iters 40000 `
    -NDeals 8192 `
    -NumBuckets 160 `
    -EmdFrozen "data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn" `
    -RiverOchs "" `
    -DeltaCkpt `
    -CheckpointEvery 100 `
    -GateEvery 2500 `
    -CkptVolumeGb 400 `
    -InstanceType "m6i.32xlarge" `
    -Spotpick `
    -MaxHours 20 `
    -NoConvergeStop `
    @args
