# Autonomous babysitter for the BLUEPRINT-V4 **ARM 2** mini-proof (bpv4_h200_arm2) -- the
# v3-closeout-mandated DECOMPOSITION arm: FULLHIST + M-RR keying (validated) + SIGNED REGRETS +
# hard negative-regret pruning + snapshot averaging, WITHOUT the LCFR discount (v3 proved LCFR
# improves coverage but destroys preflop shape faster than CFR+ -- closeout section 3), with CFR+
# w=t linear averaging KEPT and n_deals RAISED 8192 -> 16384 (the closeout's primary coverage
# lever; total traversals at 20k iters == the v3 plan's 40k x 8192).
#
# Design of record : artifacts/blueprint_v4_arm2_amendment.md (PRE-REGISTERED gates + numbers,
#                    committed BEFORE launch)
# Prior verdicts   : artifacts/bpv4_v3_closeout.md (v3 K1 hard-kill => CLOSE; this arm is the one
#                    defensible follow-up named there), artifacts/bpv4_stage0_verdict.md
#
# Thin wrapper over blueprint2_babysitter.ps1 (proven resume-from-S3 + storm/quota/spot guards).
# Arm-2 config (every number derives from the amendment doc):
#   * -V3SignedRegrets                 : POKERBOT_SIGNED_REGRETS=1 -- signed regrets + deep floor +
#       pruning, NO discount events, w=t averaging kept (pre-registered choice, amendment sec 2.1).
#   * -V3PruneC -50000 -V3PruneStart 2500 : C = -64 x stationary neg_p999 (794.3 bb, fresh arm-2
#       calibration artifacts/_arm2_calib/calib.out) rounded to 1 s.f.; floor = C*31/30 = -51,667
#       (trainer default). prune_start traversal-matched to v3's 5000@8192.
#       K0 REGSTATS sanity (BINDING, amendment sec 2.2): at the iter-2500 checkpoint neg_p999 must
#       be in [390, 3910] bb (= [C/64/2, C/64*5]); outside => STOP, rescale C to -64 x measured
#       p999 (1 s.f.), relaunch from the 2500 checkpoint.
#   * -NDeals 16384                    : the coverage lever (closeout sec 4.1). NOTE: passed
#       EXPLICITLY -- the v3 wrapper relied on @args for n_deals; never again.
#   * -Iters 20000 -CheckpointEvery 2500 : spec point 20k = 327.7M traversals (identical to the v3
#       plan's 40k x 8192); 8 checkpoints + 8 snapshots (same averaging density as the v3 plan).
#   * -V3SnapshotAvg -V3SnapStart 1600 : 8% of target (v3 rule verbatim).
#   * -RowBudget 550000000             : G8 RAM backstop on max(strat,cum), NOW ACTUALLY EXPORTED
#       (the v3 amendment registered 550M but no export existed -- the driver ran its 400M
#       default; found + fixed in arm-2 prep via --row-budget).
#   * -V3CovHands 2048 -V3Regstats     : COVERAGE + REGSTATS at every checkpoint. The launcher's
#       driver now also uploads train.log AT EVERY CHECKPOINT (v3 lesson: the K1 hard-terminate
#       lost every COVERAGE line because the log only uploaded at run end).
#   * -NumBuckets 200 -EmdFrozen "" -RiverOchs "" : made-hand K200 (matched to H200/S200/S1000
#       baselines already in hand -- do NOT re-run controls).
#   * -InstanceType c6i.32xlarge       : 128 vCPU / 256 GB. Memory math (amendment sec 4): cum
#       ~450M + strat ~105M rows @20k => RSS ~132 GB = 51% of box, inside the 60% guard; the 550M
#       row budget fires at ~61% if growth surprises.
#
# Run (repo root; leave the terminal open):
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\bpv4_arm2_babysitter.ps1
#
# Mid-run kill gates (second terminal; amendment sec 6 has the pre-registered schedule --
# iteration indices are the v3 schedule traversal-matched, thresholds UNCHANGED):
#   K0 @2500  : REGSTATS band above (grep REGSTATS from s3 logs/train.log -- uploaded per ckpt)
#   K1 @10000 : python scripts\blueprint2\bpv4_kill_gate.py --run-subdir bpv4_h200_arm2 --iter 10000 --trend-hard-iters 15000
#               + COVERAGE line: C1 on-policy postflop miss <= 55% (missed by >10pt => KILL)
#   K2 @15000 : same command --iter 15000 (full level caps + trend clauses HARD from 15k)
#   K3 @>=7500: on-policy postflop miss RISING >5pt vs the checkpoint 5,000 iters prior => KILL
#
# AFTER THE RUN: download checkpoints + snapshots, then
#   tools\battery_native\target\release\examples\snap_avg.exe out.nlcfr iter_020000.nlcfr snap_*.nlsnap
# and evaluate per the amendment (G1-G4/G6 + G5'-checkcall N=40,000 + C1-C3). Serve/eval MUST set
# POKERBOT_FULLHIST_KEY=1 AND POKERBOT_MRR=1 (train/probe abstraction-skew guard).
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
& (Join-Path $PSScriptRoot "blueprint2_babysitter.ps1") `
    -Run "bpv4_h200_arm2" `
    -FullHistKey `
    -Mrr `
    -WorkSteal `
    -V3SignedRegrets `
    -V3PruneC "-50000" `
    -V3PruneStart "2500" `
    -V3SnapshotAvg `
    -V3SnapStart "1600" `
    -V3CovHands 2048 `
    -V3Regstats `
    -RowBudget 550000000 `
    -Iters 20000 `
    -NDeals 16384 `
    -NumBuckets 200 `
    -EmdFrozen "" `
    -RiverOchs "" `
    -CheckpointEvery 2500 `
    -InstanceType "c6i.32xlarge" `
    -NoConvergeStop `
    @args
