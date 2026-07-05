# Autonomous babysitter for the BLUEPRINT-V4 **v3** mini-proof arm (bpv4_h200_v3) -- the
# Pluribus-faithful TRAINER-SCHEDULE arm: FULLHIST + M-RR keying (VALIDATED necessary at Stage-0)
# + LCFR + hard negative-regret pruning + snapshot averaging + coverage-as-first-class-metric.
#
# Design of record : artifacts/blueprint_v4_amendment_v3.md (pre-registered gates + coverage-sized
#                    run plan; numbers committed BEFORE launch)
# Stage-0 verdict  : artifacts/bpv4_stage0_verdict.md (CLOSED per Option A; this arm tests the
#                    remaining Pluribus deltas: trainer schedule + traversal coverage)
# Primary source   : artifacts/pluribus_blueprint_spec_full.md section 1
#
# Thin wrapper over blueprint2_babysitter.ps1 (proven resume-from-S3 + storm/quota/spot guards).
# v3 config (all knobs also documented in the amendment doc section 2):
#   * -V3Lcfr -V3LcfrD 250 -V3LcfrW 10000 : iteration-denominated Linear CFR (Pluribus 10/400
#       wall-minutes -> 40 discount events over the first 25% of the 40k target; signed regrets;
#       unweighted averaging).
#   * -V3PruneC -30000 -V3PruneStart 5000 : hard negative-regret pruning from W/2 (Pluribus
#       200/400 ratio); C = -30,000 bb from the calibration run (= ~64x the stable neg_p999 ~ 450
#       bb at nd=200; REGSTATS mid-run sanity clause in the amendment doc section 6 -- verify
#       neg_p999 in [150, 1500] bb at iter 5000 BEFORE pruning starts, else relaunch with C
#       rescaled; the explore coin (5%) + regret floor C*31/30 make pruning revivable either way).
#   * -V3SnapshotAvg : NO postflop strat_sum (halves-to-quarters the row memory; preflop round-1
#       average stays in-memory) + NLSNAP01 snapshot upload at every checkpoint from ~8% of the
#       run; final blueprint = snap_avg.exe offline average (checkpoint-preflop rows take
#       precedence).
#   * -V3CovHands 2048 -V3Regstats : COVERAGE + REGSTATS lines at every checkpoint -- the
#       pre-registered coverage gates read the COVERAGE lines (amendment doc section 6).
#   * -Iters 40000 : the spec measurement point; kill gates at 10k/20k/30k (trend clauses HARD
#       from 20k per the amendment -- the bp3/H200 lesson: measure trend at 10k intervals).
#   * -CheckpointEvery 5000 : measurement points 5k..40k; snapshot-avg checkpoints are ~half the
#       Stage-0 blob size so the upload duty stays <10%.
#   * -NumBuckets 200 -EmdFrozen "" -RiverOchs "" : made-hand K200 (matched to the H200 baseline
#       and the S200/S1000 controls already in hand -- do NOT re-run controls).
#   * -InstanceType c6i.32xlarge : 128 vCPU / 256 GB. Memory math (amendment doc section 4):
#       cum ~330-350M rows @40k + preflop-only strat <=65M rows => RSS ~92-110 GB, inside the 60%
#       guard -- snapshot averaging is what makes 40k fit where Stage-0 projected 206-227 GiB.
#
# Run (repo root; leave the terminal open):
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\bpv4_v3_babysitter.ps1
#
# Mid-run kill gates (second terminal; amendment doc section 6 has the pre-registered thresholds):
#   python scripts\blueprint2\bpv4_kill_gate.py --run-subdir bpv4_h200_v3 ... (shape gates)
#   + grep COVERAGE lines from the box log (coverage gates: postflop on-policy miss).
#
# AFTER THE RUN: download checkpoints + snapshots, then
#   tools\battery_native\target\release\examples\snap_avg.exe out.nlcfr iter_040000.nlcfr snap_*.nlsnap
# and evaluate per the amendment doc (gates G1-G4/G6 + G5-checkcall + coverage). Serve/eval MUST set
# POKERBOT_FULLHIST_KEY=1 AND POKERBOT_MRR=1 (train/probe abstraction-skew guard).
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
& (Join-Path $PSScriptRoot "blueprint2_babysitter.ps1") `
    -Run "bpv4_h200_v3" `
    -FullHistKey `
    -Mrr `
    -WorkSteal `
    -V3Lcfr `
    -V3LcfrD 250 `
    -V3LcfrW 10000 `
    -V3PruneC "-30000" `
    -V3PruneStart "5000" `
    -V3SnapshotAvg `
    -V3CovHands 2048 `
    -V3Regstats `
    -Iters 40000 `
    -NumBuckets 200 `
    -EmdFrozen "" `
    -RiverOchs "" `
    -CheckpointEvery 5000 `
    -InstanceType "c6i.32xlarge" `
    -NoConvergeStop `
    @args
