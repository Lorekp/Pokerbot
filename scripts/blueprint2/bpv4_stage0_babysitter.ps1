# Autonomous babysitter for the BLUEPRINT-V4 Stage-0 FULLHIST + M-RR arm (bpv4_h200) -- the DECISIVE
# A1 arm (full action-history key + M-RR action-thinning) vs the DONE, validated S200 summary control.
#
# Design of record : artifacts/blueprint_pluribus_grade_design_v2.md (Stage 0-3, pre-registered gates)
# Runbook          : scripts/blueprint2/README_LAUNCH.md, "Blueprint-v4 Stage-0 -- cloud FULLHIST arms"
# Relief-valve note : memory note [[bpv4-mrr-relief-valve]] (fullhist ~10^31 -> ~413M rows via M-RR)
#
# Thin wrapper over blueprint2_babysitter.ps1 (proven resume-from-S3 + storm/quota/spot guards + on-box
# dead-man cost caps), setting ONLY the bpv4 Stage-0 config:
#   * -FullHistKey    : --fullhist-key  (POKERBOT_FULLHIST_KEY=1; fingerprint.rs full A-segment / street)
#   * -Mrr            : --mrr           (POKERBOT_MRR=1; trainer::abstracted_legal M-RR thinning)
#   * -WorkSteal      : --work-steal    (+33% deals/s @64c; statistically equivalent)
#   * -Iters 45000    : backstop; kill-gates fire at 20k AND 30k (design 6.3, ~$5/~$12 sunk)
#   * -NumBuckets 200 : MADE-HAND K200 (matches serve POSTFLOP_NUM_BUCKETS; this arm is NOT EMD/OCHS)
#   * -EmdFrozen "" -RiverOchs "" : DISABLE the babysitter's K160 EMD/OCHS defaults. REQUIRED -- H200 is
#       the made-hand K200 arm; the ctor enforces num_buckets==blob.K, so a K160 blob + -NumBuckets 200
#       would abort. (HE, the EMD stretch arm, is a SEPARATE launch that re-adds -EmdFrozen/-RiverOchs.)
#   * -CheckpointEvery 10000 : fullhist checkpoints are ~30-55 GB; 500 (bp2 default) would be an S3 upload
#       storm. 10k => 10k/20k/30k/40k, covering both kill-gate points with <10% upload duty.
#   * -InstanceType c6i.32xlarge : 128 vCPU / 256 GB (fits the ~413M-row fullhist+M-RR table w/ headroom).
#   * -NoConvergeStop : run the full 45k backstop; the aggregate strategy-delta convergence probe is keyed
#       for the SUMMARY (preflop-only) key space, not the fullhist all-street space, so it is not a valid
#       stop signal here -- the external kill-gates (20k/30k) + the 45k backstop govern instead.
#
# Run (repo root; leave the terminal open):
#   powershell -ExecutionPolicy Bypass -File scripts\blueprint2\bpv4_stage0_babysitter.ps1
#
# Mid-run KILL GATE (~20k AND ~30k iters): in a second terminal run the bpv4 kill-gate (see the runbook)
# and terminate the run if it prints KILL.
#
# The resulting checkpoint MUST be served/evaluated with POKERBOT_FULLHIST_KEY=1 AND POKERBOT_MRR=1
# (any in-model probe/eval that bypasses M-RR biases the A1/A2 verdict by train/probe abstraction skew).
# Any extra blueprint2_babysitter.ps1 parameter can be appended and is forwarded verbatim.
# ASCII ONLY (PS 5.1 reads BOM-less .ps1 as cp1252).
& (Join-Path $PSScriptRoot "blueprint2_babysitter.ps1") `
    -Run "bpv4_h200" `
    -FullHistKey `
    -Mrr `
    -WorkSteal `
    -Iters 45000 `
    -NumBuckets 200 `
    -EmdFrozen "" `
    -RiverOchs "" `
    -CheckpointEvery 10000 `
    -InstanceType "c6i.32xlarge" `
    -NoConvergeStop `
    @args
