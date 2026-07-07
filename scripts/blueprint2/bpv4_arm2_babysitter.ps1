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
#   * -V3PruneC -1000000 -V3PruneStart 5000: K0 FIRED 2026-07-06 20:15 (sec 19). Registered
#       procedure executed verbatim: measured neg_p999 = 19,950 bb at the iter-5000 REGSTATS
#       (band [390, 3910] from the C=-50000 calibration; the sec-2 794.3 bb stationary estimate
#       was ~25x low at production scale) => C = -64 x 19,950 = -1,276,800 -> 1 s.f. =
#       **-1,000,000**; floor = C*31/30 (trainer default); relaunched FROM the iter-5000 base
#       (post-5000 old-C deltas purged from S3). History: sec 11's -30000 (LCFR regime) RETIRED;
#       sec 12.2's -50000 (small-scale signed-regrets calibration) SUPERSEDED by this on-scale
#       measurement.
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
#   * -RowBudget 900000000             : G8 RAM backstop on max(strat,cum). RE-DERIVED 2026-07-06
#       from the profile run (i-073e03466cb6fce7d): cum_regret alone hit 550M rows by iter 1400
#       (~90% of the ~609M M-RR keyspace -- coverage fills fast at nd=8192), so the registered
#       550M value stops the run at ~3.5% of target; it was sized as a RAM guard, not a kill gate.
#       Measured RAM: 191 GB RSS at ~500M rows = 0.38 GB/M rows -> 900M rows = ~342 GB, safe on
#       the 512 GB chain (256 GB families are excluded above) while still backstopping runaway
#       growth past the keyspace estimate. See blueprint_v4_arm2_amendment.md sec 16.
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
# CADENCE 250 (2026-07-06 ~20:00, user-approved after profile run #3): the parallel serializer
# (f60ecf4) cut the delta pause to ~14s, and the graceful-flush path banks the un-checkpointed
# window on any 2-min reclaim notice -- so cadence no longer bounds graceful-reclaim loss, only
# hard deaths (<=~42s compute at 250). 100 -> 250 lifts the compute fraction ~55% -> ~75%; the
# flush delta (~20-25GB at 250) still fits the notice window. Amendment sec 17.
# CKPT SIZE CAP 24->48GB (2026-07-07, user-approved): the HYBRID rolling trigger's SIZE cap was
# binding every ~4-5 min (dirty ~24GB per ~450-550 iters at iter ~12000), which the user judged
# too frequent -- ~10-min cadence is enough. Raised the cap to 48GB (+ min-gap floor 60->300s).
# At 48GB the emergency flush (~150s serialize+upload) may NOT fit the 2-min notice for near-cap
# windows: the flush stays BEST-EFFORT (pkill fallback at 100s) and the user explicitly accepted
# up to ~10 min of replay loss on a reclaim in exchange for the lower checkpoint overhead. Once
# pruning matures and the dirty rate drops, the CkptSecs 600s TIME trigger takes over at ~10 min.
# 1TB FLOOR (2026-07-06 ~20:45, sec 18): cum_rows 756M @2250 (+19-21M/100, decay slow) -- the
# 609M M-RR census is refuted and the 900M budget would fire ~iter 3000. Budget 1.4B (reaches
# the iter-5000 pruning onset with margin; ~532GB at 0.38GB/M rows) requires 1TB boxes: all
# 512GB m-families excluded; chain = r6a head + r6i/r6id/r6in/r7a alternates.
# CHURN CURATION (2026-07-06 ~21:40, spot-advisor probe): r6i is in the >20% interruption band
# for eu-central-1 (worst) -- excluded. x2idn.32xlarge (2TB, <5% band, $2.86-2.91 = cheaper than
# r6a in 1a/1b) stays IN the chain as a preferred alternate.
# SEC-21 (2026-07-07, post-audit relaunch, user-approved): census undercount confirmed (sec 20,
# run CLEAN) -> budget 2.2B (safe 1TB ceiling; carries to ~K1@20000 at the projected 2.0-2.2B),
# ckpt volume 400->800GB (base blob ~240GB at 20k; prestage + fresh serialize must coexist).
# Back half (20k->40k, ~3.5B rows) awaits the user's X-family spot quota -> x2idn 2TB + budget ~4B.
$env:SPOTPICK_EXCLUDE_FAMILIES = "c6a,c5,c5a,c5ad,c5d,c5n,c6i,c6in,c6id,c7a,m6i,m6id,m6in,m6idn,m6a,m7a,r6i,x1,x1e"
& (Join-Path $PSScriptRoot "blueprint2_babysitter.ps1") `
    -Run "bpv4_emd160_arm2b" `
    -FullHistKey `
    -Mrr `
    -WorkSteal `
    -V3SignedRegrets `
    -V3PruneC "-1000000" `
    -V3PruneStart "5000" `
    -V3SnapshotAvg `
    -V3SnapStart "3200" `
    -V3CovHands 2048 `
    -V3Regstats `
    -RowBudget 2200000000 `
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
    -CkptVolumeGb 800 `
    -InstanceType "x2idn.32xlarge" `
    -Spotpick `
    -MaxHours 20 `
    -NoConvergeStop `
    @args
