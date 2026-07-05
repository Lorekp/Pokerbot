# Blueprint-v2 launch package — `bp2_riverochs_1` (river-OCHS + full-length train)

Design of record: `artifacts/blueprint2_design.md`. Diagnosis + validation evidence:
`artifacts/blueprint2_progress.log`, `artifacts/bp2_street_decomp_*.json`,
`artifacts/bp2_selfplay_term_*.json`.

**What this trains**: a fresh 6-max 100bb CFR+ blueprint with the certified EMD-K160 flop/turn
abstraction, a NEW river-OCHS K160 abstraction (replaces the made-hand mirror river), CFR+ +
work_steal, 62k-iter backstop, convergence stop with a 40k min-iters floor (the 18k
premature-stop fix). 128-vCPU spot box, ~6-11h, **expected cost ~$15-40**, hard-bounded by
on-box dead-man caps (13h absolute) regardless of babysitter/PC state.

Everything below runs from the repo root `.`.

## 0. Pre-flight (1 minute, no cost)

```powershell
# artifacts present + parity green (all were validated when this package was built):
Test-Path data\abstraction\emd_frozen_flop_turn_160_r400.emdfrzn          # True
Test-Path data\abstraction\emd_frozen_flop_turn_160_r400.emdfrzn.sha256   # True
Test-Path data\abstraction\river_ochs_160.rochs                           # True
Test-Path data\abstraction\river_ochs_160.rochs.sha256                    # True
python -m pytest tests\test_river_ochs_parity.py -q                        # 3 passed
# AWS identity sanity:
aws sts get-caller-identity
```

If `river_ochs_160.rochs` is ever missing, regenerate deterministically (~1 min, local):
`python scripts\blueprint2\fit_river_ochs.py` — but NEVER refit between train and serve of
the same run (the table IS the river abstraction).

## 1. Launch (Terminal 1 — babysitter; leave it running)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\blueprint2\blueprint2_babysitter.ps1 -WorkSteal
```

That is the whole launch. Defaults (all overridable): `-Run bp2_riverochs_1 -Iters 62000
-NDeals 200 -NumBuckets 160 -ConvMinIters 40000 -MaxHours 12 -MaxLaunch 12`; instance type is
auto-picked by spotpick (cheapest 128-vCPU spot, capacity-diverse fallback chain; fallback
c6a.32xlarge ~ $1.9/hr). The first launch uploads the code tarball (crate + 152MB EMD blob +
10KB river table); the box sha-verifies BOTH abstraction artifacts before training and
self-terminates on mismatch.

Cost-safety (verbatim from the proven bp_emd_k160_1 tooling): on-box `shutdown -h +<floor>`
logind floor + `systemd-run` wall-guard transient unit are armed BEFORE any apt/build step, so
a crash anywhere (or a dead babysitter / powered-off PC) can never idle-bill past ~13h.
Babysitter storm guards: max 12 launches, >=600s between launches, 300s boot grace,
spot-quota drain guard.

## 2. Watch (Terminal 2 — read-only dashboard)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\blueprint2\watch_blueprint2.ps1
```

Shows progress %, it/s, infosets, ETA, spot cost estimate, and a red STALL alarm if no new
checkpoint lands for 20 min. Manual checks:

```powershell
aws s3 ls s3://video-processing-6789/pokerbot/prod6000/native_cfr/bp2_riverochs_1/checkpoints/
aws ec2 get-console-output --instance-id <iid> --region eu-central-1 --latest
```

Expected console milestones: `WALL_GUARD armed` → `EMD blob sha256 OK` →
`river-OCHS table sha256 OK` → `river-OCHS ACTIVE: ... stamp=path=...` →
`FRESH start` / `RESUMED from latest checkpoint` → `PROGRESS iters_done=...`.

## 3. Spot reclaim / resume (automatic)

On a reclaim the on-box handler SIGTERMs the driver (last checkpoint is already on S3, cadence
500 iters); the babysitter detects the dead box and relaunches with the same `--run-subdir`,
and the new box prints `RESUMED from latest checkpoint at iters_done=N`. Nothing to do.

Manual resume (babysitter died, run incomplete): re-run the Terminal-1 command — it picks up
from the S3 `latest.nlcfr` automatically.

## 4. After the run — download, eval gate, promotion

### 4a. Download the final checkpoint

```powershell
aws s3 cp s3://video-processing-6789/pokerbot/prod6000/native_cfr/bp2_riverochs_1/checkpoints/latest.nlcfr `
    runs\native_cfr\bp2_riverochs_1.nlcfr --region eu-central-1
```

### 4b. Mechanism check (optional, ~15 min local): did the jam attractor recede?

```powershell
python scripts\blueprint2\classify_checkpoint_streets.py --checkpoint runs\native_cfr\bp2_riverochs_1.nlcfr --num-buckets 160 --out artifacts\bp2_street_decomp_riverochs.json
$env:POKERBOT_RIVER_OCHS="data\abstraction\river_ochs_160.rochs"
python scripts\blueprint2\diagnose_selfplay_termination.py --blueprint runs\native_cfr\bp2_riverochs_1.nlcfr --postflop-mode emd --num-buckets 160 --emd-frozen data\abstraction\emd_frozen_flop_turn_160_r400.emdfrzn --hands 4000 --out artifacts\bp2_selfplay_term_riverochs.json
Remove-Item env:POKERBOT_RIVER_OCHS
```

Compare vs the baselines in `artifacts/bp2_selfplay_term_emd.json` (85.1% preflop-end, 35%
preflop all-in) — expect both to drop. These are mechanism metrics, NOT the gate.

Also rerun the decision-quality audit's Leg A on the new checkpoint (~seconds; audit commit
af88c37, artifacts/decision_quality_audit.md) as a mechanism acceptance check against the
published-consensus preflop baselines (deployed: BTN opens 9.5% vs 43.6% consensus, BB-vs-BTN
folds 81.8% vs 46.8%, positional monotonicity INVERTED, 32-36% of opens are 100bb jams):
expect BTN open% to rise sharply toward consensus, the BB overfold to shrink, monotonicity to
flip to normal (wider toward the button), and jam mass to collapse. Directional check, NOT the
gate — 4c decides.

### 4c. THE GATE — CRN-paired ranking vs the deployed blueprint at >=8k decks

The two arms have DIFFERENT river abstractions (deployed = mirror river; new = river-OCHS), so
use the per-arm-subprocess cross-abstraction ranker (one serve env per process — this is the
tool that certified the EMD promotion):

```powershell
python scripts\rank_cross_abstraction.py `
  --a runs\native_cfr\bp_emd_k160_1.nlcfr  --a-mode emd --a-num-buckets 160 `
  --a-emd-frozen data\abstraction\emd_frozen_flop_turn_160_r400.emdfrzn `
  --b runs\native_cfr\bp2_riverochs_1.nlcfr --b-mode emd --b-num-buckets 160 `
  --b-emd-frozen data\abstraction\emd_frozen_flop_turn_160_r400.emdfrzn `
  --b-river-ochs data\abstraction\river_ochs_160.rochs `
  --decks 8000 --seed 11 --workers 10 `
  --label-a bp_emd_k160_1 --label-b bp2_riverochs_1 `
  --out-json artifacts\rank_bp2riverochs_vs_emdk160_8k.json
```

**Promotion requires**: verdict `bp2_riverochs_1 STRONGER (95% CI excludes 0)` at n=8000.
Anything else (INCONCLUSIVE / A stronger) = NO promotion; keep serving bp_emd_k160_1 and
report the artifact.

### 4d. Ship gates + promotion (only after 4c passes)

```powershell
# convert + run all serve-skew gates (sha sidecars incl. the river table, strict stamp
# cross-check incl. the river_ochs field, legal/finite probe, >50% coverage, infoset floor)
# -- NO activation yet:
python scripts\ship_blueprint.py --checkpoint runs\native_cfr\bp2_riverochs_1.nlcfr `
    --name bp2_riverochs_1 --river-ochs data\abstraction\river_ochs_160.rochs

# promote (after the gate output reads PASS):
python scripts\ship_blueprint.py --checkpoint runs\native_cfr\bp2_riverochs_1.nlcfr `
    --name bp2_riverochs_1 --river-ochs data\abstraction\river_ochs_160.rochs --activate
```

IMPORTANT after `--activate`: every deployed-serve consumer must run with
`POKERBOT_RIVER_OCHS=data/abstraction/river_ochs_160.rochs` in addition to the existing EMD
env (the stamp guard fails loud — never silent — if it is missing). Update the live_coach /
eval preamble defaults the same way the dab000a promotion updated the EMD flags, and re-smoke
live play-money before trusting autoplay (standing rule).

### 4e. Rollback

```powershell
python scripts\ship_blueprint.py --revert     # restores the .bak (bp_emd_k160_1)
```

(plus revert any env-default commits, mirroring the documented dab000a rollback).

---

# Blueprint-v3 launch package — `bp3_prefhist_1` (preflop HISTORY key)

Design of record: `artifacts/preflop_mispricing_diagnosis.md` (root cause: preflop betting-state
infoset ALIASING; causal E5 A/B — aliased arm BTN 8.5% inverted / AA-limp 91% vs hist arm BTN
23.9% normal-monotone / AA-limp 54% at matched 20k iters).

**What this trains**: the `bp2_riverochs_1` config VERBATIM (EMD-K160 flop/turn + river-OCHS K160
+ CFR+ n_deals=200 + work_steal + 40k convergence floor + 62k backstop) PLUS one change:
`POKERBOT_PREFLOP_HIST_KEY=1` on the box, so the trainer keys PREFLOP infosets with 3 extra bytes
`[n_limps, n_raises, opener_pos+1]` (fingerprint.rs, env-gated). Preflop rows grow ~7x (~350-700k;
table still small); throughput ~unchanged; same ~$15-40 spot cost envelope as bp2.

**Train/serve contract**: the key is SHARED train/serve. A hist-trained checkpoint MUST be served
with `POKERBOT_PREFLOP_HIST_KEY=1` (and vice versa: the deployed non-hist blueprints keep serving
with it UNSET, byte-identical). The Python serve twin (`jax_weva_fast.preflop_hist_bytes` + the
`b"H"` digest branch), the stamp (`active_abstraction_config()["preflop_hist_key"]`), the
two-sided serve guard (`lbr.check_blueprint_serve_env`), and the ship gate
(`ship_blueprint.py --preflop-hist-key`) enforce this; a skew in either direction fails LOUD
(100% preflop miss).

## B0. Pre-flight (1 minute, no cost)

```powershell
# bp2 pre-flight (section 0 above: EMD blob + river table + sha sidecars + river parity), PLUS:
python -m pytest tests\test_preflop_hist_key_parity.py -q     # 2 passed (rust==python==layout,
                                                              #  512+ states, BOTH env states)
```

## B1. Launch (Terminal 1 — babysitter; leave it running)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\blueprint2\bp3_prefhist_babysitter.ps1
```

(A thin wrapper: `blueprint2_babysitter.ps1 -Run bp3_prefhist_1 -PreflopHistKey -WorkSteal` —
everything in sections 1-3 above applies verbatim, including watch_blueprint2.ps1.)
New expected console milestone after `RAYON_NUM_THREADS=...`: `PREFLOP_HIST_KEY=1`.

## B2. MID-RUN KILL GATE at 10-20k iters (~$5 sunk decision point)

When the ladder shows ~10-20k iters (checkpoints land every 500), run in a second terminal:

```powershell
python scripts\blueprint2\check_prefhist_kill_gates.py
```

It downloads the latest `bp3_prefhist_1` ladder checkpoint (+ a half-iters trend baseline), runs
the Leg-A direct-key extraction under the hist key space, and evaluates the pre-registered gates
(diagnosis section 6.4, calibrated to the local 20k A/B): positional monotonicity NORMAL (no
inversion, BTN>UTG), BTN open >= 15%, AA-BTN-limp <= 60% (falling-trend clause advisory before
30k, hard after — the E5 positive control rises 21.6%→53.7% over 5k→20k), BB-vs-BTN-open fold
<= 75%. It prints every number and a PASS/KILL verdict.

On **KILL**: terminate the box + stop the babysitter — do NOT fund the run to the backstop:

```powershell
aws ec2 terminate-instances --instance-ids <iid> --region eu-central-1
```

Re-run the same command near the end of the run (trend clause is hard from 30k).

## B3. After the run — download + final gate

```powershell
aws s3 cp s3://video-processing-6789/pokerbot/prod6000/native_cfr/bp3_prefhist_1/checkpoints/latest.nlcfr `
    runs\native_cfr\bp3_prefhist_1.nlcfr --region eu-central-1
```

**THE GATE** — 8k-deck CRN-paired ranking vs the DEPLOYED `bp_emd_k160_1` (per-arm subprocess;
arm B needs BOTH the river-OCHS table AND the hist key — its per-arm env sets both):

```powershell
python scripts\rank_cross_abstraction.py `
  --a runs\native_cfr\bp_emd_k160_1.nlcfr  --a-mode emd --a-num-buckets 160 `
  --a-emd-frozen data\abstraction\emd_frozen_flop_turn_160_r400.emdfrzn `
  --b runs\native_cfr\bp3_prefhist_1.nlcfr --b-mode emd --b-num-buckets 160 `
  --b-emd-frozen data\abstraction\emd_frozen_flop_turn_160_r400.emdfrzn `
  --b-river-ochs data\abstraction\river_ochs_160.rochs `
  --b-preflop-hist-key `
  --decks 8000 --seed 11 --workers 10 `
  --label-a bp_emd_k160_1 --label-b bp3_prefhist_1 `
  --out-json artifacts\rank_bp3prefhist_vs_emdk160_8k.json
```

**Promotion requires**: verdict `bp3_prefhist_1 STRONGER (95% CI excludes 0)` at n=8000. Anything
else = NO promotion (a wider, monotone preflop that gate-fails stays unpromoted — the gate
decides, not chart-distance). Optional mechanism check first: rerun B2's kill-gate script with
`--checkpoint runs\native_cfr\bp3_prefhist_1.nlcfr` (trend clause hard at final iters).

## B4. Ship gates + promotion (only after B3 passes)

```powershell
# all serve-skew gates (strict stamp cross-check now incl. preflop_hist_key + river_ochs), NO activation:
python scripts\ship_blueprint.py --checkpoint runs\native_cfr\bp3_prefhist_1.nlcfr `
    --name bp3_prefhist_1 --river-ochs data\abstraction\river_ochs_160.rochs --preflop-hist-key

# promote (after the gate output reads PASS):
python scripts\ship_blueprint.py --checkpoint runs\native_cfr\bp3_prefhist_1.nlcfr `
    --name bp3_prefhist_1 --river-ochs data\abstraction\river_ochs_160.rochs --preflop-hist-key --activate
```

IMPORTANT after `--activate`: every deployed-serve consumer must now run with BOTH
`POKERBOT_RIVER_OCHS=data/abstraction/river_ochs_160.rochs` AND `POKERBOT_PREFLOP_HIST_KEY=1`
in addition to the existing EMD env (the two-sided stamp guards fail loud — never silent — on
any omission). Update the live_coach / eval preamble defaults the same way the dab000a promotion
updated the EMD flags, and **re-smoke live play-money before trusting autoplay** (standing rule).
Native serve fast paths stay FAIL-CLOSED under the river-OCHS env (no native river-OCHS serve
keying yet) — the Python twin serves, same regime as bp2 would have.

Rollback: `python scripts\ship_blueprint.py --revert` (+ revert the env-default commits),
mirroring 4e above.

## Troubleshooting

* Babysitter exits `max launches reached`: repeated capacity failure — rerun with
  `-InstanceType c7a.32xlarge` (or another spotpick suggestion) or `-OnDemand` (pricier).
* `FATAL: ... sha mismatch` in console output: a corrupted artifact upload — delete
  `s3://video-processing-6789/pokerbot/prod6000/native_cfr/code/native_cfr_code.tar.gz`, rerun
  (the babysitter re-uploads on launch #1 of a fresh babysitter process).
* Stall alarm with a live box: check `get-console-output` for a cargo build error; the box
  self-terminates at its dead-man cap regardless — no idle-bill risk.
* Native serve fast paths: they stay FAIL-CLOSED for river-OCHS checkpoints (fingerprint.rs
  does not carry the river-OCHS keying yet); evals/serve use the Python path — same regime the
  deployed EMD blueprint already runs. Follow-up: port river-OCHS into
  `fingerprint.rs::fingerprint_fast_depth_emd` + `probe_native_emd_parity.py`.

# Blueprint-v4 Stage-0 — cloud FULLHIST arms

Trains the DECISIVE full-action-history-key + M-RR arms to the 45k backstop on a 256 GB spot box,
runs mid-run kill-gates at 20k/30k, and produces the A1/A2 causal verdict. Design of record:
`artifacts/blueprint_pluribus_grade_design_v2.md` §5–7. Why cloud: M-RR made fullhist CONVERGENT but
still Pluribus-scale (~413M rows ≈ 55 GB blob / ~90–110 GB RAM); the local 31 GB box ROW_CAPs at
~iter 2,200, far below the 40k gates. The SUMMARY controls (S200/S1000) are DONE + validated locally
(S200@20k reproduces BTN 8.77% open / AA-limp 90.7% — the limp-trap the controls must show), so only
the FULLHIST arms need the box.

**Arm order (run the box for these in this sequence):**
1. **H200** — fullhist + M-RR, made-hand K200. The decisive **A1** arm vs the DONE S200 summary
   control (same K, only the key space differs → isolates the history axis). Launch this first.
2. **HD200** — H200 with DCFR (`-Dcfr`, a follow-up; the launcher/babysitter DCFR passthrough is the
   deliverable-1 follow-up — do NOT block H200 on it).
3. **H1000** — fullhist + M-RR, made-hand K1000. The **A2** card-axis arm (K1000-vs-K200 within
   fullhist decides whether an EMD re-fit enters Stage 1).
4. **HE** — fullhist + M-RR + EMD-K160/river-OCHS (the EMD stretch; a SEPARATE launch that re-adds
   `-EmdFrozen`/`-RiverOchs` and the sha sidecars).

## V0. Pre-flight (BEFORE launching the box)

```powershell
# 1) fullhist key cross-language parity — Rust fingerprint.rs == Python twin on >=512 all-street
#    states, ON and OFF (a miss here means every lookup on a fullhist blueprint falls to uniform):
python -m pytest tests/test_fullhist_key_parity.py::test_fullhist_key_parity -q
#    (skips cleanly if battery_native is not built; build it first so the gate actually runs.)

# 2) default-off byte-identity (bp2/bp3 launches must be unchanged): the new --fullhist-key/--mrr
#    launcher flags OFF must generate a byte-identical bootstrap — dead-knob screen per the design.

# 3) EMD/OCHS sha sidecars — ONLY for the HE arm (H200/HD200/H1000 are made-hand K200/K1000, no
#    EMD/OCHS). For HE, ship the frozen bucketer + river-OCHS sha sidecars the same way bp2/bp3 do.
```

## V1. Launch (H200 first) — the USER runs the spot box

```powershell
# Leave this terminal open; the babysitter owns spotpick + resume-from-S3 + on-box dead-man cost
# caps + storm/quota/spot guards, inherited UNCHANGED from blueprint2_babysitter.ps1.
powershell -ExecutionPolicy Bypass -File scripts\blueprint2\bpv4_stage0_babysitter.ps1
```

`bpv4_stage0_babysitter.ps1` bakes the H200 config: `-FullHistKey -Mrr -WorkSteal -Iters 45000
-NumBuckets 200 -EmdFrozen "" -RiverOchs "" -CheckpointEvery 10000 -InstanceType c6i.32xlarge
-NoConvergeStop`. Checkpoint cadence 10000 => ladder 10k/20k/30k/40k (covers both kill-gate points;
fullhist blobs are 30–55 GB so 500-iter cadence would be an S3 upload storm). For **H1000** append
`-NumBuckets 1000`; for **HE** append `-EmdFrozen data\abstraction\emd_frozen_flop_turn_160_r400.emdfrzn
-RiverOchs data\abstraction\river_ochs_160.rochs -NumBuckets 160` (and use its own `-Run bpv4_he200`).
Any extra `blueprint2_babysitter.ps1` parameter is forwarded verbatim.

## V2. Watch (second terminal, read-only, never terminates a box)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\blueprint2\watch_blueprint2.ps1 `
    -Run bpv4_h200 -Iters 45000
```

## V3. Mid-run KILL GATE at ~20k AND ~30k (third terminal)

Design §6.3 (bp3 lesson: the 30k check would have saved that run). Cost at risk ~$5 (20k) / ~$12
(30k). Streams the latest S3 ladder checkpoint memory-safely (never loads the 30–55 GB blob whole),
extracts the fullhist preflop nodes, and prints G1–G4/G6 + the trend forms of G2/G3:

```powershell
python scripts\blueprint2\bpv4_kill_gate.py --run-subdir bpv4_h200            # latest ladder point
python scripts\blueprint2\bpv4_kill_gate.py --run-subdir bpv4_h200 --iter 20000   # pin 20k
python scripts\blueprint2\bpv4_kill_gate.py --run-subdir bpv4_h200 --iter 30000   # pin 30k
```

Gates (calibrated at the mid-run point; hard caps at every check, trend HARD from `--trend-hard-iters`
= 30k): **G1** UTG≤HJ≤CO≤BTN open (−0.5pt tol, BTN>UTG); **G2** BTN open ≥ 20% (+ decline ≥ −3pt hard
at 30k); **G3** AA-BTN-limp ≤ 30% (+ non-rising hard at 30k); **G4** BB-vs-BTN-open fold ≤ 70%
(open-size-weighted); **G6** BTN jam share of open ≤ 10%. A nonzero **negative-control** count (summary
rows read from a fullhist blob) ABORTS with rc 2 — the key space is wrong, do not trust the numbers.
The babysitter also runs these from the S3 ladder automatically and terminates on KILL; on a manual
**KILL** verdict:

```powershell
aws ec2 terminate-instances --instance-ids <iid> --region eu-central-1   # + stop the babysitter
```

## V4. Download the final checkpoint (after the 45k backstop, or the last good ladder point)

```powershell
aws s3 cp `
  s3://video-processing-6789/pokerbot/prod6000/native_cfr/bpv4_h200/checkpoints/iter_045000.nlcfr `
  runs\native_cfr\bpv4_stage0\H200\iter_045000.nlcfr --region eu-central-1
# (grab the whole ladder dir if you want the final gate's 30k/40k trajectory clauses)
```

## V5. Final Stage-0 gates + G5 probe -> A1/A2 verdict

```powershell
# G1-G4/G6 + ladder + M1 row scan at 30k/40k on the AVERAGE strategy (streaming, wrong-key-space
# negative control asserted per checkpoint):
python scripts\blueprint2\bpv4_stage0_gates.py --arm H200 `
    --dir runs\native_cfr\bpv4_stage0\H200 --key-mode fullhist `
    --out artifacts\bpv4_stage0_gates_H200.json

# Direct-key extraction sanity (does NOT use abstracted_legal — direct dict lookup, MRR-independent):
python scripts\blueprint2\preflop_root_diag.py --full-hist --deltas `
    runs\native_cfr\bpv4_stage0\H200\iter_030000.nlcfr `
    runs\native_cfr\bpv4_stage0\H200\iter_040000.nlcfr `
    --out artifacts\bpv4_extract_H200.json

# G5 forced-response probe (8-shard, fallback-0 gate). SET POKERBOT_MRR=1 (see the MRR-FIDELITY note):
$env:POKERBOT_MRR = "1"
python scripts\blueprint2\bpv4_stage0_probe.py --arm H200 `
    --ckpt runs\native_cfr\bpv4_stage0\H200\iter_040000.nlcfr `
    --key-mode fullhist --postflop-mode fast --num-buckets 200
```

**Verdict** — write `artifacts/bpv4_stage0_verdict.md` by hand from the two JSONs:
* **A1 (history axis) PASS** iff H200 passes G1–G5 while the DONE S200 summary control fails ≥ 2 of
  them ⇒ history keying is causal ⇒ fund Stage 1. G5a: EV(best sized open) − EV(limp) ≥ −25 bb/100;
  G5b: EV(open) > 0. If the fullhist arm ALSO fails broadly ⇒ **STOP, do not fund Stage 1** (the
  residual driver is cards/deeper; the `bp3_forced_response_probe.md` §5 closure terms stand).
* **A2 (card axis)**: run the same on **H1000** (K1000). If the K1000-vs-K200 main effect on G2/G5 >
  half the history main effect ⇒ an EMD re-fit enters Stage 1.

## !! CRITICAL — G5 probe M-RR fidelity (must read before trusting the verdict)

The G5 probe rollout is a **pure-Python path** (`bpv4_stage0_probe.py` → `bp3_forced_response_probe.py`
`rollout` → `TabularBlueprintAgent` → `TabularBlueprintAdapter` → `lbr.make_blueprint_policy`). It does
**NOT** call the Rust `trainer::abstracted_legal` (the MRR-thinning path) **nor** `subgame::abstracted_
legal` (subgame.rs:240). It is the **Python analog of the SERVE path**: it legal-masks the stored strat
row with the un-thinned serve menu `allowed_post = (0,1,3,4,5)` and reads **no** `POKERBOT_MRR` env var.

* **Setting `POKERBOT_MRR=1` for the probe is a NO-OP on this path** (only the Rust trainer OnceLock
  reads it). We set it anyway — defensively/for provenance, and so any future native probe honors it.
* **Fidelity is preserved IMPLICITLY, and only conditionally.** On a fingerprint **hit** the frozen
  average-strategy row carries **zero** mass on MRR-removed actions (the trainer masked them out), so
  masking the row by the wider serve menu + renormalizing keeps them zero → downstream play stays
  within the trained (thinned) menu. **BUT on a fingerprint MISS**, `make_blueprint_policy` silently
  returns **uniform over the full un-thinned legal menu** (MRR-removed sizes included) — train/probe
  abstraction skew — **and this miss is NOT counted by the probe's `adapter_fallback_rate` gate**
  (`TabularBlueprintAgent.fallback_count` only trips on an exception / all-zero row; the miss is
  absorbed inside `make_blueprint_policy`, which always returns a valid distribution). So the driver's
  "fallback == 0.0000" abort does **not** actually detect fingerprint misses.
* **What to do:** treat the G5 verdict as trustworthy only if the checkpoint is well-converged in the
  fullhist key space so misses are near-zero. **SCHEDULED FIX** (before the probe is load-bearing for a
  promotion decision): make the probe MRR-aware — either (a) enable `lbr._BP_COVERAGE` in the rollout
  and assert per-street miss-rate ≈ 0, or (b) apply the M-RR keep-lists to the serve menu / uniform
  fallback when `POKERBOT_MRR=1` (mirror `mrr_reraise_keep`/`mrr_tr_first_keep` in
  `lbr.make_blueprint_policy`). The forced arms themselves (open325/open400/jam) are **preflop**, which
  M-RR never thins, so the forcing is unaffected — only the postflop rollout tail carries this risk.
* The direct-key extractor (`preflop_root_diag.py --full-hist`, and the streaming
  `bpv4_stage0_gates.py`/`bpv4_kill_gate.py`) does **NOT** use `abstracted_legal` — it looks keys up
  directly in the checkpoint dict and all its gate nodes are **preflop** (untouched by M-RR), so
  extraction and every G1–G4/G6 number are **MRR-independent** and unaffected.
