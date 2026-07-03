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
