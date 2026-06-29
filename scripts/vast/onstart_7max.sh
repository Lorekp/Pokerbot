#!/bin/bash
# vast.ai ONSTART provisioning + RESUME-train + periodic-S3-push + LBR-gate loop for the 7-max
# neural-CFR tournament blueprint. Mirrors the AWS launcher's on-box body MINUS the EC2-isms:
# Docker captures stdout natively (vast `vastai logs`), so NO exec>logfile redirect (we DO tee to a
# durable file that we also push to S3). vast OVERRIDES the Docker CMD/ENTRYPOINT, so this script is
# the only thing that runs -- it is invoked by `--onstart-cmd "bash /workspace/onstart_7max.sh"`.
#
# CONTRACT (all env injected by the launcher via `vastai create --env -e KEY=VAL`):
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION   S3 creds (no IAM role on vast)
#   S3_BUCKET           default video-processing-6789
#   S3_PREFIX           default pokerbot/prod6000/neural_cfr/vast_7max   (a RUN subdir)
#   REPO_TRANSPORT      "git" (clone a private repo via GITHUB_TOKEN) | "s3" (pull a code tarball
#                       the launcher uploaded to ${S3_PREFIX}/code/pokerbot_code.tar.gz). Default s3.
#   GITHUB_TOKEN / GIT_REPO_URL / GIT_BRANCH    only for REPO_TRANSPORT=git
#   ADDITIONAL_ITERS    additional iters to train (babysitter math = start + this). Default 50000.
#   PUSH_EVERY_SEC      S3 checkpoint-push cadence (default 900 = 15 min; a preemption loses <= this)
#   GATE_EVERY_SEC      LBR-gate eval cadence (default 5400 = 90 min)
#   GATE_TIER           1 (hard stop-gate) | 2 (quality). Default 1.
#   MAX_RUNTIME_SEC     budget cap: training is killed + box destroyed after this wall-clock. The
#                       launcher computes this from $71 / bid so spend never exceeds the cap.
#   AVG_STEPS / HIST_STEPS / DEPTH_BB / LR    optional trainer levers (defaults match the babysitter)
#
# CHECKPOINT LAYOUT (3 pickles, base = checkpoints/sd_cfr_7max_neural/seven_v1.pkl):
#   seven_v1.pkl (~0.9MB serve)   seven_v1_full.pkl (~25MB resumable)   seven_v1_buffers.pkl (~316MB)
# We mirror ALL THREE to s3://${S3_BUCKET}/${S3_PREFIX}/checkpoints/ so a fresh box resumes warm.
set -uo pipefail

# ---------------------------------------------------------------------------------------------------
# 0. env defaults
# ---------------------------------------------------------------------------------------------------
: "${S3_BUCKET:=video-processing-6789}"
: "${S3_PREFIX:=pokerbot/prod6000/neural_cfr/vast_7max}"
: "${AWS_DEFAULT_REGION:=eu-central-1}"
: "${REPO_TRANSPORT:=s3}"
: "${ADDITIONAL_ITERS:=50000}"
: "${PUSH_EVERY_SEC:=900}"
: "${GATE_EVERY_SEC:=5400}"
: "${GATE_TIER:=1}"
: "${MAX_RUNTIME_SEC:=777600}"          # ~9 days; the launcher overrides from the $71 budget
: "${AVG_STEPS:=512}"
: "${HIST_STEPS:=64}"
: "${DEPTH_BB:=8,120}"
: "${LR:=1e-4}"
: "${GIT_BRANCH:=aof-exploit-engine}"
export AWS_DEFAULT_REGION

REPO=/workspace/pokerbot
CKPT_DIR="$REPO/checkpoints/sd_cfr_7max_neural"
CKPT="$CKPT_DIR/seven_v1.pkl"
LOG=/workspace/onstart.log
TRAIN_LOG=/workspace/train.log
GATE_LOG=/workspace/gate.log
STATE_DIR=/workspace/state
mkdir -p /workspace "$STATE_DIR"

# tee everything to a durable log we also push to S3; do NOT redirect with exec (keep vast's native
# stdout capture working -- the AWS "no exec>logfile" lesson, here for vast logs).
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

S3="s3://${S3_BUCKET}/${S3_PREFIX}"
aws_s3() { aws s3 "$@" --region "$AWS_DEFAULT_REGION"; }

log "onstart begin. bucket=$S3_BUCKET prefix=$S3_PREFIX transport=$REPO_TRANSPORT add_iters=$ADDITIONAL_ITERS max_runtime=${MAX_RUNTIME_SEC}s gate_tier=$GATE_TIER"

# ---------------------------------------------------------------------------------------------------
# 1. base tooling (image-aware: nvcr/pytorch images are Debian/Ubuntu; only apt if apt exists)
# ---------------------------------------------------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq || true
  apt-get install -y -qq git curl unzip >/dev/null 2>&1 || true
fi
# awscli + boto3 via pip (the nvcr image has pip+python; do NOT install a second torch).
python -m pip install --quiet --no-input awscli boto3 >>"$LOG" 2>&1 || \
  pip install --quiet --no-input awscli boto3 >>"$LOG" 2>&1 || true
command -v aws >/dev/null 2>&1 || { log "FATAL: aws cli not available after pip install"; exit 1; }
aws --version 2>&1 | tee -a "$LOG" || true

# verify S3 creds early (fail fast rather than train for hours then fail to push).
if ! aws_s3 ls "$S3/" >/dev/null 2>&1; then
  log "WARNING: cannot list $S3/ yet (empty prefix is fine on first run; bad creds is NOT). probing write..."
  echo "probe $(date -u)" > /workspace/_probe.txt
  if aws_s3 cp /workspace/_probe.txt "$S3/_probe.txt" >>"$LOG" 2>&1; then
    log "S3 write OK"
  else
    log "FATAL: S3 write failed -- check AWS_ACCESS_KEY_ID/SECRET scope (need Get/Put/List on $S3_BUCKET)"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------------------------------
# 2. repo sync (two transports; the user picks via REPO_TRANSPORT)
# ---------------------------------------------------------------------------------------------------
if [ "$REPO_TRANSPORT" = "git" ]; then
  : "${GIT_REPO_URL:?REPO_TRANSPORT=git requires GIT_REPO_URL}"
  if [ -d "$REPO/.git" ]; then
    log "git pull existing repo"
    git -C "$REPO" pull --ff-only >>"$LOG" 2>&1 || true
  else
    AUTH_URL="$GIT_REPO_URL"
    if [ -n "${GITHUB_TOKEN:-}" ]; then
      AUTH_URL="$(echo "$GIT_REPO_URL" | sed -E "s#https://#https://${GITHUB_TOKEN}@#")"
    fi
    log "git clone $GIT_REPO_URL ($GIT_BRANCH)"
    git clone --depth 1 --branch "$GIT_BRANCH" "$AUTH_URL" "$REPO" >>"$LOG" 2>&1 \
      || { log "FATAL: git clone failed"; exit 1; }
  fi
else
  # S3 tarball transport (the launcher uploaded it). Allow-list tarball, same _INCLUDE shape as AWS.
  log "pull code tarball from $S3/code/pokerbot_code.tar.gz"
  mkdir -p "$REPO"
  aws_s3 cp "$S3/code/pokerbot_code.tar.gz" /workspace/code.tar.gz >>"$LOG" 2>&1 \
    || { log "FATAL: code tarball download failed"; exit 1; }
  tar -xzf /workspace/code.tar.gz -C "$REPO" >>"$LOG" 2>&1 \
    || { log "FATAL: code tarball extract failed"; exit 1; }
fi
ls -la "$REPO/scripts/train_neural_cfr.py" 2>&1 | tee -a "$LOG" \
  || { log "FATAL: train_neural_cfr.py missing after sync"; exit 1; }

# ---------------------------------------------------------------------------------------------------
# 3. python deps for the trainer (JAX-CUDA + the repo's own requirements; the nvcr torch stays)
# ---------------------------------------------------------------------------------------------------
# The trainer is JAX, not torch. Install a CUDA jaxlib matching the box CUDA. We let pip resolve the
# CUDA12 local wheels (jax[cuda12]) -- the nvcr 25.x image ships CUDA 12.x. If the repo pins jax in
# requirements, install that too (no-deps where it would fight torch).
log "installing trainer python deps (jax[cuda12] + repo requirements)"
python -m pip install --quiet --no-input "jax[cuda12]" >>"$LOG" 2>&1 \
  || log "WARN: jax[cuda12] install returned non-zero (may already be present / falling back)"
if [ -f "$REPO/requirements.txt" ]; then
  python -m pip install --quiet --no-input -r "$REPO/requirements.txt" >>"$LOG" 2>&1 \
    || log "WARN: requirements.txt install returned non-zero (continuing; core imports checked next)"
fi
# sanity: the trainer's import chain must resolve, and JAX must see the GPU.
cd "$REPO"
export PYTHONPATH="$REPO"
python - <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys
try:
    import jax
    print("jax", jax.__version__, "devices:", jax.devices())
    gpu = any(d.platform == "gpu" for d in jax.devices())
    print("GPU_VISIBLE", gpu)
    if not gpu:
        print("WARNING: no GPU device visible to JAX -- training will run on CPU (slow).")
except Exception as e:  # noqa: BLE001
    print("FATAL import jax:", e); sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------------------------------
# 4. pull the latest 7-max checkpoint from S3 (warm resume). Prefer S3; if absent, the box may have a
#    baked-in repo checkpoint (unlikely on a fresh container) -- either way the babysitter resumes
#    whenever ${CKPT%.pkl}_full.pkl exists.
# ---------------------------------------------------------------------------------------------------
mkdir -p "$CKPT_DIR"
log "syncing checkpoints from $S3/checkpoints/ -> $CKPT_DIR"
aws_s3 sync "$S3/checkpoints/" "$CKPT_DIR/" >>"$LOG" 2>&1 || true
if [ -f "${CKPT%.pkl}_full.pkl" ]; then
  log "RESUME: found ${CKPT%.pkl}_full.pkl ($(du -h "${CKPT%.pkl}_full.pkl" | cut -f1)) -> warm resume"
  RESUME_ARGS=(--resume "$CKPT")
else
  log "NO S3 checkpoint found -- the user must seed s3://.../checkpoints/ with seven_v1{,_full,_buffers}.pkl. Starting from whatever the repo ships (likely COLD)."
  if [ -f "${CKPT%.pkl}_full.pkl" ]; then RESUME_ARGS=(--resume "$CKPT"); else RESUME_ARGS=(); fi
fi

# ---------------------------------------------------------------------------------------------------
# 5. background: periodic checkpoint PUSH to S3 (so a preemption loses <= PUSH_EVERY_SEC of progress)
# ---------------------------------------------------------------------------------------------------
push_ckpts() {
  while true; do
    sleep "$PUSH_EVERY_SEC"
    # sync only the 3 checkpoint pickles + the train/gate logs; cheap, idempotent.
    aws_s3 sync "$CKPT_DIR/" "$S3/checkpoints/" --exclude "*" \
      --include "seven_v1.pkl" --include "seven_v1_full.pkl" --include "seven_v1_buffers.pkl" \
      >>"$LOG" 2>&1 || true
    for f in "$TRAIN_LOG" "$GATE_LOG" "$LOG"; do
      [ -f "$f" ] && aws_s3 cp "$f" "$S3/logs/$(basename "$f")" >>"$LOG" 2>&1 || true
    done
    # heartbeat for the watcher: iter + ms/iter parsed from the train log tail + $-budget clock.
    python - "$TRAIN_LOG" "$STATE_DIR/heartbeat.json" >>"$LOG" 2>&1 <<'PYEOF' || true
import json, os, re, sys, time
tl, out = sys.argv[1], sys.argv[2]
it = ms = None
try:
    with open(tl, "r", errors="ignore") as fh:
        tail = fh.readlines()[-200:]
    for line in reversed(tail):
        m = re.search(r"iter\s+(\d+)\s*/\s*(\d+)", line)
        if m and it is None:
            it = int(m.group(1)); tot = int(m.group(2))
        m2 = re.search(r"([\d.]+)\s*ms/iter", line)
        if m2 and ms is None:
            ms = float(m2.group(1))
        if it is not None and ms is not None:
            break
except OSError:
    tot = None
hb = {"ts": time.time(), "iter": it, "total": locals().get("tot"), "ms_per_iter": ms,
      "start_epoch": float(os.environ.get("RUN_START_EPOCH", time.time())),
      "max_runtime_sec": float(os.environ.get("MAX_RUNTIME_SEC", 0))}
with open(out, "w") as fh:
    json.dump(hb, fh)
PYEOF
    aws_s3 cp "$STATE_DIR/heartbeat.json" "$S3/state/heartbeat.json" >>"$LOG" 2>&1 || true
  done
}
export RUN_START_EPOCH MAX_RUNTIME_SEC
RUN_START_EPOCH=$(date +%s); export RUN_START_EPOCH
push_ckpts &
PUSH_PID=$!

# ---------------------------------------------------------------------------------------------------
# 6. background: periodic LBR GATE eval (CPU-jax so it does NOT contend the GPU the trainer holds).
#    On a PASS: write the GATE_MET sentinel to S3 + locally, then kill training (the main loop sees
#    the sentinel and exits cleanly -> box self-destroys).
# ---------------------------------------------------------------------------------------------------
GATE_OUT=/workspace/tournament_eval_net.json
run_gate() {
  while true; do
    sleep "$GATE_EVERY_SEC"
    [ -f "$CKPT" ] || continue
    log "GATE: running net LBR suite on the live checkpoint (CPU jax, 3-cell subset)"
    # CPU jax for the eval so the GPU stays with the trainer. Mandatory POKERBOT_INFO_SUITS=1.
    POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu PYTHONPATH="$REPO" \
      python scripts/tournament/eval_tournament_suite.py --policy net \
        --format NORMAL --regime deep --ckpt "$CKPT" --spots 400 \
        --out "${GATE_OUT}.deep.json" >>"$GATE_LOG" 2>&1 || true
    # The suite runs one format/regime per invocation; build the 3-cell subset by merging.
    POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu PYTHONPATH="$REPO" \
      python scripts/tournament/eval_tournament_suite.py --policy net \
        --format NORMAL --regime shallow --ckpt "$CKPT" --spots 400 \
        --out "${GATE_OUT}.shallow.json" >>"$GATE_LOG" 2>&1 || true
    POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu PYTHONPATH="$REPO" \
      python scripts/tournament/eval_tournament_suite.py --policy net \
        --format NORMAL --regime bubble --ckpt "$CKPT" --spots 400 \
        --out "${GATE_OUT}.bubble.json" >>"$GATE_LOG" 2>&1 || true
    # merge the three single-cell JSONs into one cells map the gate-decider reads.
    python - "${GATE_OUT}" "${GATE_OUT}.deep.json" "${GATE_OUT}.shallow.json" "${GATE_OUT}.bubble.json" \
      >>"$GATE_LOG" 2>&1 <<'PYEOF' || true
import json, sys
out = sys.argv[1]; parts = sys.argv[2:]
merged = {"cells": {}}
for p in parts:
    try:
        d = json.load(open(p))
    except (OSError, ValueError):
        continue
    merged.update({k: v for k, v in d.items() if k != "cells"})
    merged.setdefault("cells", {}).update(d.get("cells", {}))
json.dump(merged, open(out, "w"))
PYEOF
    aws_s3 cp "${GATE_OUT}" "$S3/state/tournament_eval_net.json" >>"$GATE_LOG" 2>&1 || true
    if python scripts/vast/gate_decider.py --json "${GATE_OUT}" --tier "$GATE_TIER" \
         >>"$GATE_LOG" 2>&1; then
      log "GATE PASSED (tier $GATE_TIER) -- writing sentinel + stopping training"
      echo "GATE_MET tier=$GATE_TIER $(date -u)" > "$STATE_DIR/GATE_MET"
      aws_s3 cp "$STATE_DIR/GATE_MET" "$S3/state/GATE_MET" >>"$GATE_LOG" 2>&1 || true
      # final ckpt + log push then stop the trainer; main loop notices and exits.
      aws_s3 sync "$CKPT_DIR/" "$S3/checkpoints/" >>"$LOG" 2>&1 || true
      [ -n "${TPID:-}" ] && kill "$TPID" 2>/dev/null || true
      break
    else
      log "GATE not yet met (tier $GATE_TIER); continuing. last: $(tail -1 "$GATE_LOG")"
    fi
  done
}
run_gate &
GATE_PID=$!

# ---------------------------------------------------------------------------------------------------
# 7. background: budget / runtime watchdog -- hard kill + destroy after MAX_RUNTIME_SEC.
# ---------------------------------------------------------------------------------------------------
budget_watch() {
  local start; start=$(date +%s)
  while true; do
    sleep 120
    local now; now=$(date +%s)
    if [ "$((now - start))" -ge "$MAX_RUNTIME_SEC" ]; then
      log "BUDGET CAP: ${MAX_RUNTIME_SEC}s elapsed -> stopping training (budget exhausted)"
      echo "BUDGET_CAP $(date -u)" > "$STATE_DIR/BUDGET_CAP"
      aws_s3 cp "$STATE_DIR/BUDGET_CAP" "$S3/state/BUDGET_CAP" >>"$LOG" 2>&1 || true
      [ -n "${TPID:-}" ] && kill "$TPID" 2>/dev/null || true
      break
    fi
  done
}
budget_watch &
BUDGET_PID=$!

# ---------------------------------------------------------------------------------------------------
# 8. the trainer. RESUME from S3 checkpoint; additive --iterations (= ADDITIONAL_ITERS); the babysitter
#    math elsewhere is start+ADDITIONAL. --traversal-workers 0 (mp pool is 6-max only). GPU env from
#    the babysitter. We do NOT exec-redirect; tee to a durable log the push loop ships to S3.
# ---------------------------------------------------------------------------------------------------
log "launching trainer: +$ADDITIONAL_ITERS iters, avg_steps=$AVG_STEPS hist_steps=$HIST_STEPS depth=$DEPTH_BB lr=$LR"
POKERBOT_INFO_SUITS=1 POKERBOT_7MAX_ANTE_BB=0.125 POKERBOT_7MAX_DEPTH_BB="$DEPTH_BB" \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 JAX_PLATFORMS=cuda \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH="$REPO" \
  python -u scripts/train_neural_cfr.py --game nlhe-7max --iterations "$ADDITIONAL_ITERS" \
    --traversals 128 --batch-size 1024 --hidden-dim 256 --train-steps 16 \
    --avg-steps "$AVG_STEPS" --hist-steps "$HIST_STEPS" --lr "$LR" --seed 0 \
    --checkpoint "$CKPT" --checkpoint-every 25 --buffer-checkpoint-every 500 \
    --log-every 25 --traversal-workers 0 "${RESUME_ARGS[@]}" \
    2>&1 | tee "$TRAIN_LOG" &
TPID=$!
export TPID
log "trainer PID $TPID"
wait "$TPID"
TRC=$?
log "trainer exited rc=$TRC"

# ---------------------------------------------------------------------------------------------------
# 9. FINAL push (always) + decide why we stopped + self-destroy the vast box on a terminal stop.
# ---------------------------------------------------------------------------------------------------
kill "$PUSH_PID" "$GATE_PID" "$BUDGET_PID" 2>/dev/null || true
log "final checkpoint + log push to S3"
aws_s3 sync "$CKPT_DIR/" "$S3/checkpoints/" >>"$LOG" 2>&1 || true
for f in "$TRAIN_LOG" "$GATE_LOG" "$LOG"; do
  [ -f "$f" ] && aws_s3 cp "$f" "$S3/logs/$(basename "$f")" >>"$LOG" 2>&1 || true
done

if [ -f "$STATE_DIR/GATE_MET" ]; then
  log "TERMINAL: gate met -> writing DONE marker + self-destroying box"
  echo "DONE gate $(date -u)" > "$STATE_DIR/DONE.marker"
  aws_s3 cp "$STATE_DIR/DONE.marker" "$S3/sevenmax_DONE.marker" >>"$LOG" 2>&1 || true
elif [ -f "$STATE_DIR/BUDGET_CAP" ]; then
  log "TERMINAL: budget cap -> writing DONE marker + self-destroying box"
  echo "DONE budget $(date -u)" > "$STATE_DIR/DONE.marker"
  aws_s3 cp "$STATE_DIR/DONE.marker" "$S3/sevenmax_DONE.marker" >>"$LOG" 2>&1 || true
elif [ "$TRC" -eq 0 ]; then
  log "TERMINAL: trainer reached the iteration cap cleanly -> DONE marker + self-destroy"
  echo "DONE iters $(date -u)" > "$STATE_DIR/DONE.marker"
  aws_s3 cp "$STATE_DIR/DONE.marker" "$S3/sevenmax_DONE.marker" >>"$LOG" 2>&1 || true
else
  # crash / preemption: do NOT write DONE -- the babysitter relaunches a fresh box and resumes from S3.
  log "NON-TERMINAL exit (rc=$TRC, likely crash/OOM/preempt). NO DONE marker -- babysitter will resume."
fi

# Self-destroy on a terminal stop so storage billing ends. vast keeps billing after an in-container
# shutdown, so we must call `vastai destroy` on OUR OWN instance id. The launcher injects
# VAST_API_KEY + the container env VAST_CONTAINERLABEL/CONTAINER_ID; we resolve our instance id.
if [ -f "$STATE_DIR/DONE.marker" ] && [ -n "${VAST_API_KEY:-}" ]; then
  python -m pip install --quiet --no-input vastai >>"$LOG" 2>&1 || true
  vastai set api-key "$VAST_API_KEY" >>"$LOG" 2>&1 || true
  # the instance id is in $CONTAINER_ID on most vast images; fall back to the label suffix.
  MYID="${VAST_INSTANCE_ID:-${CONTAINER_ID:-}}"
  if [ -n "$MYID" ]; then
    log "self-destroying vast instance $MYID"
    vastai destroy instance "$MYID" >>"$LOG" 2>&1 || log "WARN: self-destroy failed; the babysitter will reap it"
  else
    log "WARN: could not resolve own vast instance id; the babysitter must destroy this box on DONE"
  fi
fi
log "onstart end."
