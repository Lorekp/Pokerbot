#!/usr/bin/env bash
# Overnight heads-up NLHE neural-CFR (VR-DeepPDCFR+) on the local RTX 4070 SUPER via WSL CUDA.
# ASCII only. Run from inside WSL (the CUDA jax venv); it self-activates and self-detaches.
#
#   wsl -d Debian -- bash -lc './scripts/neural_cfr/run_hu_overnight.sh'
#
# Design: a very high iteration CAP + frequent ATOMIC checkpoints. The job runs until you kill it
# (or morning); the latest checkpoint is always good. Override any default via env, e.g.
#   ITERS=200 CKPT_EVERY=10 .../run_hu_overnight.sh        # short calibration run
set -euo pipefail

REPO=.
VENV="$HOME/jaxgpu-venv"

# --- config (env-overridable) ---------------------------------------------------------------------
GAME=${GAME:-nlhe-hu}
ITERS=${ITERS:-100000}            # effective "run until killed"; checkpoints preserve progress
# Profiled on the 4070 SUPER: the traversal is ~0.7 s/iter (nearly free), so traversals are cheap
# and worth raising for lower-variance targets; the wall cost is almost entirely the SGD steps
# (~per-step overhead is high). So: MANY iters with MODERATE steps (CFR gains more from iterations
# than from over-fitting each iter; the Leduc gate was healthy at 8-64 steps) beats few heavy iters.
TRAVERSALS=${TRAVERSALS:-128}     # cheap (~0.7 s); better advantage targets
BATCH=${BATCH:-1024}
HIDDEN=${HIDDEN:-256}
TRAIN_STEPS=${TRAIN_STEPS:-16}    # SGD steps/iter for advantage nets
AVG_STEPS=${AVG_STEPS:-128}       # the SERVED average-strategy net gets the most fitting
HIST_STEPS=${HIST_STEPS:-32}      # HistoryValueNet baseline
LR=${LR:-1e-4}
SEED=${SEED:-0}
CKPT_EVERY=${CKPT_EVERY:-25}
BUF_CKPT_EVERY=${BUF_CKPT_EVERY:-500}   # heavy reservoir snapshot cadence (cheap core ckpt stays at CKPT_EVERY)
LOG_EVERY=${LOG_EVERY:-25}              # progress-line cadence
CKPT=${CKPT:-checkpoints/sd_cfr_hu_neural/hu_vrpdcfr.pkl}

# --- GPU / memory safety --------------------------------------------------------------------------
export PYTHONPATH=.
export POKERBOT_INFO_SUITS=1                 # suit-aware info-state (flushes); stamped into ckpt
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # do not grab all 12 GB VRAM (display shares the GPU)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export JAX_PLATFORMS=cuda                     # fail loudly if CUDA is missing rather than silently CPU
# keep CPU thread pools modest so WSL does not starve native Windows (WSL is already capped to 8 cpus)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$REPO"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

mkdir -p "$REPO/$(dirname "$CKPT")" "$REPO/runs/hu_neural"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$REPO/runs/hu_neural/hu_train_${STAMP}.log"

RESUME_ARG=()
if [ -f "$REPO/${CKPT%.pkl}_full.pkl" ]; then
  RESUME_ARG=(--resume "$CKPT")
  echo "[launcher] found existing checkpoint -> RESUMING from ${CKPT%.pkl}_full.pkl" | tee -a "$LOG"
fi

echo "[launcher] game=$GAME iters=$ITERS trav=$TRAVERSALS batch=$BATCH hidden=$HIDDEN \
steps(t/a/h)=$TRAIN_STEPS/$AVG_STEPS/$HIST_STEPS ckpt_every=$CKPT_EVERY -> $CKPT" | tee -a "$LOG"
echo "[launcher] log: $LOG" | tee -a "$LOG"

# nohup + setsid so it survives the launching shell / WSL terminal closing.
setsid nohup python scripts/train_neural_cfr.py \
  --game "$GAME" --iterations "$ITERS" --traversals "$TRAVERSALS" --batch-size "$BATCH" \
  --hidden-dim "$HIDDEN" --train-steps "$TRAIN_STEPS" --avg-steps "$AVG_STEPS" \
  --hist-steps "$HIST_STEPS" --lr "$LR" --seed "$SEED" \
  --checkpoint "$CKPT" --checkpoint-every "$CKPT_EVERY" \
  --buffer-checkpoint-every "$BUF_CKPT_EVERY" --log-every "$LOG_EVERY" --traversal-workers 0 \
  "${RESUME_ARG[@]}" >>"$LOG" 2>&1 &

PID=$!
echo "$PID" > "$REPO/runs/hu_neural/hu_train.pid"
echo "[launcher] started PID $PID. monitor: tail -f $LOG   stop: kill $PID" | tee -a "$LOG"
