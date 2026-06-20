#!/usr/bin/env bash
# One-shot verification that the EXACT overnight HU config compiles + steps + checkpoints through
# WSL->GPU, with peak host-RAM capture. ASCII only. Writes a SUMMARY line at the end.
set -uo pipefail
REPO=.
VENV="$HOME/jaxgpu-venv"
RUNDIR="$REPO/runs/hu_neural"
mkdir -p "$RUNDIR"
OUT="$RUNDIR/verify_$(date +%Y%m%d_%H%M%S).log"
PEAK="$RUNDIR/verify.peakmem"

export PYTHONPATH=.
export POKERBOT_INFO_SUITS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

cd "$REPO"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

# --- peak-RAM sampler (records max used MiB across the run) ---
echo "0" > "$PEAK"
( while true; do
    used=$(free -m | awk '/^Mem:/{print $3}')
    cur=$(cat "$PEAK" 2>/dev/null || echo 0)
    if [ "${used:-0}" -gt "${cur:-0}" ]; then echo "$used" > "$PEAK"; fi
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

echo "[verify] start $(date +%H:%M:%S)  MemTotal=$(awk '/MemTotal/{print int($2/1024)" MiB"}' /proc/meminfo)" | tee "$OUT"
START=$(date +%s)
python -u scripts/train_neural_cfr.py \
  --game nlhe-hu --iterations 3 --traversals 128 --batch-size 1024 \
  --hidden-dim 256 --train-steps 16 --avg-steps 128 --hist-steps 32 \
  --lr 1e-4 --seed 0 \
  --checkpoint checkpoints/sd_cfr_hu_neural/hu_vrpdcfr_verify.pkl --checkpoint-every 1 \
  --traversal-workers 0 2>&1 | tee -a "$OUT"
RC=${PIPESTATUS[0]}
END=$(date +%s)
PK=$(cat "$PEAK")

echo "" | tee -a "$OUT"
if [ "$RC" -eq 0 ]; then
  CKPT_OK=no
  [ -f "$REPO/checkpoints/sd_cfr_hu_neural/hu_vrpdcfr_verify_full.pkl" ] && CKPT_OK=yes
  echo "[verify] SUMMARY: OK rc=0 wall=$((END-START))s peak_host_ram=${PK}MiB checkpoint_written=$CKPT_OK" | tee -a "$OUT"
else
  echo "[verify] SUMMARY: FAIL rc=$RC wall=$((END-START))s peak_host_ram=${PK}MiB (see $OUT)" | tee -a "$OUT"
  tail -n 5 "$OUT"
fi
echo "$OUT" > "$RUNDIR/verify.curlog"
