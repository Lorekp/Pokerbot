#!/usr/bin/env bash
# Run N parallel HU-LBR runs (one seed each) to use CPU headroom, then pool into one tight estimate.
# Foreground-blocking (waits on all children) so the launching wsl.exe stays an init-child -> the WSL
# VM stays up for the whole job (do NOT self-detach). Each worker is pinned ~single-threaded so N
# workers don't oversubscribe the box (thread-scan oversubscription has hung jobs here before).
set -u
cd .
source ~/jaxgpu-venv/bin/activate
export PYTHONPATH=.
export POKERBOT_INFO_SUITS=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

N=${1:-10}
HANDS=${2:-2000}
OUT=runs/hu_neural
mkdir -p "$OUT"
rm -f "$OUT"/hu_lbr_par_s*.json "$OUT"/hu_lbr_par_s*.log
echo "[$(date -u +%FT%TZ)] launching $N x $HANDS-hand LBR runs (seeds 1..$N)..."
pids=()
for i in $(seq 1 "$N"); do
  python scripts/neural_cfr/hu_lbr_neural.py --hands "$HANDS" --seed "$i" \
     --output "$OUT/hu_lbr_par_s${i}.json" > "$OUT/hu_lbr_par_s${i}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[$(date -u +%FT%TZ)] all workers done (failures=$fail). aggregating..."
# Pool the N parallel runs plus the existing independent seed-0 4000-hand run if present.
python scripts/neural_cfr/hu_lbr_aggregate.py "$OUT"/hu_lbr_par_s*.json \
  $([ -f "$OUT/hu_lbr_4000.json" ] && echo "$OUT/hu_lbr_4000.json")
echo "[$(date -u +%FT%TZ)] DONE"
