#!/bin/bash
# cpu128_probe_onbox.sh — on-box driver for the metal SUBTRACTIVE-ATTRIBUTION + flamegraph profiling pass.
#
# REAL MEASUREMENT, not an A/B of code variants. The headline is the bigbox-attribution-protocol STEP-2
# SUBTRACTIVE harness: run the SAME production recursive path (EMD-K160, N=1024) across POKERBOT_AB_STUB in
# {none, emd_nomemo, regret_zero, showdown_const} and attribute each component by the CLEAN nodes/s DELTA.
# This is robust to BOTH (a) fat-LTO inlining collapsing the candidate leaves into one traverse frame, and
# (b) the FxHashMap<u64,_>::get symbol COLLISION (the frozen-EMD probe and the cum_regret get are the SAME
# monomorphized hashbrown symbol) — the two reasons a flamegraph alone is a non-answer here. WHY the stub
# sweep and not just a flamegraph: the per-rayon-thread EMD_BUCKET_MEMO is fully warm by steady state, so the
# real 152MB FxHashMap probe almost never fires in a steady perf window -> the frozen table reads ~0 DRAM
# weight REGARDLESS of whether it is the bottleneck. emd_nomemo defeats the memo (a real probe every postflop
# node) so its nodes/s drop vs PROD = the true per-node frozen-probe cost, and its DRAM flamegraph/PMU is the
# ONE capture that can actually attribute to the 152MB lookup.
#
# Shapes (one ab_stub-feature build; mode none = the baseline, so the ab_stub branch overhead CANCELS in
# every delta):
#   PROD          none           128t  CPU-flame + DRAM-flame(memoized steady) + full PMU  -> baseline nodes/s
#   EMDNOMEMO     emd_nomemo     128t  DRAM-flame(real probe fires) + stall/mlp/dram PMU   -> frozen-probe cost
#   REGRETZERO    regret_zero    128t  stall/mlp/dram PMU                                  -> cum_regret cost
#   SHOWDOWNCONST showdown_const 128t  stall/mlp/dram PMU                                  -> showdown cost
#   PROD64        none            64t  stall/mlp/dram PMU                                  -> SMT/contention x-check
#
# set -uxo pipefail (NOT -e): a failing build/perf stage must DEGRADE with an explicit marker, never abort
# before PROBE_DONE. The 4 BLOCKING FIXES stay wired: #1 disk-full df guard; #2 perf HARD-GATE -> PERF_BROKEN
# degrade; #3 PMU-masked per-event parse + NUMA_NODES topology gate; #4 cost-runaway PID-liveness + STEADY_TIMEOUT
# (+ the user_data WALL_GUARD backstop).
set -uxo pipefail

# ---------------------------------------------------------------------------- STEP 0: SETUP + helpers
mkdir -p /var/log/probe
cd /opt/pokerbot

upload_one() {
  local f="$1" key="$2"
  [ -e "$f" ] || { echo "upload_one: missing $f (skip $key)"; return 0; }
  sync
  aws s3 cp "$f" "$PROBE_PREFIX/$key" --region "$REGION" || echo "WARN upload failed: $f -> $key"
}
marker() {
  local name="$1"; shift
  printf '%s\n' "${*:-$(date -u +%FT%TZ)}" > "/var/log/probe/$name"
  upload_one "/var/log/probe/$name" "$name"
}

echo "PROBE_PREFIX=$PROBE_PREFIX  TABLE_ID=$TABLE_ID  SHAPES=$SHAPES"

# Download the PRE-UPLOADED frozen blob (CACHE_HIT; the box NEVER recomputes the ~16.5M spots).
aws s3 cp "s3://$BUCKET/$RUN_PREFIX/frozen/$TABLE_ID.emdfrzn" /opt/pokerbot/frozen.emdfrzn --region "$REGION" \
  || { echo "FROZEN_DOWNLOAD_FAIL"; marker FROZEN_DOWNLOAD_FAIL "no s3://$BUCKET/$RUN_PREFIX/frozen/$TABLE_ID.emdfrzn"; exit 1; }
cp /opt/pokerbot/data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn.sha256 \
   /opt/pokerbot/frozen.emdfrzn.sha256 2>/dev/null || echo "WARN missing sha256 sidecar in tarball"

# ---------------------------------------------------------------------------- APT (BLOCKING-FIX #2 pkgs)
# linux-tools-common = the perf WRAPPER; linux-tools-aws = the AWS-kernel-flavour perf; linux-tools-$(uname
# -r) = the EXACT booted-kernel build. NOT linux-tools-generic (wrong on an aws kernel). + sysstat + numactl
# + git/perl (FlameGraph) + binutils (readelf, for the inline-DWARF assert). gcc is in build-essential.
apt-get install -y -qq linux-tools-common linux-tools-aws "linux-tools-$(uname -r)" \
  sysstat numactl git perl binutils || echo "WARN apt perf/sysstat install had a failure"
perf --version || echo "WARN perf wrapper present but versioned binary may mismatch kernel"

# ---------------------------------------------------------------------------- SYSCTL (perf permissions)
# nmi_watchdog=0 FREES the GP counter the watchdog pins (the most likely cause of the prior "perf record
# cycles -> 0 samples" AND counter pressure). perf_event_paranoid=-1 + kptr_restrict=0 -> full sampling.
sysctl -w kernel.perf_event_paranoid=-1 || true
sysctl -w kernel.kptr_restrict=0 || true
sysctl -w kernel.nmi_watchdog=0 || true
echo 0 > /proc/sys/kernel/nmi_watchdog 2>/dev/null || true
printf 'kernel.perf_event_paranoid=-1\nkernel.kptr_restrict=0\nkernel.nmi_watchdog=0\n' > /etc/sysctl.d/99-perf.conf
sysctl --system || true
sysctl -w kernel.perf_event_max_sample_rate=10000 || true

# ---------------------------------------------------------------------------- FLAMEGRAPH TOOLCHAINS (clone first)
# Clone the perl FlameGraph FIRST (the sufficient path) so a slow `cargo install inferno` can never eat the
# steady window; inferno is a bounded best-effort bonus.
export PATH="$HOME/.cargo/bin:$PATH"
git clone --depth 1 https://github.com/brendangregg/FlameGraph /opt/FlameGraph || true
timeout 300 cargo install --locked inferno || echo "WARN inferno install failed/timed out; using FlameGraph perl"

# ---------------------------------------------------------------------------- BUILD (debug-symbol'd v3, ab_stub)
# KEEP -C target-cpu=x86-64-v3 (the profiled binary IS the shipping binary). debuginfo=2 = FULL DWARF incl
# DW_TAG_inlined_subroutine (so perf --inline can expand the inlined leaves — line-tables-only would NOT, so
# we deliberately do NOT set CARGO_PROFILE_RELEASE_DEBUG). force-frame-pointers -> cheap --call-graph fp.
# strip=none -> keep symtab. NOTE: legacy mangling is the stable-rustc DEFAULT (perf demangles it fine, as the
# prior native-LBR runs confirmed); explicitly naming `-C symbol-mangling-version=legacy` is REJECTED on stable
# ("requires -Z unstable-options") and was the BUILD_FAIL on probe_1782727507 — so we rely on the default and
# verify with the `nm -C` demangle assert below. --features python,ab_stub: ONE binary serves all shapes;
# POKERBOT_AB_STUB (env, per shape) selects the subtractive mode. Cargo.toml NOT edited (firewall-clean).
export PYO3_PYTHON="$(python3 -c 'import sys;print(sys.executable)')"
RF='-C target-cpu=x86-64-v3 -C force-frame-pointers=yes -C debuginfo=2 -C strip=none'
RUSTFLAGS="$RF" cargo build --release --features python,ab_stub --lib --manifest-path tools/battery_native/Cargo.toml || { echo "BUILD_FAIL"; marker BUILD_FAIL "cargo build failed"; exit 1; }
SO="$(ls tools/battery_native/target/release/libbattery_native.so 2>/dev/null || ls tools/battery_native/target/release/battery_native.so 2>/dev/null)"
cp "$SO" /opt/pokerbot/battery_native_prod.so
cp /opt/pokerbot/battery_native_prod.so /opt/pokerbot/battery_native.so
ls -la /opt/pokerbot/battery_native_prod.so
# BUILD_OK verify: debug_info present + PyInit resolves + FULL inline DWARF (B1) + demangle works (S7).
INLREC="$(readelf --debug-dump=info /opt/pokerbot/battery_native_prod.so 2>/dev/null | grep -c DW_TAG_inlined_subroutine || echo 0)"
nm -C /opt/pokerbot/battery_native_prod.so 2>/dev/null | grep -q 'battery_native::trainer' && DEMANGLE=ok || DEMANGLE=WARN_NO_DEMANGLE
if file /opt/pokerbot/battery_native_prod.so | grep -q 'debug_info' \
   && nm /opt/pokerbot/battery_native_prod.so 2>/dev/null | grep -q 'PyInit_battery_native'; then
  marker BUILD_OK "ab_stub cdylib; debug_info + PyInit resolved; DW_TAG_inlined_subroutine=$INLREC demangle=$DEMANGLE (inline-attribution ready)"
else
  echo "BUILD_VERIFY_WARN: missing debug_info / PyInit symbol"
  marker BUILD_OK "WARN: verify weak — file/nm did not match; inlined_subroutine=$INLREC demangle=$DEMANGLE"
fi
[ "${INLREC:-0}" -gt 0 ] || marker BUILD_INLINE_WARN "DW_TAG_inlined_subroutine=$INLREC — inline attribution may be degraded (B1)"

# ---------------------------------------------------------------------------- PERF HARD-GATE (FIX #2)
# THREE decoupled capabilities (COUNTING / CPU-time SAMPLING / PEBS DRAM SAMPLING are independent on metal):
#   PERF_STAT_OK  (G1) gates perf-stat hardware COUNTING (IPC, stalls, MLP, DRAM) = the memory-bound cert.
#   PERF_RECORD_OK(G2) gates the CPU-TIME flamegraph (try hardware cycles, else software cpu-clock).
#   PERF_PEBS_OK  (G3) gates the DRAM-MISS flamegraph — self-tested on a 512MB pointer-CHASE burner (the `yes`
#                      G2 burner is L1-resident -> ~0 DRAM misses -> would false-NEGATIVE a working PEBS).
PERF_AVAILABLE=0; PERF_STAT_OK=0; PERF_RECORD_OK=0; PERF_PEBS_OK=0; PERF_RECORD_EVENT=""
PEBS_REC_ARGS=()      # the working DRAM PEBS event(s) as perf -e args (BOTH local+remote when both pass — S6)
perf list 2>/dev/null > /var/log/probe/perf_event_inventory.txt || true
upload_one /var/log/probe/perf_event_inventory.txt perf_event_inventory.txt
# G1: hardware PMU COUNTING. cycles must NOT report <not supported>.
if perf stat -e cycles true 2>&1 | grep -q '<not supported>'; then
  echo "PERF_GATE G1 FAIL: cycles <not supported> (no PMU counting)"
else
  PERF_STAT_OK=1
fi
# G2: CPU-TIME SAMPLING — profile a BUSY 2s burner. Try hardware cycles; else software cpu-clock.
for EV in cycles cpu-clock; do
  rm -f /tmp/perfselftest.data
  perf record -e "$EV" -F 997 -g -o /tmp/perfselftest.data -- bash -c 'timeout 2 yes > /dev/null 2>&1' 2>/dev/null || true
  if perf script -i /tmp/perfselftest.data 2>/dev/null | head -5 | grep -q . ; then
    PERF_RECORD_OK=1; PERF_RECORD_EVENT="$EV"; break
  fi
done
[ "$PERF_RECORD_OK" = 1 ] || echo "PERF_GATE G2 FAIL: neither cycles nor cpu-clock produced samples"
# G3: PEBS DRAM-MISS SAMPLING — pointer-chase a 512MB (>L3) array so every load misses to DRAM, test BOTH
# local_dram and remote_dram (a 2-socket interleaved 152MB lookup misses to BOTH — S6), record every one
# that delivers samples.
cat > /tmp/memburn.c <<'CEOF'
#include <stdlib.h>
#include <stdint.h>
int main(void){
  size_t n=(size_t)1<<26;                 /* 64M * 8B = 512MB > L3 */
  uint64_t *a=(uint64_t*)malloc(n*sizeof(uint64_t));
  if(!a) return 1;
  for(size_t i=0;i<n;i++) a[i]=(i*2654435761ULL)>>3;
  uint64_t idx=0,s=0;
  for(long r=0;r<64;r++) for(size_t i=0;i<n;i++){ idx=(a[idx]^(idx*2654435761ULL))&(n-1); s+=idx; }
  return (int)s;
}
CEOF
gcc -O2 -o /tmp/memburn /tmp/memburn.c 2>/dev/null || echo "WARN gcc memburn build failed"
perf_event_ok() { perf stat -e "$1" sleep 0.2 2>&1 | grep -Eq '<not supported>|<not counted>' && return 1; return 0; }
for DEV in mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram; do
  perf_event_ok "$DEV" || continue
  rm -f /tmp/pebstest.data
  if [ -x /tmp/memburn ]; then
    perf record -e "${DEV}:pp" -c 5003 -g -o /tmp/pebstest.data -- bash -c 'timeout 3 /tmp/memburn' 2>/dev/null || true
    if perf script -i /tmp/pebstest.data 2>/dev/null | head -5 | grep -q . ; then
      PERF_PEBS_OK=1; PEBS_REC_ARGS+=(-e "${DEV}:pp")
    fi
  fi
done
[ "$PERF_PEBS_OK" = 1 ] || echo "PERF_GATE G3: PEBS DRAM sampling unavailable -> DRAM attribution via stat totals only"
if [ "$PERF_STAT_OK" = 1 ] || [ "$PERF_RECORD_OK" = 1 ]; then PERF_AVAILABLE=1; fi
if [ "$PERF_STAT_OK" = 1 ] && [ "$PERF_RECORD_OK" = 1 ]; then
  marker PERF_OK "PERF_STAT_OK=1 PERF_RECORD_OK=1 PERF_PEBS_OK=$PERF_PEBS_OK record_event=$PERF_RECORD_EVENT pebs_args='${PEBS_REC_ARGS[*]:-none}'"
elif [ "$PERF_STAT_OK" = 1 ] || [ "$PERF_RECORD_OK" = 1 ]; then
  marker PERF_PARTIAL "PERF_STAT_OK=$PERF_STAT_OK PERF_RECORD_OK=$PERF_RECORD_OK PERF_PEBS_OK=$PERF_PEBS_OK record_event=${PERF_RECORD_EVENT:-none}"
else
  marker PERF_BROKEN "PERF_STAT_OK=0 PERF_RECORD_OK=0 -> degrade to numastat+mpstat+train_profile_json only"
fi

# ---------------------------------------------------------------------------- TOPOLOGY (gates NUMA hypo)
{
  echo "=== nproc ==="; nproc
  echo "=== lscpu ==="; lscpu
  echo "=== RUSTFLAGS ==="; echo "$RF"
  echo "=== build features ==="; echo "python,ab_stub (mode via POKERBOT_AB_STUB env per shape)"
  echo "=== DW_TAG_inlined_subroutine ==="; echo "$INLREC (demangle=$DEMANGLE)"
  echo "=== git rev ==="; (cd /opt/pokerbot && git rev-parse HEAD 2>/dev/null || echo 'no git in tarball')
  echo "=== file .so ==="; file /opt/pokerbot/battery_native.so
  echo "=== nm train ==="; nm -C /opt/pokerbot/battery_native.so 2>/dev/null | grep -i train | head
  echo "=== nmi_watchdog ==="; cat /proc/sys/kernel/nmi_watchdog
  echo "=== lstopo ==="; (lstopo --of txt 2>/dev/null || echo 'lstopo not installed')
} > /var/log/probe/probe_env.txt 2>&1
upload_one /var/log/probe/probe_env.txt probe_env.txt

numactl --hardware > /var/log/probe/numactl_topology.txt 2>&1 || echo "WARN numactl --hardware failed"
upload_one /var/log/probe/numactl_topology.txt numactl_topology.txt
NUMA_NODES="$(grep -c '^node [0-9]* cpus:' /var/log/probe/numactl_topology.txt 2>/dev/null || echo 0)"
if [ "${NUMA_NODES:-0}" -lt 2 ]; then
  echo "NUMA_STRUCTURALLY_IMPOSSIBLE NUMA_NODES=$NUMA_NODES"
  marker NUMA_STRUCTURALLY_IMPOSSIBLE "NUMA_NODES=$NUMA_NODES (<2) -> NUMA hypothesis REFUTE-by-topology"
fi
echo "NUMA_NODES=$NUMA_NODES PERF_AVAILABLE=$PERF_AVAILABLE PERF_PEBS_OK=$PERF_PEBS_OK"

perf_event_supported() {
  local ev="$1"
  perf stat -e "$ev" sleep 0.2 2>&1 | grep -Eq '<not supported>|<not counted>' && return 1
  return 0
}
write_event_availability() {
  local out="$1"
  : > "$out"
  local evs="cycles instructions cycle_activity.stalls_mem_any l1d_pend_miss.pending l1d_pend_miss.pending_cycles offcore_requests_outstanding.all_data_rd offcore_requests_outstanding.cycles_with_data_rd mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram"
  for ev in $evs; do
    if [ "$PERF_AVAILABLE" = 1 ] && perf_event_supported "$ev"; then echo "$ev AVAILABLE" >> "$out"; else echo "$ev NOT_SUPPORTED" >> "$out"; fi
  done
}

# pstat NAME EVENTS SECS PID KEYDIR : ONE small (<=3 GP events + fixed cycles) perf-stat group, its OWN perf
# invocation. Each group carries cycles (every ratio gets a denominator). After it runs, classify
# COUNTED/UNCOUNTED (S3) so a <not counted> is never read as a true 0.
pstat() {
  local nm="$1" evs="$2" secs="$3" pid="$4" keydir="$5"
  local out="/var/log/probe/${keydir}_perfstat_${nm}.txt"
  kill -0 "$pid" 2>/dev/null || { echo "pstat $nm: PID dead, skip"; return 0; }
  # CRITICAL: `timeout` defaults to SIGTERM, which `perf stat` does NOT handle -> it is killed before it
  # flushes the -o file -> EMPTY output (the probe_1782728898 bug). `-s INT` sends SIGINT, which perf's
  # handler catches to print+flush the counters (the standard `perf stat -p PID -o file` + Ctrl-C pattern).
  timeout -s INT "$secs" perf stat -p "$pid" -e "$evs" -o "$out" 2>&1 || true
  if [ ! -s "$out" ] || grep -Eq '<not counted>|<not supported>' "$out" 2>/dev/null; then
    echo "$nm UNCOUNTED" >> "/var/log/probe/${keydir}_perfstat_status.txt"   # EMPTY counts as UNCOUNTED, not a true 0
  else
    echo "$nm COUNTED" >> "/var/log/probe/${keydir}_perfstat_status.txt"
  fi
  upload_one "$out" "${keydir}/${keydir}_perfstat_${nm}.txt"
}
run_pmu_group() {  # group_name pid keydir
  case "$1" in
    ipc)      pstat ipc      "cycles,instructions"                                                            16 "$2" "$3" ;;
    stall)    pstat stall    "cycles,cycle_activity.stalls_mem_any"                                           16 "$2" "$3" ;;
    mlp)      pstat mlp      "cycles,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"                      16 "$2" "$3" ;;
    mlpoff_n) pstat mlpoff_n "cycles,offcore_requests_outstanding.all_data_rd"                               16 "$2" "$3" ;;  # S1: split (same-event/distinct-cmask won't co-schedule)
    mlpoff_c) pstat mlpoff_c "cycles,offcore_requests_outstanding.cycles_with_data_rd"                       16 "$2" "$3" ;;
    dram)     pstat dram     "cycles,mem_load_l3_miss_retired.local_dram,mem_load_l3_miss_retired.remote_dram" 16 "$2" "$3" ;;
  esac
}

# ---------------------------------------------------------------------------- STEP 1: THE DRIVER
if [ -f /opt/pokerbot/scripts/drive_cpu128_probe.py ]; then
  cp /opt/pokerbot/scripts/drive_cpu128_probe.py /opt/pokerbot/drive_cpu128_probe.py
else
  echo "WARN drive_cpu128_probe.py not in tarball — heredoc fallback"
  cat > /opt/pokerbot/drive_cpu128_probe.py <<'PYEOF'
import sys
print("FATAL: drive_cpu128_probe.py was not shipped in the tarball", file=sys.stderr)
sys.exit(2)
PYEOF
fi

# ---------------------------------------------------------------------------- SIGTERM/spot trap (armed early)
# On a reclaim: SIGTERM the driver (its finally flushes), sweep ALL artifacts to S3, write a terminal-state
# marker (S10) so a partial run is never mistaken for a hang, THEN allow shutdown.
cat > /tmp/spot_watch.sh <<SWEOF
TOK=\$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
while true; do
  C=\$(curl -s -o /dev/null -w "%{http_code}" -H "X-aws-ec2-metadata-token: \$TOK" http://169.254.169.254/latest/meta-data/spot/instance-action)
  if [ "\$C" = "200" ]; then
    echo "SPOT_RECLAIM_NOTICE"
    pkill -TERM -f drive_cpu128_probe.py
    sleep 8
    sync
    aws s3 cp --recursive /var/log/probe/ "$PROBE_PREFIX/" --region "$REGION" || true
    printf 'RECLAIM_SWEEP_DONE %s\n' "\$(date -u +%FT%TZ)" | aws s3 cp - "$PROBE_PREFIX/RECLAIM_SWEEP_DONE" --region "$REGION" || true
    break
  fi
  sleep 5
done
SWEOF
nohup bash /tmp/spot_watch.sh >/var/log/spot_watch.log 2>&1 &

# ---------------------------------------------------------------------------- STEP 2: PER-SHAPE LOOP
IFS=',' read -r -a SHAPE_ARR <<< "$SHAPES"
PROBE_SUMMARY="/var/log/probe/_probe_done_summary.txt"
: > "$PROBE_SUMMARY"

for SHAPE in "${SHAPE_ARR[@]}"; do
  echo "================= SHAPE $SHAPE ================="
  N=1024
  unset RAYON_NUM_THREADS || true
  unset POKERBOT_AB_STUB || true
  export POKERBOT_BATCH_DEALS=0            # recursive PROD path for ALL shapes (no batched-traversal)
  DO_CPUFLAME=0; DO_DRAMFLAME=0
  PMU_GROUPS="stall mlp dram"
  PERF_WIN_SECS=110
  # PERF_WIN_SECS MUST exceed the sum of the shell's back-to-back captures so the driver PID stays live
  # through the LAST capture. Budgets: CPU-rec 30 + DRAM-rec 35 + each pstat 16 + portable 14.
  case "$SHAPE" in
    # PMU-CERT re-run (timeout -s INT fix recovers stall/MLP/DRAM that the prior run lost). Full PMU groups
    # on PROD64 (the headline 64t regime the operator chose) AND PROD (128t) so the 64-vs-128 stall/MLP delta
    # answers contention-vs-per-thread-latency-wall. Flamegraphs attempted (degrade if this box, like the last,
    # delivers no sampling). Budgets: CPU-rec 30 + DRAM-rec 35 + 6*pstat 96 + portable 14 = 175 -> 230.
    PROD64)        export RAYON_NUM_THREADS=64; DO_CPUFLAME=1; DO_DRAMFLAME=1; PMU_GROUPS="ipc stall mlp mlpoff_n mlpoff_c dram"; PERF_WIN_SECS=230 ;;  # 64t cert (HEADLINE)
    PROD)          DO_CPUFLAME=1; DO_DRAMFLAME=1; PMU_GROUPS="ipc stall mlp mlpoff_n mlpoff_c dram"; PERF_WIN_SECS=230 ;;                                # 128t comparison
    EMDNOMEMO64)   export POKERBOT_AB_STUB=emd_nomemo; export RAYON_NUM_THREADS=64; DO_DRAMFLAME=1; PMU_GROUPS="ipc stall mlp dram"; PERF_WIN_SECS=200 ;;  # 64t frozen-probe DRAM signature
    EMDNOMEMO)     export POKERBOT_AB_STUB=emd_nomemo;     DO_DRAMFLAME=1; PMU_GROUPS="stall mlp dram"; PERF_WIN_SECS=160 ;;
    REGRETZERO)    export POKERBOT_AB_STUB=regret_zero;    PMU_GROUPS="stall mlp dram"; PERF_WIN_SECS=110 ;;
    SHOWDOWNCONST) export POKERBOT_AB_STUB=showdown_const; PMU_GROUPS="stall mlp dram"; PERF_WIN_SECS=110 ;;
    *) echo "UNKNOWN SHAPE $SHAPE — skip"; continue ;;
  esac
  CAP="$WARMUP_CAP"

  write_event_availability "/var/log/probe/${SHAPE}_perf_event_availability.txt"
  upload_one "/var/log/probe/${SHAPE}_perf_event_availability.txt" "$SHAPE/${SHAPE}_perf_event_availability.txt"

  rm -f "/var/log/probe/$SHAPE.STEADY_BEGIN" "/var/log/probe/$SHAPE.STEADY_BEGIN_PERF" "/var/log/probe/$SHAPE.WINDOW_DONE"
  cp /opt/pokerbot/battery_native_prod.so /opt/pokerbot/battery_native.so
  echo "$SHAPE ab_stub=${POKERBOT_AB_STUB:-none} rayon=${RAYON_NUM_THREADS:-default} N=$N cpuflame=$DO_CPUFLAME dramflame=$DO_DRAMFLAME pmu='$PMU_GROUPS' win=$PERF_WIN_SECS"
  marker "${SHAPE}_CFG" "ab_stub=${POKERBOT_AB_STUB:-none} rayon=${RAYON_NUM_THREADS:-default} N=$N cpuflame=$DO_CPUFLAME dramflame=$DO_DRAMFLAME pmu='$PMU_GROUPS' perf_win_secs=$PERF_WIN_SECS"

  python3 /opt/pokerbot/drive_cpu128_probe.py \
      --shape "$SHAPE" --ndeals "$N" --window "$WINDOW_ITERS" --profile-every "$PROFILE_EVERY" \
      --warmup-cap "$CAP" --perf-window-secs "$PERF_WIN_SECS" > "/var/log/probe/${SHAPE}_driver.log" 2>&1 &
  PID=$!
  echo "$SHAPE driver PID=$PID N=$N CAP=$CAP"

  # ----------------------------------------- STEP 3: WAIT FOR STEADY (WINDOW A) + PID liveness (FIX #4)
  WAITED=0; STEADY_OK=0
  while true; do
    if [ -f "/var/log/probe/$SHAPE.STEADY_BEGIN" ]; then STEADY_OK=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "DRIVER_DIED before STEADY_BEGIN (shape $SHAPE)"
      upload_one "/var/log/probe/${SHAPE}_driver.log" "$SHAPE/${SHAPE}_driver.log"
      marker "${SHAPE}_FAILED" "DRIVER_DIED before STEADY_BEGIN"; break
    fi
    if [ "$WAITED" -gt 900 ]; then
      echo "STEADY_TIMEOUT (shape $SHAPE, ${WAITED}s)"
      kill -TERM "$PID" 2>/dev/null || true; sleep 5
      upload_one "/var/log/probe/${SHAPE}_driver.log" "$SHAPE/${SHAPE}_driver.log"
      marker "${SHAPE}_FAILED" "STEADY_TIMEOUT after ${WAITED}s"; break
    fi
    sleep 2; WAITED=$((WAITED + 2))
  done
  [ "$STEADY_OK" = 1 ] || { echo "skip captures for $SHAPE"; continue; }

  # WINDOW A: CLEAN throughput (NO perf) — mpstat + numastat. The CLEAN nodes/s here is what the subtractive
  # attribution uses (perf attach depresses throughput; deltas must be on the un-attached number — S2).
  echo "$SHAPE WINDOW A (clean) — mpstat + numastat"
  numastat -p "$PID" > "/var/log/probe/${SHAPE}_numastat_start.txt" 2>&1 || true
  mpstat -P ALL 1 30 > "/var/log/probe/${SHAPE}_mpstat.txt" 2>&1 &
  MPID=$!
  pidstat -t -p "$PID" 1 30 > "/var/log/probe/${SHAPE}_pidstat.txt" 2>&1 &
  PSID=$!
  wait "$MPID" 2>/dev/null || true
  wait "$PSID" 2>/dev/null || true
  numastat -p "$PID" > "/var/log/probe/${SHAPE}_numastat_end.txt" 2>&1 || true
  for f in mpstat pidstat numastat_start numastat_end; do upload_one "/var/log/probe/${SHAPE}_${f}.txt" "$SHAPE/${SHAPE}_${f}.txt"; done

  # ----------------------------------------- STEP 3b: WAIT FOR PERF WINDOW (WINDOW B)
  WAITED=0; PERFWIN_OK=0
  while true; do
    if [ -f "/var/log/probe/$SHAPE.STEADY_BEGIN_PERF" ]; then PERFWIN_OK=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then echo "driver exited before WINDOW B (shape $SHAPE)"; break; fi
    if [ "$WAITED" -gt 300 ]; then echo "WINDOW B wait timeout (shape $SHAPE)"; break; fi
    sleep 2; WAITED=$((WAITED + 2))
  done

  # ----------------------------------------- STEP 4: CAPTURE (WINDOW B; perf only if available)
  CAP_T0=$(date +%s)
  if [ "$PERFWIN_OK" = 1 ] && [ "$PERF_AVAILABLE" = 1 ]; then
    AVAIL="$(df -BG / | awk 'NR==2{print $4}' | tr -d G)"
    echo "$SHAPE disk free: ${AVAIL}G"
    if [ "${AVAIL:-0}" -lt 20 ]; then
      echo "DISK_FULL (${AVAIL}G < 20G) — skipping perf record for $SHAPE"
      marker "${SHAPE}_DISK_FULL" "df / = ${AVAIL}G < 20G"
    else
      # (A) CPU-TIME flamegraph record (where wall goes) — fp unwind on the live steady PID.
      if [ "$DO_CPUFLAME" = 1 ] && [ "$PERF_RECORD_OK" = 1 ] && kill -0 "$PID" 2>/dev/null; then
        timeout 30 perf record -e "$PERF_RECORD_EVENT" -p "$PID" -F 997 --call-graph fp -m 4096 --aio \
          -o "/var/log/probe/${SHAPE}_perf_fp.data" 2>&1 || true
        kill -0 "$PID" 2>/dev/null && LE=live || LE=ended_during
        marker "${SHAPE}_PERF_RECORD_LIVE" "end=$LE event=$PERF_RECORD_EVENT"
        upload_one "/var/log/probe/${SHAPE}_perf_fp.data" "$SHAPE/${SHAPE}_perf_fp.data"
      fi
      # (B) DRAM-MISS flamegraph record (the attribution) — BOTH local+remote PEBS events (S6), fp call-graph.
      if [ "$DO_DRAMFLAME" = 1 ] && [ "$PERF_PEBS_OK" = 1 ] && kill -0 "$PID" 2>/dev/null; then
        timeout 35 perf record "${PEBS_REC_ARGS[@]}" -p "$PID" -c 5003 --call-graph fp -m 4096 --aio \
          -o "/var/log/probe/${SHAPE}_perf_dram.data" 2>&1 || true
        kill -0 "$PID" 2>/dev/null && LE=live || LE=ended_during
        marker "${SHAPE}_PERF_DRAM_LIVE" "end=$LE events='${PEBS_REC_ARGS[*]}'"
        upload_one "/var/log/probe/${SHAPE}_perf_dram.data" "$SHAPE/${SHAPE}_perf_dram.data"
      elif [ "$DO_DRAMFLAME" = 1 ]; then
        marker "${SHAPE}_PERF_DRAM_LIVE" "end=no_pebs (PEBS DRAM sampling unavailable; attribution via stat totals)"
      fi
      # (C) PMU STAT — the per-shape small groups, each its own invocation carrying cycles.
      if [ "$PERF_STAT_OK" = 1 ]; then
        : > "/var/log/probe/${SHAPE}_perfstat_status.txt"
        for g in $PMU_GROUPS; do run_pmu_group "$g" "$PID" "$SHAPE"; done
        timeout -s INT 14 perf stat -p "$PID" --per-socket -I 1000 \
          -e task-clock,cycles,instructions,cache-references,cache-misses,dTLB-loads,dTLB-load-misses,context-switches,cpu-migrations \
          -o "/var/log/probe/${SHAPE}_perfstat_portable.txt" 2>&1 || true
        upload_one "/var/log/probe/${SHAPE}_perfstat_portable.txt" "$SHAPE/${SHAPE}_perfstat_portable.txt"
        upload_one "/var/log/probe/${SHAPE}_perfstat_status.txt"   "$SHAPE/${SHAPE}_perfstat_status.txt"
      else
        echo "$SHAPE: PERF_STAT_OK=0 — skipping perf stat (no PMU counting)"
        marker "${SHAPE}_PERFSTAT_SKIPPED" "PERF_STAT_OK=0 (cycles not countable)"
      fi
    fi
  else
    echo "$SHAPE: perf capture SKIPPED (PERF_AVAILABLE=$PERF_AVAILABLE PERFWIN_OK=$PERFWIN_OK) — DEGRADED"
    marker "${SHAPE}_PERF_DEGRADED" "PERF_AVAILABLE=$PERF_AVAILABLE PERFWIN_OK=$PERFWIN_OK"
  fi
  # Capture-budget self-check (S4): warn if the back-to-back captures came within 10s of the driver's window.
  CAP_ELAPSED=$(( $(date +%s) - CAP_T0 ))
  echo "$SHAPE capture elapsed ${CAP_ELAPSED}s vs PERF_WIN_SECS=$PERF_WIN_SECS"
  [ "$CAP_ELAPSED" -gt $((PERF_WIN_SECS - 10)) ] && marker "${SHAPE}_BUDGET_TIGHT" "captures ${CAP_ELAPSED}s vs win ${PERF_WIN_SECS}s — last capture may have raced teardown"

  # ----------------------------------------- STEP 5: FLAMEGRAPHS + TOP + ANNOTATE (all --inline — B1)
  make_flame() {  # data title keysuffix
    local data="$1" title="$2" key="$3"
    [ -s "$data" ] || { echo "no $data — skip flame $key"; return 0; }
    perf script -i "$data" --inline > "/var/log/probe/${SHAPE}_${key}.script" 2>/dev/null || \
      perf script -i "$data" > "/var/log/probe/${SHAPE}_${key}.script" 2>/dev/null || true
    if command -v inferno-collapse-perf >/dev/null 2>&1; then
      inferno-collapse-perf "/var/log/probe/${SHAPE}_${key}.script" > "/var/log/probe/${SHAPE}_${key}.folded" 2>/dev/null || true
      inferno-flamegraph --title "$title" "/var/log/probe/${SHAPE}_${key}.folded" > "/var/log/probe/${SHAPE}_${key}.svg" 2>/dev/null || true
    elif [ -f /opt/FlameGraph/stackcollapse-perf.pl ]; then
      perl /opt/FlameGraph/stackcollapse-perf.pl "/var/log/probe/${SHAPE}_${key}.script" > "/var/log/probe/${SHAPE}_${key}.folded" 2>/dev/null || true
      perl /opt/FlameGraph/flamegraph.pl --title "$title" "/var/log/probe/${SHAPE}_${key}.folded" > "/var/log/probe/${SHAPE}_${key}.svg" 2>/dev/null || true
    fi
    upload_one "/var/log/probe/${SHAPE}_${key}.folded" "$SHAPE/${SHAPE}_${key}.folded"
    upload_one "/var/log/probe/${SHAPE}_${key}.svg"    "$SHAPE/${SHAPE}_${key}.svg"
  }
  # CPU-time flamegraph + top symbols (--inline).
  CPUDATA="/var/log/probe/${SHAPE}_perf_fp.data"
  if [ -s "$CPUDATA" ]; then
    make_flame "$CPUDATA" "battery_native CPU-time $SHAPE" "cpuflame"
    perf report -i "$CPUDATA" --stdio --inline -g none --sort overhead,symbol 2>/dev/null | head -60 > "/var/log/probe/${SHAPE}_cpu_top.txt" || true
    upload_one "/var/log/probe/${SHAPE}_cpu_top.txt" "$SHAPE/${SHAPE}_cpu_top.txt"
  fi
  # DRAM-miss flamegraph + top symbols + annotate (all --inline; HOT filtered to our DSO — S5).
  DRAMDATA="/var/log/probe/${SHAPE}_perf_dram.data"
  if [ -s "$DRAMDATA" ]; then
    make_flame "$DRAMDATA" "battery_native DRAM-miss $SHAPE" "dramflame"
    perf report -i "$DRAMDATA" --stdio --inline -g none --sort overhead,dso,symbol 2>/dev/null | head -80 > "/var/log/probe/${SHAPE}_dram_top.txt" || true
    upload_one "/var/log/probe/${SHAPE}_dram_top.txt" "$SHAPE/${SHAPE}_dram_top.txt"
    # hottest DRAM-miss symbol WITHIN our .so (reject [unknown]/0x/empty), top-5 reported (S5).
    HOT="$(grep -E 'battery_native' "/var/log/probe/${SHAPE}_dram_top.txt" 2>/dev/null | grep -vE '^#' \
           | awk 'NF>=4{for(i=1;i<=NF;i++) if($i=="[.]"){sym="";for(j=i+1;j<=NF;j++)sym=sym (j>i+1?" ":"") $j; print sym; exit}}' \
           | grep -vE '^\s*$|^0x|\[unknown\]' | head -1)"
    echo "HOTTEST DRAM-miss symbol (our DSO): $HOT"
    marker "${SHAPE}_DRAM_HOT_SYMBOL" "${HOT:-NONE_IN_DSO}"
    # auto-annotate the hottest symbols (no fragile --symbol match; --inline for inlined leaf source lines).
    perf annotate -i "$DRAMDATA" --stdio --inline 2>/dev/null | head -400 > "/var/log/probe/${SHAPE}_dram_annotate.txt" || true
    upload_one "/var/log/probe/${SHAPE}_dram_annotate.txt" "$SHAPE/${SHAPE}_dram_annotate.txt"
  fi

  # ----------------------------------------- STEP 6/7: let the driver finish, upload artifacts.
  wait "$PID" 2>/dev/null || true
  for jn in profileA profileB summary driver; do
    ext=jsonl; [ "$jn" = summary ] && ext=json; [ "$jn" = driver ] && ext=log
    upload_one "/var/log/probe/${SHAPE}_${jn}.${ext}" "$SHAPE/${SHAPE}_${jn}.${ext}"
  done
  ONE="$(python3 - "$SHAPE" <<'PYJ'
import json,sys,os
shape=sys.argv[1]
try:
    d=json.load(open(f"/var/log/probe/{shape}_summary.json"))
    print(f"{shape}: nodes/s_clean={d.get('nodes_per_s_clean','?')} nodes/s_perf={d.get('nodes_per_s','?')} "
          f"deals/s_clean={d.get('deals_per_s_clean','?')} cpu%_mean={d.get('cpu_pct_mean','?')} "
          f"rayon={d.get('rayon_threads_realized','?')} "
          f"dramflame={'Y' if os.path.exists(f'/var/log/probe/{shape}_dramflame.svg') else 'N'}")
except Exception as e:
    print(f"{shape}: summary unavailable ({e})")
PYJ
)"
  echo "$ONE" >> "$PROBE_SUMMARY"
  echo "$ONE"
done

# ---------------------------------------------------------------------------- STEP 7 (final): PROBE_DONE
upload_one /var/log/cloud-init-output.log logs/cloud-init-output.log
upload_one /var/log/probe/perf_event_inventory.txt perf_event_inventory.txt
upload_one /var/log/probe/numactl_topology.txt numactl_topology.txt
upload_one /var/log/probe/probe_env.txt probe_env.txt

{
  echo "PROBE_DONE $(date -u +%FT%TZ)"
  echo "PERF_AVAILABLE=$PERF_AVAILABLE PERF_PEBS_OK=$PERF_PEBS_OK NUMA_NODES=$NUMA_NODES inlined_subroutine=$INLREC demangle=$DEMANGLE"
  echo "--- per-shape (nodes/s_clean is the attribution number; nodes/s_perf is perf-ATTACHED, depressed) ---"
  cat "$PROBE_SUMMARY"
  echo "--- HOW TO DECODE (subtractive attribution = the verdict; flamegraph = corroborating color) ---"
  echo "1. COMPONENT COST (causal, inlining/symbol-collision-proof): from nodes/s_clean:"
  echo "   frozen-EMD probe cost  = nodes/s(PROD) - nodes/s(EMDNOMEMO)   [emd_nomemo defeats the per-thread memo"
  echo "        -> a REAL 152MB FxHashMap probe EVERY postflop node; the drop IS the per-PROBE cost. CAUTION: this is"
  echo "        the per-probe cost in a NO-MEMO world. The PROD-relevant cost = delta x (1 - emd_bucket_hit_rate) because"
  echo "        the memo serves most probes in prod (emd_bucket_hit_rate is in PROD_summary.json). MPHF compaction is"
  echo "        only justified if delta x prod-miss-rate is large; a big RAW delta with a tiny prod miss-rate is NOT.]"
  echo "   cum_regret get cost    = nodes/s(REGRETZERO)    - nodes/s(PROD)   [regret_zero skips the get; the rise = its cost. (counter-drift caveat — skipping the get perturbs the equilibrium)]"
  echo "   showdown cost          = nodes/s(SHOWDOWNCONST) - nodes/s(PROD)   [showdown_const skips the eval; the rise = its cost. (counter-drift caveat — same equilibrium perturbation; only emd_nomemo is zero-drift)]"
  echo "   The LARGEST delta names the per-deal bottleneck. (ab_stub branch overhead cancels: all deltas are vs PROD on the SAME binary.)"
  echo "2. DRAM SIGNATURE OF THE FROZEN PROBE (the headline test, memo-defeated): EMDNOMEMO_dramflame.svg +"
  echo "   EMDNOMEMO_dram_top.txt + EMDNOMEMO_perfstat_{stall,mlp,dram}.txt. Does forcing a real probe every node"
  echo "   SPIKE mem_load_l3_miss_retired.remote_dram + stalls_mem_any vs PROD? If yes -> the 152MB lookup is the"
  echo "   >L3 DRAM bottleneck (direct proof). PROD_dramflame.svg shows the MEMOIZED steady hot path (frozen probe"
  echo "   ~absent there is EXPECTED = the memo working, NOT evidence the table is innocent)."
  echo "3. EFFECTIVE MLP: <shape>_perfstat_mlp.txt l1d_pend_miss.pending / l1d_pend_miss.pending_cycles = avg"
  echo "   outstanding L1D misses when busy. Low(~1-2)+high stall => latency-bound, little overlap. (offcore MLP:"
  echo "   PROD_perfstat_mlpoff_n.all_data_rd / mlpoff_c.cycles_with_data_rd — split into 2 groups so it schedules.)"
  echo "4. STALL FRACTION + IPC: <shape>_perfstat_stall.txt stalls_mem_any/cycles ; PROD_perfstat_ipc.txt instr/cycles."
  echo "   CHECK <shape>_perfstat_status.txt = COUNTED before trusting a 0 (UNCOUNTED = unscheduled, not a true zero)."
  echo "5. 128t vs 64t (PROD vs PROD64): a 64t stall-fraction drop = memory-channel contention OR SMT-sibling relief"
  echo "   (64t = 1 thread/physical-core; cannot separate the two here). Confirm rayon_realized=128 vs 64 first."
  echo "--- artifacts ---"
  echo "subtractive sweep  : {PROD,EMDNOMEMO,REGRETZERO,SHOWDOWNCONST,PROD64}_summary.json (nodes/s_clean)"
  echo "flamegraphs        : PROD_cpuflame.svg ; PROD_dramflame.svg (memoized) ; EMDNOMEMO_dramflame.svg (real probe)"
  echo "DRAM attribution   : *_dram_top.txt *_dram_annotate.txt *_DRAM_HOT_SYMBOL"
  echo "PMU                : *_perfstat_{ipc,stall,mlp,mlpoff_n,mlpoff_c,dram,portable}.txt + *_perfstat_status.txt"
  echo "topology           : numactl_topology.txt probe_env.txt perf_event_inventory.txt"
} > /var/log/probe/PROBE_DONE
upload_one /var/log/probe/PROBE_DONE PROBE_DONE
echo "ALL SHAPES DONE — PROBE_DONE uploaded."
