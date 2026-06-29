#!/bin/bash
# cpu128_probe_onbox.sh — the on-box driver for the instrumented 128-vCPU native-CFR profiling probe.
#
# Holds ALL heavy logic (build + perf hard-gate + per-shape warmup->steady->2-window capture loop +
# flamegraph + upload-as-you-go) so user_data stays well under 16KB. EXEC'd by the launcher's bootstrap
# after it exports BUCKET/PREFIX/REGION/RUN_PREFIX/PROBE_PREFIX/TABLE_ID/NUM_BUCKETS/WARMUP_CAP/
# WINDOW_ITERS/PROFILE_EVERY/SHAPES/WALL_GUARD_SECS/PYTHONPATH.
#
# set -uxo pipefail (NOT -e): a failing cargo build / missing perf must NOT abort the script — it must
# emit explicit per-stage OK/FAIL markers so PROBE_DONE is only trustworthy if BUILD_OK + (PERF_OK or
# PERF_BROKEN) + FROZEN_OK are all present. The 4 BLOCKING FIXES are wired here:
#   #1 disk-full   : df -BG / assert >20G before each perf record (root is a 150GB gp3 from the launcher).
#   #2 perf-broken : a HARD GATE (perf stat -e cycles true + a 2s perf record self-test) -> PERF_BROKEN
#                    + DEGRADE to numastat+mpstat+train_profile_json-only (never a silent empty flamegraph).
#   #3 PMU masked  : per-event <not supported> parse + a NUMA hypothesis gated on numactl --hardware node
#                    count (NUMA_NODES<2 => NUMA_STRUCTURALLY_IMPOSSIBLE).
#   #4 cost-runaway: STEP3 steady-wait has PID kill -0 liveness + a 900s STEADY_TIMEOUT; the independent
#                    WALL_GUARD watchdog (armed in user_data) is the outer backstop.
set -uxo pipefail

# ---------------------------------------------------------------------------- STEP 0: SETUP + helper
mkdir -p /var/log/probe
cd /opt/pokerbot

# upload_one f key : sync the page cache, then aws s3 cp the file under $PROBE_PREFIX/<key>. `sync`
# BEFORE every cp is the durability discipline that closes the prior tee-only data-loss.
upload_one() {
  local f="$1" key="$2"
  [ -e "$f" ] || { echo "upload_one: missing $f (skip $key)"; return 0; }
  sync
  aws s3 cp "$f" "$PROBE_PREFIX/$key" --region "$REGION" || echo "WARN upload failed: $f -> $key"
}
# marker NAME [body] : write a one-line S3 marker (and a local copy).
marker() {
  local name="$1"; shift
  printf '%s\n' "${*:-$(date -u +%FT%TZ)}" > "/var/log/probe/$name"
  upload_one "/var/log/probe/$name" "$name"
}

echo "PROBE_PREFIX=$PROBE_PREFIX  TABLE_ID=$TABLE_ID  SHAPES=$SHAPES"

# Download the PRE-UPLOADED frozen blob (CACHE_HIT; the box NEVER recomputes the ~16.5M spots).
aws s3 cp "s3://$BUCKET/$RUN_PREFIX/frozen/$TABLE_ID.emdfrzn" /opt/pokerbot/frozen.emdfrzn --region "$REGION" \
  || { echo "FROZEN_DOWNLOAD_FAIL"; marker FROZEN_DOWNLOAD_FAIL "no s3://$BUCKET/$RUN_PREFIX/frozen/$TABLE_ID.emdfrzn"; exit 1; }
# Sidecar shipped in the tarball; copy it next to the blob for the driver's integrity verify.
cp /opt/pokerbot/data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn.sha256 \
   /opt/pokerbot/frozen.emdfrzn.sha256 2>/dev/null || echo "WARN missing sha256 sidecar in tarball"

# ---------------------------------------------------------------------------- APT (BLOCKING-FIX #2 pkgs)
# linux-tools-common = the perf WRAPPER; linux-tools-aws = the AWS-kernel-flavour perf; linux-tools-$(uname
# -r) = the EXACT booted-kernel build. NOT linux-tools-generic (wrong on an aws kernel). + sysstat (mpstat/
# pidstat) + numactl + git/perl (FlameGraph).
apt-get install -y -qq linux-tools-common linux-tools-aws "linux-tools-$(uname -r)" \
  sysstat numactl git perl || echo "WARN apt perf/sysstat install had a failure"
perf --version || echo "WARN perf wrapper present but versioned binary may mismatch kernel"

# ---------------------------------------------------------------------------- SYSCTL (perf permissions)
sysctl -w kernel.perf_event_paranoid=-1 || true
sysctl -w kernel.kptr_restrict=0 || true
printf 'kernel.perf_event_paranoid=-1\nkernel.kptr_restrict=0\n' > /etc/sysctl.d/99-perf.conf
sysctl --system || true
sysctl -w kernel.perf_event_max_sample_rate=10000 || true

# ---------------------------------------------------------------------------- BUILD (debug-symbol'd v3) — TWO .so for the f32_regret A/B
# KEEP -C target-cpu=x86-64-v3 (must match production codegen — the profiled binary IS the shipping
# binary). debuginfo=2 -> DWARF symbol names for inlined leaves; force-frame-pointers -> cheap --call
# -graph fp unwind; strip=none -> keep the symtab. Cargo.toml is NOT edited (firewall-clean; lto=fat /
# codegen-units=1 / panic=abort stay production).
#
# The f32_regret A/B needs TWO binaries from the SAME crate via the `f32_regret` feature gate, both with the
# production-default features (phe_showdown + phe_site1) so ONLY the persistent regret-row precision differs:
#   f64 (baseline) = DEFAULT features              -> persistent cum_regret rows are [f64;11]
#   f32 (lever)    = --features python,f32_regret  -> persistent cum_regret rows are [f32;11] (strat_sum stays f64)
# Both keep target-cpu=x86-64-v3 + debuginfo. The driver imports a FIXED /opt/pokerbot/battery_native.so, so
# the A/B is done by building BOTH once and per-shape COPYING the chosen flavor over that fixed path before
# launching the (fresh-process) driver — NO driver change. This measures whether halving the cum_regret read/
# write bytes relieves the 128t memory-contention the prior attribution probe (probe_1782704164) certified.
export PYO3_PYTHON="$(python3 -c 'import sys;print(sys.executable)')"
export CARGO_PROFILE_RELEASE_DEBUG=line-tables-only
RF='-C target-cpu=x86-64-v3 -C force-frame-pointers=yes -C debuginfo=2 -C strip=none'
# f64 baseline (DEFAULT features) — the production regret precision; the REGIME-GATE binary (perf-statted 128t).
RUSTFLAGS="$RF" cargo build --release --features python --lib --manifest-path tools/battery_native/Cargo.toml || { echo "BUILD_FAIL_F64"; marker BUILD_FAIL "f64 cargo build failed"; exit 1; }
SO_ON="$(ls tools/battery_native/target/release/libbattery_native.so 2>/dev/null || ls tools/battery_native/target/release/battery_native.so 2>/dev/null)"
cp "$SO_ON" /opt/pokerbot/battery_native_phe_on.so
# f32 lever (--features python,f32_regret keeps DEFAULT features = phe ON, ADDS the f32 cum_regret rows).
RUSTFLAGS="$RF" cargo build --release --features python,f32_regret --lib --manifest-path tools/battery_native/Cargo.toml || { echo "BUILD_FAIL_F32"; marker BUILD_FAIL "f32 cargo build failed"; exit 1; }
SO_F32="$(ls tools/battery_native/target/release/libbattery_native.so 2>/dev/null || ls tools/battery_native/target/release/battery_native.so 2>/dev/null)"
cp "$SO_F32" /opt/pokerbot/battery_native_f32.so
ls -la /opt/pokerbot/battery_native_phe_on.so /opt/pokerbot/battery_native_f32.so
# Default the live import to the f64 baseline until the per-shape cp overrides it.
cp /opt/pokerbot/battery_native_phe_on.so /opt/pokerbot/battery_native.so
# BUILD_OK verify: debug_info present, PyInit resolves, AND f64 vs f32 DIFFER (cmp -s non-zero = differ).
# If byte-identical the f32_regret feature gate did nothing -> the A/B would be void, so DIFFER is required.
AB_DIFFER=0; cmp -s /opt/pokerbot/battery_native_phe_on.so /opt/pokerbot/battery_native_f32.so || AB_DIFFER=1
if file /opt/pokerbot/battery_native_phe_on.so | grep -q 'debug_info' \
   && nm /opt/pokerbot/battery_native_f32.so 2>/dev/null | grep -q 'PyInit_battery_native' \
   && nm /opt/pokerbot/battery_native_phe_on.so 2>/dev/null | grep -q 'PyInit_battery_native' \
   && [ "$AB_DIFFER" = 1 ]; then
  marker BUILD_OK "f64 baseline + f32 lever built; debug_info + PyInit_battery_native resolved + binaries DIFFER (f32 A/B valid)"
else
  echo "BUILD_VERIFY_WARN: missing debug_info / PyInit symbol, or f64==f32 byte-identical (AB_DIFFER=$AB_DIFFER -> f32 A/B void)"
  marker BUILD_OK "WARN: verify weak — AB_DIFFER=$AB_DIFFER; file/nm did not match"
fi
marker AB_BUILD "f64 + f32 built; AB_DIFFER=$AB_DIFFER (1=valid; f32_regret feature flipped codegen)"

# ---------------------------------------------------------------------------- FLAMEGRAPH TOOLCHAINS (both)
export PATH="$HOME/.cargo/bin:$PATH"
cargo install --locked inferno || echo "WARN inferno install failed; will fall back to FlameGraph perl"
git clone --depth 1 https://github.com/brendangregg/FlameGraph /opt/FlameGraph || true

# ---------------------------------------------------------------------------- PERF HARD-GATE (FIX #2)
# DECOUPLED capabilities (the c6i.metal lesson): PMU COUNTING (perf stat) and PMU SAMPLING (perf record) are
# INDEPENDENT. On EC2 bare metal the hardware PMI for sampling is frequently NOT delivered (perf record on
# `cycles` yields 0 samples even for a busy process) even though COUNTING works perfectly (perf stat reads
# the counters fine). Coupling them under one flag threw away the working stat counters when sampling failed.
#   PERF_STAT_OK  (G1) gates the perf-stat hardware counters (IPC, stalls_mem_any, L3-miss) = the memory-bound cert.
#   PERF_RECORD_OK(G2) gates the flamegraph; we try hardware `cycles` then SOFTWARE `cpu-clock` (an OS-timer
#                      event that samples WITHOUT the hardware PMI), so the flamegraph still works on a sample-less PMU.
PERF_AVAILABLE=0; PERF_STAT_OK=0; PERF_RECORD_OK=0; PERF_RECORD_EVENT=""
perf list 2>/dev/null > /var/log/probe/perf_event_inventory.txt || true
upload_one /var/log/probe/perf_event_inventory.txt perf_event_inventory.txt
# G1: hardware PMU COUNTING. cycles must NOT report <not supported>.
if perf stat -e cycles true 2>&1 | grep -q '<not supported>'; then
  echo "PERF_GATE G1 FAIL: cycles <not supported> (no PMU counting)"
else
  PERF_STAT_OK=1
fi
# G2: SAMPLING for the flamegraph — profile a BUSY 2s CPU-burner (NOT an idle process, which yields ~0
# samples even on a working PMU). Try hardware `cycles`; if the metal PMI is undelivered (0 samples), fall
# back to SOFTWARE `cpu-clock` (OS-timer sampling, needs no hardware PMI). PERF_RECORD_OK if either works.
for EV in cycles cpu-clock; do
  rm -f /tmp/perfselftest.data
  # CRITICAL: the workload `timeout 2 yes` exits NON-ZERO (124, timeout-killed), and `perf record -- cmd`
  # propagates cmd's exit code. So we must NOT chain `perf record && perf script` — the non-zero exit
  # short-circuits the `&&` and masks whether sampling actually worked. Run record (ignore its exit), then
  # INDEPENDENTLY check the captured sample count.
  perf record -e "$EV" -F 997 -g -o /tmp/perfselftest.data -- bash -c 'timeout 2 yes > /dev/null 2>&1' 2>/dev/null || true
  if perf script -i /tmp/perfselftest.data 2>/dev/null | head -5 | grep -q . ; then
    PERF_RECORD_OK=1; PERF_RECORD_EVENT="$EV"; break
  fi
done
[ "$PERF_RECORD_OK" = 1 ] || echo "PERF_GATE G2 FAIL: neither cycles nor cpu-clock perf-record produced samples"
if [ "$PERF_STAT_OK" = 1 ] || [ "$PERF_RECORD_OK" = 1 ]; then PERF_AVAILABLE=1; fi
if [ "$PERF_STAT_OK" = 1 ] && [ "$PERF_RECORD_OK" = 1 ]; then
  marker PERF_OK "PERF_STAT_OK=1 PERF_RECORD_OK=1 record_event=$PERF_RECORD_EVENT (counting + sampling both work)"
elif [ "$PERF_STAT_OK" = 1 ] || [ "$PERF_RECORD_OK" = 1 ]; then
  marker PERF_PARTIAL "PERF_STAT_OK=$PERF_STAT_OK PERF_RECORD_OK=$PERF_RECORD_OK record_event=${PERF_RECORD_EVENT:-none} -> capture what works, degrade the rest"
else
  # DEGRADE (do NOT abort): both counting AND sampling unavailable -> route to numastat+mpstat+train_profile_json,
  # NEVER produce a silent empty flamegraph that looks like success.
  marker PERF_BROKEN "PERF_STAT_OK=0 PERF_RECORD_OK=0 -> degrade to numastat+mpstat+train_profile_json only"
fi

# ---------------------------------------------------------------------------- TOPOLOGY (gates NUMA hypo)
{
  echo "=== nproc ==="; nproc
  echo "=== lscpu ==="; lscpu
  echo "=== RUSTFLAGS ==="; echo "-C target-cpu=x86-64-v3 -C force-frame-pointers=yes -C debuginfo=2 -C strip=none"
  echo "=== git rev ==="; (cd /opt/pokerbot && git rev-parse HEAD 2>/dev/null || echo 'no git in tarball')
  echo "=== file .so ==="; file /opt/pokerbot/battery_native.so
  echo "=== nm train ==="; nm -C /opt/pokerbot/battery_native.so 2>/dev/null | grep -i train | head
  echo "=== env RAYON ==="; env | grep -i rayon || echo '(no RAYON_* set)'
  echo "=== lstopo ==="; (lstopo --of txt 2>/dev/null || echo 'lstopo not installed')
} > /var/log/probe/probe_env.txt 2>&1
upload_one /var/log/probe/probe_env.txt probe_env.txt

numactl --hardware > /var/log/probe/numactl_topology.txt 2>&1 || echo "WARN numactl --hardware failed"
upload_one /var/log/probe/numactl_topology.txt numactl_topology.txt
# NUMA_NODES gate (BLOCKING-FIX #3): <2 nodes => the cross-socket hypothesis is structurally impossible.
NUMA_NODES="$(grep -c '^node [0-9]* cpus:' /var/log/probe/numactl_topology.txt 2>/dev/null || echo 0)"
if [ "${NUMA_NODES:-0}" -lt 2 ]; then
  echo "NUMA_STRUCTURALLY_IMPOSSIBLE NUMA_NODES=$NUMA_NODES"
  marker NUMA_STRUCTURALLY_IMPOSSIBLE "NUMA_NODES=$NUMA_NODES (<2) -> NUMA hypothesis REFUTE-by-topology"
fi
echo "NUMA_NODES=$NUMA_NODES PERF_AVAILABLE=$PERF_AVAILABLE"

# perf_event_supported EV : true if `perf stat -e EV sleep 0.2` does not report <not supported>/<not counted>.
perf_event_supported() {
  local ev="$1"
  perf stat -e "$ev" sleep 0.2 2>&1 | grep -Eq '<not supported>|<not counted>' && return 1
  return 0
}
# Emit per-event availability (BLOCKING-FIX #3) into a per-shape file so a counter-blind run is never
# mistaken for a NUMA refutation. Writes to $1.
write_event_availability() {
  local out="$1"
  : > "$out"
  local evs="cycles instructions stalled-cycles-backend branch-misses cache-references cache-misses node-loads node-load-misses dTLB-loads dTLB-load-misses context-switches cpu-migrations cycle_activity.stalls_mem_any mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram"
  for ev in $evs; do
    if [ "$PERF_AVAILABLE" = 1 ] && perf_event_supported "$ev"; then
      echo "$ev AVAILABLE" >> "$out"
    else
      echo "$ev NOT_SUPPORTED" >> "$out"
    fi
  done
}

# ---------------------------------------------------------------------------- STEP 1: WRITE THE DRIVER
# A reviewable copy is also in the tarball at scripts/drive_cpu128_probe.py; use it directly if present,
# else heredoc an equivalent. Prefer the shipped copy (single source of truth).
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

# ---------------------------------------------------------------------------- STEP 8 (armed early): SIGTERM/spot trap
# Even though the launcher defaults to ON-DEMAND, keep the IMDSv2 spot poller as belt-and-suspenders. On a
# reclaim notice it SIGTERMs the driver (its finally: flushes+fsyncs+uploads partial jsonl), then sweeps
# every on-disk artifact to S3 BEFORE shutdown can proceed.
cat > /tmp/spot_watch.sh <<SWEOF
TOK=\$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
while true; do
  C=\$(curl -s -o /dev/null -w "%{http_code}" -H "X-aws-ec2-metadata-token: \$TOK" http://169.254.169.254/latest/meta-data/spot/instance-action)
  if [ "\$C" = "200" ]; then
    echo "SPOT_RECLAIM_NOTICE"
    pkill -TERM -f drive_cpu128_probe.py
    sleep 8
    sync
    aws s3 cp --recursive /var/log/probe/ "$PROBE_PREFIX/SWEEP/" --region "$REGION" || true
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
  # Per-shape config (canonical mapping; the launcher's --dry-run table mirrors this).
  # The PHE A/B is a 2x2 (phe-ON/OFF x rayon 128t/64t), all at the production EMD-K160 fanout N=1024.
  # SO_SEL selects which freshly-built flavor is copied to the FIXED /opt/pokerbot/battery_native.so the
  # (fresh-process) driver imports — that copy IS the entire A/B mechanism (no driver change).
  N=1024
  WRAP=()
  unset RAYON_NUM_THREADS || true
  unset POKERBOT_AB_STUB || true
  SO_SEL=/opt/pokerbot/battery_native_phe_on.so
  # f32_regret A/B shapes (a 2x2: f64/f32 x rayon 128t/64t, all at the production EMD-K160 fanout N=1024). P1 =
  # the STEP-1 REGIME GATE (f64 128t + the WINDOW-B perf-stat reads memory_bound=stalls_mem_any/cycles, IPC,
  # L3-miss local/remote DRAM). PF32 = the SAME at 128t on the f32 build = THE A/B vs P1 (does halving cum_regret
  # bytes lower stalls + raise nodes/s under 128t contention?). P3/PF32_64 = the 64t SMT cross-check pair. No
  # POKERBOT_AB_STUB here (the attribution stubs were the prior probe; this run isolates the f32 precision lever).
  case "$SHAPE" in
    P1)      N=1024; SO_SEL=/opt/pokerbot/battery_native_phe_on.so ;;                                       # REGIME GATE: f64 baseline 128t + perf-stat
    PF32)    N=1024; SO_SEL=/opt/pokerbot/battery_native_f32.so ;;                                          # THE A/B vs P1: f32 cum_regret 128t + perf-stat
    P3)      N=1024; SO_SEL=/opt/pokerbot/battery_native_phe_on.so; export RAYON_NUM_THREADS=64 ;;          # SMT cross-check: f64 baseline 64t (vs P1 128t)
    PF32_64) N=1024; SO_SEL=/opt/pokerbot/battery_native_f32.so;    export RAYON_NUM_THREADS=64 ;;          # THE A/B vs P3: f32 cum_regret 64t
    *) echo "UNKNOWN SHAPE $SHAPE — skip"; continue ;;
  esac
  CAP="$WARMUP_CAP"; [ "$N" = 200 ] && CAP=600   # per-N cap (cosmetic; the driver's steady-detect is primary)

  # Per-shape event availability (BLOCKING-FIX #3).
  write_event_availability "/var/log/probe/${SHAPE}_perf_event_availability.txt"
  upload_one "/var/log/probe/${SHAPE}_perf_event_availability.txt" "$SHAPE/${SHAPE}_perf_event_availability.txt"

  rm -f "/var/log/probe/$SHAPE.STEADY_BEGIN" "/var/log/probe/$SHAPE.STEADY_BEGIN_PERF" "/var/log/probe/$SHAPE.WINDOW_DONE"

  # PHE A/B select: copy the chosen flavor over the FIXED import path BEFORE the fresh-process driver loads it.
  cp "$SO_SEL" /opt/pokerbot/battery_native.so
  echo "$SHAPE using $(basename "$SO_SEL") -> /opt/pokerbot/battery_native.so (rayon=${RAYON_NUM_THREADS:-default})"
  marker "${SHAPE}_SO" "$(basename "$SO_SEL") rayon=${RAYON_NUM_THREADS:-default} N=$N"

  # Launch the driver in the background under the shape wrapper (P3/P4: RAYON_NUM_THREADS=64).
  "${WRAP[@]}" python3 /opt/pokerbot/drive_cpu128_probe.py \
      --shape "$SHAPE" --ndeals "$N" --window "$WINDOW_ITERS" --profile-every "$PROFILE_EVERY" \
      --warmup-cap "$CAP" > "/var/log/probe/${SHAPE}_driver.log" 2>&1 &
  PID=$!
  echo "$SHAPE driver PID=$PID N=$N CAP=$CAP RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-default} WRAP=${WRAP[*]:-none}"

  # ----------------------------------------- STEP 3: WAIT FOR STEADY (WINDOW A) with PID liveness (FIX #4)
  WAITED=0
  STEADY_OK=0
  while true; do
    if [ -f "/var/log/probe/$SHAPE.STEADY_BEGIN" ]; then STEADY_OK=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "DRIVER_DIED before STEADY_BEGIN (shape $SHAPE)"
      upload_one "/var/log/probe/${SHAPE}_driver.log" "$SHAPE/${SHAPE}_driver.log"
      marker "${SHAPE}_FAILED" "DRIVER_DIED before STEADY_BEGIN"
      break
    fi
    if [ "$WAITED" -gt 900 ]; then
      echo "STEADY_TIMEOUT (shape $SHAPE, ${WAITED}s)"
      kill -TERM "$PID" 2>/dev/null || true
      sleep 5
      upload_one "/var/log/probe/${SHAPE}_driver.log" "$SHAPE/${SHAPE}_driver.log"
      marker "${SHAPE}_FAILED" "STEADY_TIMEOUT after ${WAITED}s"
      break
    fi
    sleep 2; WAITED=$((WAITED + 2))
  done
  [ "$STEADY_OK" = 1 ] || { echo "skip captures for $SHAPE"; continue; }

  # WINDOW A (clean throughput, NO perf): only mpstat+numastat attach so the headline deals/s is not
  # depressed by the profiler. The driver streams profileA.jsonl during this window.
  echo "$SHAPE WINDOW A (clean) — mpstat + numastat only"
  numastat -p "$PID" > "/var/log/probe/${SHAPE}_numastat_start.txt" 2>&1 || true
  mpstat -P ALL 1 30 > "/var/log/probe/${SHAPE}_mpstat.txt" 2>&1 &
  MPID=$!
  pidstat -t -p "$PID" 1 30 > "/var/log/probe/${SHAPE}_pidstat.txt" 2>&1 &
  PSID=$!
  wait "$MPID" 2>/dev/null || true
  wait "$PSID" 2>/dev/null || true
  numastat -p "$PID" > "/var/log/probe/${SHAPE}_numastat_end.txt" 2>&1 || true
  upload_one "/var/log/probe/${SHAPE}_mpstat.txt"          "$SHAPE/${SHAPE}_mpstat.txt"
  upload_one "/var/log/probe/${SHAPE}_pidstat.txt"         "$SHAPE/${SHAPE}_pidstat.txt"
  upload_one "/var/log/probe/${SHAPE}_numastat_start.txt"  "$SHAPE/${SHAPE}_numastat_start.txt"
  upload_one "/var/log/probe/${SHAPE}_numastat_end.txt"    "$SHAPE/${SHAPE}_numastat_end.txt"

  # ----------------------------------------- STEP 3b: WAIT FOR PERF WINDOW (WINDOW B)
  WAITED=0
  PERFWIN_OK=0
  while true; do
    if [ -f "/var/log/probe/$SHAPE.STEADY_BEGIN_PERF" ]; then PERFWIN_OK=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then echo "driver exited before WINDOW B (shape $SHAPE)"; break; fi
    if [ "$WAITED" -gt 300 ]; then echo "WINDOW B wait timeout (shape $SHAPE)"; break; fi
    sleep 2; WAITED=$((WAITED + 2))
  done

  # ----------------------------------------- STEP 4: CAPTURE (WINDOW B; perf only if available)
  # ORDER MATTERS (measurement-validity fix, refuter 2b): WINDOW B is only ~75s wall at N=1024, but on bare
  # metal the PMU events COUNT (on KVM they were <not supported> and returned instantly). Three back-to-back
  # full-length captures could overrun the window so the LAST capture attaches to a process already in its
  # finally/teardown -> empty perf.data -> NO flamegraph for the headline P1. So:
  #   (1) run `perf record` FIRST (it is the headline "name the NEXT hot kernel" deliverable) on a guaranteed
  #       -live steady process right after STEADY_BEGIN_PERF, and gate it on a `kill -0` liveness check;
  #   (2) shorten the two `perf stat` captures to `timeout 15` each so record(30s)+portable(15s)+vendor(15s)
  #       = ~60s fits inside the ~75s window with margin;
  #   (3) emit PERF_RECORD_LIVE_SECS so the decoder knows the flamegraph saw a live steady process, not a
  #       teardown.
  if [ "$PERFWIN_OK" = 1 ] && [ "$PERF_AVAILABLE" = 1 ]; then
    # (a) DISK GUARD (BLOCKING-FIX #1): need >20G free before perf record (perf.data can be 200-600MB).
    AVAIL="$(df -BG / | awk 'NR==2{print $4}' | tr -d G)"
    echo "$SHAPE disk free: ${AVAIL}G"
    if [ "${AVAIL:-0}" -lt 20 ]; then
      echo "DISK_FULL (${AVAIL}G < 20G) — skipping perf record for $SHAPE"
      marker "${SHAPE}_DISK_FULL" "df / = ${AVAIL}G < 20G"
    else
      # (d-FIRST) flamegraph record (fp unwind — we forced frame pointers; -m 16M --aio = low-overhead).
      # Run FIRST so the headline next-hot-kernel capture sees a fully-live steady process. Liveness-gate it.
      if [ "$PERF_RECORD_OK" = 1 ] && kill -0 "$PID" 2>/dev/null; then
        # Use the sampling event that PASSED the G2 self-test ($PERF_RECORD_EVENT = hardware `cycles` if the
        # PMI is delivered, else software `cpu-clock` = OS-timer CPU-time sampling). Either yields a valid
        # call-graph flamegraph (fp unwind); cpu-clock weights by CPU time, which is exactly the "where does
        # per-deal time go / name the NEXT hot kernel" signal.
        PERF_REC_T0=$(date +%s)
        timeout 30 perf record -e "$PERF_RECORD_EVENT" -p "$PID" -F 997 --call-graph fp -m 16M --aio \
          -o "/var/log/probe/${SHAPE}_perf_fp.data" 2>&1 || true
        PERF_REC_T1=$(date +%s)
        # Confirm the process stayed live through the record (else the capture may have caught teardown).
        if kill -0 "$PID" 2>/dev/null; then PERF_REC_LIVE_END=live; else PERF_REC_LIVE_END=ended_during; fi
        marker "${SHAPE}_PERF_RECORD_LIVE_SECS" "secs=$((PERF_REC_T1 - PERF_REC_T0)) end=$PERF_REC_LIVE_END event=$PERF_RECORD_EVENT (live=flamegraph saw steady process)"
      elif [ "$PERF_RECORD_OK" != 1 ]; then
        echo "$SHAPE: PERF_RECORD_OK=0 (no working sampling event — hardware PMI undelivered AND cpu-clock failed) — skipping flamegraph"
        marker "${SHAPE}_PERF_RECORD_LIVE_SECS" "secs=0 end=no_sampling_event (NO flamegraph; PMU sampling unavailable on this instance)"
      else
        echo "$SHAPE: PID not live before perf record — skipping flamegraph capture"
        marker "${SHAPE}_PERF_RECORD_LIVE_SECS" "secs=0 end=dead_before_record (NO flamegraph; process exited before WINDOW B perf record)"
      fi
      upload_one "/var/log/probe/${SHAPE}_perf_fp.data" "$SHAPE/${SHAPE}_perf_fp.data"
      # perf-stat hardware COUNTERS — gated on PERF_STAT_OK (counting), INDEPENDENT of sampling: on c6i.metal
      # these COUNT even when perf record cannot sample. This is the memory-bound certification (IPC + stalls).
      if [ "$PERF_STAT_OK" = 1 ]; then
        # (b) portable per-socket counters (15s — fits the window after the record).
        timeout 15 perf stat -p "$PID" --per-socket -I 1000 \
          -e task-clock,cycles,instructions,stalled-cycles-backend,branch-misses,cache-references,cache-misses,node-loads,node-load-misses,dTLB-loads,dTLB-load-misses,context-switches,cpu-migrations \
          -o "/var/log/probe/${SHAPE}_perfstat_portable.txt" 2>&1 || true
        upload_one "/var/log/probe/${SHAPE}_perfstat_portable.txt" "$SHAPE/${SHAPE}_perfstat_portable.txt"
        # (c) vendor-specific direct remote-DRAM counters (15s — isolated in their own perf invocation so a
        # PEBS/offcore-group contention <not counted> can't poison the portable IPC read).
        VENDOR="$(grep -m1 vendor_id /proc/cpuinfo | awk '{print $3}')"
        if [ "$VENDOR" = "GenuineIntel" ]; then
          timeout 15 perf stat -p "$PID" \
            -e cycle_activity.stalls_mem_any,mem_load_l3_miss_retired.local_dram,mem_load_l3_miss_retired.remote_dram \
            -o "/var/log/probe/${SHAPE}_perfstat_vendor.txt" 2>&1 || true
        else
          RAW="$(grep -iE 'fills_from_sys' /var/log/probe/perf_event_inventory.txt | awk '{print $1}' | paste -sd, -)"
          [ -n "$RAW" ] && timeout 15 perf stat -p "$PID" -e "$RAW" \
            -o "/var/log/probe/${SHAPE}_perfstat_vendor.txt" 2>&1 || echo "no fills_from_sys events" > "/var/log/probe/${SHAPE}_perfstat_vendor.txt"
        fi
        upload_one "/var/log/probe/${SHAPE}_perfstat_vendor.txt" "$SHAPE/${SHAPE}_perfstat_vendor.txt"
      else
        echo "$SHAPE: PERF_STAT_OK=0 — skipping perf stat counters (no PMU counting)"
        marker "${SHAPE}_PERFSTAT_SKIPPED" "PERF_STAT_OK=0 (cycles not countable)"
      fi
    fi
  else
    echo "$SHAPE: perf capture SKIPPED (PERF_AVAILABLE=$PERF_AVAILABLE PERFWIN_OK=$PERFWIN_OK) — DEGRADED to numastat+mpstat"
    marker "${SHAPE}_PERF_DEGRADED" "PERF_AVAILABLE=$PERF_AVAILABLE PERFWIN_OK=$PERFWIN_OK"
  fi

  # ----------------------------------------- STEP 5: FLAMEGRAPH + TOP (only if perf.data non-empty)
  PERFDATA="/var/log/probe/${SHAPE}_perf_fp.data"
  if [ -s "$PERFDATA" ]; then
    perf script -i "$PERFDATA" > "/var/log/probe/${SHAPE}_perf_fp.script" 2>/dev/null || true
    if command -v inferno-collapse-perf >/dev/null 2>&1; then
      inferno-collapse-perf "/var/log/probe/${SHAPE}_perf_fp.script" > "/var/log/probe/${SHAPE}_perf_fp.folded" 2>/dev/null || true
      inferno-flamegraph --title "battery_native $SHAPE" "/var/log/probe/${SHAPE}_perf_fp.folded" > "/var/log/probe/${SHAPE}_flame.svg" 2>/dev/null || true
    elif [ -f /opt/FlameGraph/stackcollapse-perf.pl ]; then
      perl /opt/FlameGraph/stackcollapse-perf.pl "/var/log/probe/${SHAPE}_perf_fp.script" > "/var/log/probe/${SHAPE}_perf_fp.folded" 2>/dev/null || true
      perl /opt/FlameGraph/flamegraph.pl --title "battery_native $SHAPE" "/var/log/probe/${SHAPE}_perf_fp.folded" > "/var/log/probe/${SHAPE}_flame.svg" 2>/dev/null || true
    fi
    perf report -i "$PERFDATA" --stdio --sort overhead,symbol 2>/dev/null | head -60 > "/var/log/probe/${SHAPE}_perf_top.txt" || true
    upload_one "/var/log/probe/${SHAPE}_perf_fp.script" "$SHAPE/${SHAPE}_perf_fp.script"
    upload_one "/var/log/probe/${SHAPE}_perf_fp.folded" "$SHAPE/${SHAPE}_perf_fp.folded"
    upload_one "/var/log/probe/${SHAPE}_flame.svg"      "$SHAPE/${SHAPE}_flame.svg"
    upload_one "/var/log/probe/${SHAPE}_perf_top.txt"   "$SHAPE/${SHAPE}_perf_top.txt"
  else
    echo "$SHAPE: no perf.data — no flamegraph (DEGRADED verdict from numastat+mpstat+profileA/B)"
  fi

  # ----------------------------------------- STEP 6/7: let the driver finish, upload its artifacts.
  wait "$PID" 2>/dev/null || true
  upload_one "/var/log/probe/${SHAPE}_profileA.jsonl" "$SHAPE/${SHAPE}_profileA.jsonl"
  upload_one "/var/log/probe/${SHAPE}_profileB.jsonl" "$SHAPE/${SHAPE}_profileB.jsonl"
  upload_one "/var/log/probe/${SHAPE}_summary.json"   "$SHAPE/${SHAPE}_summary.json"
  upload_one "/var/log/probe/${SHAPE}_driver.log"     "$SHAPE/${SHAPE}_driver.log"
  # One-line per-shape summary for PROBE_DONE (deals/s + CPU% extracted from the driver's summary).
  ONE="$(python3 - "$SHAPE" <<'PYJ'
import json,sys
shape=sys.argv[1]
try:
    d=json.load(open(f"/var/log/probe/{shape}_summary.json"))
    print(f"{shape}: deals/s={d.get('deals_per_s_clean','?')} cpu%_mean={d.get('cpu_pct_mean','?')} "
          f"cpu%_max1s={d.get('cpu_pct_max_1s','?')} merge_share={d.get('merge_share','?')} "
          f"rayon={d.get('rayon_threads_realized','?')} ipc_src={'perf' if __import__('os').path.exists(f'/var/log/probe/{shape}_perfstat_portable.txt') else 'NONE'}")
except Exception as e:
    print(f"{shape}: summary unavailable ({e})")
PYJ
)"
  echo "$ONE" >> "$PROBE_SUMMARY"
  echo "$ONE"
done

# ---------------------------------------------------------------------------- STEP 7 (final): PROBE_DONE
# Re-upload box-level provenance + a per-decoder ANSWERABLE/NOT index so a counter-blind run is never read
# as a NUMA refutation.
upload_one /var/log/cloud-init-output.log logs/cloud-init-output.log
upload_one /var/log/probe/perf_event_inventory.txt perf_event_inventory.txt
upload_one /var/log/probe/numactl_topology.txt numactl_topology.txt
upload_one /var/log/probe/probe_env.txt probe_env.txt

{
  echo "PROBE_DONE $(date -u +%FT%TZ)"
  echo "PERF_AVAILABLE=$PERF_AVAILABLE  NUMA_NODES=$NUMA_NODES"
  echo "--- per-shape ---"
  cat "$PROBE_SUMMARY"
  echo "--- per-decoder answerability ---"
  echo "F32_DIFFER=${AB_DIFFER:-?} (1=f32_regret feature flipped codegen -> A/B valid; 0=void)"
  echo "--- f32_regret memory-traffic-relief certification ---"
  echo "THE CLINCHER A/B (f64 vs f32 cum_regret)  : ANSWERABLE (nodes/s + deals/s, P1-vs-PF32 @128t, P3-vs-PF32_64 @64t; PMU-immune). 128t pair = the contention-regime read; 64t pair = the SMT cross-check."
  echo "A/B SIGNIFICANCE (n=1 between-run noise)   : ANSWERABLE-FROM-DATA — the decoder MUST compute a within-window bootstrap CI on per-iter nodes/s (nodes / total_ms*1000) from the ~180 P1/PF32 (and P3/PF32_64) profileB.jsonl rows, and REQUIRE the f32 delta to EXCEED that CI before calling a speedup. EXPECT LOW-SINGLE-DIGIT (the f32 verdict: ~0-8%; the 152MB frozen EMD lookup is the bigger >L3 hog), so a null/marginal result is the EXPECTED outcome, not a failure."
  echo "MEMORY-CONTENTION RELIEF (the mechanism)   : ANSWERABLE if PERF — does f32 LOWER cycle_activity.stalls_mem_any/cycle AND RAISE IPC AND RAISE nodes/s at 128t (P1 vs PF32 perfstat_vendor+portable)? Direct test of 'halving cum_regret bytes relieves the infoset-table memory traffic'. A bigger 128t gain than 64t (PF32_64 vs P3) corroborates a CONTENTION mechanism (the relief amplifies with thread count)."
  if [ "$PERF_AVAILABLE" = 1 ]; then
    echo "memory-latency-bound (IPC/backend-stall) : ANSWERABLE (perf: IPC + cycle_activity.stalls_mem_any + mem_load_l3_miss_retired.{local,remote}_dram)"
    echo "compute-hidden-under-L3-stall test       : ANSWERABLE (did f32 LOWER stalls_mem_any AND RAISE nodes/s, or was it hidden? P1 vs PF32 perf delta)"
    echo "NEXT hot kernel (flamegraph)             : ANSWERABLE (P1 perf_top.txt + flame.svg widest non-showdown frame; expect EnvStateN copy or fingerprint probe). CHECK P1_PERF_RECORD_LIVE_SECS end=live before trusting the flamegraph (record runs FIRST in WINDOW B on a guaranteed-live steady process)."
  else
    echo "memory-latency-bound (IPC/backend-stall) : NOT (perf unavailable on this box) -> f32 A/B clincher (nodes/s) still ANSWERABLE PMU-free; mechanism test N/A"
    echo "NEXT hot kernel (flamegraph)             : NOT (perf unavailable) -> flamegraph N/A"
  fi
  echo "SMT-sibling contention (rayon 64 vs 128)  : ANSWERABLE (P3-vs-P1 + PF32_64-vs-PF32 deals/s, no PMU)"
  echo "traverse collect-barrier imbalance       : ANSWERABLE (mpstat/pidstat, no PMU)"
  echo "128-thread allocator/atomic contention   : $([ "$PERF_AVAILABLE" = 1 ] && echo 'ANSWERABLE (flamegraph+%sys)' || echo 'PARTIAL (mpstat %sys + P3-vs-P1)')"
  echo "CloudWatch wall-average contamination    : ANSWERABLE (mpstat %idle/%steal vs in-process)"
  echo "merge-Amdahl ceiling                     : ANSWERABLE (in-process merge_share, no PMU)"
  if [ "${NUMA_NODES:-0}" -ge 2 ]; then
    echo "NUMA cross-socket                        : PARTIAL (numastat delta + node-load PMU; no interleave A/B shape in this PHE set)"
  else
    echo "NUMA cross-socket                        : REFUTE-BY-TOPOLOGY (NUMA_NODES<2)"
  fi
} > /var/log/probe/PROBE_DONE
upload_one /var/log/probe/PROBE_DONE PROBE_DONE
echo "ALL SHAPES DONE — PROBE_DONE uploaded. Box stays up (keep-alive); confirm S3 then terminate manually."
