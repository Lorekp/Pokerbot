#!/usr/bin/env bash
# Recon the metal box before repurposing it for perf/flamegraph.
echo "USER=$(whoami) PWD=$(pwd) NOW=$(date -u +%H:%M:%SZ)"
echo "--- repo ---"
REPO=$(find /home /root /opt /mnt -maxdepth 5 -type d -name battery_native 2>/dev/null | head -1)
echo "battery_native: ${REPO:-NOT-FOUND}"
[ -n "$REPO" ] && echo "repo root: $(dirname "$(dirname "$REPO")")" && ls "$(dirname "$(dirname "$REPO")")" 2>/dev/null | head
echo "--- A/B processes still running? ---"
ps -eo pid,etimes,pcpu,pmem,comm,args 2>/dev/null | grep -E 'python|drive_cpu128|battery|cargo|ab_train|[p]erf ' | grep -v grep | head -30
echo "--- perf availability ---"
which perf || echo "NO perf binary"
perf --version 2>&1 | head -1
echo "paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null) kptr=$(cat /proc/sys/kernel/kptr_restrict 2>/dev/null) kernel=$(uname -r)"
echo "--- cpu topology ---"
echo "nproc=$(nproc)"
lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|NUMA node\(s\)|NUMA node0|NUMA node1' | head
echo "--- mem/disk ---"
free -g 2>/dev/null | head -2
df -h / /tmp 2>/dev/null | tail -2
echo "--- cargo toolchain ---"
. ${HOME}/.cargo/env 2>/dev/null; . "$HOME/.cargo/env" 2>/dev/null
(cargo --version 2>/dev/null && rustc --version 2>/dev/null) || echo "cargo not on PATH yet"
echo "--- existing built .so ---"
find /home /root -maxdepth 6 -name 'battery_native*.so' 2>/dev/null | head
echo "--- S3 run marker ---"
aws s3 ls s3://video-processing-6789/pokerbot/prod6000/native_cfr/probe_1782724459/probe/ --region eu-central-1 2>&1 | head
echo "--- cost-safety: scheduled shutdown / wall guard ---"
cat /run/systemd/shutdown/scheduled 2>/dev/null && echo "(shutdown scheduled above)" || echo "no /run/systemd/shutdown/scheduled"
systemctl list-timers --all 2>/dev/null | grep -iE 'shutdown|probe|guard|stop' | head
ps -eo pid,etimes,args 2>/dev/null | grep -E 'shutdown|wall.?guard|self.?term|terminate' | grep -v grep | head
echo "RECON_DONE"
