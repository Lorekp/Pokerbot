#!/usr/bin/env bash
set -uo pipefail
RUNDIR=./runs/hu_neural
echo "time=$(date +%H:%M:%S)"
echo "--- train proc ---"
ps -eo pid,etimes,rss,cmd | grep "[t]rain_neural_cfr.py" | head -1 || echo "NO train proc"
echo "--- peak host RAM (MiB) ---"
cat "$RUNDIR/verify.peakmem" 2>/dev/null || echo "(no peakmem file)"
echo "--- free now (MiB) ---"
free -m | awk '/^Mem:/{print "used="$3" avail="$7" total="$2}'
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo "--- verify log tail ---"
L=$(ls -t "$RUNDIR"/verify_*.log 2>/dev/null | head -1)
echo "log=$L"
if [ -n "$L" ]; then grep -E "iter |SUMMARY|Saved|Traceback|Error|Killed|slow_operation|Compiling" "$L" | tail -10; tail -n 3 "$L"; fi
echo "--- boot.out ---"
tail -n 3 "$RUNDIR/verify_boot.out" 2>/dev/null
