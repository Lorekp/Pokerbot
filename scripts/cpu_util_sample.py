"""
cpu_util_sample.py — per-second CPU utilization sampler.

Spawns a child process and records every 1 second:
  - total system CPU% across all cores
  - per-process CPU% (child + all descendants)
  - child thread count

Writes a CSV to --output-csv. Prints summary on completion.

Usage:
    python scripts/cpu_util_sample.py \
        --command "./tools/battery_native/target/release/examples/bench_trainer.exe" \
        --args "500 200 ./artifacts/flamegraph/blob.bin" \
        --output-csv "./artifacts/flamegraph/cpu_run1.csv" \
        --sample-sec 1 \
        --timeout-sec 600
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import psutil


def collect_process_tree(root_pid: int) -> tuple[float, int]:
    """Walk the process tree from root_pid; return (cumulative_cpu_seconds, total_threads)."""
    try:
        root = psutil.Process(root_pid)
    except psutil.NoSuchProcess:
        return (0.0, 0)
    seen: set[int] = set()
    stack = [root]
    total_cpu = 0.0
    total_threads = 0
    while stack:
        p = stack.pop()
        if p.pid in seen:
            continue
        seen.add(p.pid)
        try:
            with p.oneshot():
                total_cpu += p.cpu_times().user + p.cpu_times().system
                total_threads += p.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        try:
            for child in p.children(recursive=False):
                stack.append(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return (total_cpu, total_threads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True)
    parser.add_argument("--args", default="", help="space-separated args for the child process")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sample-sec", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--n-cores", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--env", default="", help="comma-separated KEY=VAL entries to set in child env")
    args = parser.parse_args()

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_list = [args.command] + (args.args.split() if args.args else [])
    child_env = os.environ.copy()
    for kv in args.env.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            child_env[k] = v

    print(f"=== cpu_util_sample.py ===")
    print(f"Command: {' '.join(cmd_list)}")
    print(f"Output:  {out_path}")
    print(f"Sample:  every {args.sample_sec}s, timeout {args.timeout_sec}s, cores={args.n_cores}")
    print()

    # prime psutil cpu_percent baseline
    psutil.cpu_percent(interval=None)

    t0 = time.monotonic()
    start_wall = time.time()
    proc = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        text=True,
    )
    print(f"Started PID {proc.pid} at {time.strftime('%H:%M:%S', time.localtime(start_wall))}")

    samples: list[tuple[float, float, float, int]] = []
    prev_cpu_sec: Optional[float] = None
    prev_time: Optional[float] = None

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "elapsed_sec", "total_cpu_pct", "process_cpu_pct", "threads"])
        f.flush()

        while proc.poll() is None:
            time.sleep(args.sample_sec)
            now = time.monotonic()
            elapsed = now - t0
            total_pct = psutil.cpu_percent(interval=None)
            cpu_sec, threads = collect_process_tree(proc.pid)
            proc_pct = ""
            if prev_cpu_sec is not None and prev_time is not None:
                d_cpu = cpu_sec - prev_cpu_sec
                d_t = now - prev_time
                if d_t > 0:
                    proc_pct = round(100.0 * d_cpu / d_t / args.n_cores, 2)
            prev_cpu_sec = cpu_sec
            prev_time = now

            w.writerow([
                time.strftime("%H:%M:%S", time.localtime()),
                round(elapsed, 1),
                round(total_pct, 2),
                proc_pct,
                threads,
            ])
            f.flush()
            samples.append((elapsed, total_pct, float(proc_pct) if proc_pct != "" else 0.0, threads))

            if elapsed > args.timeout_sec:
                print(f"TIMEOUT after {args.timeout_sec}s, killing child")
                proc.kill()
                break

    stdout, stderr = proc.communicate()
    wall = time.monotonic() - t0
    rc = proc.returncode

    # write summary row
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# exit_code={rc} wall_sec={round(wall,2)} samples={len(samples)}"])
    # remove the summary line from samples
    samples = [s for s in samples if s[1] >= 0]

    valid_total = [s[1] for s in samples if s[1] > 0]
    valid_proc = [s[2] for s in samples if s[2] > 0]
    avg_total = round(sum(valid_total) / len(valid_total), 2) if valid_total else 0.0
    max_total = round(max(valid_total), 2) if valid_total else 0.0
    avg_proc = round(sum(valid_proc) / len(valid_proc), 2) if valid_proc else 0.0
    max_proc = round(max(valid_proc), 2) if valid_proc else 0.0

    print()
    print("=== SUMMARY ===")
    print(f"exit_code      : {rc}")
    print(f"wall_sec       : {round(wall, 2)}")
    print(f"samples        : {len(samples)}")
    print(f"avg total CPU% : {avg_total} (across {args.n_cores} cores)")
    print(f"max total CPU% : {max_total}")
    print(f"avg proc CPU%  : {avg_proc}")
    print(f"max proc CPU%  : {max_proc}")
    gate = "PASS" if max_total >= 95 else "FAIL"
    print(f"GATE {gate} : max total CPU% {max_total} {'>=' if max_total >= 95 else '<'} 95")
    print()
    print("--- stdout ---")
    print(stdout)
    print("--- stderr ---")
    print(stderr)

    return 0 if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())
