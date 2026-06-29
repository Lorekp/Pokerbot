"""Watch the DCFR-curve box: pull streamed checkpoints + manifest + the work_steal probe as they land,
and when training is done (CURVE_TRAIN_DONE / PROBE_DONE / instance gone) run the LOCAL paired-LBR scorer
to produce the exploitability-vs-compute curve. Progress-logged; cost-safety termination is the separate
_probe_watch.py poller -- this is results-only.

  python scripts/poll_dcfr_curve.py <run_subdir> [instance_id]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import boto3
import urllib3

urllib3.disable_warnings()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKET = "video-processing-6789"
REGION = "eu-central-1"
RUN_SUBDIR = sys.argv[1] if len(sys.argv) > 1 else "dcfrcurve"
IID = sys.argv[2] if len(sys.argv) > 2 else None
PREFIX = f"pokerbot/prod6000/native_cfr/{RUN_SUBDIR}/dcfrcurve/"
OUTDIR = os.path.join(ROOT, "artifacts", RUN_SUBDIR)
os.makedirs(OUTDIR, exist_ok=True)

s3 = boto3.client("s3", region_name=REGION, verify=False)
ec2 = boto3.client("ec2", region_name=REGION, verify=False)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def listing():
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
    return {os.path.basename(o["Key"]): (o["Key"], o["Size"]) for o in r.get("Contents", [])}


def inst_state():
    if not IID:
        return "?"
    try:
        r = ec2.describe_instances(InstanceIds=[IID])
        return r["Reservations"][0]["Instances"][0]["State"]["Name"]
    except Exception:
        return "gone"


def pull_new(files, pulled):
    for name, (key, _sz) in files.items():
        if name in pulled:
            continue
        if name.endswith(".pkl") or name in ("manifest.json", "ws_probe.jsonl", "train.log",
                                             "build.log", "smoke.log", "STATUS", "CURVE_TRAIN_DONE",
                                             "PROBE_DONE", "ws_probe.err"):
            try:
                s3.download_file(BUCKET, key, os.path.join(OUTDIR, name))
                pulled.add(name)
                log(f"  pulled {name}")
            except Exception as e:  # noqa: BLE001
                log(f"  WARN pull {name}: {e!r}")


def summarize_ws_probe():
    p = os.path.join(OUTDIR, "ws_probe.jsonl")
    if not os.path.exists(p):
        return
    rows = []
    for ln in open(p):
        ln = ln.strip()
        if ln:
            try:
                rows.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                pass
    if not rows:
        return
    by = {(r["threads"], r["ws"]): r["deals_per_s"] for r in rows}
    threads = sorted({r["threads"] for r in rows}, key=lambda x: int(x) if str(x).isdigit() else 0)
    print("\n  work_steal scaling (deals/s; static WS=0 vs work_steal WS=1):")
    print(f"    {'threads':>8} | {'static':>12} | {'worksteal':>12} | {'ws/static':>9}")
    for th in threads:
        st, ws = by.get((th, 0)), by.get((th, 1))
        ratio = (ws / st) if (st and ws) else float("nan")
        print(f"    {str(th):>8} | {(st or 0):12.0f} | {(ws or 0):12.0f} | {ratio:9.3f}")


def _stage_pyd():
    """Stage the locally-built battery_native module so the scorer's LBR backend imports. Returns a dir
    containing battery_native.pyd (Windows) / .so (Linux), or None if no local build exists. Honors an
    existing PYD_DIR env (if the user already has one staged)."""
    import shutil
    if os.environ.get("PYD_DIR") and any(
            os.path.exists(os.path.join(os.environ["PYD_DIR"], f"battery_native{e}")) for e in (".pyd", ".so")):
        return os.environ["PYD_DIR"]
    rel = os.path.join(ROOT, "tools", "battery_native", "target", "release")
    src = next((os.path.join(rel, f"battery_native{e}") for e in (".dll", ".so")
                if os.path.exists(os.path.join(rel, f"battery_native{e}"))), None)
    if not src:
        log("WARN no local battery_native build found under target/release -- scorer LBR may fail "
            "(set PYD_DIR to a staged module)")
        return None
    pyd_dir = os.path.join(ROOT, "artifacts", "_pyd")
    os.makedirs(pyd_dir, exist_ok=True)
    ext = ".so" if src.endswith(".so") else ".pyd"
    shutil.copy(src, os.path.join(pyd_dir, f"battery_native{ext}"))
    log(f"staged LBR backend {src} -> {pyd_dir}/battery_native{ext}")
    return pyd_dir


def run_scorer():
    log("running LOCAL paired-LBR curve scorer (the exploitability-vs-compute verdict)...")
    env = dict(os.environ)
    pyd_dir = _stage_pyd()
    if pyd_dir:
        env["PYD_DIR"] = pyd_dir
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "score_dcfr_curve_lbr.py"), OUTDIR],
                       env=env, capture_output=True, text=True)
    out = p.stdout + "\n" + p.stderr
    open(os.path.join(OUTDIR, "curve_scorer_stdout.txt"), "w").write(out)
    for ln in out.splitlines():
        print("   " + ln, flush=True)
    log(f"scorer rc={p.returncode}; verdict json -> {OUTDIR}/dcfr_curve_verdict.json")


def main():
    log(f"watching s3://{BUCKET}/{PREFIX}  instance={IID}")
    pulled = set()
    scored = False
    t0 = time.time()
    while time.time() - t0 < 200 * 60:
        files = listing()
        ncp = sum(1 for n in files if n.endswith(".pkl"))
        log(f"artifacts: {len(files)} ({ncp} checkpoints)  inst={inst_state()}")
        pull_new(files, pulled)
        done = ("CURVE_TRAIN_DONE" in files or "PROBE_DONE" in files
                or inst_state() in ("terminated", "shutting-down", "gone"))
        if done and not scored:
            log("training finished -- final pull + score")
            pull_new(listing(), pulled)
            summarize_ws_probe()
            run_scorer()
            scored = True
            break
        time.sleep(45)
    if not scored:
        log("watch window elapsed without a DONE marker -- pulling what exists + scoring best-effort")
        pull_new(listing(), pulled)
        summarize_ws_probe()
        run_scorer()
    log("watch done")


if __name__ == "__main__":
    main()
