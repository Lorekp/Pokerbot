"""Watch the hogwild cert box: pull the two QUALITY blueprints when they land, run the paired LBR LOCALLY
(JAX is local), and pull the THROUGHPUT crossover when it lands. Progress-logged; stops at PROBE_DONE or
instance termination. Cost-safety termination is the separate _probe_watch.py poller — this is results-only.
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
RUN_SUBDIR = sys.argv[1] if len(sys.argv) > 1 else "hogthru_1782747943"
IID = sys.argv[2] if len(sys.argv) > 2 else "i-0a29bc3c610419186"
PREFIX = f"pokerbot/prod6000/native_cfr/{RUN_SUBDIR}/hogwild/"
QDIR = os.path.join(ROOT, "quality_ab")
OUTDIR = os.path.join(ROOT, "artifacts", RUN_SUBDIR)
os.makedirs(OUTDIR, exist_ok=True)

s3 = boto3.client("s3", region_name=REGION, verify=False)
ec2 = boto3.client("ec2", region_name=REGION, verify=False)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def listing():
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
    return {os.path.basename(o["Key"]): (o["Key"], o["Size"]) for o in r.get("Contents", [])}


def download(key, dst):
    s3.download_file(BUCKET, key, dst)
    return dst


def inst_state():
    try:
        r = ec2.describe_instances(InstanceIds=[IID])
        return r["Reservations"][0]["Instances"][0]["State"]["Name"]
    except Exception:
        return "gone"


def run_local_lbr(fork_pkl, hog_pkl):
    env = dict(os.environ)
    env.update(QAB_LOAD_FORK=fork_pkl, QAB_LOAD_HOG=hog_pkl,
               QAB_NUM_BUCKETS="200", QAB_HANDS="8000", QAB_SEEDS="16")
    log("running LOCAL paired LBR on the downloaded blueprints (this is the correctness verdict)...")
    p = subprocess.run([sys.executable, os.path.join(QDIR, "build_quality_ab.py")],
                       env=env, capture_output=True, text=True)
    out = p.stdout + "\n" + p.stderr
    open(os.path.join(OUTDIR, "lbr_verdict.txt"), "w").write(out)
    # echo the result block
    for ln in out.splitlines():
        if any(k in ln for k in ("QUALITY A/B RESULT", "fork-merge", "hogwild", "PAIRED", "VERDICT",
                                 "budget", "calibration", "====")):
            print("   " + ln, flush=True)
    log(f"LBR verdict saved -> {OUTDIR}/lbr_verdict.txt")


def main():
    log(f"watching s3://{BUCKET}/{PREFIX}  instance={IID}")
    scored = False
    got_thru = False
    t0 = time.time()
    while time.time() - t0 < 95 * 60:
        files = listing()
        present = sorted(files.keys())
        log(f"artifacts: {present}  inst={inst_state()}")

        if not scored and "blueprint_fork.pkl" in files and "blueprint_hog.pkl" in files:
            fk = download(files["blueprint_fork.pkl"][0], os.path.join(QDIR, "blueprint_fork.pkl"))
            hk = download(files["blueprint_hog.pkl"][0], os.path.join(QDIR, "blueprint_hog.pkl"))
            log(f"downloaded blueprints ({files['blueprint_fork.pkl'][1]//1024}KB / "
                f"{files['blueprint_hog.pkl'][1]//1024}KB)")
            run_local_lbr(fk, hk)
            scored = True

        if not got_thru and "throughput_results.json" in files:
            tp = download(files["throughput_results.json"][0],
                          os.path.join(OUTDIR, "throughput_results.json"))
            try:
                d = json.load(open(tp))
                log(f"THROUGHPUT: prod-shape crossover={d.get('prod_crossover_threads')} "
                    f"envelope crossover={d.get('envelope_crossover_threads')} "
                    f"undetermined={d.get('undetermined')} missing={len(d.get('missing', []))}")
            except Exception as e:
                log(f"throughput json parse fail: {e!r}")
            got_thru = True

        if "PROBE_DONE" in files or inst_state() in ("terminated", "shutting-down", "gone"):
            log("run finished (PROBE_DONE or instance gone) — final pull")
            # one last attempt at anything still missing
            files = listing()
            if not scored and "blueprint_fork.pkl" in files and "blueprint_hog.pkl" in files:
                fk = download(files["blueprint_fork.pkl"][0], os.path.join(QDIR, "blueprint_fork.pkl"))
                hk = download(files["blueprint_hog.pkl"][0], os.path.join(QDIR, "blueprint_hog.pkl"))
                run_local_lbr(fk, hk)
            if not got_thru and "throughput_results.json" in files:
                download(files["throughput_results.json"][0],
                         os.path.join(OUTDIR, "throughput_results.json"))
            break
        if scored and got_thru:
            log("both verdicts in — stopping watch")
            break
        time.sleep(45)
    log("watch done")


if __name__ == "__main__":
    main()
