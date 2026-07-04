#!/usr/bin/env python3
"""blueprint-v4 Stage-0 arm runner (artifacts/blueprint_pluribus_grade_design.md §7.1.2).

Sequences the six mini-train arms on the local box (never in parallel — each arm uses all cores),
driving tools/battery_native ab_train.exe purely via env. Kill-safe: every arm checkpoints at
5k/10k/20k/30k/35k/40k/45k and resumes from its latest iter_*.nlcfr (POKERBOT_AB_RESUME=1), and
this runner records per-arm status in artifacts/bpv4_stage0_state.json — a restarted runner (or a
resumed agent) picks up where the last one died.

Standing job-logging rule: heartbeats (arm, iters, %, ETA, strat rows, RSS) are appended to
artifacts/bpv4_stage0_progress.log. RSS guard (§7.1.2): if an arm's trainer process exceeds 60% of
box RAM, the arm is STOPPED at its last checkpoint (status rss_stopped) and gates are evaluated
there — partial evidence beats OOM. Stall guard: no trainer output for >30 min -> kill + report.

Arms (all: n_deals=200, seed 0xC0FFEE baked in ab_train, 45k iters):
  S200   summary keying, made-hand K200,  CFR+   (E5 arm-A validity anchor @20k: BTN ~8.5%)
  H200   FULLHIST,       made-hand K200,  CFR+
  S1000  summary,        made-hand K1000, CFR+
  H1000  FULLHIST,       made-hand K1000, CFR+
  HD200  FULLHIST,       made-hand K200,  DCFR(1.5,0,2)+RBP (-300k bb, explore 0.05)
  HE160  FULLHIST,       EMD-K160 + river-OCHS, CFR+  (stretch, overnight)

Usage:
  python scripts/blueprint2/bpv4_stage0_run.py [--arms S200,H200,...] [--iters 45000]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
ART = os.path.join(_ROOT, "artifacts")
RUNS = os.path.join(_ROOT, "runs", "native_cfr", "bpv4_stage0")
PROGRESS_LOG = os.path.join(ART, "bpv4_stage0_progress.log")
STATE_FILE = os.path.join(ART, "bpv4_stage0_state.json")
EXE = os.path.join(_ROOT, "tools", "battery_native", "target", "release", "examples",
                   "ab_train.exe")
# Early points (1k/2500) added after H200's uniform-phase row explosion (55.8M rows @1.4k iters,
# 16.35GiB transient RSS): if the 60%-RAM guard stops a full-hist arm early, the run must still
# leave evidence at SOME checkpoint (design §7.1.2: partial evidence beats OOM).
CHECKPOINTS = "1000,2500,5000,10000,15000,20000,25000,30000,35000,40000,45000"
EMD_FROZEN = os.path.join(_ROOT, "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")
RIVER_OCHS = os.path.join(_ROOT, "data", "abstraction", "river_ochs_160.rochs")

ARMS: dict[str, dict[str, str]] = {
    "S200": {},
    "H200": {"POKERBOT_FULLHIST_KEY": "1"},
    "S1000": {"POKERBOT_AB_NUM_BUCKETS": "1000"},
    "H1000": {"POKERBOT_FULLHIST_KEY": "1", "POKERBOT_AB_NUM_BUCKETS": "1000"},
    "HD200": {"POKERBOT_FULLHIST_KEY": "1", "POKERBOT_AB_DCFR": "1"},
    "HE160": {"POKERBOT_FULLHIST_KEY": "1", "POKERBOT_AB_NUM_BUCKETS": "160",
              "POKERBOT_AB_EMD_FROZEN": EMD_FROZEN, "POKERBOT_AB_RIVER_OCHS": RIVER_OCHS},
}
DEFAULT_ORDER = ["S200", "H200", "S1000", "H1000", "HD200", "HE160"]

RSS_FRAC_LIMIT = 0.60          # of total box RAM (design §7.1.2 RSS guard)
STALL_S = 30 * 60              # kill-and-report on no trainer output for 30 min
HEARTBEAT_S = 60.0             # progress-log cadence


def plog(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"arms": {}}


def save_state(state: dict) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE_FILE)


def run_arm(arm: str, iters: int, state: dict) -> str:
    import psutil

    total_ram = psutil.virtual_memory().total
    rss_limit = int(total_ram * RSS_FRAC_LIMIT)
    arm_dir = os.path.join(RUNS, arm)
    os.makedirs(arm_dir, exist_ok=True)
    out_path = os.path.join(arm_dir, "final.nlcfr")
    if os.path.exists(out_path):
        plog(f"{arm}: final.nlcfr already present -> done (skip)")
        return "done"

    env = dict(os.environ)
    env.pop("POKERBOT_FULLHIST_KEY", None)
    env.pop("POKERBOT_PREFLOP_HIST_KEY", None)
    env.update(ARMS[arm])
    env["POKERBOT_AB_CHECKPOINTS"] = CHECKPOINTS
    env["POKERBOT_AB_RESUME"] = "1"
    env["POKERBOT_WORKSTEAL"] = "1"
    # never RAYON_NUM_THREADS=1 (standing rule); leave rayon at its default (all cores).
    env.pop("RAYON_NUM_THREADS", None)

    plog(f"{arm}: START iters={iters} env={ARMS[arm]} dir={arm_dir}")
    proc = subprocess.Popen(
        [EXE, out_path, str(iters), "200"],
        cwd=_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1,
    )
    ps = psutil.Process(proc.pid)
    last_line_t = time.time()
    last_hb = 0.0
    last_trainer_line = ""
    rss_peak = 0
    verdict = "done"

    import threading
    lines: list[str] = []
    lock = threading.Lock()

    def reader():
        for line in proc.stdout:  # type: ignore[union-attr]
            with lock:
                lines.append(line.rstrip())

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    while proc.poll() is None:
        time.sleep(2.0)
        with lock:
            new = lines[:]
            lines.clear()
        if new:
            last_line_t = time.time()
            last_trainer_line = new[-1]
            for ln in new:
                if "CHECKPOINT" in ln or "DONE" in ln or "RESUMED" in ln:
                    plog(f"{arm}: {ln}")
        try:
            rss = ps.memory_info().rss
            rss_peak = max(rss_peak, rss)
        except psutil.Error:
            rss = 0
        now = time.time()
        if now - last_hb >= HEARTBEAT_S:
            last_hb = now
            plog(f"{arm}: hb rss={rss/2**30:.2f}GiB peak={rss_peak/2**30:.2f}GiB "
                 f"last='{last_trainer_line[-140:]}'")
            state["arms"].setdefault(arm, {})
            state["arms"][arm].update({"status": "running", "rss_gib": round(rss / 2**30, 2),
                                       "rss_peak_gib": round(rss_peak / 2**30, 2),
                                       "last_line": last_trainer_line[-200:]})
            save_state(state)
        if rss > rss_limit:
            plog(f"{arm}: RSS GUARD FIRED rss={rss/2**30:.2f}GiB > "
                 f"{rss_limit/2**30:.2f}GiB (60% of box) -> stopping arm at last checkpoint")
            proc.kill()
            verdict = "rss_stopped"
            break
        if now - last_line_t > STALL_S:
            plog(f"{arm}: STALL >30min without trainer output -> kill and report "
                 f"(last='{last_trainer_line[-140:]}')")
            proc.kill()
            verdict = "stalled"
            break
    proc.wait()
    with lock:
        for ln in lines:
            if "CHECKPOINT" in ln or "DONE" in ln:
                plog(f"{arm}: {ln}")
    if verdict == "done" and proc.returncode != 0:
        verdict = f"failed_rc{proc.returncode}"
    plog(f"{arm}: END status={verdict} rc={proc.returncode} rss_peak={rss_peak/2**30:.2f}GiB")
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--iters", type=int, default=45000)
    args = ap.parse_args()
    if not os.path.exists(EXE):
        print(f"FATAL: {EXE} not built (cargo build --release --example ab_train)")
        return 2
    os.makedirs(ART, exist_ok=True)
    state = load_state()
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            print(f"FATAL: unknown arm {arm}")
            return 2
        st = state["arms"].get(arm, {})
        if st.get("status") in ("done", "rss_stopped"):
            plog(f"{arm}: status={st['status']} -> skip")
            continue
        verdict = run_arm(arm, args.iters, state)
        state["arms"].setdefault(arm, {})
        state["arms"][arm]["status"] = verdict
        save_state(state)
        if verdict.startswith("failed"):
            plog(f"{arm}: arm failed; continuing with next arm (evidence at last checkpoint)")
    plog(f"ALL ARMS SEQUENCED: { {a: state['arms'].get(a, {}).get('status') for a in ARMS} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
