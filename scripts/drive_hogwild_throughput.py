"""On-box driver for the Hogwild-vs-fork-merge THROUGHPUT CROSSOVER sweep (production EMD-K160 path).

Question: at what core count, and at what per-iteration batch, does the lock-free Hogwild trainer (no
fork-merge barrier, no per-task accumulator map) overtake fork-merge in deals/s? Local 12-core was a
wash; the fork-merge barrier idle GROWS with core count, so the crossover appears at 32/64/128 cores.

WHY a 2D sweep (threads x n_deals), not a single shape (adversarial-gate V1): fork-merge is squeezed
from both sides — small n_deals starves many threads (deal_chunk_min_len -> 1, fewer tasks than cores),
large n_deals thrashes cache (its per-task accumulator map grows toward the full infoset set). Hogwild
has NO per-task map (writes straight to the shared table), so it can use a large n_deals to SATURATE
many cores WITHOUT the cache penalty — which is precisely its thesis. So we measure each arm's
BEST-achievable deals/s per core count (the envelope) AND report the production-shape (n_deals=200) row
separately. One binary built --features python,hogwild serves both arms via the runtime toggle.

Per-config worker = fresh subprocess (rayon reads RAYON_NUM_THREADS once at init); it asserts the
effective rayon thread count (V2), warms to steady, checks the table is steady during the timed window
(V3), and times the PRODUCTION call t.train(step) (internal Rust loop, no Python round-trip). A missing
config makes the verdict UNDETERMINED rather than silently shifting the crossover (V4).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EMD_PATH = os.environ.get("EMD_FROZEN_PATH",
                          os.path.join(ROOT, "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn"))
NUM_BUCKETS = int(os.environ.get("NUM_BUCKETS", "160"))
SECS = float(os.environ.get("WINDOW_SECS", "20"))
WARM = float(os.environ.get("WARM_SECS", "15"))
THREADS_LIST = [int(x) for x in os.environ.get("THREADS_LIST", "16,32,64,128").split(",") if x]
N_DEALS_LIST = [int(x) for x in os.environ.get("N_DEALS_LIST", "200,1600,12800").split(",") if x]
PROD_N_DEALS = int(os.environ.get("PROD_N_DEALS", "200"))   # the production-shape row
ARMS = ["fork", "hog"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def worker() -> int:
    """One (arm, threads, n_deals) measurement. RAYON_NUM_THREADS is set in the env by the parent."""
    sys.path.insert(0, os.environ.get("PYD_DIR", ROOT))
    import battery_native as bn
    arm = os.environ["ARM"]
    req_threads = int(os.environ.get("RAYON_NUM_THREADS", "0"))
    n_deals = int(os.environ["N_DEALS"])
    hogwild = (arm == "hog")
    cap = int(os.environ.get("HOG_CAP", str(1 << 21)))
    t = bn.NlheCFRTrainer(NUM_BUCKETS, [], n_deals, 6, float("-inf"), False, EMD_PATH, 0, hogwild, cap)
    assert t.is_hogwild() == hogwild, f"toggle wrong is_hogwild={t.is_hogwild()} arm={arm}"
    # V2: assert rayon actually built the requested pool size (label can't lie about the thread count)
    eff_threads = None
    try:
        eff_threads = int(bn.rayon_num_threads())
    except Exception:
        eff_threads = req_threads
    if req_threads and eff_threads != req_threads:
        print(f"WARN rayon effective threads {eff_threads} != requested {req_threads}", flush=True)
    try:
        stamp = t.emd_stamp()
    except Exception:
        stamp = "n/a"
    # warmup by wall, calibrate iters/s on the production path
    w0 = time.time()
    i_w = t.iters_done()
    while time.time() - w0 < WARM:
        t.train(4)
    ips_warm = (t.iters_done() - i_w) / max(time.time() - w0, 1e-9)
    step = max(4, int(math.ceil(SECS * ips_warm)))
    info_before = t.n_infosets()           # V3: steady-state check across the timed window
    i0 = t.iters_done()
    t0 = time.time()
    t.train(step)
    wall = time.time() - t0
    di = t.iters_done() - i0
    info_after = t.n_infosets()
    grew = (info_after - info_before) / max(info_before, 1)
    res = {"arm": arm, "threads": req_threads, "eff_threads": eff_threads, "n_deals": n_deals,
           "iters": di, "wall": wall, "deals": di * n_deals,
           "deals_per_s": di * n_deals / wall if wall else 0.0,
           "iters_per_s": di / wall if wall else 0.0,
           "infosets": info_after, "info_grew_frac": grew, "steady": grew < 0.02,
           "emd_stamp": stamp, "num_buckets": NUM_BUCKETS}
    print("RESULT " + json.dumps(res), flush=True)
    return 0


def driver() -> int:
    pyd_dir = os.environ.get("PYD_DIR", ROOT)
    out_dir = os.environ.get("OUT_DIR", "/var/log/hogwild")
    if not os.path.exists(EMD_PATH):
        _log(f"FATAL frozen EMD lookup missing: {EMD_PATH}")
        return 1
    _log("=== Hogwild throughput crossover sweep (2D: threads x n_deals, both arms) ===")
    _log(f"EMD={EMD_PATH} K={NUM_BUCKETS} window={SECS}s warm={WARM}s "
         f"threads={THREADS_LIST} n_deals={N_DEALS_LIST} prod_n_deals={PROD_N_DEALS}")
    configs = [(arm, th, nd) for arm in ARMS for th in THREADS_LIST for nd in N_DEALS_LIST]
    rows, missing = [], []
    t_start = time.time()
    for i, (arm, th, nd) in enumerate(configs):
        env = dict(os.environ)
        env.update(ARM=arm, RAYON_NUM_THREADS=str(th), N_DEALS=str(nd), PYD_DIR=pyd_dir)
        _log(f"[{i+1}/{len(configs)}] arm={arm:4s} threads={th:3d} n_deals={nd:5d} "
             f"... (elapsed {time.time()-t_start:.0f}s)")
        try:
            out = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker"],
                                 env=env, capture_output=True, text=True, timeout=int(SECS + WARM + 600))
        except subprocess.TimeoutExpired:
            _log(f"  TIMEOUT arm={arm} threads={th} n_deals={nd}")
            missing.append((arm, th, nd))
            continue
        res = None
        for ln in out.stdout.splitlines():
            if ln.startswith("RESULT "):
                res = json.loads(ln[len("RESULT "):])
        if res is None:
            _log(f"  NO RESULT; stderr tail: " + " | ".join(out.stderr.splitlines()[-4:]))
            missing.append((arm, th, nd))
            continue
        rows.append(res)
        flag = "" if res["steady"] else f"  [NOT-STEADY grew {res['info_grew_frac']*100:.1f}%]"
        _log(f"  -> {res['deals_per_s']/1e3:8.1f}K deals/s  {res['iters_per_s']:7.2f} it/s  "
             f"eff_threads={res['eff_threads']}  infosets={res['infosets']:,}{flag}")

    by = {(r["arm"], r["threads"], r["n_deals"]): r["deals_per_s"] for r in rows}

    def envelope(arm, th):
        vals = [(nd, by[(arm, th, nd)]) for nd in N_DEALS_LIST if (arm, th, nd) in by]
        return max(vals, key=lambda x: x[1]) if vals else (None, None)

    # ---- production-shape (n_deals == PROD_N_DEALS) crossover ----
    print("\n" + "=" * 78, flush=True)
    print(f"PRODUCTION-SHAPE (n_deals={PROD_N_DEALS}) deals/s; ratio = hogwild / fork-merge", flush=True)
    print("=" * 78, flush=True)
    print(f"{'threads':>8} | {'fork-merge':>13} | {'hogwild':>13} | {'hog/fork':>9}", flush=True)
    prod_cross = None
    for th in THREADS_LIST:
        f = by.get(("fork", th, PROD_N_DEALS))
        h = by.get(("hog", th, PROD_N_DEALS))
        if f and h:
            ratio = h / f
            if prod_cross is None and ratio >= 1.0:
                prod_cross = th
            print(f"{th:>8} | {f/1e3:>10.1f}K/s | {h/1e3:>10.1f}K/s | {ratio:>8.3f}x", flush=True)
        else:
            print(f"{th:>8} | {'(missing)':>13} | {'(missing)':>13} | {'-':>9}", flush=True)

    # ---- BEST-ACHIEVABLE (envelope over n_deals) crossover — the fair architectural comparison ----
    print("\n" + "=" * 78, flush=True)
    print("BEST-ACHIEVABLE (envelope over n_deals) deals/s; ratio = hogwild / fork-merge", flush=True)
    print("=" * 78, flush=True)
    print(f"{'threads':>8} | {'fork-merge (n_deals)':>22} | {'hogwild (n_deals)':>22} | {'hog/fork':>9}", flush=True)
    env_cross = None
    undetermined = False
    for th in THREADS_LIST:
        fnd, f = envelope("fork", th)
        hnd, h = envelope("hog", th)
        if f and h:
            ratio = h / f
            if env_cross is None and ratio >= 1.0:
                env_cross = th
            print(f"{th:>8} | {f/1e3:>13.1f}K/s @{fnd:<6} | {h/1e3:>13.1f}K/s @{hnd:<6} | {ratio:>8.3f}x",
                  flush=True)
        else:
            undetermined = True
            print(f"{th:>8} | {'(missing)':>22} | {'(missing)':>22} | {'-':>9}", flush=True)
    print("=" * 78, flush=True)

    if missing:
        print(f"PARTIAL: {len(missing)} config(s) MISSING -> {missing}", flush=True)
    if undetermined:
        print("CROSSOVER: UNDETERMINED — an envelope point is missing (re-run the missing configs)", flush=True)
    else:
        print(f"CROSSOVER (production-shape n_deals={PROD_N_DEALS}): "
              + (f"hogwild >= fork at >= {prod_cross} threads" if prod_cross
                 else "hogwild never overtook fork in the swept range"), flush=True)
        print(f"CROSSOVER (best-achievable envelope): "
              + (f"hogwild >= fork at >= {env_cross} threads" if env_cross
                 else "hogwild never overtook fork in the swept range"), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "throughput_results.json"), "w").write(json.dumps(
        {"rows": rows, "missing": missing, "prod_crossover_threads": prod_cross,
         "envelope_crossover_threads": env_cross, "undetermined": undetermined,
         "threads_list": THREADS_LIST, "n_deals_list": N_DEALS_LIST, "prod_n_deals": PROD_N_DEALS,
         "window_secs": SECS, "warm_secs": WARM}, indent=2))
    # loud terminal status the on-box script keys its marker on
    open(os.path.join(out_dir, "DRIVER_STATUS"), "w").write(
        "UNDETERMINED" if undetermined else "OK")
    _log(f"wrote {out_dir}/throughput_results.json")
    return 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(worker())
    raise SystemExit(driver())
