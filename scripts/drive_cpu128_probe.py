#!/usr/bin/env python3
"""drive_cpu128_probe.py — the Python driver for the instrumented 128-vCPU native-CFR profiling probe.

Heredoc'd / copied onto the box by scripts/cpu128_probe_onbox.sh (this reviewable repo copy is the single
source of truth — the onbox script prefers it). For ONE shape it:

  1. verifies the PRE-UPLOADED 152MB frozen blob against its sha256 sidecar (344ab2f...) before trusting
     emd_stamp (emd_stamp only reads a header, not the whole 152MB),
  2. constructs NlheCFRTrainer(160, [], NDEALS, 6, -inf, False, frozen_path) and asserts the emd_stamp
     carries 'k=160' + 'n_runouts=400' (guards a wrong table),
  3. WARMUPs to steady (n_infosets growth <1% over 100 iters AND total_ms drift <3%, or WARMUP_CAP),
  4. runs WINDOW A — a CLEAN throughput window (NO perf attached; the shell attaches only mpstat+numastat)
     -> the headline deals/s — then WINDOW B — a perf-attached window for the phase-share read,
  5. STREAMS each train_profile_json(profile_every) batch to disk with flush()+fsync() AND s3.put_object,
     so a kill loses < profile_every iters and never a completed window,
  6. writes a derived <SHAPE>_summary.json and the STEADY_BEGIN / STEADY_BEGIN_PERF / WINDOW_DONE sentinels
     the shell polls.

The ENTIRE measure loop is in try/finally; a SIGTERM handler RAISES so the finally still flushes+fsyncs+
uploads collected rows (the 4th of the 5 data-loss guards; the spot trap sweep is the 5th).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import sys
import time
import traceback

import boto3
import battery_native as bn  # the cdylib built on-box (libbattery_native.so -> /opt/pokerbot)

FROZEN_PATH = "/opt/pokerbot/frozen.emdfrzn"
FROZEN_SHA_SIDECAR = "/opt/pokerbot/frozen.emdfrzn.sha256"
EXPECTED_FROZEN_SHA256 = "344ab2fd328f4b360fd2c11a09c1f6fd2c45a56e62f75d6bc1a502c71612cb74"
PROBE_DIR = "/var/log/probe"

BUCKET = os.environ.get("BUCKET", "")
RUN_PREFIX = os.environ.get("RUN_PREFIX", "")
REGION = os.environ.get("REGION", "eu-central-1")
NUM_BUCKETS = int(os.environ.get("NUM_BUCKETS", "160"))

_s3 = None


def s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=REGION)
    return _s3


def _put(local_path: str, key_suffix: str) -> None:
    """Upload PROBE_DIR/<file> to s3://BUCKET/RUN_PREFIX/probe/<key_suffix> (best-effort; never fatal)."""
    if not BUCKET or not RUN_PREFIX:
        return
    try:
        s3().upload_file(local_path, BUCKET, f"{RUN_PREFIX}/probe/{key_suffix}")
    except Exception as e:  # noqa: BLE001
        print(f"WARN s3 put {key_suffix}: {e}", flush=True)


def _sentinel(name: str) -> None:
    """Write+fsync a one-line sentinel the shell polls (epoch ms), then upload it."""
    path = os.path.join(PROBE_DIR, name)
    with open(path, "w") as f:
        f.write(f"{int(time.time() * 1000)}\n")
        f.flush()
        os.fsync(f.fileno())
    _put(path, name)


def verify_frozen() -> None:
    """ABORT if the sidecar is missing or the blob's sha256 does not match (header-only emd_stamp can't
    catch a corrupt body). The full-152MB sha256 is cheap relative to the run."""
    if not os.path.exists(FROZEN_SHA_SIDECAR):
        raise SystemExit("FROZEN_VERIFY: sidecar /opt/pokerbot/frozen.emdfrzn.sha256 missing — ABORT")
    sidecar = open(FROZEN_SHA_SIDECAR).read().strip().split()[0]
    if sidecar != EXPECTED_FROZEN_SHA256:
        raise SystemExit(f"FROZEN_VERIFY: sidecar {sidecar} != expected {EXPECTED_FROZEN_SHA256} — ABORT")
    h = hashlib.sha256()
    with open(FROZEN_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != EXPECTED_FROZEN_SHA256:
        raise SystemExit(f"FROZEN_VERIFY: blob sha {got} != sidecar {EXPECTED_FROZEN_SHA256} — ABORT")
    print(f"FROZEN_OK sha256={got}", flush=True)


def make_trainer(ndeals: int):
    # signature: (num_buckets, depth_edges, n_deals, n_seats=6, prune=-inf, fast_merge=False, emd_frozen_path)
    t = bn.NlheCFRTrainer(NUM_BUCKETS, [], ndeals, 6, float("-inf"), False, FROZEN_PATH)
    stamp = t.emd_stamp()
    print(f"emd_stamp={stamp}", flush=True)
    if "k=160" not in stamp or "n_runouts=400" not in stamp:
        raise SystemExit(f"EMD_STAMP guard FAILED (expected k=160 + n_runouts=400): {stamp} — ABORT")
    return t


def warmup_to_steady(t, warmup_cap: int) -> None:
    """Train in batches of 50 until n_infosets growth over the last 100 iters < 1% AND rolling-mean
    total_ms drift < 3%, OR WARMUP_CAP. The driver's own steady-detect is primary; the cap is a backstop."""
    BATCH = 50
    iters = 0
    infos_hist = []   # (iters, n_infosets)
    ms_hist = []      # rolling total_ms means per batch
    while iters < warmup_cap:
        prof = json.loads(t.train_profile_json(BATCH))
        iters += BATCH
        n_inf = t.n_infosets()
        batch_total_ms = statistics.fmean(r["total_ms"] for r in prof) if prof else float("nan")
        infos_hist.append((iters, n_inf))
        ms_hist.append(batch_total_ms)
        # n_infosets growth over the last ~100 iters (2 batches back).
        growth = 1.0
        if len(infos_hist) >= 3:
            prev = infos_hist[-3][1]
            growth = (n_inf - prev) / prev if prev > 0 else 1.0
        ms_drift = 1.0
        if len(ms_hist) >= 3 and ms_hist[-3] > 0:
            ms_drift = abs(ms_hist[-1] - ms_hist[-3]) / ms_hist[-3]
        print(f"WARMUP iters={iters} n_infosets={n_inf} growth={growth:.4f} "
              f"batch_total_ms={batch_total_ms:.2f} ms_drift={ms_drift:.4f}", flush=True)
        if len(infos_hist) >= 3 and growth < 0.01 and ms_drift < 0.03:
            print(f"STEADY reached at iters={iters} (growth={growth:.4f} ms_drift={ms_drift:.4f})", flush=True)
            return
    print(f"STEADY by WARMUP_CAP={warmup_cap}", flush=True)


def run_window(t, shape: str, window: int, profile_every: int, jsonl_name: str, label: str) -> list[dict]:
    """Stream `window` iters in batches of `profile_every`; append+flush+fsync+put_object each batch.
    Returns the parsed rows. Any exception/SIGTERM still leaves the flushed rows on disk + S3."""
    path = os.path.join(PROBE_DIR, jsonl_name)
    rows: list[dict] = []
    done = 0
    with open(path, "w") as f:
        while done < window:
            step = min(profile_every, window - done)
            batch = json.loads(t.train_profile_json(step))
            for r in batch:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
            rows.extend(batch)
            done += step
            _put(path, f"{shape}/{jsonl_name}")  # partial put — worst-case kill loses < profile_every iters
            if batch:
                print(f"{label} {shape} iters+={step}/{window} last_total_ms={batch[-1]['total_ms']:.3f}",
                      flush=True)
    return rows


def _pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _merge_mpstat_cpu(shape: str) -> dict:
    """Parse the shell's <SHAPE>_mpstat.txt (mpstat -P ALL 1 30) for box-wide CPU%: max-over-1s and
    mean-over-window, both steal-incl and steal-excl. The 'all' rows carry %usr/%sys/%idle/%steal; CPU% =
    100 - %idle (steal-incl) and 100 - %idle - %steal (steal-excl). Best-effort; absent => NaNs."""
    path = os.path.join(PROBE_DIR, f"{shape}_mpstat.txt")
    incl, excl = [], []
    try:
        hdr_idx = {}
        for line in open(path):
            parts = line.split()
            if not parts:
                continue
            if "%idle" in parts and "CPU" in parts:
                hdr_idx = {name: i for i, name in enumerate(parts)}
                continue
            if not hdr_idx:
                continue
            # the aggregate row labels the CPU column 'all'
            cpu_col = hdr_idx.get("CPU")
            if cpu_col is None or cpu_col >= len(parts) or parts[cpu_col] != "all":
                continue
            try:
                idle = float(parts[hdr_idx["%idle"]])
                steal = float(parts[hdr_idx.get("%steal", -1)]) if "%steal" in hdr_idx else 0.0
            except (ValueError, KeyError, IndexError):
                continue
            incl.append(100.0 - idle)
            excl.append(100.0 - idle - steal)
    except FileNotFoundError:
        pass
    return {
        "cpu_pct_max_1s": max(incl) if incl else float("nan"),
        "cpu_pct_mean": statistics.fmean(incl) if incl else float("nan"),
        "cpu_pct_max_1s_steal_excl": max(excl) if excl else float("nan"),
        "cpu_pct_mean_steal_excl": statistics.fmean(excl) if excl else float("nan"),
        "mpstat_samples": len(incl),
    }


def summarize(shape: str, ndeals: int, rows_clean: list[dict], rows_perf: list[dict]) -> dict:
    """Derive the verdict summary. Drop the first 20 rows of each window as cache-transient. deals/s comes
    from the CLEAN window A; phase shares from the perf window B (or clean if B empty)."""
    clean = rows_clean[20:] if len(rows_clean) > 20 else rows_clean
    perf = rows_perf[20:] if len(rows_perf) > 20 else (rows_perf or clean)

    def col(rs, k):
        return [r[k] for r in rs if k in r]

    total_clean = col(clean, "total_ms")
    deals_per_s_clean = (ndeals * 1000.0 / statistics.fmean(total_clean)) if total_clean else float("nan")

    trav = col(perf, "traverse_ms")
    merge = col(perf, "merge_ms")
    floor = col(perf, "floor_ms")
    total = col(perf, "total_ms")
    nodes = col(perf, "nodes")
    mean_total = statistics.fmean(total) if total else float("nan")
    nodes_per_s = (statistics.fmean(nodes) * 1000.0 / mean_total) if (nodes and mean_total) else float("nan")

    fp_calls = col(perf, "fingerprint_calls")
    fp_new = col(perf, "fp_memo_new_entries")
    fp_memo_miss_rate = (sum(fp_new) / sum(fp_calls)) if (fp_calls and sum(fp_calls)) else float("nan")
    emd_q = col(perf, "emd_bucket_queries")
    emd_hits = col(perf, "emd_bucket_hits")
    emd_hit_rate = (sum(emd_hits) / sum(emd_q)) if (emd_q and sum(emd_q)) else float("nan")
    rayon_realized = perf[-1]["rayon_threads"] if perf else (clean[-1]["rayon_threads"] if clean else None)
    max_depth = max(col(perf, "max_depth") or [0])

    def share(part):
        return (statistics.fmean(part) / mean_total) if (part and mean_total) else float("nan")

    trav_p95 = _pct(trav, 0.95)
    trav_p50 = _pct(trav, 0.50)

    out = {
        "shape": shape,
        "n_deals": ndeals,
        "clean_rows": len(clean),
        "perf_rows": len(perf),
        "traverse_ms_mean": statistics.fmean(trav) if trav else float("nan"),
        "traverse_ms_p50": trav_p50,
        "traverse_ms_p95": trav_p95,
        "traverse_p95_over_p50": (trav_p95 / trav_p50) if (trav_p50 and trav_p50 == trav_p50 and trav_p50 != 0) else float("nan"),
        "merge_ms_mean": statistics.fmean(merge) if merge else float("nan"),
        "merge_ms_p50": _pct(merge, 0.50),
        "merge_ms_p95": _pct(merge, 0.95),
        "floor_ms_mean": statistics.fmean(floor) if floor else float("nan"),
        "floor_ms_p50": _pct(floor, 0.50),
        "floor_ms_p95": _pct(floor, 0.95),
        "total_ms_mean": mean_total,
        "total_ms_p50": _pct(total, 0.50),
        "total_ms_p95": _pct(total, 0.95),
        "merge_share": share(merge),
        "floor_share": share(floor),
        "deals_per_s_clean": deals_per_s_clean,
        "nodes_per_s": nodes_per_s,
        "fp_memo_miss_rate": fp_memo_miss_rate,
        "emd_bucket_hit_rate": emd_hit_rate,
        "rayon_threads_realized": rayon_realized,
        "max_depth": max_depth,
    }
    out.update(_merge_mpstat_cpu(shape))  # CPU% (max-1s + mean, steal-incl + steal-excl) from the shell's mpstat
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--ndeals", type=int, required=True)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--profile-every", type=int, default=20)
    ap.add_argument("--warmup-cap", type=int, default=800)
    args = ap.parse_args()
    shape = args.shape

    # SIGTERM -> raise so the finally still flushes+fsyncs+uploads (data-loss guard #4).
    def _on_term(*_):
        raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM, _on_term)

    os.makedirs(PROBE_DIR, exist_ok=True)
    verify_frozen()
    t = make_trainer(args.ndeals)

    rows_clean: list[dict] = []
    rows_perf: list[dict] = []
    try:
        warmup_to_steady(t, args.warmup_cap)

        # WINDOW A: CLEAN throughput (no perf; shell attaches mpstat+numastat only after STEADY_BEGIN).
        _sentinel(f"{shape}.STEADY_BEGIN")
        rows_clean = run_window(t, shape, args.window, args.profile_every,
                                f"{shape}_profileA.jsonl", "WINDOW_A")

        # Estimate window-secs so the shell can size its `timeout` captures.
        a = rows_clean[20:] if len(rows_clean) > 20 else rows_clean
        mean_total = statistics.fmean(r["total_ms"] for r in a) if a else 50.0
        est_secs = max(10, min(90, int(mean_total * args.window / 1000.0)))
        print(f"EST_WINDOW_SECS={est_secs}", flush=True)

        # WINDOW B: PERF attached (shell attaches perf record/stat after STEADY_BEGIN_PERF).
        _sentinel(f"{shape}.STEADY_BEGIN_PERF")
        rows_perf = run_window(t, shape, args.window, args.profile_every,
                               f"{shape}_profileB.jsonl", "WINDOW_B")
    except KeyboardInterrupt:
        print(f"{shape}: SIGTERM/interrupt — flushing collected rows in finally", flush=True)
    except Exception:
        traceback.print_exc()
    finally:
        # Always: derive + write + fsync + upload the summary from whatever we collected, and signal done.
        try:
            summ = summarize(shape, args.ndeals, rows_clean, rows_perf)
            spath = os.path.join(PROBE_DIR, f"{shape}_summary.json")
            with open(spath, "w") as f:
                json.dump(summ, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            _put(spath, f"{shape}/{shape}_summary.json")
            # Re-upload final jsonls (belt-and-suspenders over the per-batch puts).
            for jn in (f"{shape}_profileA.jsonl", f"{shape}_profileB.jsonl"):
                jp = os.path.join(PROBE_DIR, jn)
                if os.path.exists(jp):
                    _put(jp, f"{shape}/{jn}")
            print(f"SUMMARY {shape}: deals/s={summ.get('deals_per_s_clean')} "
                  f"merge_share={summ.get('merge_share')} rayon={summ.get('rayon_threads_realized')} "
                  f"cpu%_mean={summ.get('cpu_pct_mean')}", flush=True)
        except Exception:
            traceback.print_exc()
        _sentinel(f"{shape}.WINDOW_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
