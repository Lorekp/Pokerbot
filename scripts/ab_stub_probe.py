"""Big-box attribution STEP-2 subtractive harness (local preliminary).

Runs ONE `ab_stub`-feature cdylib (built `--features python,ab_stub`) with POKERBOT_AB_STUB set per arm, to
attribute the per-node traverse cost to each phase. For each stub the per-node-cost delta vs the unstubbed
baseline = that phase's contribution (the protocol's S_emd etc.):
  emd_const      subtractive: skip the 152MB frozen-lookup probe (has counter drift; reported)
  emd_nomemo     ZERO-drift:  force a real lookup EVERY postflop node (measures probe per-call cost; the memo
                              saving = nsnode_nomemo - nsnode_base, and the REAL contribution ~= saving*miss_rate)
  regret_zero    subtractive: skip the cum_regret hashmap get
  showdown_const subtractive: skip the terminal (site-2) showdown eval

Warm EMD-K160 production path, interleaved baseline/stub reps, bootstrap CI on the nsnode ratio + node-count
drift bound. Single-thread by default (cleanest per-core component cost; the metal run adds 128-thread contention).
NEVER cloud-ship the ab_stub cdylib (stubs live in the *_profile twins; default-OFF feature).
"""
from __future__ import annotations
import argparse, json, os, shutil, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUBS = ["emd_const", "emd_nomemo", "regret_zero", "showdown_const"]


def _load(dll: str):
    import importlib.util
    td = tempfile.mkdtemp(prefix="abstub_")
    pyd = os.path.join(td, "battery_native.pyd")
    shutil.copy2(dll, pyd)
    spec = importlib.util.spec_from_file_location("battery_native", pyd)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["battery_native"] = mod
    spec.loader.exec_module(mod)
    return mod


def _worker(a) -> int:
    mod = _load(a.dll)
    emd = a.emd_frozen or None
    tr = mod.NlheCFRTrainer(a.num_buckets, [], a.deals, 6, float("-inf"), False, emd)
    rows = json.loads(tr.train_profile_json(a.iters))
    out = [{"nodes": int(r.get("nodes", 0)), "traverse_ms": float(r.get("traverse_ms", 0.0))} for r in rows]
    sys.stdout.write("ABSTUB_ROWS " + json.dumps(out) + "\n")
    return 0


def _run(a, stub: str | None):
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = str(a.threads)
    if stub:
        env["POKERBOT_AB_STUB"] = stub
    else:
        env.pop("POKERBOT_AB_STUB", None)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--dll", a.dll,
           "--emd-frozen", a.emd_frozen or "", "--num-buckets", str(a.num_buckets),
           "--deals", str(a.deals), "--iters", str(a.iters)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"worker failed (stub={stub}) rc={p.returncode}\n{p.stdout}\n{p.stderr}")
    line = next((l for l in p.stdout.splitlines() if l.startswith("ABSTUB_ROWS ")), None)
    if not line:
        raise SystemExit(f"no ABSTUB_ROWS (stub={stub})\n{p.stdout}\n{p.stderr}")
    return json.loads(line[len("ABSTUB_ROWS "):])


def _nsnode(rows, warm):
    steady = rows[warm:]
    nodes = sum(r["nodes"] for r in steady)
    tms = sum(r["traverse_ms"] for r in steady)
    return (tms * 1e6 / max(nodes, 1)), nodes  # ns/node, total nodes


def _ci(ratios, nboot=20000):
    k = len(ratios); seed = 0x9E3779B97F4A7C15; ms = []
    for _ in range(nboot):
        s = 0.0
        for _ in range(k):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            s += ratios[(seed >> 11) % k]
        ms.append(s / k)
    ms.sort()
    return ms[int(0.025 * nboot)], ms[int(0.975 * nboot)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--dll", default=str(ROOT / "tools/battery_native/target_abstub/release/battery_native.dll"))
    ap.add_argument("--emd-frozen", default=str(ROOT / "data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn"))
    ap.add_argument("--num-buckets", type=int, default=160)
    ap.add_argument("--deals", type=int, default=96)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--stubs", default=",".join(STUBS))
    a = ap.parse_args()
    if a.worker:
        return _worker(a)
    if not Path(a.dll).exists():
        raise SystemExit(f"ab_stub cdylib missing: {a.dll} (build --features python,ab_stub)")

    print(f"[ab-stub attribution] dll={Path(a.dll).name} emd={Path(a.emd_frozen).name} buckets={a.num_buckets} "
          f"deals={a.deals} iters={a.iters} warm={a.warmup} reps={a.reps} threads={a.threads}")
    stubs = [s for s in a.stubs.split(",") if s]
    payload = {"config": vars(a), "results": {}}
    for stub in stubs:
        base_ns, stub_ns, ratios, ndrift = [], [], [], 0.0
        for _ in range(a.reps):
            rb = _run(a, None)              # baseline (no stub)
            rs = _run(a, stub)             # stub arm
            bns, bn = _nsnode(rb, a.warmup)
            sns, sn = _nsnode(rs, a.warmup)
            base_ns.append(bns); stub_ns.append(sns); ratios.append(sns / bns)
            ndrift = max(ndrift, abs(sn - bn) / max(bn, 1))
        mb, ms = statistics.median(base_ns), statistics.median(stub_ns)
        rmed = statistics.median(ratios)
        lo, hi = _ci(ratios)
        # subtractive contribution = 1 - stub/base ; emd_nomemo is additive (stub slower) -> report saving
        if stub == "emd_nomemo":
            interp = f"per-node memo SAVING = +{(rmed-1)*100:.1f}% slower w/o memo (real frozen-lookup contrib ~= saving*miss_rate)"
            contrib = None
        else:
            contrib = (1 - rmed) * 100
            interp = f"phase contribution ~= {contrib:.1f}% of node time"
        payload["results"][stub] = {"base_nsnode": mb, "stub_nsnode": ms, "ratio_med": rmed,
                                    "ratio_ci": [lo, hi], "node_drift": ndrift, "contribution_pct": contrib}
        print(f"  {stub:14}: base {mb:7.1f} ns/node  stub {ms:7.1f}  ratio {rmed:.4f} CI[{lo:.4f},{hi:.4f}]  "
              f"node_d={ndrift*100:.2f}%  -> {interp}")
    out = ROOT / "artifacts" / "ab_stub_probe.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
