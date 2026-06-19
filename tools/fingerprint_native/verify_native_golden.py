"""Verify the NATIVE fingerprint against the USER's golden vectors (read-only consumer).

This CONSUMES ``tools/fingerprint_native/golden/golden.json.gz`` (the user's golden-vector harness
output) WITHOUT editing it: it reconstructs every captured EnvState and asserts the Rust native
fingerprint bytes == the stored Python golden hex, byte-for-byte, for every (state, seat) case. One
subprocess per section bakes that section's abstraction config (the flags are import-baked). Mirrors the
structure of ``golden_harness.verify_against_golden`` but routes through ``fingerprint_native``.

    python tools/fingerprint_native/verify_native_golden.py
Exits 0 iff every case is byte-identical.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# Reuse the user's harness IO + config matrix (read-only import — we never call its generate/verify).
sys.path.insert(0, str(_HERE))
import golden_harness as gh  # noqa: E402


def _worker_verify_native(golden_path: str, section_idx: int, out_path: str) -> None:
    import numpy as np  # noqa: F401

    from pokerbot.encoding import fingerprint_native as fn
    from pokerbot.engine.jax_weva_encode import active_abstraction_config

    eng = gh._load_engine()
    golden = gh._read_json_any(golden_path)
    section = golden["sections"][section_idx]
    schema = golden["schema"]

    if not fn.available():
        raise RuntimeError("fingerprint_native extension not built")
    live_cfg = active_abstraction_config()
    assert live_cfg == section["config"], (
        f"section[{section_idx}] '{section['name']}' config drift: baked={live_cfg} != golden={section['config']}"
    )

    n_states = 0
    n_cases = 0
    mismatches = []
    for si, st in enumerate(section["states"]):
        state = gh._b64_to_state(st["buf"], schema, eng)
        for c in st["seats"]:
            got = fn.fingerprint_native(state, int(c["seat"])).hex()
            want = c["out"]
            if got != want:
                if len(mismatches) < 8:
                    mismatches.append({"state": si, "seat": c["seat"], "got": got, "want": want})
            else:
                n_cases += 1
        n_states += 1
    total_cases = sum(len(st["seats"]) for st in section["states"])
    res = {
        "name": section["name"],
        "states": n_states,
        "cases_ok": n_cases,
        "cases_total": total_cases,
        "mismatches": len(mismatches),
        "sample_mismatches": mismatches,
        "ok": len(mismatches) == 0,
    }
    Path(out_path).write_text(json.dumps(res), encoding="ascii")


def verify(golden_path: str | None = None) -> int:
    gp = Path(golden_path) if golden_path else gh.GOLDEN_PATH
    if not gp.exists():
        print(f"[native-golden] golden not found at {gp} -- run golden_harness.py generate first")
        return 2
    golden = gh._read_json_any(gp)
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        for i, section in enumerate(golden["sections"]):
            cfg_env = next((c["env"] for c in gh.CONFIGS if c["name"] == section["name"]), None)
            if cfg_env is None:
                cfg_env = dict(gh._DEPLOYED_BASE)
                cfg_env.update(section.get("env_flags", {}))
            res_path = Path(td) / f"res_{i}.json"
            env = dict(os.environ)
            env.update(cfg_env)
            env["POKERBOT_FINGERPRINT_NATIVE"] = "1"  # force native path in the worker
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                "--golden",
                str(gp),
                "--section",
                str(i),
                "--out",
                str(res_path),
            ]
            proc = subprocess.run(cmd, env=env, cwd=str(Path(__file__).resolve().parents[2]))
            if proc.returncode != 0:
                all_ok = False
                print(f"  section {i} '{section['name']}': worker exit {proc.returncode}")
                continue
            res = json.loads(res_path.read_text(encoding="ascii"))
            status = "OK (byte-identical)" if res["ok"] else f"FAIL ({res['mismatches']} mm)"
            print(f"  {res['name']:40s} {res['states']:6d} states {res['cases_total']:7d} cases  {status}")
            if not res["ok"]:
                all_ok = False
                for mm in res["sample_mismatches"][:3]:
                    print(f"      MM state={mm['state']} seat={mm['seat']} got={mm['got']} want={mm['want']}")
    print(f"\n[native-golden] GATE: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--golden", default=None)
    ap.add_argument("--section", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.worker:
        _worker_verify_native(args.golden, args.section, args.out)
        return 0
    return verify(args.golden)


if __name__ == "__main__":
    raise SystemExit(main())
