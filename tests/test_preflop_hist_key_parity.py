"""Preflop history-key cross-language parity gate (blueprint-v3 serve twin).

The trainer (tools/battery_native/src/fingerprint.rs, env-gated POKERBOT_PREFLOP_HIST_KEY=1)
appends b"H" + [n_limps, n_raises, opener_pos+1] to PREFLOP keys; the Python serve path
(jax_weva_fast.preflop_hist_bytes + the b"H" digest branch in jax_weva_encode) must reproduce it
EXACTLY or every preflop lookup on a hist-trained blueprint misses (uniform preflop fallback).

Both env states are pinned (one subprocess each — the flags bake at import / Rust OnceLock):
  * hist=1: rust == python == independent-layout reference over >=512 preflop GAME-FLOW states,
    with coverage floors for limped / multi-limper / 3bet / 4bet+ / blind-vs-blind histories;
  * hist=0: same triple equality with the reference built WITHOUT the H segment — i.e. today's
    deployed layout, byte-identical.

Mirrors the river-OCHS parity-test pattern (test_river_ochs_parity.py) one street earlier.
Skips cleanly when the battery_native .pyd is not built.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROBE = os.path.join(_ROOT, "scripts", "probe_preflop_hist_key_parity.py")


def _native_built() -> bool:
    return os.path.exists(os.path.join(_ROOT, "battery_native.pyd")) or os.path.exists(
        os.path.join(_ROOT, "tools", "battery_native", "target", "release", "battery_native.dll")
    )


def _run_probe(hist: int, tmp_path) -> dict:
    out = str(tmp_path / f"prefhist_parity_{hist}.json")
    env = dict(os.environ)
    env.pop("POKERBOT_PREFLOP_HIST_KEY", None)  # the probe sets/clears it itself from --hist
    proc = subprocess.run(
        [sys.executable, _PROBE, "--hist", str(hist), "--min-states", "512",
         "--seed", "7", "--out", out],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, (
        f"parity probe (hist={hist}) failed rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
    )
    with open(out, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.skipif(not _native_built(), reason="battery_native extension not built")
@pytest.mark.parametrize("hist", [0, 1])
def test_preflop_hist_key_parity(hist, tmp_path):
    res = _run_probe(hist, tmp_path)
    assert res["pass"], res
    assert res["states"] >= 512
    assert res["mismatches"] == 0
    cov = res["coverage"]
    assert cov["limp1"] >= 30 and cov["limp2"] >= 8, cov
    assert cov["raise1"] >= 60 and cov["raise2"] >= 12 and cov["raise3"] >= 4, cov
    assert cov["unopened"] >= 30 and cov["bvb"] >= 3, cov
    assert len(res["opener_positions"]) >= 4, res["opener_positions"]
