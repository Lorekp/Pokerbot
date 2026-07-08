"""Full action-history key cross-language parity gate (blueprint-v4 Stage 0).

The trainer (tools/battery_native/src/fingerprint.rs, env-gated POKERBOT_FULLHIST_KEY=1)
appends b"A" + street + the raw engine action-id bytes of the whole hand to EVERY street's key
(preflop b"w" AND postflop b"B" branches — exact in-abstraction action-history keying, the
Pluribus construction; design of record artifacts/blueprint_pluribus_grade_design.md §7.1).
The Python serve path (jax_weva_fast.full_hist_bytes + the b"A" digest branch in
jax_weva_encode) must reproduce it EXACTLY or every lookup on a fullhist-trained blueprint
misses (uniform fallback on every street).

Both env states are pinned (one subprocess each — the flags bake at import / Rust OnceLock):
  * fullhist=1: rust == python == independent-layout reference over >=512 game-flow states per
    the probe's floors, INCLUDING >=512 postflop states (the whole point of v4 is postflop
    keying), postflop raise chains, and long histories;
  * fullhist=0: same triple equality with the reference built WITHOUT the A segment — i.e.
    today's deployed layout, byte-identical.

Mirrors tests/test_preflop_hist_key_parity.py one axis further (all streets). Skips cleanly
when the battery_native .pyd is not built.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROBE = os.path.join(_ROOT, "scripts", "probe_fullhist_key_parity.py")


def _native_built() -> bool:
    return os.path.exists(os.path.join(_ROOT, "battery_native.pyd")) or os.path.exists(
        os.path.join(_ROOT, "tools", "battery_native", "target", "release", "battery_native.dll")
    )


def _run_probe(fullhist: int, tmp_path, presum: int = 0) -> dict:
    out = str(tmp_path / f"fullhist_parity_{fullhist}_{presum}.json")
    env = dict(os.environ)
    env.pop("POKERBOT_FULLHIST_KEY", None)  # the probe sets/clears it itself from --fullhist
    env.pop("POKERBOT_PRESUM_KEY", None)    # the probe sets/clears it itself from --presum
    env.pop("POKERBOT_PREFLOP_HIST_KEY", None)
    proc = subprocess.run(
        [sys.executable, _PROBE, "--fullhist", str(fullhist), "--presum", str(presum),
         "--min-states", "512", "--seed", "7", "--out", out],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, (
        f"parity probe (fullhist={fullhist} presum={presum}) failed rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
    )
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def _assert_coverage(res: dict) -> None:
    assert res["pass"], res
    assert res["states"] >= 512
    assert res["mismatches"] == 0
    cov = res["coverage"]
    # every street exercised, incl. the postflop raise-chain shapes the v4 key must distinguish
    assert cov["flop"] + cov["turn"] + cov["river"] >= 512, cov
    assert cov["preflop"] >= 200 and cov["flop"] >= 200, cov
    assert cov["turn"] >= 100 and cov["river"] >= 60, cov
    assert cov["postflop_raise"] >= 40 and cov["postflop_reraise"] >= 8, cov
    assert cov["hist_len_10plus"] >= 40 and cov["hist_len_16plus"] >= 8, cov
    assert cov["limp1"] >= 20 and cov["raise2plus"] >= 12, cov


@pytest.mark.skipif(not _native_built(), reason="battery_native extension not built")
@pytest.mark.parametrize("fullhist", [0, 1])
def test_fullhist_key_parity(fullhist, tmp_path):
    _assert_coverage(_run_probe(fullhist, tmp_path))


@pytest.mark.skipif(not _native_built(), reason="battery_native extension not built")
def test_presum_key_parity(tmp_path):
    """arm-4 presum POSTFLOP preflop-summary key: rust == python == independent reference over the
    same game-flow floors, with the postflop A-segment = street + 4 summary bytes + exact postflop
    bytes and the preflop A-segment unchanged (fullhist). Requires --fullhist 1 (presum refines it)."""
    _assert_coverage(_run_probe(1, tmp_path, presum=1))
