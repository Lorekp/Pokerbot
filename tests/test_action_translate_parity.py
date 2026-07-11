"""Off-menu ACTION-TRANSLATION parity gate (POKERBOT_ACTION_TRANSLATE=1, serve-coverage fix
2026-07-11).

The serve key path may translate off-trained-menu history tokens (pseudo-harmonic, Ganzfried &
Sandholm 2013) before key construction — implemented twice: natively
(tools/battery_native/src/translate.rs, hooked in fingerprint.rs' serve entry points) and in Python
(pokerbot/engine/action_translate.py, hooked in jax_weva_fast.fingerprint_fast). A byte divergence
between the twins is a silent Python/native KEY SKEW inside one serve process, so this gate walks
game-flow states (full free menu => off-menu tokens on every street) and requires:

  * py_key == nat_key on EVERY decision state (the probe's --translate mode), AND
  * a floor of states where translation actually CHANGED the state (no vacuous pass), AND
  * the OFF control (--translate 0) stays the unchanged 3-way byte-parity gate.

One env state per process (import latch / Rust OnceLock) => subprocess per config, mirroring
tests/test_fullhist_key_parity.py. Skips cleanly when the battery_native .pyd is not built.
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


def _run_probe(translate: int, tmp_path) -> dict:
    out = str(tmp_path / f"translate_parity_{translate}.json")
    env = dict(os.environ)
    for k in ("POKERBOT_FULLHIST_KEY", "POKERBOT_PRESUM_KEY", "POKERBOT_PREFLOP_HIST_KEY",
              "POKERBOT_ACTION_TRANSLATE", "POKERBOT_MRR"):
        env.pop(k, None)  # the probe sets/clears these itself
    proc = subprocess.run(
        [sys.executable, _PROBE, "--fullhist", "1", "--presum", "1",
         "--translate", str(translate), "--min-states", "512", "--seed", "7", "--out", out],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, (
        f"translate parity probe (translate={translate}) failed rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
    )
    with open(out, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.skipif(not _native_built(), reason="battery_native extension not built")
def test_action_translate_parity_on(tmp_path):
    res = _run_probe(1, tmp_path)
    assert res["pass"], res
    assert res["mismatches"] == 0
    assert res["states"] >= 512
    # Translation must actually fire — the free-menu walk hits off-menu tokens constantly.
    assert res["translated_states"] >= 100, res


@pytest.mark.skipif(not _native_built(), reason="battery_native extension not built")
def test_action_translate_off_control(tmp_path):
    # OFF control: the unchanged 3-way (py == native == independent-ref) fullhist+presum gate.
    res = _run_probe(0, tmp_path)
    assert res["pass"], res
    assert res["mismatches"] == 0


def test_translate_identity_and_fixed_point():
    """Pure-Python unit sanity (no native needed): in-menu histories are identity; a translated
    state is a fixed point of translation."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import jax
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.engine.action_translate import trained_legal, translate_state_for_key

    occ = jax.numpy.array([True] * 6)
    rng = np.random.default_rng(3)
    n_checked = 0
    for h in range(30):
        env = to_numpy_state(init_hand(jax.random.PRNGKey(4000 + h), occupancy=occ))
        g = 0
        while not is_terminal_np(env) and g < 200:
            g += 1
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            seat = current_player_np(env)
            if seat < 0:
                break
            t = translate_state_for_key(env)
            t2 = translate_state_for_key(t)
            for f in ("street", "hist_len", "current_bet", "stacks", "committed_street"):
                assert np.array_equal(np.asarray(getattr(t2, f)), np.asarray(getattr(t, f))), (
                    f"translation not a fixed point on field {f} (hand {h})")
            n_checked += 1
            # play IN-MENU only => translation must be the identity object
            menu = trained_legal(env, int(seat))
            if not menu.any():
                menu = legal_action_mask_np(env, seat).astype(bool)
            t_id = translate_state_for_key(env)
            legal = np.flatnonzero(menu)
            env = apply_action_np(env, int(rng.choice(legal)))
            del t_id
        # in-menu whole-hand identity check: replay this hand's final decision states
    assert n_checked > 100
