"""Workstream-F native Rust trainer acceptance — the TABULAR (cheapest) path.

The acceptance (ULTRACODE_PROMPT.md Workstream F) asks for a native Rust trainer that:
  1. clears the Leduc exploitability gate (~0.76 mbb @ 1000 iters),
  2. is equivalence-gated vs the JAX/Python reference, and
  3. has durable atomic checkpoint/resume.

The native CFR+ trainer lives in ``tools/battery_native/src/leduc.rs`` (a faithful port of the game
rules in ``pokerbot/engine/leduc_jax.py`` and the CFR+ scheme in
``pokerbot/training/leduc_tabular_cfr.py::tabular_cfr``). Because Leduc enumerates, the native solver's
exploitability is EXACT (full information-set BR), so the equivalence gate is the strongest kind: the
native number must MATCH the Python reference's number to numerical precision (not merely be "close").

These tests are the gate. They are skipped only if the native extension is not built.
"""

from __future__ import annotations

import os

import pytest

from pokerbot.search import native_solver as ns

_EXT = ns._try_load()
native_only = pytest.mark.skipif(
    _EXT is None or not hasattr(_EXT, "LeducCFRTrainer"),
    reason="native LeducCFRTrainer not built (cargo build --release --features python)",
)


@native_only
def test_native_leduc_clears_exploitability_gate_at_1000_iters():
    """Native CFR+ at 1000 iters must clear the ~0.76 mbb anchor (gate threshold 1.0 mbb)."""
    expl, u0, u1, br0, br1 = _EXT.leduc_solve_exploitability(1000)
    # zero-sum sanity (the trainer is correct only if the profile EVs cancel).
    assert abs(u0 + u1) < 1e-6, f"profile not zero-sum: u0={u0} u1={u1}"
    # the OpenSpiel-validated anchor is ~0.76 mbb; gate at 1.0 mbb.
    assert expl < 1.0, f"native exploitability {expl:.4f} mbb does not clear the 1.0 mbb gate"
    # BR values must dominate the profile values (best response can only help).
    assert br0 >= u0 - 1e-9 and br1 >= u1 - 1e-9


@native_only
def test_native_matches_python_reference_exploitability_full():
    """EQUIVALENCE GATE (full): native exploitability == Python tabular-CFR+ reference, to precision.

    Both are exact (full enumeration + information-set BR), so the native port of the game + CFR+ scheme
    must reproduce the Python reference's exact exploitability AND profile/BR values bit-for-bit (fp
    noise). We compare at 100 iters (one live Python ``tabular_cfr`` solve — the JAX-backed reference is
    slow, so a single iter count keeps this under the suite timeout).
    """
    from pokerbot.training.leduc_tabular_cfr import (
        infoset_exploitability_mbb,
        tabular_cfr,
    )

    iters = 100
    n_expl, n_u0, n_u1, n_br0, n_br1 = _EXT.leduc_solve_exploitability(iters)
    avg, _ = tabular_cfr(iterations=iters, seed=0)
    p_expl, p_u0, p_u1, p_br0, p_br1 = infoset_exploitability_mbb(avg, samples=90)
    assert abs(n_expl - p_expl) < 1e-3, (
        f"@{iters}: native expl {n_expl:.6f} != python {p_expl:.6f} mbb"
    )
    assert abs(n_u0 - p_u0) < 1e-6 and abs(n_u1 - p_u1) < 1e-6
    assert abs(n_br0 - p_br0) < 1e-6 and abs(n_br1 - p_br1) < 1e-6


@native_only
def test_native_matches_python_reference_floor_at_1000():
    """EQUIVALENCE GATE (1000-iter anchor): native @1000 == the Python reference's cached floor.

    ``equilibrium_reference_exploitability_mbb(1000)`` is the SAME Python tabular-CFR+ number this whole
    stack is rulered against (~0.76 mbb), disk-cached so this comparison is instant. The native trainer
    must reproduce it — this is the headline equivalence at the gate's nominal iteration count.
    """
    from pokerbot.training.leduc_tabular_cfr import (
        equilibrium_reference_exploitability_mbb,
    )

    n_expl = _EXT.leduc_solve_exploitability(1000)[0]
    p_floor = equilibrium_reference_exploitability_mbb(1000)
    assert abs(n_expl - p_floor) < 1e-3, (
        f"native @1000 {n_expl:.6f} != python reference floor {p_floor:.6f} mbb"
    )
    # the anchor is ~0.76 mbb.
    assert 0.70 < n_expl < 0.80


@native_only
def test_native_monotone_convergence():
    """More CFR+ iters must not increase exploitability (convergence sanity)."""
    e100 = _EXT.leduc_solve_exploitability(100)[0]
    e1000 = _EXT.leduc_solve_exploitability(1000)[0]
    assert e1000 < e100


@native_only
def test_checkpoint_resume_durable_through_disk(tmp_path):
    """Atomic checkpoint/resume: train 600 -> save_bytes -> os.replace to disk -> reload -> train 400.

    The resumed run must reproduce a straight-through 1000-iter solve's exploitability exactly, proving
    the checkpoint is durable and resume continues the SAME solve (no lost regret/strategy state).
    """
    # straight-through reference.
    straight = _EXT.LeducCFRTrainer()
    straight.train(1000)
    straight_expl = straight.exploitability()[0]

    # partial solve -> atomic checkpoint to disk.
    part = _EXT.LeducCFRTrainer()
    part.train(600)
    assert part.iters_done() == 600
    blob = part.save_bytes()
    ckpt = os.path.join(tmp_path, "leduc.ckpt")
    tmp = ckpt + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(blob)
    os.replace(tmp, ckpt)  # atomic on the same volume

    # fresh process-like reload from disk -> resume.
    with open(ckpt, "rb") as fh:
        loaded_blob = fh.read()
    resumed = _EXT.LeducCFRTrainer()
    assert resumed.load_bytes(loaded_blob) is True
    assert resumed.iters_done() == 600
    resumed.train(400)
    assert resumed.iters_done() == 1000
    resumed_expl = resumed.exploitability()[0]

    # exact reproduction of the straight-through solve.
    assert abs(resumed_expl - straight_expl) < 1e-9, (
        f"resume diverged: straight {straight_expl:.9f} vs resumed {resumed_expl:.9f} mbb"
    )
    # and it clears the gate.
    assert resumed_expl < 1.0


@native_only
def test_load_bytes_rejects_garbage():
    """Resume must fail closed on a corrupt/foreign blob (no silent wrong-state load)."""
    t = _EXT.LeducCFRTrainer()
    assert t.load_bytes(b"not a checkpoint") is False
    assert t.load_bytes(b"") is False
