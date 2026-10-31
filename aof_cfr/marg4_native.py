"""Native (Rust) kernel for the 4-max AoF marginal payoff precompute — a bit-identical, deal-parallel
drop-in for ``aof_cfr/fp_4max_vec.py::precompute_payoffs``.

The kernel fills the same ``(15, n_deals, 4)`` net-bb payoff tensor the pure-Python path builds, but
in native code with rayon over deals. It reproduces the production showdown MC EXACTLY:

  * Board RNG: a verbatim port of CPython's MT19937 (``init_by_array`` seeding + ``genrand_uint32``)
    and ``random.Random._randbelow`` drives the identical Fisher-Yates board sequence per showdown,
    seeded by the same 131-rolling hash of the flat alive hole-card ids (see ``_showdown_shares``).
  * Evaluator: the bundled 7-card evaluator makes the SAME winner/tie decisions as the OMPEval
    ``hu_table_native.eval7`` used in production (verified over millions of random boards); only the
    per-board argmax + equality classes matter for the pot-share arithmetic, and those match.

So the output is byte-for-byte identical to ``precompute_payoffs`` (``np.array_equal``), proven by
``tests/test_marg4_native.py``. If the extension is not built, callers transparently fall back to the
pure-Python path — nothing special-cases availability.

Build (Windows, cargo only — no maturin/cmake/cl):

    PYO3_PYTHON="$(python -c 'import sys; print(sys.executable)')" cargo build --release --features python --lib \
        --manifest-path tools/marg4_native/Cargo.toml
    cp tools/marg4_native/target/release/marg4_native.dll \
       tools/marg4_native/target/release/marg4_native.pyd   # CPython needs the .pyd name on Windows
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import numpy as np

_CRATE_DIR = Path(__file__).resolve().parents[1] / "tools" / "marg4_native"
_native: ModuleType | None = None
_native_tried = False

N_OUTCOMES = 15
NUM_PLAYERS = 4


def _candidate_pyd_dirs():
    yield _CRATE_DIR / "target" / "release"
    yield _CRATE_DIR / "target" / "debug"
    yield _CRATE_DIR


def _ensure_pyd() -> Path | None:
    for d in _candidate_pyd_dirs():
        if not d.is_dir():
            continue
        pyd = d / "marg4_native.pyd"
        if pyd.exists():
            return pyd
        so = d / "marg4_native.so"
        if so.exists():
            return so
        dll = d / "marg4_native.dll"
        if dll.exists():
            try:
                shutil.copy2(dll, pyd)
                return pyd
            except OSError:
                return dll
    return None


def load_native() -> ModuleType | None:
    global _native, _native_tried
    if _native is not None or _native_tried:
        return _native
    _native_tried = True
    pyd = _ensure_pyd()
    if pyd is None:
        return None
    try:
        spec = importlib.util.spec_from_file_location("marg4_native", str(pyd))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _native = mod
    except Exception:  # noqa: BLE001 — missing/corrupt extension -> fall back to Python
        _native = None
    return _native


def native_available() -> bool:
    return load_native() is not None


def reset_native_cache() -> None:
    global _native, _native_tried
    _native = None
    _native_tried = False


def precompute_payoffs_native(
    holes: list,
    stack_bb: float,
    *,
    game_model: int,
    fee_bb: float,
    threads: int = 1,
) -> np.ndarray | None:
    """Native ``(15, n, 4)`` payoff tensor for ``holes`` (a list of 4-tuples of (c0,c1) per deal),
    or ``None`` if the extension is unavailable. ``game_model`` / ``fee_bb`` must be the resolved
    runtime values (the Python path reads them inside ``contested_pot_bb``; here they are passed
    explicitly so the native and Python tensors are identical).

    ``threads`` is the native rayon pool size for THIS call (1 = serial, matching the per-chunk
    ``ProcessPoolExecutor`` driver where each process already owns one core; pass >1 only when
    running a single big batch outside the process pool)."""
    mod = load_native()
    if mod is None:
        return None
    n = len(holes)
    if n == 0:
        return np.zeros((N_OUTCOMES, 0, NUM_PLAYERS), dtype=np.float64)
    flat = np.empty((n, 8), dtype=np.int64)
    for d, deal in enumerate(holes):
        flat[d, 0] = deal[0][0]; flat[d, 1] = deal[0][1]
        flat[d, 2] = deal[1][0]; flat[d, 3] = deal[1][1]
        flat[d, 4] = deal[2][0]; flat[d, 5] = deal[2][1]
        flat[d, 6] = deal[3][0]; flat[d, 7] = deal[3][1]
    out = np.empty((N_OUTCOMES, n, NUM_PLAYERS), dtype=np.float64)
    mod.precompute_payoffs(
        np.ascontiguousarray(flat), int(n), float(stack_bb), int(game_model), float(fee_bb),
        out, int(threads),
    )
    return out


def verify_against_python(n_deals: int = 2000, seed: int = 4242, depths=(6.0, 8.0)):
    """Assert the native tensor reproduces ``fp_4max_vec.precompute_payoffs`` EXACTLY
    (``np.array_equal``) over ``n_deals`` real-card deals x several depths."""
    from aof_cfr.fp_4max_vec import precompute_payoffs, sample_deals
    from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default

    mod = load_native()
    if mod is None:
        print("[marg4_native] backend = python-fallback (nothing to verify)")
        return {"_backend": "python-fallback", "checked": 0}
    gm = game_model_default()
    fee = flat_fee_default()
    checked = 0
    for depth in depths:
        _, holes = sample_deals(n_deals, seed)
        ref = precompute_payoffs(holes, float(depth))
        nat = precompute_payoffs_native(holes, float(depth), game_model=gm, fee_bb=fee)
        assert nat is not None
        assert np.array_equal(ref, nat), f"payoff mismatch at depth {depth}"
        checked += 1
    print("[marg4_native] backend = native")
    print(f"  verified {checked} (depth) cases  OK (np.array_equal, bit-identical)")
    return {"_backend": "native", "checked": checked}


if __name__ == "__main__":
    verify_against_python()
