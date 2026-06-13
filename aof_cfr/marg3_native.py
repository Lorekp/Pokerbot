"""Native (Rust) cube kernel for the 3-max legal marginals — bit-identical to the NumPy path in
``aof_cfr/fp_br.py::_legal_marginals_3max``.

The kernel materializes, per requested (seat, history) marginal key, the two contiguous (169,169,169)
cubes ``w`` (deal-weight) and ``prod = w * push_cell`` that the NumPy code builds via a chain of
``pushfold_payoffs.mix`` broadcasts. Those exact buffers are then reduced with ``np.sum(..., axis=red)``
on the Python side — the order-sensitive reductions are NEVER reimplemented in native code, so the
result is bit-for-bit identical to the original (proven by ``np.array_equal`` in
``verify_against_numpy`` / ``tests/test_marg3_native.py``).

The native module only accelerates the elementwise build (a fused, cache-blocked, rayon-parallel pass
that replaces seven 38 MB broadcast temporaries per ``btn`` marginal). If the extension is not built,
``_legal_marginals_3max`` transparently uses its original pure-NumPy path — callers never special-case
availability.

Build (Windows, cargo only — no maturin/cmake):

    PYO3_PYTHON="python" cargo build --release --features python --lib \
        --manifest-path tools/marg3_native/Cargo.toml
    cp tools/marg3_native/target/release/marg3_native.dll \
       tools/marg3_native/target/release/marg3_native.pyd   # CPython needs the .pyd name on Windows
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import numpy as np

_CRATE_DIR = Path(__file__).resolve().parents[1] / "tools" / "marg3_native"
_native: ModuleType | None = None
_native_tried = False

N = 169
N3 = N * N * N


def _candidate_pyd_dirs():
    yield _CRATE_DIR / "target" / "release"
    yield _CRATE_DIR / "target" / "debug"
    yield _CRATE_DIR


def _ensure_pyd() -> Path | None:
    for d in _candidate_pyd_dirs():
        if not d.is_dir():
            continue
        pyd = d / "marg3_native.pyd"
        if pyd.exists():
            return pyd
        so = d / "marg3_native.so"
        if so.exists():
            return so
        dll = d / "marg3_native.dll"
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
        spec = importlib.util.spec_from_file_location("marg3_native", str(pyd))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _native = mod
    except Exception:
        _native = None
    return _native


def native_available() -> bool:
    return load_native() is not None


def reset_native_cache() -> None:
    global _native, _native_tried
    _native = None
    _native_tried = False


def _c64(a) -> np.ndarray:
    """Return a C-contiguous float64 view/copy (the kernel reads exactly these bytes)."""
    arr = np.ascontiguousarray(a, dtype=np.float64)
    return arr


def build_cubes(key: str, st, deal_w: np.ndarray, btn_push, sb_push, bb_push):
    """Build ``(w, prod)`` contiguous (169,169,169) cubes for marginal ``key`` natively.

    Returns ``None`` if the native module is unavailable. ``st`` is the ``StaticPayoffs`` instance;
    the push arrays are the per-seat 169-vectors / (k,169) tables passed to ``_legal_marginals_3max``.
    """
    mod = load_native()
    if mod is None:
        return None
    from aof_cfr.pushfold_payoffs import TERMINAL as t

    dw = _c64(deal_w)
    # Cube fields (only those a key needs are read, but all are passed for a uniform signature).
    bb0 = _c64(st.btn_bb_call0)
    bs0 = _c64(st.btn_sb_call0)
    th0 = _c64(st.three0)
    sbbb1 = _c64(st.sb_bb_call1)
    btsb1 = _c64(st.btn_sb_call1)
    th1 = _c64(st.three1)
    sbbb2 = _c64(st.sb_bb_call2)
    btbb2 = _c64(st.btn_bb_call2)
    th2 = _c64(st.three2)

    # Flat push layout: btn, psf, psa, pfa, paf, paa, then the two fold constants.
    btn = np.asarray(btn_push, dtype=np.float64).reshape(-1)
    psf = np.asarray(sb_push[0], dtype=np.float64).reshape(-1)
    psa = np.asarray(sb_push[1], dtype=np.float64).reshape(-1)
    pfa = np.asarray(bb_push[0], dtype=np.float64).reshape(-1)
    paf = np.asarray(bb_push[1], dtype=np.float64).reshape(-1)
    paa = np.asarray(bb_push[2], dtype=np.float64).reshape(-1)
    pushes = np.concatenate([
        btn, psf, psa, pfa, paf, paa,
        np.array([float(t.btn_win_uncontested_after_sb_fold),
                  float(t.sb_win_uncontested_after_btn_fold)], dtype=np.float64),
    ]).tolist()

    out_w = np.empty((N, N, N), dtype=np.float64)
    out_prod = np.empty((N, N, N), dtype=np.float64)
    mod.build_marginal_cubes(
        key, dw, bb0, bs0, th0, sbbb1, btsb1, th1, sbbb2, btbb2, th2,
        pushes, out_w, out_prod,
    )
    return out_w, out_prod


def verify_against_numpy(n_policies: int = 50, depths=(6.0, 8.0, 10.0, 12.0), verbose: bool = True):
    """Assert the native kernel reproduces the NumPy ``_legal_marginals_3max`` output EXACTLY
    (``np.array_equal``) over ``n_policies`` random push policies x all six keys x several depths."""
    from aof_cfr.exploit_router import _payoffs_3max
    from aof_cfr.fp_br import _legal_marginals_3max

    mod = load_native()
    backend = "native" if mod is not None else "python-fallback (nothing to verify)"
    keys = ("btn", "sb_f", "sb_a", "bb_fa", "bb_af", "bb_aa")
    if mod is None:
        if verbose:
            print(f"[marg3_native] backend = {backend}")
        return {"_backend": backend, "checked": 0}

    rng = np.random.default_rng(20260611)
    checked = 0
    for depth in depths:
        st, dw = _payoffs_3max(float(depth))
        for _ in range(n_policies):
            btn = rng.random(N)
            sb = rng.random((2, N))
            bb = rng.random((3, N))
            ref = _legal_marginals_3max(st, dw, btn, sb, bb, keys=keys)
            for key in keys:
                ep_ref, ef_ref, w_ref = ref[key]
                built = build_cubes(key, st, dw, btn, sb, bb)
                assert built is not None
                w_cube, prod_cube = built
                keep = {"btn": 0, "sb_f": 1, "sb_a": 1, "bb_fa": 2, "bb_af": 2, "bb_aa": 2}[key]
                red = tuple(a for a in (0, 1, 2) if a != keep)
                den = np.sum(w_cube, axis=red)
                num = np.sum(prod_cube, axis=red)
                eps = 1e-12
                ep = np.where(den > eps, num / np.maximum(den, eps), 0.0)
                assert np.array_equal(ep, ep_ref), f"ep mismatch key={key} depth={depth}"
                assert np.array_equal(den, w_ref), f"w mismatch key={key} depth={depth}"
                checked += 1
    if verbose:
        print(f"[marg3_native] backend = {backend}")
        print(f"  verified {checked} (policy x key x depth) cases  OK (np.array_equal, bit-identical)")
    return {"_backend": backend, "checked": checked}


if __name__ == "__main__":
    verify_against_numpy()
