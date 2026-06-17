"""Native (Rust) binding for the AIVAT control variates — bit-identical to ``pokerbot.eval.aivat`` /
``pokerbot.eval.chance_term``.

This module exposes the SAME API as the Python oracle, dispatching to the compiled Rust extension
(``tools/aivat_native``, built as ``aivat_native.pyd``) when available. The native core reproduces
NumPy's IEEE-754 f64 operation order EXACTLY — including NumPy's pairwise summation in ``np.mean`` /
``np.std`` / ``np.sum`` — so every output is bit-for-bit identical to the oracle (proven by
``verify_against_golden`` against ``tools/aivat_native/golden/golden.json``).

The RNG-driven gadget value function ``continuation_leaf_value`` STAYS in Python (numpy PCG64 +
np_engine playout + matrix-game CFR — see the profile / boundary writeup): it is the 96% of the chance
term that cannot be reproduced bit-identically in native code without re-deriving PCG64 and the engine's
float order. The native core therefore consumes PRECOMPUTED gadget values (``chance_reduce`` /
``chance_corrections_from_values``) and accelerates the deterministic reductions + accumulator stats.

If the extension is not built, every function falls back to a PURE-PYTHON path that is itself
bit-identical (the same pairwise algorithm), so callers never need to special-case availability.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

_CRATE_DIR = Path(__file__).resolve().parents[2] / "tools" / "aivat_native"
_native: ModuleType | None = None
_native_tried = False


# ---------------------------------------------------------------------------
# Native loading: locate target/release/aivat_native.dll, alias to .pyd, import.
# ---------------------------------------------------------------------------
def _candidate_pyd_dirs():
    yield _CRATE_DIR / "target" / "release"
    yield _CRATE_DIR / "target" / "debug"
    yield _CRATE_DIR


def _ensure_pyd() -> Path | None:
    """On Windows a cdylib builds as ``aivat_native.dll``; CPython needs ``aivat_native.pyd``.
    Copy/alias it next to the dll the first time. On Linux it would be ``.so`` directly."""
    for d in _candidate_pyd_dirs():
        if not d.is_dir():
            continue
        pyd = d / "aivat_native.pyd"
        if pyd.exists():
            return pyd
        so = d / "aivat_native.so"
        if so.exists():
            return so
        dll = d / "aivat_native.dll"
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
        spec = importlib.util.spec_from_file_location("aivat_native", str(pyd))
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


# ---------------------------------------------------------------------------
# Pure-python fallbacks (bit-identical: same pairwise algorithm as the Rust core).
# ---------------------------------------------------------------------------
def _decision_correction_py(probs, baseline_values, taken_idx: int) -> float:
    p = np.asarray(probs, dtype=np.float64)
    bv = np.asarray(baseline_values, dtype=np.float64)
    s = p.sum()
    if s <= 0 or not (0 <= taken_idx < len(bv)):
        return 0.0
    p = p / s
    return float(bv[taken_idx] - float((p * bv).sum()))


def _chance_reduce_py(realized, vals) -> float:
    v = np.asarray(vals, dtype=np.float64)
    expected = float(np.mean(v)) if v.size else float(realized)
    return float(realized) - expected


def _accumulator_summary_py(raw, adj) -> dict:
    raw = np.asarray(raw, dtype=np.float64)
    adj = np.asarray(adj, dtype=np.float64)
    rstd = float(raw.std(ddof=1)) if len(raw) > 1 else 0.0
    astd = float(adj.std(ddof=1)) if len(adj) > 1 else 0.0
    return {
        "hands": len(raw),
        "raw_mean_bb": float(raw.mean()) if len(raw) else 0.0,
        "adj_mean_bb": float(adj.mean()) if len(adj) else 0.0,
        "raw_std": rstd,
        "adj_std": astd,
        "variance_reduction": (1.0 - (astd / rstd) ** 2) if rstd > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Public API — mirrors pokerbot.eval.aivat / chance_term exactly.
# ---------------------------------------------------------------------------
def decision_correction(probs, baseline_values, taken_idx: int) -> float:
    """Bit-identical to ``aivat.decision_correction``."""
    mod = load_native()
    if mod is not None:
        p = np.asarray(probs, dtype=np.float64).reshape(-1)
        bv = np.asarray(baseline_values, dtype=np.float64).reshape(-1)
        return float(mod.decision_correction(p.tolist(), bv.tolist(), int(taken_idx)))
    return _decision_correction_py(probs, baseline_values, taken_idx)


def chance_correction(expected_value, realized_value) -> float:
    """Bit-identical to ``aivat.chance_correction`` (``realized - expected``)."""
    mod = load_native()
    if mod is not None:
        return float(mod.chance_correction(float(expected_value), float(realized_value)))
    return float(realized_value) - float(expected_value)


def chance_reduce(realized, vals) -> float:
    """Per-reveal chance correction ``c_C = realized - float(np.mean(vals))`` (gadget vals precomputed).

    This is the deterministic reduction inside ``chance_term.chance_corrections_for_hand`` lifted to
    native code; the RNG-driven gadget ``v`` values are computed in Python and passed in."""
    mod = load_native()
    if mod is not None:
        v = np.asarray(vals, dtype=np.float64).reshape(-1)
        return float(mod.chance_reduce(float(realized), v.tolist()))
    return _chance_reduce_py(realized, vals)


def chance_corrections_from_values(reveals) -> list[float]:
    """Vectorized native chance term: ``reveals`` = list of ``(realized, [candidate_v, ...])`` per board
    reveal (the precomputed gadget values). Returns one ``c_C`` per reveal — bit-identical to the loop in
    ``chance_corrections_for_hand`` (which the caller would otherwise do in Python)."""
    return [chance_reduce(realized, vals) for realized, vals in reveals]


def np_mean(vals) -> float:
    """``np.mean`` with NumPy-exact pairwise summation (native when available)."""
    mod = load_native()
    if mod is not None:
        v = np.asarray(vals, dtype=np.float64).reshape(-1)
        return float(mod.np_mean(v.tolist()))
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def add_hand_adjusted(raw_chips: float, corrections=None, chance_corrections=None) -> float:
    """``AIVATAccumulator.add_hand`` adjusted score for one hand (raw - sum(corr) - sum(chance))."""
    mod = load_native()
    corr = [] if not corrections else [float(x) for x in corrections]
    ch = [] if not chance_corrections else [float(x) for x in chance_corrections]
    if mod is not None:
        return float(mod.add_hand_adjusted(float(raw_chips), corr, ch))
    c = float(sum(corr)) if corr else 0.0
    c += float(sum(ch)) if ch else 0.0
    return float(raw_chips) - c


class AIVATAccumulator:
    """Drop-in for ``aivat.AIVATAccumulator`` whose ``summary()`` is computed natively (bit-identical)."""

    def __init__(self) -> None:
        self._raw: list[float] = []
        self._adj: list[float] = []

    def add_hand(self, raw_chips: float, corrections=None, chance_corrections=None) -> None:
        self._raw.append(float(raw_chips))
        self._adj.append(add_hand_adjusted(raw_chips, corrections, chance_corrections))

    def summary(self) -> dict:
        mod = load_native()
        if mod is not None:
            hands, rmean, amean, rstd, astd, vr = mod.accumulator_summary(
                list(self._raw), list(self._adj)
            )
            return {
                "hands": int(hands),
                "raw_mean_bb": float(rmean),
                "adj_mean_bb": float(amean),
                "raw_std": float(rstd),
                "adj_std": float(astd),
                "variance_reduction": float(vr),
            }
        return _accumulator_summary_py(self._raw, self._adj)


# ---------------------------------------------------------------------------
# Bit-identity verification against the captured golden vectors.
# ---------------------------------------------------------------------------
def verify_against_golden(golden_path: str | None = None, verbose: bool = True) -> dict:
    """Assert native output == Python-oracle golden EXACTLY (f64 bit equality) for every captured case.

    Returns a per-section pass/fail dict; raises AssertionError on the first mismatch (with the case)."""
    import json
    import struct

    if golden_path is None:
        golden_path = str(_CRATE_DIR / "golden" / "golden.json")
    with open(golden_path) as f:
        g = json.load(f)

    def fh(hexs: str) -> float:
        return struct.unpack("<d", bytes.fromhex(hexs))[0]

    def bits(x: float) -> bytes:
        return struct.pack("<d", float(x))

    results: dict[str, dict] = {}
    backend = "native" if native_available() else "python-fallback"

    # decision_correction
    n_ok = 0
    for i, c in enumerate(g["decision_correction"]):
        probs = [fh(x) for x in c["probs"]]
        bv = [fh(x) for x in c["baseline_values"]]
        got = decision_correction(probs, bv, c["taken_idx"])
        want = fh(c["out"])
        assert bits(got) == bits(want), f"decision_correction[{i}]: {got!r} != {want!r}"
        n_ok += 1
    results["decision_correction"] = {"cases": n_ok, "ok": True}

    # chance_correction
    n_ok = 0
    for i, c in enumerate(g["chance_correction"]):
        got = chance_correction(fh(c["expected"]), fh(c["realized"]))
        want = fh(c["out"])
        assert bits(got) == bits(want), f"chance_correction[{i}]: {got!r} != {want!r}"
        n_ok += 1
    results["chance_correction"] = {"cases": n_ok, "ok": True}

    # chance_reduce (the native boundary, incl. pairwise-trap sizes + real reveal vals)
    n_ok = 0
    for i, c in enumerate(g["chance_reduce"]):
        realized = fh(c["realized"])
        vals = [fh(x) for x in c["vals"]]
        got = chance_reduce(realized, vals)
        want = fh(c["out"])
        assert bits(got) == bits(want), f"chance_reduce[{i}] (n={len(vals)}): {got!r} != {want!r}"
        n_ok += 1
    results["chance_reduce"] = {"cases": n_ok, "ok": True}

    # accumulator summary
    n_ok = 0
    for i, c in enumerate(g["accumulator"]):
        acc = AIVATAccumulator()
        for h in c["hands_in"]:
            raw = fh(h["raw"])
            corr = [fh(x) for x in h["corr"]] if h["corr"] else None
            ch = [fh(x) for x in h["ch"]] if h["ch"] else None
            acc.add_hand(raw, corr, ch)
        summ = acc.summary()
        want = c["summary"]
        assert summ["hands"] == want["hands"], f"accumulator[{i}] hands"
        for k in ("raw_mean_bb", "adj_mean_bb", "raw_std", "adj_std", "variance_reduction"):
            assert bits(summ[k]) == bits(fh(want[k])), (
                f"accumulator[{i}].{k}: {summ[k]!r} != {fh(want[k])!r}"
            )
        n_ok += 1
    results["accumulator"] = {"cases": n_ok, "ok": True}

    if verbose:
        print(f"[aivat_native] backend = {backend}")
        for sect, r in results.items():
            print(f"  {sect:28s}  {r['cases']:3d} cases  OK (bit-identical)")
    results["_backend"] = backend
    return results


if __name__ == "__main__":
    verify_against_golden()
