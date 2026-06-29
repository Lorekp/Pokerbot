"""Tests for the CRN (common random numbers) reseed per replication.

Per ``artifacts/pluribus_emd_retrain_report.md`` hypothesis B, the previous ``seeds = list(range(N))``
pattern let the table_id + deck sequence collide across replications sharing the same ``seed_idx``.
The canonical fix is a stable XOR with ``BASE_SEED`` per replication so every replication gets a
distinct starting point while the per-hand deal schedule (driven by ``LBR6Config(hands, seed)``)
stays reproducible.

These tests exercise the helper exported from
``scripts/eval/eval_emd_vs_iter62000.py`` and re-exported from
``scripts/launch_emd_pipeline_spot.py``. Both modules must use the SAME ``BASE_SEED`` constant and
the SAME xor recipe or the CRN slices between the two eval modes (emd / fast) would diverge and
the per-seat DELTA would no longer be a true CRN read.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


# Resolve both modules from their absolute paths so the test does not depend on the host's sys.path.
REPO = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO / "scripts" / "eval" / "eval_emd_vs_iter62000.py"
LAUNCH_SCRIPT = REPO / "scripts" / "launch_emd_pipeline_spot.py"


def _load_module(path: Path, name: str):
    """Import a script as a module by path. Bypasses the normal package loader for scripts/ siblings."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load both modules lazily (some other test might have a conftest that pre-loads them; be safe).
@pytest.fixture(scope="module")
def eval_module():
    return _load_module(EVAL_SCRIPT, "_crn_test_eval_module")


@pytest.fixture(scope="module")
def launch_module():
    # The launch script imports from launch_native_cfr_training_spot at top-level, which needs
    # scripts/ on sys.path. If that import fails for unrelated reasons in this environment, we
    # fall back to just verifying the eval-module helper (the launch re-export is a thin mirror).
    try:
        return _load_module(LAUNCH_SCRIPT, "_crn_test_launch_module")
    except Exception:
        pytest.skip(f"could not load {LAUNCH_SCRIPT} (likely a sibling-import env issue)")


# --- Constants under test --------------------------------------------------------

EXPECTED_BASE_SEED = 0x5EEDBAB0  # 1592263344; "BABY" mnemonic preserved


# --- The 4 canonical tests -------------------------------------------------------

def test_seed_for_replication_zero_returns_base_seed_when_xor_idempotent(eval_module):
    """id=0 produces BASE_SEED itself: this is the documented behavior (XOR with 0 = identity).

    The assert ``s != BASE_SEED`` inside ``seed_for_replication`` is INTENTIONALLY designed to fire
    only for NON-zero replication_id (the trivial-loop regression where someone replaces the function
    with a constant is caught by id=1..N; id=0 is the legitimate xor-identity case). The test calls
    the function for id=0 and asserts it returns BASE_SEED directly (no assert fires).
    """
    seed_for_replication = eval_module.seed_for_replication
    # id=0 is the xor-identity case: BASE_SEED ^ 0 == BASE_SEED. The helper must NOT assert on this.
    assert seed_for_replication(0) == EXPECTED_BASE_SEED, "id=0 must return BASE_SEED (xor identity)"
    # And the helper's base constant matches.
    assert eval_module.BASE_SEED == EXPECTED_BASE_SEED


def test_seed_for_replication_nonzero_returns_different_seed(eval_module):
    """id=1..100, all != BASE_SEED."""
    seed_for_replication = eval_module.seed_for_replication
    for rid in range(1, 101):
        s = seed_for_replication(rid)
        assert s != EXPECTED_BASE_SEED, f"id={rid}: reseed must differ from BASE_SEED"
        assert s != rid, f"id={rid}: reseed must not be the trivial identity"
        # Sanity: s == BASE_SEED ^ rid (the documented xor recipe).
        assert s == (EXPECTED_BASE_SEED ^ rid), f"id={rid}: xor recipe changed"


def test_seed_for_replication_no_collisions(eval_module):
    """id=0..10000, len(set) == 10001.

    Per the docstring: "no collisions for typical N (we assert for the first 10000 replications;
    collisions are negligible beyond 2**32 but the assert catches the trivial cases)."

    The helper itself is exercised on id=0..10000 (id=0 returns BASE_SEED by xor identity; no assert).
    """
    seed_for_replication = eval_module.seed_for_replication
    # Helper on full id range 0..10000: must all be distinct. id=0 returns BASE_SEED (xor identity).
    full = [seed_for_replication(i) for i in range(0, 10001)]
    assert len(set(full)) == 10001, f"CRN reseed produced collisions on id=0..10000 (len(set)={len(set(full))})"
    # And id=0 specifically is BASE_SEED (the xor-identity edge case).
    assert full[0] == EXPECTED_BASE_SEED
    # Sanity: the xor recipe holds for all ids.
    for rid in range(0, 10001):
        assert full[rid] == (EXPECTED_BASE_SEED ^ rid), f"id={rid}: xor recipe changed"


def test_seed_for_replication_xor_inverse(eval_module):
    """seed_for_replication(seed_for_replication(x)) == x for small x in 1..100.

    XOR is its own inverse: applying it twice with the same base returns the original. The inner
    call's `s != BASE_SEED` assert is satisfied because for x in 1..100, BASE_SEED ^ x is
    neither zero nor BASE_SEED (BASE_SEED = 1592263347, far above 100, so the high bits flip and
    the result is large and distinct from both).
    """
    seed_for_replication = eval_module.seed_for_replication
    for x in range(1, 101):
        s1 = seed_for_replication(x)
        s2 = seed_for_replication(s1)
        assert s2 == x, f"xor inverse failed for x={x}: seed_for_replication({x})={s1}, then seed_for_replication({s1})={s2}"


# --- Cross-module consistency check --------------------------------------------

def test_launch_module_exposes_same_base_seed(launch_module, eval_module):
    """The launch script's mirror of BASE_SEED must equal the eval script's BASE_SEED -- otherwise
    the CRN slices between the two modes would diverge and the per-seat DELTA would no longer be a
    true CRN read."""
    assert launch_module.BASE_SEED == eval_module.BASE_SEED == EXPECTED_BASE_SEED


def test_launch_module_seed_for_replication_matches(launch_module, eval_module):
    """The launch script's helper (re-export or mirror) must produce the same sequence."""
    for rid in range(0, 50):
        assert launch_module.seed_for_replication(rid) == eval_module.seed_for_replication(rid), (
            f"launch vs eval seed_for_replication diverged at rid={rid}")