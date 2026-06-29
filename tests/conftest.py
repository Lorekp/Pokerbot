"""Pytest config for the search-speed/quality suite.

The blueprint-CV control variate toggle (POKERBOT_BLUEPRINT_CV) is a PROCESS-GLOBAL flag set on
``pokerbot.search.pluribus_rollout._BLUEPRINT_CV``. Tests that mutate it (test_pluribus_rollout.py)
need to leave it OFF for the byte-identity gates downstream (test_native_cfv_zero_deviation.py,
test_pluribus_search.py, test_search_budget_robustness.py). Without this autouse fixture, pytest's
test ordering leaks the CV-ON state across files and breaks the bit-identity assertions.

The clean fix: every test starts with CV OFF. Tests that want the CV ON (the variance-reduction
bench) opt-in explicitly via `set_blueprint_cv(True)`. Tests that mutate the toggle restore it
in their own teardown.
"""

from __future__ import annotations

import os

# Set the test-mode flag at module load so any import-time guards see it. The blueprint
# snapshot validator defaults to strict=True in production; tests that pass an empty {}
# snapshot to PluribusReBeLSolver would otherwise raise at construction. Production code
# paths never set this env var; the bypass is unreachable there.
os.environ.setdefault("POKERBOT_TEST_MODE", "1")


def pytest_configure(config):
    """Re-assert POKERBOT_TEST_MODE for the test session.

    Belt-and-suspenders: the env var is also set at module load (above) but some
    test-runner environments (e.g. xdist workers, plugins that fork) can spawn a
    child interpreter that may not inherit the var. Setting it in pytest_configure
    guarantees the var is present BEFORE any test module is collected.
    """
    os.environ["POKERBOT_TEST_MODE"] = "1"


import pytest


@pytest.fixture(autouse=True, scope="session")
def _set_test_mode_for_session():
    """Session-scoped fixture: re-assert the POKERBOT_TEST_MODE env var.

    Some pytest configurations (e.g. xdist workers, subprocess plugins) can spawn a
    child interpreter that inherits the env, then a test module's own
    ``os.environ.setdefault`` call may have a different ordering effect than ours.
    Re-asserting here on a session-scoped autouse fixture makes the contract robust
    against any module-load order.
    """
    os.environ["POKERBOT_TEST_MODE"] = "1"
    yield


@pytest.fixture(autouse=True)
def _reset_blueprint_cv():
    """Ensure the blueprint-CV control variate is OFF at the start of every test.

    The CV is a process-global toggle on ``pokerbot.search.pluribus_rollout._BLUEPRINT_CV``.
    The byte-identity gate tests (test_native_cfv_zero_deviation.py,
    test_pluribus_search.py::test_zero_deviation_bit_identity,
    test_search_budget_robustness.py) require CV=OFF — the algebraically-zero / FP-non-zero
    shift introduced by the CV breaks strict ``allclose(atol=1e-9)`` assertions and the
    budget-bound metadata downstream of the starved-solve classifier.

    Tests that exercise the CV path (test_pluribus_rollout.py) toggle it ON in their own
    setup and restore it in teardown; this fixture is the safety net for the byte-identity
    suite and any future test that runs alongside.
    """
    # Import here so the conftest does not force-load pluribus_rollout for tests that
    # do not import it (test collection stays fast).
    try:
        from pokerbot.search.pluribus_rollout import set_blueprint_cv
    except ImportError:
        yield
        return
    set_blueprint_cv(False)
    yield
    # Leave OFF after the test too — matches the default and prevents leakage to the next test.
    set_blueprint_cv(False)