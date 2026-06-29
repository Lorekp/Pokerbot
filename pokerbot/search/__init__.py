"""TurboReBeL subgame search and PBS tracking.

Lean EMD pipeline boxes import ``pokerbot.search`` only to reach ``native_solver``.
Those boxes intentionally do not install the GPU/training stack (flax, optax, torch), so
package import must not eagerly pull modules that depend on it. Heavy symbols remain
available in full environments; on lean boxes they become call-time placeholders with a
clear dependency error.
"""

try:
    from pokerbot.search.leaf_evaluator import HistoryValueEvaluator, MockHistoryValueEvaluator  # noqa: F401
    _LEAF_EVAL_OK = True
    _LEAF_EVAL_ERR = None
except ImportError as _e:  # noqa: BLE001 - surface the underlying missing module
    HistoryValueEvaluator = None  # type: ignore[assignment]
    MockHistoryValueEvaluator = None  # type: ignore[assignment]
    _LEAF_EVAL_OK = False
    _LEAF_EVAL_ERR = _e


def _leaf_eval_unavailable(name):  # pragma: no cover - only hit on lean boxes
    def _raise(*_a, **_kw):
        raise ImportError(
            f"pokerbot.search.{name} requires flax/optax/torch (GPU training env), "
            f"which is not installed here. Underlying error: {_LEAF_EVAL_ERR}"
        )
    return _raise


def _heavy_unavailable(name, underlying):  # pragma: no cover - only hit on lean boxes
    def _raise(*_a, **_kw):
        raise ImportError(
            f"pokerbot.search.{name} requires a heavy dep (flax/optax/torch) that is not "
            f"installed here (lean box). Underlying error: {underlying}"
        )
    return _raise


if not _LEAF_EVAL_OK:
    HistoryValueEvaluator = _leaf_eval_unavailable("HistoryValueEvaluator")
    MockHistoryValueEvaluator = _leaf_eval_unavailable("MockHistoryValueEvaluator")


try:
    from pokerbot.search.pbs_tracker import PBSTracker  # noqa: F401
    _PBS_OK = True
    _PBS_ERR = None
except ImportError as _e:  # noqa: BLE001 - lean box has no torch
    PBSTracker = None  # type: ignore[assignment,misc]
    _PBS_OK = False
    _PBS_ERR = _e

if not _PBS_OK:
    PBSTracker = _heavy_unavailable("PBSTracker", _PBS_ERR)


try:
    from pokerbot.search.turbo_rebel_search import SolveResult, TurboReBeLConfig, TurboReBeLSolver  # noqa: E402,F401
    _TURBO_OK = True
    _TURBO_ERR = None
except ImportError as _e:  # noqa: BLE001 - lean box has no flax/torch stack
    _TURBO_OK = False
    _TURBO_ERR = _e

if not _TURBO_OK:
    SolveResult = _heavy_unavailable("SolveResult", _TURBO_ERR)
    TurboReBeLConfig = _heavy_unavailable("TurboReBeLConfig", _TURBO_ERR)
    TurboReBeLSolver = _heavy_unavailable("TurboReBeLSolver", _TURBO_ERR)

__all__ = [
    "HistoryValueEvaluator",
    "MockHistoryValueEvaluator",
    "PBSTracker",
    "TurboReBeLConfig",
    "TurboReBeLSolver",
    "SolveResult",
]
