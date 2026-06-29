"""Tournament P0/P1 hardening tests (REFUTE-TOURN-PROD §5 + §6).

Targets:
- P0 pickle hardening in `tournament.deep_provider`:
  * refuses non-project-root paths (RCE-sink mitigation)
  * verifies SHA-256 sidecar on load (refuses mismatch)
  * warns when no sidecar present
- P1 exception narrowing: every swallowed exception site must now
  * narrow to (KeyError, ValueError, IndexError, LinAlgError, OSError) [+ AttributeError/RuntimeError
    for neural_serve], and
  * not silently cache `None` permanently in `mtt_lifecycle._cache`.
"""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path

import pytest


def _import_deep_provider():
    from pokerbot.tournament import deep_provider
    return deep_provider


def _project_root() -> Path:
    # pokerbot/tournament/deep_provider.py -> 3 levels up = repo root
    return Path(__file__).resolve().parents[1] / "pokerbot" / "tournament" / "deep_provider.py"


def test_pickle_hardening_refuses_path_outside_snapshot_roots(tmp_path, monkeypatch):
    """P0: a path outside project-owned dirs must be refused before unpickle."""
    dp = _import_deep_provider()
    bad = tmp_path / "evil.pkl"
    bad.write_bytes(b"\x80\x04\x95\x00")  # would RCE if loaded
    # `monkeypatch.setattr` is overkill; just call __init__ and expect refusal.
    with pytest.raises(ValueError, match="not under any project snapshot root"):
        dp.BlueprintDeepProvider(str(bad))


def test_pickle_hardening_loads_legitimate_snapshot(tmp_path, monkeypatch):
    """P0: a snapshot under data/ with matching sidecar loads successfully."""
    # Place snapshot under the project's data/ dir (the only snapshot root that exists in CI).
    root = Path(_import_deep_provider().__file__).resolve().parent.parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "test_tourn_p0_hardening.pkl"
    target.write_bytes(pickle.dumps({"hello": "world"}))
    try:
        # No sidecar -> warning + load proceeds.
        dp = _import_deep_provider()
        # Bypass check_blueprint_serve_env (it would require the abstraction env vars).
        monkeypatch.setattr("pokerbot.eval.lbr.check_blueprint_serve_env", lambda strict=False: None)
        monkeypatch.setattr("pokerbot.eval.lbr.make_blueprint_policy", lambda snap, **kw: lambda env, seat: [0.25] * 4)
        prov = dp.BlueprintDeepProvider(str(target))
        assert prov._n_entries == 1  # dict has one key
    finally:
        target.unlink(missing_ok=True)
        sidecar = target.with_suffix(target.suffix + ".sha256")
        sidecar.unlink(missing_ok=True)


def test_pickle_hardening_sidecar_mismatch_refuses_load(tmp_path, monkeypatch):
    """P0: SHA-256 mismatch with sidecar must raise before unpickle."""
    import hashlib

    root = Path(_import_deep_provider().__file__).resolve().parent.parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "test_tourn_p0_mismatch.pkl"
    target.write_bytes(pickle.dumps({"a": 1}))
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar.write_text("0" * 64 + "  test_tourn_p0_mismatch.pkl\n", encoding="utf-8")
    try:
        dp = _import_deep_provider()
        monkeypatch.setattr("pokerbot.eval.lbr.check_blueprint_serve_env", lambda strict=False: None)
        monkeypatch.setattr("pokerbot.eval.lbr.make_blueprint_policy", lambda snap, **kw: lambda env, seat: [0.25] * 4)
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            dp.BlueprintDeepProvider(str(target))
    finally:
        target.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)


def test_pickle_hardening_uses_with_open_no_leak():
    """P0: source must use `with open(...)` — verified by static inspection."""
    src = _project_root().read_text(encoding="utf-8")
    assert "pickle.load(open(" not in src, "deep_provider.py still has unhardened `pickle.load(open(...))`"
    assert "with open(resolved, \"rb\") as fh" in src


def test_lbr_icm_hand_label_uses_narrow_except():
    """P1: `_hand_label` must narrow to (IndexError, ValueError) so encoder-skew cannot be swallowed."""
    src = Path(__file__).resolve().parents[1] / "pokerbot" / "tournament" / "lbr_icm.py"
    text = src.read_text(encoding="utf-8")
    # Find the _hand_label block
    i = text.find("def _hand_label(holes_seat)")
    assert i >= 0
    j = text.find("def _detect", i)
    block = text[i:j]
    assert "except (IndexError, ValueError)" in block, "_hand_label must narrow to (IndexError, ValueError)"


def test_lbr_icm_policy_fn_uses_narrow_except_and_logs():
    """P1: analytic-ICM cell policy must narrow + log + record degraded cells."""
    src = Path(__file__).resolve().parents[1] / "pokerbot" / "tournament" / "lbr_icm.py"
    text = src.read_text(encoding="utf-8")
    assert "_log = logging.getLogger(__name__)" in text
    # The pushed narrow `except (KeyError, ValueError, IndexError, LinAlgError, OSError)` is present
    assert "except (KeyError, ValueError, IndexError, np.linalg.LinAlgError, OSError):" in text
    assert "degraded_cells" in text, "degraded_cells must surface in ledger"


def test_mtt_lifecycle_no_longer_caches_none():
    """P1: transient solve failures must NOT poison _cache with None."""
    src = Path(__file__).resolve().parents[1] / "pokerbot" / "tournament" / "mtt_lifecycle.py"
    text = src.read_text(encoding="utf-8")
    assert "_abstains" in text, "mtt_lifecycle must add _abstains set"
    # The old poisoning line must be gone
    assert "_cache[key] = self_none" not in text, "mtt_lifecycle must not cache None permanently"


def test_pushfold_icm_uses_filerelative_path():
    """P1: _HU_EQ_PATH must resolve relative to __file__, not cwd."""
    src = Path(__file__).resolve().parents[1] / "pokerbot" / "tournament" / "pushfold_icm.py"
    text = src.read_text(encoding="utf-8")
    assert "Path(__file__).resolve().parent" in text, "pushfold_icm must anchor _HU_EQ_PATH to __file__"


def test_neural_serve_uses_narrow_excepts():
    """P1: neural_serve excepts narrowed from `Exception` to specific types."""
    src = Path(__file__).resolve().parents[1] / "pokerbot" / "tournament" / "neural_serve.py"
    text = src.read_text(encoding="utf-8")
    # The narrow form is present (must replace BOTH old `except Exception` sites in this module)
    assert "except (KeyError, ValueError, IndexError, OSError, AttributeError, RuntimeError):" in text
    assert text.count("except Exception:") <= 0, "neural_serve must have ZERO bare `except Exception:` left"


def test_regime_battery_has_logger_and_narrow_excepts():
    """P1: regime_battery must log and use narrow excepts."""
    src = Path(__file__).resolve().parents[1] / "pokerbot" / "tournament" / "regime_battery.py"
    text = src.read_text(encoding="utf-8")
    assert "_log = logging.getLogger(__name__)" in text
    # Both narrow sites present
    assert text.count("except (KeyError, ValueError, IndexError, np.linalg.LinAlgError, OSError):") >= 2
    assert "except Exception:" not in text or text.count("except Exception:") == 0
