"""Env-gated tournament/cash training distribution for the HU neural-CFR reset: per-deal effective
stack depth (POKERBOT_HU_DEPTH_BB) and per-player ante (POKERBOT_HU_ANTE_BB / ANTE_ZERO_PROB). Defaults
reproduce the legacy fixed-100bb no-ante reset. Tournament HU is late & high-blind -> shallow band.
"""

from __future__ import annotations

import jax
import numpy as np

from pokerbot.engine import jax_engine as je
from pokerbot.training.cfr_env import nlhe_hu_env_spec

_LIVE = (3, 5)


def _eff_bb(st) -> float:
    # starting effective depth = stack + committed_total (blinds/antes were moved out of the stack);
    # both live seats share the sampled depth, so the min is the covering stack.
    chips = min(int(st.stacks[i]) + int(st.committed_total[i]) for i in _LIVE)
    return chips / je.CHIP_UNIT


def test_default_is_fixed_100bb_no_ante(monkeypatch):
    for k in ("POKERBOT_HU_DEPTH_BB", "POKERBOT_HU_ANTE_BB", "POKERBOT_HU_ANTE_ZERO_PROB"):
        monkeypatch.delenv(k, raising=False)
    env = nlhe_hu_env_spec()
    for s in range(8):
        st = env.reset(jax.random.PRNGKey(s))
        assert _eff_bb(st) == je.STARTING_STACK_BB     # always 100bb
        assert int(st.pot_dead) == 0                   # no ante


def test_depth_band_is_shallow_and_within_range(monkeypatch):
    monkeypatch.setenv("POKERBOT_HU_DEPTH_BB", "5,60")
    monkeypatch.delenv("POKERBOT_HU_ANTE_BB", raising=False)
    env = nlhe_hu_env_spec()
    bbs = np.array([_eff_bb(env.reset(jax.random.PRNGKey(s))) for s in range(80)])
    assert bbs.min() >= 5 - 0.5 and bbs.max() <= 60 + 0.5      # inside the band (rounding tol)
    assert bbs.min() < 12 and bbs.max() > 40                   # genuinely spans the band
    assert (bbs < 20).mean() > 0.30                            # log-uniform -> shallow-weighted


def test_ante_posts_into_pot(monkeypatch):
    monkeypatch.setenv("POKERBOT_HU_ANTE_BB", "0.125")
    monkeypatch.delenv("POKERBOT_HU_DEPTH_BB", raising=False)
    monkeypatch.delenv("POKERBOT_HU_ANTE_ZERO_PROB", raising=False)
    env = nlhe_hu_env_spec()
    st = env.reset(jax.random.PRNGKey(0))
    assert int(st.pot_dead) == 2 * round(0.125 * je.CHIP_UNIT)   # both live seats ante


def test_ante_zero_prob_mixes_cash_and_tournament(monkeypatch):
    monkeypatch.setenv("POKERBOT_HU_ANTE_BB", "0.125")
    monkeypatch.setenv("POKERBOT_HU_ANTE_ZERO_PROB", "0.5")
    monkeypatch.delenv("POKERBOT_HU_DEPTH_BB", raising=False)
    env = nlhe_hu_env_spec()
    pots = [int(env.reset(jax.random.PRNGKey(s)).pot_dead) for s in range(40)]
    assert any(p == 0 for p in pots) and any(p > 0 for p in pots)   # both regimes appear


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
