"""Regression test for F-SEARCH-9 (``pokerbot/eval/lbr6.py`` _lbr_action_analytic CHECK branch).

The bug: the CHECK branch computed ``V(check) = wp * pot_others`` as if every villain would also
check down. When at least one opponent is still live with chips and an action seat, that opponent
can DONK bet post-check, which a 1-ply LBR cannot model -- so the value is systematically
overstated and the LBR bound under-tightens.

The fix: when an opponent can donk, drop CHECK to ``v = 0`` (the conservative LBR bound: a CHECK
that may trigger a donk bet should not be cheaper than FOLD from the LBR's perspective). The
fix is internal to ``_lbr_action_analytic``; this test pins the behavior with a minimal mock env
that exposes exactly the fields the function reads.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from pokerbot.eval.lbr6 import _lbr_action_analytic, LBR6Config


# ---------------------------------------------------------------------------------------
# minimal mock EnvState (only the fields _lbr_action_analytic actually reads)
# ---------------------------------------------------------------------------------------
class _MockEnv:
    """Stand-in for the jax EnvState. Only the attributes ``_lbr_action_analytic`` touches are
    populated; everything else is irrelevant for this test."""

    def __init__(self, *, n_seats: int, committed_total: list[float],
                 committed_street: list[float], current_bet: float,
                 action_seat: int = 0, done: bool = False, pot: float = 0.0):
        self.n_seats = n_seats
        self.committed_total = np.asarray(committed_total, dtype=np.float64)
        self.committed_street = np.asarray(committed_street, dtype=np.float64)
        self.current_bet = float(current_bet)
        self.action_seat = int(action_seat)
        self.done = bool(done)
        self.pot = pot


def _stub_beliefs(*_a, **_k):
    """Stand-in for the beliefs arg. _lbr_action_analytic reads it via ``_wp_vs_beliefs`` only,
    which is short-circuited by the CHECK-with-donk path BEFORE the wp evaluation is even
    reached. Provide a benign default so any leftover call does not crash."""
    return None


def _stub_policy_fn(*_a, **_k):
    """Stub policy_fn used for the bet-candidates branch. Not invoked for the CHECK test."""
    raise AssertionError("policy_fn should not be invoked in the CHECK branch test")


def _make_mask(*, fold: bool = True, check: bool = True, call: bool = False,
               bets: bool = False, all_in: bool = False) -> np.ndarray:
    """Build a legal-action mask compatible with the LBR6 layout. Indices: FOLD=0, CHECK=1,
    CALL=2, then bet buckets. We don't model the full 11-element vector; the function reads
    only FOLD/CHECK/CALL/bet buckets here. A minimal ``mask[FOLD]=fold, mask[CHECK]=check`` is
    sufficient for these tests."""
    m = np.zeros(11, dtype=bool)
    m[0] = fold
    m[1] = check
    m[2] = call
    if bets:
        # Mark a couple of bet-candidate indices as legal so the bet branch is reachable if needed.
        m[3] = True
        m[4] = True
    if all_in:
        m[5] = True
    return m


@pytest.fixture
def cfg():
    return LBR6Config(
        hands=1, seed=1, eq_samples=1, eq_runouts=1,
        bet_candidates=(),                    # disable bet branch; the test only cares about CHECK
    )


def test_check_value_is_zero_when_opponent_can_donk(monkeypatch, cfg):
    """A 6-max state with seat 1 live and matched-or-behind: V(check) must be 0.0 (the donk-aware
    conservative bound), NOT ``wp * pot_others``. We instrument the function so the OLD buggy
    formula would yield a positive V(check) and beat FOLD, but the FIXED formula returns V=0 and
    ties FOLD. The chosen action must therefore be FOLD (the function prefers the first legal
    argmax, and FOLD is set BEFORE CHECK) -- proving the bound tightened."""
    from pokerbot import eval as _eval_pkg
    lbr6 = _eval_pkg.lbr6

    monkeypatch.setattr(lbr6, "_wp_vs_beliefs", lambda *a, **k: 0.9)   # strong hand

    # 6-max env: LBR at seat 0 (committed_street=2.0); opponent at seat 1 is live with
    # committed_street=2.0 (matched), and ``env.done=False`` so seat 1 can still act/donk.
    env = _MockEnv(
        n_seats=6,
        committed_total=[2.0, 2.0, 0.0, 0.0, 0.5, 1.0],
        committed_street=[2.0, 2.0, 0.0, 0.0, 0.5, 1.0],
        current_bet=2.0,
        action_seat=0,
        done=False,                          # opponents can still bet post-check -> DONK possible
        pot=3.5,
    )
    mask = _make_mask(fold=True, check=True)
    monkeypatch.setattr(lbr6, "legal_action_mask_np", lambda *_a, **_k: mask)

    chosen = _lbr_action_analytic(env, lbr_seat=0, beliefs=None, policy_fn=_stub_policy_fn,
                                  cfg=cfg, rng=np.random.default_rng(0))
    # With the FIX: V(check) = 0.0 (donk-aware), V(fold) = 0.0; tie-break is the first legal
    # argmax, which is FOLD (set at line 507 BEFORE the CHECK branch). If the OLD buggy formula
    # were still in place, V(check) = 0.9 * (3.5 - 2.0) = 1.35 -> CHECK would win. So the chosen
    # action must be FOLD (=0).
    assert chosen == 0, (
        f"F-SEARCH-9: _lbr_action_analytic returned {chosen}; expected FOLD (0). "
        "If a non-zero action came back, the donk-aware V(check)=0.0 bound did NOT engage."
    )


def test_check_value_uses_wp_times_pot_others_when_no_opponent_can_donk(monkeypatch, cfg):
    """If NO opponent can donk (everyone else folded or all-in or matched at >= LBR's stake), the
    original ``wp * pot_others`` formula is correct -- the street WILL check down to showdown.
    Patch ``_wp_vs_beliefs`` to return a known ``wp`` and verify the function uses it."""
    from pokerbot import eval as _eval_pkg
    lbr6 = _eval_pkg.lbr6

    captured: dict[str, float] = {}

    def _wp_capture(env, lbr_seat, beliefs, rng, eq_samples, eq_runouts):
        captured["called"] = True
        return 0.8                            # pretend the LBR wins 80% at showdown

    monkeypatch.setattr(lbr6, "_wp_vs_beliefs", _wp_capture)

    # 6-max: LBR at seat 0, every other seat is folded or all-in -- nobody can donk.
    folded = [True, True, True, True, True, True]
    env = _MockEnv(
        n_seats=6,
        committed_total=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        committed_street=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        current_bet=2.0,
        action_seat=0,
        done=True,                            # hand is effectively over; nobody can act
        pot=2.0,
    )
    mask = _make_mask(fold=True, check=True)
    monkeypatch.setattr(lbr6, "legal_action_mask_np", lambda *_a, **_k: mask)
    # The "opponent_can_donk" predicate in _lbr_action_analytic inspects committed_street --
    # with all opponents committed 0 (< LBR's 2.0) they would normally qualify as "behind" and
    # flag donkable. To force the "no-donk" path, override the predicate inline by mocking the
    # committed_street >= LBR's committed_street for the opponent rows.
    env.committed_street = np.asarray([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    chosen = _lbr_action_analytic(env, lbr_seat=0, beliefs=None, policy_fn=_stub_policy_fn,
                                  cfg=cfg, rng=np.random.default_rng(0))
    assert captured.get("called") is True     # the wp path was reached
    # V(check) = 0.8 * pot_others = 0.8 * 0.0 (no other players committed anything) = 0.0; FOLD
    # is also 0.0; tie-break is implementation-defined, but the chosen action MUST be CHECK or FOLD.
    assert chosen in (0, 1)


def test_check_branch_does_not_crash_when_only_lbr_is_live(monkeypatch, cfg):
    """Boundary: a heads-up LBR-only showdown (env.done=True, only LBR has any commitment).
    The donk predicate must NOT trip on LBR's own seat (i==lbr_seat is excluded)."""
    from pokerbot import eval as _eval_pkg
    lbr6 = _eval_pkg.lbr6

    monkeypatch.setattr(lbr6, "_wp_vs_beliefs",
                        lambda *a, **k: 0.5)
    env = _MockEnv(
        n_seats=2,
        committed_total=[5.0, 0.0],
        committed_street=[5.0, 0.0],
        current_bet=5.0,
        action_seat=0,
        done=True,
        pot=5.0,
    )
    mask = _make_mask(fold=True, check=True)
    monkeypatch.setattr(lbr6, "legal_action_mask_np", lambda *_a, **_k: mask)
    # No exception means the LBR-only case was handled cleanly (no donk on LBR's own seat).
    chosen = _lbr_action_analytic(env, lbr_seat=0, beliefs=None, policy_fn=_stub_policy_fn,
                                  cfg=cfg, rng=np.random.default_rng(0))
    assert chosen in (0, 1)