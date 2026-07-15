"""Fixed-source-depth postflop blueprint projection regressions."""

from types import SimpleNamespace

import numpy as np
import pytest

from pokerbot.engine.jax_engine import CHIP_UNIT
from pokerbot.engine.np_engine import ALL_IN, BET_BUCKET_START, _action_amount
from pokerbot.search.postflop_projection import (
    project_postflop_source_policy,
    project_postflop_state_to_reference,
)
from pokerbot.search.pluribus_search import PluribusReBeLSolver


def _state(*, depth_bb=200, committed=10, pot=20):
    target = int(depth_bb * CHIP_UNIT)
    invested = int(committed * CHIP_UNIT)
    return SimpleNamespace(
        street=np.int32(1),
        stacks=np.asarray([target - invested, target - invested], dtype=np.int32),
        committed_total=np.asarray([invested, invested], dtype=np.int32),
        committed_street=np.asarray([0, 0], dtype=np.int32),
        seat_status=np.asarray([2, 2], dtype=np.int8),
        current_bet=np.int32(0),
        min_raise=np.int32(CHIP_UNIT),
        pot_dead=np.int32(pot * CHIP_UNIT),
        acted=np.asarray([False, False]),
        last_raise_full=np.asarray(True),
    )


def test_deep_state_caps_starting_totals_without_changing_public_money():
    state = _state(depth_bb=200, committed=10, pot=20)
    original_stacks = state.stacks.copy()

    shadow, effective = project_postflop_state_to_reference(state)

    assert effective == pytest.approx(200.0)
    np.testing.assert_array_equal(
        shadow.stacks + shadow.committed_total,
        np.full(2, 100 * CHIP_UNIT, dtype=np.int32),
    )
    np.testing.assert_array_equal(shadow.committed_total, state.committed_total)
    np.testing.assert_array_equal(shadow.committed_street, state.committed_street)
    assert int(shadow.pot_dead) == int(state.pot_dead)
    np.testing.assert_array_equal(state.stacks, original_stacks)  # no live-state mutation


def test_shallow_or_fully_invested_source_state_abstains():
    shallow = _state(depth_bb=120, committed=10)
    assert project_postflop_state_to_reference(shallow)[0] is None

    over = _state(depth_bb=200, committed=100)
    assert project_postflop_state_to_reference(over)[0] is None


def test_source_allin_mass_maps_to_largest_nonterminal_source_bucket():
    source = _state(depth_bb=100, committed=0, pot=20)
    source_legal = np.zeros(11, dtype=bool)
    source_legal[[1, 3, 4, 5, 6, 7, ALL_IN]] = True
    live_legal = source_legal.copy()
    probs = np.zeros(11, dtype=np.float64)
    probs[1], probs[ALL_IN] = 0.25, 0.75

    out, target, moved = project_postflop_source_policy(
        probs, source, 0, source_legal, live_legal)

    nonterminal = [a for a in range(BET_BUCKET_START, ALL_IN)
                   if source_legal[a]
                   and int(_action_amount(source, 0, a)) < int(source.stacks[0])]
    expected = max(nonterminal, key=lambda a: int(_action_amount(source, 0, a)))
    assert target == expected
    assert moved == pytest.approx(0.75)
    assert out[ALL_IN] == 0.0
    assert out[target] == pytest.approx(0.75)
    assert out[1] == pytest.approx(0.25)
    assert out.sum() == pytest.approx(1.0)


def test_solver_trust_boundary_addresses_only_source_depth_when_deep():
    state = _state(depth_bb=200, committed=5, pot=20)
    legal = np.zeros(11, dtype=bool)
    legal[[1, 3, 4, 5, 6, 7, ALL_IN]] = True
    source_row = np.zeros(11, dtype=np.float64)
    source_row[1], source_row[ALL_IN] = 0.2, 0.8

    class Adapter:
        calls = []

        def blueprint_with_status(self, env, hero):
            depth = float(np.min(env.stacks + env.committed_total)) / CHIP_UNIT
            self.calls.append(depth)
            return ((source_row.copy(), True) if depth == pytest.approx(100.0)
                    else (np.ones(11) / 11.0, False))

    solver = object.__new__(PluribusReBeLSolver)
    solver.adapter = Adapter()
    solver.config = SimpleNamespace(preflop_blueprint_band_bb=(80.0, 130.0))
    solver._pushfold_prior = lambda base, hero, mask: None
    solver._legal = lambda base, hero: legal.copy()

    out, meta = solver._trusted_root_blueprint_with_meta(state, 0, legal)

    # A deep direct key may alias the artifact's saturated SPR bucket.  It must
    # not be consulted, even when the adapter would report a direct hit.
    assert solver.adapter.calls == pytest.approx([100.0])
    assert meta["fallback_source"] == "postflop_depth_projected_blueprint"
    assert meta["blueprint_source_hit"] is False
    assert meta["projection_source_hit"] is True
    assert meta["blueprint_direct_skipped_offdepth"] is True
    assert meta["postflop_projection_effective_bb"] == pytest.approx(200.0)
    assert meta["postflop_projection_reference_bb"] == pytest.approx(100.0)
    assert out[ALL_IN] == 0.0
    assert out.sum() == pytest.approx(1.0)


def test_deep_direct_alias_is_rejected_when_source_row_is_missing():
    state = _state(depth_bb=400, committed=5, pot=20)
    legal = np.zeros(11, dtype=bool)
    legal[[1, 3, 4, 5, 6, 7, ALL_IN]] = True
    aliased_direct = np.zeros(11, dtype=np.float64)
    aliased_direct[ALL_IN] = 1.0

    class Adapter:
        calls = []

        def blueprint_with_status(self, env, hero):
            depth = float(np.min(env.stacks + env.committed_total)) / CHIP_UNIT
            self.calls.append(depth)
            # The live 400bb key falsely looks like a hit; the 100bb source row
            # is absent.  The trust boundary must abstain, not return the jam.
            if depth == pytest.approx(400.0):
                return aliased_direct.copy(), True
            return np.ones(11) / 11.0, False

    solver = object.__new__(PluribusReBeLSolver)
    solver.adapter = Adapter()
    solver.config = SimpleNamespace(preflop_blueprint_band_bb=(80.0, 130.0))
    solver._pushfold_prior = lambda base, hero, mask: None
    solver._legal = lambda base, hero: legal.copy()

    out, meta = solver._trusted_root_blueprint_with_meta(state, 0, legal)

    assert out is None
    assert solver.adapter.calls == pytest.approx([100.0])
    assert meta["projection_source_hit"] is False
    assert meta["blueprint_direct_skipped_offdepth"] is True


def test_in_band_postflop_still_uses_direct_row():
    state = _state(depth_bb=100, committed=5, pot=20)
    legal = np.zeros(11, dtype=bool)
    legal[[1, 5, ALL_IN]] = True
    direct = np.zeros(11, dtype=np.float64)
    direct[1], direct[5] = 0.3, 0.7

    class Adapter:
        calls = []

        def blueprint_with_status(self, env, hero):
            depth = float(np.min(env.stacks + env.committed_total)) / CHIP_UNIT
            self.calls.append(depth)
            return direct.copy(), True

    solver = object.__new__(PluribusReBeLSolver)
    solver.adapter = Adapter()
    solver.config = SimpleNamespace(preflop_blueprint_band_bb=(80.0, 130.0))
    solver._pushfold_prior = lambda base, hero, mask: None

    out, meta = solver._trusted_root_blueprint_with_meta(state, 0, legal)

    assert solver.adapter.calls == pytest.approx([100.0])
    assert meta["fallback_source"] == "blueprint"
    assert meta["blueprint_source_hit"] is True
    assert out[1] == pytest.approx(0.3)
    assert out[5] == pytest.approx(0.7)
