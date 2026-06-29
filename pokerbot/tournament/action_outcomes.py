"""Per-action chip-outcome tables from a resumed deep node — the deep-ICM evaluator's rollout producer
(workflow wexmo5p7v P1). This is what feeds ``decide_deep`` / ``icm_action_value``.

For each candidate hero action at a deep near-money node, clone the state, apply the action, and advance to
terminal ``rollouts`` times under ``policies`` (the rollout strategy for every seat, incl. hero's later
streets) with the mean-preserving equity leaf (``rollout_leaf.equity_allin_settle``). The result is
``{action_label -> [(prob, chip_delta_vector), ...]}`` — exactly the ``action_outcomes`` shape
``decide_deep`` consumes. Built on ``game.advance_to_terminal`` (the resumable seam from P0), so it
genuinely resumes a mid-street clone rather than re-dealing.
"""

from __future__ import annotations

import copy
import random

from pokerbot.engine.game import advance_to_terminal
from pokerbot.tournament.rollout_leaf import equity_allin_settle


def action_label(action) -> str:
    """A stable label for a candidate action (type + size), used as the outcome-table key."""
    t = getattr(action.action_type, "name", str(action.action_type))
    amt = getattr(action, "amount", None)
    return t if amt in (None, 0) else f"{t}:{round(float(amt), 3)}"


def build_action_outcomes(state, candidate_actions, *, policies=None, rollouts: int = 64,
                          n_runouts: int = 60, rng: random.Random | None = None) -> dict:
    """Outcome distribution per candidate action: ``{label -> [(prob, chip_delta_vector), ...]}``.

    ``chip_delta_vector[seat]`` is the seat's NET chip change for the hand if the hero takes that action and
    everyone plays ``policies`` to terminal (equity-settled). ``policies`` maps seat -> ``callable(state) ->
    Action`` (default ``None`` for every seat = the engine's check/call/fold fallback — a valid, cheap
    rollout; pass a blueprint/field policy for higher fidelity). Each rollout is an equal-probability
    outcome. Conservation: every chip_delta_vector sums to ~0 (zero-sum hand)."""
    rng = rng or random.Random(0)
    policies = policies or {}
    n = len(state.players)
    out: dict = {}
    for act in candidate_actions:
        vectors = []
        for _ in range(rollouts):
            clone = copy.deepcopy(state)
            r = random.Random(rng.getrandbits(32))            # per-rollout stream (reproducible from rng)
            def _leaf(s, nb, _r=r):
                return equity_allin_settle(s, nb, n_runouts=n_runouts, rng=_r)
            clone.apply_action(act)
            if clone.is_terminal():
                deltas = clone.resolve_showdown()
            else:
                clone.advance_to_next_actor()
                _, deltas = advance_to_terminal(clone, policies, rng=r, allin_settle=_leaf)
            vectors.append([float(deltas.get(i, 0.0)) for i in range(n)])
        p = 1.0 / len(vectors) if vectors else 0.0
        out[action_label(act)] = [(p, v) for v in vectors]
    return out
