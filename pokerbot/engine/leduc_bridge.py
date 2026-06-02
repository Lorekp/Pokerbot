"""Thin Python view over canonical JAX Leduc — no duplicate game logic."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.leduc_jax import (
    ANTE,
    CALL,
    MAX_HISTORY,
    LeducState,
    apply_action,
    init_from_deal,
    init_hand,
    legal_actions,
)

_APPLY_JIT = jax.jit(apply_action)
_APPLY_CACHE: dict[tuple[tuple, int], PyLeducState] = {}


def _fields_from_jax(s: LeducState) -> dict:
    """One device sync per transition — hot path stays pure Python afterward."""
    cards = np.asarray(s.cards)
    stacks = np.asarray(s.stacks)
    committed = np.asarray(s.committed)
    invested = np.asarray(s.invested)
    acted = np.asarray(s.acted)
    n = int(s.hist_len)
    hist: tuple[int, ...] = ()
    if n > 0:
        hist = tuple(int(x) for x in np.asarray(s.hist_action[:n]).tolist())
    return {
        "cards": (int(cards[0]), int(cards[1])),
        "public": int(s.public),
        "pot": int(s.pot),
        "stacks": (int(stacks[0]), int(stacks[1])),
        "committed": (int(committed[0]), int(committed[1])),
        "invested": (int(invested[0]), int(invested[1])),
        "current_bet": int(s.current_bet),
        "raises": int(s.raises),
        "round": int(s.round),
        "acting": int(s.acting),
        "acted": (bool(acted[0]), bool(acted[1])),
        "done": bool(s.done),
        "rewards": (float(s.rewards[0]), float(s.rewards[1])),
        "hist": hist,
        "deal_public": int(s.fixed_public),
        "_s": s,
    }


@dataclass(frozen=True)
class PyLeducState:
    """Immutable Python handle; transitions delegate to ``leduc_jax.apply_action``."""

    cards: tuple[int, int]
    public: int
    pot: int
    stacks: tuple[int, int]
    committed: tuple[int, int]
    invested: tuple[int, int]
    current_bet: int
    raises: int
    round: int
    acting: int
    acted: tuple[bool, bool]
    done: bool
    rewards: tuple[float, float]
    hist: tuple[int, ...]
    deal_public: int
    _s: LeducState

    @staticmethod
    def from_jax(s: LeducState) -> PyLeducState:
        return PyLeducState(**_fields_from_jax(s))

    @property
    def jax(self) -> LeducState:
        return self._s


def py_from_jax(state: LeducState) -> PyLeducState:
    return PyLeducState.from_jax(state)


def jax_from_py(state: PyLeducState) -> LeducState:
    return state._s


def py_fingerprint(state: PyLeducState) -> tuple:
    return (
        state.cards,
        state.public,
        state.pot,
        state.stacks,
        state.committed,
        state.invested,
        state.current_bet,
        state.raises,
        state.round,
        state.acting,
        state.acted,
        state.done,
        state.hist,
    )


def apply_action_py(state: PyLeducState, action: int) -> PyLeducState:
    # fixed_public/deal_public must be in the cache key: pre-public states share the same
    # observable fingerprint across deals but diverge when the public card is dealt.
    key = (state.deal_public, py_fingerprint(state), action)
    cached = _APPLY_CACHE.get(key)
    if cached is not None:
        return cached
    nxt = PyLeducState.from_jax(_APPLY_JIT(state._s, action))
    _APPLY_CACHE[key] = nxt
    return nxt


def py_state_from_deal(cards: tuple[int, int], deal_public: int) -> PyLeducState:
    return PyLeducState.from_jax(init_from_deal(cards, deal_public))


def init_py_from_key(rng_key) -> PyLeducState:
    s = init_hand(rng_key)
    s2 = apply_action(apply_action(s, CALL), CALL)
    cards = (int(np.asarray(s.cards)[0]), int(np.asarray(s.cards)[1]))
    return PyLeducState.from_jax(init_from_deal(cards, int(s2.public)))


def py_legal(state: PyLeducState) -> np.ndarray:
    seat = state.acting
    to_call = max(0, state.current_bet - state.committed[seat])
    from pokerbot.engine.leduc_jax import MAX_RAISES

    can_raise = (state.raises < MAX_RAISES) and (state.stacks[seat] > to_call)
    return np.array([True, True, can_raise], dtype=bool)


def py_info_key(state: PyLeducState, seat: int) -> tuple:
    pub = state.public if state.public >= 0 else -1
    to_call = max(0, state.current_bet - state.committed[seat])
    return (seat, state.cards[seat], pub, state.round, to_call, state.raises, state.hist)


def jax_state_fingerprint(state: LeducState) -> tuple:
    py = PyLeducState.from_jax(state)
    fp = py_fingerprint(py)
    return fp + (len(py.hist), py.hist, py.deal_public)


def collect_reachable_infosets(roots: list[PyLeducState]) -> set[tuple]:
    """All decision infosets reachable from ``roots`` (acting player's view)."""
    infosets: set[tuple] = set()
    stack = list(roots)
    seen: set[tuple] = set()
    while stack:
        state = stack.pop()
        fp = py_fingerprint(state)
        if fp in seen:
            continue
        seen.add(fp)
        if state.done:
            continue
        seat = state.acting
        infosets.add(py_info_key(state, seat))
        mask = py_legal(state)
        for a in range(3):
            if mask[a]:
                stack.append(apply_action_py(state, a))
    return infosets


def collect_reachable_transitions(roots: list[PyLeducState]) -> list[tuple[tuple, int, tuple]]:
    """(parent_fingerprint, action, child_fingerprint) for every edge."""
    edges: list[tuple[tuple, int, tuple]] = []
    stack = list(roots)
    seen: set[tuple] = set()
    while stack:
        state = stack.pop()
        fp = py_fingerprint(state)
        if fp in seen:
            continue
        seen.add(fp)
        if state.done:
            continue
        mask = py_legal(state)
        for a in range(3):
            if mask[a]:
                child = apply_action_py(state, a)
                edges.append((fp, a, py_fingerprint(child)))
                stack.append(child)
    return edges
