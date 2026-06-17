"""Tabular CFR + infoset-BR exploitability for Leduc (reference ruler)."""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.leduc_bridge import (
    PyLeducState,
    apply_action_py,
    init_py_from_key,
    jax_from_py,
    py_fingerprint,
    py_info_key,
    py_legal,
    py_state_from_deal,
)
from pokerbot.engine.leduc_jax import (
    MAX_HISTORY,
    NUM_ACTIONS,
    NUM_PLAYERS,
    apply_action,
    legal_actions,
)

# Gate thresholds (milli-ante units, ×1000).
LEDUC_TABULAR_EXPLOITABILITY_MBB = 5.0  # converged tabular CFR must be near zero
LEDUC_SD_CFR_EXPLOITABILITY_MBB = 100.0  # neural approximation ceiling
# Per-hand profile EV sanity (chips): equilibrium ≈ ±0.03; always-fold ≈ ±1.0.
# Values above this indicate payoff / transition bugs, not merely bad play.
LEDUC_PROFILE_EV_SANITY_MAX = 1.05


def all_leduc_deals() -> tuple[tuple[tuple[int, int], int], ...]:
    """All 90 deck permutations: ((p0_card, p1_card), public_card)."""
    deck = (0, 0, 1, 1, 2, 2)
    return tuple(((perm[0], perm[1]), perm[2]) for perm in sorted(set(permutations(deck))))


def _all_leduc_deals() -> tuple[tuple[tuple[int, int], int], ...]:
    return all_leduc_deals()


def tabular_cfr(iterations: int = 800, seed: int = 0) -> tuple[dict, dict]:
    """CFR+ on the canonical Leduc tree; returns (avg_strategy, cumulative_regrets).

    Regret-matching+ with alternating updates and linear averaging — the same scheme
    OpenSpiel's ``CFRPlusSolver`` uses (validated to match its ``nash_conv`` to <1 mbb
    on standard Leduc). The strategy is held fixed across a player's full sweep over all
    deals; cumulative regrets accumulate as deltas and are floored to >= 0 once per sweep
    *after* the traversal. Recomputing the strategy from regrets mid-traversal (the old
    vanilla loop did this, re-reading regrets it had just mutated within the same
    iteration) destroys the CFR+ contraction and leaves exploitability at the slow
    vanilla 1/sqrt(T) rate — ~67 mbb at 1000 iters instead of <1.
    """
    cur_policy: dict[tuple, np.ndarray] = {}
    cum_regret: dict[tuple, np.ndarray] = defaultdict(
        lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
    )
    cum_policy: dict[tuple, np.ndarray] = defaultdict(
        lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
    )
    legal_mask: dict[tuple, np.ndarray] = {}
    box = {"weight": 1.0}

    def policy_for(key: tuple, mask: np.ndarray) -> np.ndarray:
        sigma = cur_policy.get(key)
        return sigma if sigma is not None else mask / mask.sum()

    def cfr(state: PyLeducState, player: int, reach: np.ndarray) -> np.ndarray:
        """External chance handled by deal enumeration; reach = [pi_0, pi_1, pi_chance]."""
        if state.done:
            return np.asarray(state.rewards, dtype=np.float64)
        seat = state.acting
        mask = py_legal(state).astype(np.float64)
        key = py_info_key(state, seat)
        legal_mask[key] = mask
        sigma = policy_for(key, mask)
        value = np.zeros(NUM_PLAYERS, dtype=np.float64)
        child_value: dict[int, np.ndarray] = {}
        for a in range(NUM_ACTIONS):
            if mask[a]:
                child_reach = reach.copy()
                child_reach[seat] *= sigma[a]
                cv = cfr(apply_action_py(state, a), player, child_reach)
                value += sigma[a] * cv
                child_value[a] = cv
        if seat != player:
            return value
        # counterfactual reach = product of every reach except the acting player's
        cf_reach = reach[1 - seat] * reach[NUM_PLAYERS]
        reach_prob = reach[seat]
        regret = cum_regret[key]
        strat = cum_policy[key]
        for a in range(NUM_ACTIONS):
            if mask[a]:
                regret[a] += cf_reach * (child_value[a][seat] - value[seat])
                strat[a] += box["weight"] * reach_prob * sigma[a]
        return value

    all_deals = _all_leduc_deals()
    chance = 1.0 / len(all_deals)
    for t in range(1, iterations + 1):
        box["weight"] = float(t)  # linear averaging
        for player in range(NUM_PLAYERS):  # alternating updates
            for cards, deal_public in all_deals:
                root = py_state_from_deal(cards, deal_public)
                cfr(root, player, np.array([1.0, 1.0, chance], dtype=np.float64))
            for regret in cum_regret.values():  # regret-matching+ floor
                np.maximum(regret, 0.0, out=regret)
            for key, regret in cum_regret.items():  # refresh fixed strategy for next sweep
                mask = legal_mask[key]
                positive = np.maximum(regret, 0.0) * mask
                total = positive.sum()
                cur_policy[key] = positive / total if total > 0 else mask / mask.sum()

    avg = {}
    for k, strat in cum_policy.items():
        total = strat.sum()
        avg[k] = strat / total if total > 0 else strat
    return avg, dict(cum_regret)


def mean_l1_recovered_avg_vs_tabular(
    policy_fn,
    tabular_avg: dict,
    *,
    deals: tuple | None = None,
) -> tuple[float, int]:
    """Mean L1 distance between ``policy_fn`` and converged tabular avg over all acting nodes."""
    from pokerbot.engine.leduc_bridge import jax_from_py

    deal_list = deals if deals is not None else all_leduc_deals()
    roots = [py_state_from_deal(c, p) for c, p in deal_list]
    l1_list: list[float] = []
    stack = list(roots)
    seen: set[tuple] = set()
    while stack:
        state = stack.pop()
        fp = (state.deal_public, state.cards, state.public, state.round, state.hist, state.pot)
        if fp in seen:
            continue
        seen.add(fp)
        if state.done:
            continue
        seat = state.acting
        mask = py_legal(state)
        key = py_info_key(state, seat)
        tab = _policy_for_key(tabular_avg, key, mask)
        sd = policy_fn(jax_from_py(state), seat)
        l1_list.append(float(np.sum(np.abs(tab - sd))))
        for a in range(NUM_ACTIONS):
            if mask[a]:
                stack.append(apply_action_py(state, a))
    arr = np.asarray(l1_list, dtype=np.float64)
    return float(arr.mean()), int(arr.size)


def _policy_for_key(avg: dict, key: tuple, mask: np.ndarray) -> np.ndarray:
    sigma = avg.get(key)
    if sigma is None:
        sigma = np.zeros(NUM_ACTIONS, dtype=np.float64)
        sigma[mask] = 1.0 / mask.sum()
    else:
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma = sigma * mask
        s = sigma.sum()
        sigma = sigma / s if s > 0 else np.where(mask, 1.0 / mask.sum(), 0.0)
    return sigma


def _terminal_reward_bounds(state: PyLeducState, player: int, cache: dict | None = None) -> tuple[float, float]:
    """Min/max terminal payoff for ``player`` reachable from ``state`` (exact tree)."""
    cache = cache if cache is not None else {}
    sk = (state.deal_public, py_fingerprint(state), player, "bounds")
    if sk in cache:
        return cache[sk]
    if state.done:
        val = float(state.rewards[player])
        bounds = (val, val)
    else:
        lo, hi = float("inf"), float("-inf")
        mask = py_legal(state)
        for a in range(NUM_ACTIONS):
            if mask[a]:
                child_lo, child_hi = _terminal_reward_bounds(
                    apply_action_py(state, a), player, cache
                )
                lo = min(lo, child_lo)
                hi = max(hi, child_hi)
        bounds = (lo, hi)
    cache[sk] = bounds
    return bounds


def derive_leduc_max_instant_cf_bound() -> float:
    """Max |opp_reach * (util[a]-node_util)| with opp_reach<=1 over the Leduc tree."""
    deals = all_leduc_deals()
    max_spread = 0.0

    def action_value(state: PyLeducState, traverser: int, action: int) -> float:
        child = apply_action_py(state, action)
        if child.done:
            return float(child.rewards[traverser])
        mask = py_legal(child)
        vals = [action_value(child, traverser, a) for a in range(NUM_ACTIONS) if mask[a]]
        return float(np.mean(vals)) if vals else 0.0

    def walk(state: PyLeducState, traverser: int) -> None:
        nonlocal max_spread
        if state.done:
            return
        mask = py_legal(state)
        utils = [action_value(state, traverser, a) for a in range(NUM_ACTIONS) if mask[a]]
        if utils:
            max_spread = max(max_spread, max(utils) - min(utils))
        for a in range(NUM_ACTIONS):
            if mask[a]:
                walk(apply_action_py(state, a), traverser)

    for cards, deal_public in deals:
        root = py_state_from_deal(cards, deal_public)
        for trav in (0, 1):
            walk(root, trav)
    return float(max_spread)


def assert_leduc_profile_ev_in_range(
    u0: float,
    u1: float,
    roots: list[PyLeducState],
    profile_u0_fn,
    profile_u1_fn,
) -> None:
    """Fail fast when profile EV is inconsistent with terminal payoffs or zero-sum."""
    if abs(u0 + u1) > 1e-6:
        raise AssertionError(f"profile EV not zero-sum: u0={u0:.4f} u1={u1:.4f}")
    if abs(u0) > LEDUC_PROFILE_EV_SANITY_MAX or abs(u1) > LEDUC_PROFILE_EV_SANITY_MAX:
        raise AssertionError(
            f"profile EV out of range (max {LEDUC_PROFILE_EV_SANITY_MAX}): "
            f"u0={u0:.4f} u1={u1:.4f}"
        )
    for root in roots:
        p0 = profile_u0_fn(root)
        p1 = profile_u1_fn(root)
        if abs(p0 + p1) > 1e-6:
            raise AssertionError(
                f"per-deal profile not zero-sum deal={root.cards}/{root.deal_public}: "
                f"u0={p0:.4f} u1={p1:.4f}"
            )
        lo0, hi0 = _terminal_reward_bounds(root, 0)
        lo1, hi1 = _terminal_reward_bounds(root, 1)
        if not (lo0 - 1e-6 <= p0 <= hi0 + 1e-6):
            raise AssertionError(
                f"P0 profile EV {p0:.4f} outside terminal bounds [{lo0:.4f}, {hi0:.4f}] "
                f"deal={root.cards}/{root.deal_public}"
            )
        if not (lo1 - 1e-6 <= p1 <= hi1 + 1e-6):
            raise AssertionError(
                f"P1 profile EV {p1:.4f} outside terminal bounds [{lo1:.4f}, {hi1:.4f}] "
                f"deal={root.cards}/{root.deal_public}"
            )


def _profile_value(state: PyLeducState, player: int, avg: dict, cache: dict | None = None) -> float:
    cache = cache if cache is not None else {}
    sk = (state.deal_public, py_fingerprint(state), player, "profile")
    if sk in cache:
        return cache[sk]
    if state.done:
        val = state.rewards[player]
        cache[sk] = val
        return val
    seat = state.acting
    mask = py_legal(state)
    key = py_info_key(state, seat)
    sigma = _policy_for_key(avg, key, mask)
    val = 0.0
    for a in range(NUM_ACTIONS):
        if mask[a]:
            val += sigma[a] * _profile_value(apply_action_py(state, a), player, avg, cache)
    cache[sk] = val
    return val


class _InfosetBestResponse:
    """Counterfactual-reach best response (OpenSpiel-style, no card cheating)."""

    def __init__(self, br_seat: int, policy_sigma_fn) -> None:
        self.br_seat = br_seat
        self.policy_sigma_fn = policy_sigma_fn
        self.infosets: dict[tuple, list[tuple[PyLeducState, float]]] = defaultdict(list)
        self._br_action: dict[tuple, int] = {}
        self._value_cache: dict[tuple, float] = {}

    def _transitions(self, state: PyLeducState) -> list[tuple[int, float]]:
        if state.done:
            return []
        seat = state.acting
        mask = py_legal(state)
        if seat == self.br_seat:
            return [(a, 1.0) for a in range(NUM_ACTIONS) if mask[a]]
        sigma = self.policy_sigma_fn(state, seat)
        return [(a, float(sigma[a])) for a in range(NUM_ACTIONS) if mask[a]]

    def _collect_infosets(self, state: PyLeducState, cf_prob: float) -> None:
        if state.done:
            return
        if state.acting == self.br_seat:
            self.infosets[py_info_key(state, self.br_seat)].append((state, cf_prob))
        for action, prob in self._transitions(state):
            self._collect_infosets(apply_action_py(state, action), cf_prob * prob)

    def _q_value(self, state: PyLeducState, action: int) -> float:
        return self._value(apply_action_py(state, action))

    def _br_action_at(self, state: PyLeducState) -> int:
        ik = py_info_key(state, self.br_seat)
        if ik not in self._br_action:
            entries = self.infosets[ik]
            mask = py_legal(entries[0][0])
            legal = [a for a in range(NUM_ACTIONS) if mask[a]]
            self._br_action[ik] = max(
                legal,
                key=lambda a: sum(cf_p * self._q_value(s, a) for s, cf_p in entries),
            )
        return self._br_action[ik]

    def _value(self, state: PyLeducState) -> float:
        vk = (state.deal_public, py_fingerprint(state))
        if vk in self._value_cache:
            return self._value_cache[vk]
        if state.done:
            val = float(state.rewards[self.br_seat])
        elif state.acting == self.br_seat:
            val = self._q_value(state, self._br_action_at(state))
        else:
            val = 0.0
            for action, prob in self._transitions(state):
                val += prob * self._q_value(state, action)
        self._value_cache[vk] = val
        return val

    def build(self, roots: list[PyLeducState]) -> None:
        for root in roots:
            self._collect_infosets(root, 1.0)

    def root_value(self, state: PyLeducState) -> float:
        return self._value(state)


def infoset_exploitability_mbb(avg: dict, samples: int = 90) -> tuple[float, float, float, float, float]:
    """Information-set BR exploitability for tabular avg strategy.

    Returns (expl_mbb, u0, u1, br0, br1).
    """

    def policy_sigma(state: PyLeducState, seat: int) -> np.ndarray:
        mask = py_legal(state)
        return _policy_for_key(avg, py_info_key(state, seat), mask)

    deals = _all_leduc_deals()
    eval_deals = list(deals) if samples >= len(deals) else [deals[int(i)] for i in range(samples)]
    roots = [py_state_from_deal(c, p) for c, p in eval_deals]

    br0_calc = _InfosetBestResponse(0, policy_sigma)
    br1_calc = _InfosetBestResponse(1, policy_sigma)
    br0_calc.build(roots)
    br1_calc.build(roots)

    br0 = br1 = u0 = u1 = 0.0
    for root in roots:
        br0 += br0_calc.root_value(root)
        br1 += br1_calc.root_value(root)
        u0 += _profile_value(root, 0, avg)
        u1 += _profile_value(root, 1, avg)
    n = float(len(roots))
    br0 /= n
    br1 /= n
    u0 /= n
    u1 /= n
    assert_leduc_profile_ev_in_range(
        u0,
        u1,
        roots,
        lambda r: _profile_value(r, 0, avg),
        lambda r: _profile_value(r, 1, avg),
    )
    expl = float(max(0.0, br0 - u0) + max(0.0, br1 - u1)) * 1000.0
    return expl, u0, u1, br0, br1


def _profile_value_from_policy(
    state: PyLeducState,
    player: int,
    policy_fn,
    cache: dict | None = None,
) -> float:
    cache = cache if cache is not None else {}
    sk = (state.deal_public, py_fingerprint(state), player, "profile_policy")
    if sk in cache:
        return cache[sk]
    if state.done:
        val = state.rewards[player]
        cache[sk] = val
        return val
    seat = state.acting
    mask = py_legal(state)
    sigma = np.asarray(policy_fn(jax_from_py(state), seat), dtype=np.float64)
    sigma = sigma * mask
    total = sigma.sum()
    if total > 0:
        sigma = sigma / total
    else:
        sigma = mask.astype(np.float64) / max(mask.sum(), 1)
    val = 0.0
    for a in range(NUM_ACTIONS):
        if mask[a]:
            val += sigma[a] * _profile_value_from_policy(
                apply_action_py(state, a), player, policy_fn, cache
            )
    cache[sk] = val
    return val


def infoset_exploitability_mbb_from_policy(
    policy_fn,
    samples: int = 90,
) -> tuple[float, float, float, float, float]:
    """Information-set BR exploitability for a JAX-state policy callable.

    Returns (expl_mbb, u0, u1, br0, br1).
    """

    def policy_sigma(state: PyLeducState, seat: int) -> np.ndarray:
        mask = py_legal(state)
        sigma = np.asarray(policy_fn(jax_from_py(state), seat), dtype=np.float64)
        sigma = sigma * mask
        total = sigma.sum()
        if total > 0:
            return sigma / total
        return mask.astype(np.float64) / max(mask.sum(), 1)

    deals = _all_leduc_deals()
    eval_deals = list(deals) if samples >= len(deals) else [deals[int(i)] for i in range(samples)]
    roots = [py_state_from_deal(c, p) for c, p in eval_deals]

    br0_calc = _InfosetBestResponse(0, policy_sigma)
    br1_calc = _InfosetBestResponse(1, policy_sigma)
    br0_calc.build(roots)
    br1_calc.build(roots)

    br0 = br1 = u0 = u1 = 0.0
    for root in roots:
        br0 += br0_calc.root_value(root)
        br1 += br1_calc.root_value(root)
        u0 += _profile_value_from_policy(root, 0, policy_fn)
        u1 += _profile_value_from_policy(root, 1, policy_fn)
    n = float(len(roots))
    br0 /= n
    br1 /= n
    u0 /= n
    u1 /= n
    assert_leduc_profile_ev_in_range(
        u0,
        u1,
        roots,
        lambda r: _profile_value_from_policy(r, 0, policy_fn),
        lambda r: _profile_value_from_policy(r, 1, policy_fn),
    )
    expl = float(max(0.0, br0 - u0) + max(0.0, br1 - u1)) * 1000.0
    return expl, u0, u1, br0, br1


def equilibrium_reference_exploitability_mbb(iterations: int = 2000) -> float:
    """Tabular CFR infoset-BR exploitability — primary Leduc ruler."""
    avg, _ = tabular_cfr(iterations=iterations, seed=0)
    expl, _, _, _, _ = infoset_exploitability_mbb(avg, samples=90)
    return expl


def openspiel_nash_conv_uniform_mbb() -> float:
    """OpenSpiel ``nash_conv`` for uniform random policy on standard Leduc (reference anchor)."""
    try:
        import pyspiel
        from open_spiel.python.algorithms import exploitability as os_exploitability
    except ImportError:
        return float("nan")

    game = pyspiel.load_game("leduc_poker")
    policy = pyspiel.UniformRandomPolicy(game)
    nash_conv = os_exploitability.nash_conv(game, policy)
    return float(nash_conv) * 1000.0


# Re-exports for callers that still import from this module.
__all__ = [
    "LEDUC_PROFILE_EV_SANITY_MAX",
    "LEDUC_SD_CFR_EXPLOITABILITY_MBB",
    "LEDUC_TABULAR_EXPLOITABILITY_MBB",
    "assert_leduc_profile_ev_in_range",
    "PyLeducState",
    "apply_action_py",
    "equilibrium_reference_exploitability_mbb",
    "infoset_exploitability_mbb",
    "infoset_exploitability_mbb_from_policy",
    "init_py_from_key",
    "openspiel_nash_conv_uniform_mbb",
    "py_state_from_deal",
    "tabular_cfr",
]
