"""Preflop tabular CFR warm-up with Monte Carlo postflop rollout leaf payoffs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokerbot.engine.cards import Card, Rank, Suit
from pokerbot.engine.hand_eval import compare_hands

WEVA_NUM_DEPTH_NODES = 4
WEVA_PREFLOP_ACTIONS = 3
WEVA_WARMUP_ITERATIONS = 10

FOLD = 0
CALL = 1
RAISE = 2


@dataclass(frozen=True, slots=True)
class PreflopScenario:
    """Public preflop context for a decision depth."""

    depth: int
    pot_bb: float
    hero_invested_bb: float
    call_bb: float
    raise_bb: float
    fold_equity: float


DEFAULT_SCENARIOS: tuple[PreflopScenario, ...] = (
    PreflopScenario(0, pot_bb=1.5, hero_invested_bb=0.0, call_bb=1.0, raise_bb=3.0, fold_equity=0.55),
    PreflopScenario(1, pot_bb=2.5, hero_invested_bb=0.0, call_bb=1.0, raise_bb=4.0, fold_equity=0.45),
    PreflopScenario(2, pot_bb=4.5, hero_invested_bb=0.0, call_bb=3.0, raise_bb=9.0, fold_equity=0.35),
    PreflopScenario(3, pot_bb=12.0, hero_invested_bb=3.0, call_bb=6.0, raise_bb=15.0, fold_equity=0.25),
)


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    num_rollouts: int = 128
    seed: int = 42
    cfr_seed: int = 42
    warmup_iterations: int = WEVA_WARMUP_ITERATIONS
    num_opponents: int = 5
    num_depths: int = WEVA_NUM_DEPTH_NODES


def _index_to_card(idx: int) -> Card:
    rank = idx // 4 + 2
    suit = idx % 4
    return Card(Rank(rank), Suit(suit))


class MCRolloutEvaluator:
    """Estimate postflop terminal EV via check-down Monte Carlo rollouts."""

    def __init__(self, config: RolloutConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._cache: dict[tuple[int, int, float, float], tuple[float, float]] = {}

    def _one_rollout_net(self, hero_hole: tuple[Card, Card], pot_bb: float, hero_invest_bb: float) -> float:
        dead = {hero_hole[0].index, hero_hole[1].index}
        deck = np.array([i for i in range(52) if i not in dead], dtype=np.int32)
        self.rng.shuffle(deck)

        ptr = 0
        opp_holes: list[tuple[Card, Card]] = []
        for _ in range(self.config.num_opponents):
            c0 = _index_to_card(int(deck[ptr]))
            c1 = _index_to_card(int(deck[ptr + 1]))
            ptr += 2
            opp_holes.append((c0, c1))

        board = [_index_to_card(int(deck[ptr + i])) for i in range(5)]

        hands = [[hero_hole[0], hero_hole[1], *board]]
        for oc0, oc1 in opp_holes:
            hands.append([oc0, oc1, *board])

        winners = compare_hands(hands)
        share = pot_bb / len(winners) if winners else 0.0
        hero_share = share if 0 in winners else 0.0
        return hero_share - hero_invest_bb

    def rollout_ev(self, hero_hole: tuple[Card, Card], pot_bb: float, hero_invest_bb: float) -> tuple[float, float]:
        hi, hj = sorted(c.index for c in hero_hole)
        cache_key = (hi, hj, round(pot_bb, 4), round(hero_invest_bb, 4))
        if cache_key in self._cache:
            return self._cache[cache_key]

        samples = np.empty(self.config.num_rollouts, dtype=np.float64)
        for i in range(self.config.num_rollouts):
            samples[i] = self._one_rollout_net(hero_hole, pot_bb, hero_invest_bb)
        mean = float(samples.mean())
        stderr = float(samples.std(ddof=1) / np.sqrt(max(len(samples), 1))) if len(samples) > 1 else 0.0
        self._cache[cache_key] = (mean, stderr)
        return mean, stderr

    def scenario_action_evs(self, hero_hole: tuple[Card, Card], scenario: PreflopScenario) -> np.ndarray:
        evs = np.zeros(WEVA_PREFLOP_ACTIONS, dtype=np.float64)
        evs[FOLD] = -scenario.hero_invested_bb

        call_pot = scenario.pot_bb + scenario.call_bb
        call_invest = scenario.hero_invested_bb + scenario.call_bb
        call_ev, _ = self.rollout_ev(hero_hole, call_pot, call_invest)
        evs[CALL] = call_ev

        raise_pot = scenario.pot_bb + scenario.raise_bb
        raise_invest = scenario.hero_invested_bb + scenario.raise_bb
        raised_showdown, _ = self.rollout_ev(hero_hole, raise_pot, raise_invest)
        steal = scenario.pot_bb - scenario.hero_invested_bb
        evs[RAISE] = scenario.fold_equity * steal + (1.0 - scenario.fold_equity) * raised_showdown
        return evs


def build_action_evs_matrix(
    roots: list,
    config: RolloutConfig,
    scenarios: tuple[PreflopScenario, ...] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    scenarios = scenarios or DEFAULT_SCENARIOS[: config.num_depths]
    if len(scenarios) != config.num_depths:
        raise ValueError(f"Expected {config.num_depths} scenarios, got {len(scenarios)}")

    evaluator = MCRolloutEvaluator(config)
    n = len(roots)
    matrix = np.zeros((n, config.num_depths, WEVA_PREFLOP_ACTIONS), dtype=np.float32)
    stderrs: list[float] = []

    for i, root in enumerate(roots):
        for d, scenario in enumerate(scenarios):
            evs = evaluator.scenario_action_evs(root.hole, scenario)
            matrix[i, d, :] = evs.astype(np.float32)

    stderrs = [s for _, s in evaluator._cache.values()]

    aa_key = tuple(sorted([Card(Rank.ACE, Suit.SPADES).index, Card(Rank.ACE, Suit.HEARTS).index]))
    s72_key = tuple(sorted([Card(Rank.SEVEN, Suit.SPADES).index, Card(Rank.TWO, Suit.HEARTS).index]))
    aa_idx = None
    s72_idx = None
    for i, root in enumerate(roots):
        hk = root.hole_key
        if hk == aa_key:
            aa_idx = i
        if hk == s72_key:
            s72_idx = i

    calibration: dict[str, float] = {
        "mean_rollout_stderr": float(np.mean(stderrs)) if stderrs else 0.0,
    }
    if aa_idx is not None and s72_idx is not None:
        aa_call = float(matrix[aa_idx, 0, CALL])
        s72_call = float(matrix[s72_idx, 0, CALL])
        # Relative ordering only — rollout EV is approximate; never assert on gap magnitude.
        calibration["aa_vs_72o_call_ordering"] = 1.0 if aa_call > s72_call else 0.0

    return matrix, calibration


class TabularCFRPlus:
    """Vectorized tabular CFR+ over precomputed per-root action EVs."""

    def __init__(
        self,
        action_evs: np.ndarray,
        warmup_iterations: int = WEVA_WARMUP_ITERATIONS,
        seed: int = 42,
    ) -> None:
        self.action_evs = action_evs.astype(np.float64)
        self.w = warmup_iterations
        self.regrets = np.zeros_like(self.action_evs)
        self.strategy_sum = np.zeros_like(self.action_evs)

    @staticmethod
    def _regret_matching(regrets: np.ndarray) -> np.ndarray:
        pos = np.maximum(regrets, 0.0)
        totals = pos.sum(axis=-1, keepdims=True)
        uniform = np.full_like(pos, 1.0 / regrets.shape[-1])
        return np.where(totals > 0, pos / np.maximum(totals, 1e-8), uniform)

    def run(self) -> None:
        for _ in range(self.w):
            strat = self._regret_matching(self.regrets)
            self.strategy_sum += strat
            node_val = (strat * self.action_evs).sum(axis=-1, keepdims=True)
            self.regrets += self.action_evs - node_val
            self.regrets = np.maximum(self.regrets, 0.0)

    @property
    def average_strategy(self) -> np.ndarray:
        return self.strategy_sum / max(self.w, 1)

    def node_evs_under_avg_strategy(self) -> np.ndarray:
        avg = self.average_strategy
        return (avg * self.action_evs).sum(axis=-1)


def extract_depth_weighted_features(
    node_evs: np.ndarray,
    gamma: float,
    max_nodes: int,
) -> np.ndarray:
    n, d = node_evs.shape
    weights = np.array([gamma**i for i in range(d)], dtype=np.float32)
    weighted = node_evs.astype(np.float32) * weights[np.newaxis, :]
    if d >= max_nodes:
        return weighted[:, :max_nodes]
    pad = np.zeros((n, max_nodes - d), dtype=np.float32)
    return np.concatenate([weighted, pad], axis=1)


def run_weva_cfr_pipeline(
    roots: list,
    config: RolloutConfig,
    gamma: float,
    max_nodes: int,
    feature_mode: str,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    action_evs, calib = build_action_evs_matrix(roots, config)
    cfr = TabularCFRPlus(action_evs, warmup_iterations=config.warmup_iterations, seed=config.cfr_seed)
    cfr.run()
    node_evs = cfr.node_evs_under_avg_strategy()

    if feature_mode == "hole_only":
        features = extract_depth_weighted_features(node_evs, gamma, max_nodes)
        return features, calib, cfr.average_strategy

    hole_keys = [r.hole_key for r in roots]
    unique_holes: dict[tuple[int, int], int] = {}
    hole_roots: list = []
    for key in hole_keys:
        if key not in unique_holes:
            unique_holes[key] = len(hole_roots)
            sample = next(r for r in roots if r.hole_key == key)
            hole_roots.append(sample)

    hole_action_evs, hole_calib = build_action_evs_matrix(hole_roots, config)
    hole_cfr = TabularCFRPlus(
        hole_action_evs,
        warmup_iterations=config.warmup_iterations,
        seed=config.cfr_seed,
    )
    hole_cfr.run()
    hole_node_evs = hole_cfr.node_evs_under_avg_strategy()
    hole_features = extract_depth_weighted_features(hole_node_evs, gamma, max_nodes)

    ctx_matrix = np.stack([r.feature_suffix() for r in roots], axis=0)
    base = hole_features[[unique_holes[k] for k in hole_keys]]
    features = np.concatenate([base, ctx_matrix], axis=1).astype(np.float32)
    calib.update(hole_calib)
    return features, calib, cfr.average_strategy
