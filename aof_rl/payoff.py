"""Terminal chip EV (bb) for AoF hands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aof_rl.env import AoFHandState

RANKS = "23456789TJQKA"
SUITS = "cdhs"

# Module-level payoff config (set from RolloutConfig via league_trainer)
_payoff_hu_mode: str = "table"
_hu_showdown_path: Path = Path("data/aof/hu_showdown_equity.npz")


def configure_payoff(
    *,
    payoff_hu_mode: str = "table",
    hu_showdown_table_path: str | Path = "data/aof/hu_showdown_equity.npz",
) -> None:
    global _payoff_hu_mode, _hu_showdown_path
    _payoff_hu_mode = payoff_hu_mode
    _hu_showdown_path = Path(hu_showdown_table_path)
    if payoff_hu_mode == "table":
        from aof_equity_showdown import load_showdown_table

        load_showdown_table(_hu_showdown_path)


def _hero_key(cards: tuple[str, str]) -> str:
    return "".join(cards)


def _card_id(card: str) -> int:
    from phevaluator.card import Card as PeCard

    return PeCard.to_id(card[0].upper() + card[1].lower())


def _headsup_equity(
    cards_a: tuple[str, str],
    cards_b: tuple[str, str],
    samples: int,
    seed: int,
) -> float:
    from aof_equity_showdown import PayoffHuMode, equity_hu_showdown_exact

    mode: PayoffHuMode
    if _payoff_hu_mode == "exact_enum":
        mode = "exact_enum"
    elif _payoff_hu_mode == "mc_legacy":
        mode = "mc_legacy"
    else:
        mode = "table"
    return equity_hu_showdown_exact(
        cards_a,
        cards_b,
        table_path=_hu_showdown_path,
        mode=mode,
    )


def compute_payoffs(
    state: AoFHandState,
    *,
    samples: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """
    Net profit in bb per seat for this hand (zero-sum-ish; includes posted blinds).
    """
    alive = [s for s in state.seats if s not in state.folded]
    if len(alive) == 0:
        return {s: -state.invested[s] for s in state.seats}

    if len(alive) == 1:
        winner = alive[0]
        pot = sum(state.invested.values())
        out = {s: -state.invested[s] for s in state.seats}
        out[winner] = pot - state.invested[winner]
        return out

    pot = sum(state.invested.values())
    rng = np.random.default_rng(seed)

    if len(alive) == 2:
        a, b = alive[0], alive[1]
        eq = _headsup_equity(state.holes[a], state.holes[b], samples, seed)
        share_a = eq * pot
        share_b = (1.0 - eq) * pot
        return {
            **{s: -state.invested[s] for s in state.seats if s not in (a, b)},
            a: share_a - state.invested[a],
            b: share_b - state.invested[b],
        }

    shares = _multiway_shares_numba(state, alive, samples, seed, rng)
    out = {s: -state.invested[s] for s in state.seats}
    for seat, frac in shares.items():
        out[seat] = frac * pot - state.invested[seat]
    return out


def _multiway_shares_numba(
    state: AoFHandState,
    alive: list[str],
    samples: int,
    seed: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Monte Carlo equity shares via numba kernels."""
    from aof_equity import (
        _mc_3way_shares_all,
        _mc_4way_shares_all,
        _sample_boards_3way,
        _sample_boards_4way,
    )

    n_alive = len(alive)
    per = max(samples, 200)
    hole_ids = np.array(
        [
            cid
            for s in alive
            for cid in (_card_id(state.holes[s][0]), _card_id(state.holes[s][1]))
        ],
        dtype=np.int32,
    )

    if n_alive == 3:
        h0, h1 = int(hole_ids[0]), int(hole_ids[1])
        v0, v1 = int(hole_ids[2]), int(hole_ids[3])
        w0, w1 = int(hole_ids[4]), int(hole_ids[5])
        bds = _sample_boards_3way(h0, h1, v0, v1, w0, w1, per, rng)
        shares = _mc_3way_shares_all(h0, h1, v0, v1, w0, w1, bds)
        return {alive[i]: float(shares[i]) for i in range(3)}

    if n_alive == 4:
        h0, h1 = int(hole_ids[0]), int(hole_ids[1])
        v0, v1 = int(hole_ids[2]), int(hole_ids[3])
        w0, w1 = int(hole_ids[4]), int(hole_ids[5])
        x0, x1 = int(hole_ids[6]), int(hole_ids[7])
        bds = _sample_boards_4way(hole_ids, per, rng)
        shares = _mc_4way_shares_all(h0, h1, v0, v1, w0, w1, x0, x1, bds)
        return {alive[i]: float(shares[i]) for i in range(4)}

    eq = 1.0 / n_alive
    return {s: eq for s in alive}
