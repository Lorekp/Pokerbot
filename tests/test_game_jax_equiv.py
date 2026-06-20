"""game.py  <->  jax_engine / side_pots_jax  EQUIVALENCE GATE  (Workstream B headline deliverable).

There is no equivalence gate between ``pokerbot.engine.game.py`` (the ungated float-BB eval engine) and
the canonical int-chip ``jax_engine`` / ``side_pots_jax`` today — the highest divergence risk in the
codebase. This builds one.

SETTLEMENT PARITY: random multiway all-in terminal states are settled by BOTH
``np_engine.compute_returns_np`` (int chips, bit-identical to ``side_pots_jax.compute_returns`` per
``test_np_engine_equiv``) and ``game.GameState.resolve_showdown`` (float BB), and the per-seat chip
deltas must agree EXACTLY. This is the direct test for BUG-B5 (uncalled-bet refund via ``srt[-2]``) and
BUG-B6 (odd-chip layer splitting: int ``award_layer`` vs float ``_split_layer_chips``).

Returns are compared in INTEGER CHIPS (game.py BB * CHIP_UNIT, rounded) so odd-chip routing divergences
surface as ±1 chip mismatches rather than being hidden by a loose float tolerance. Uses pure-numpy
(``compute_returns_np``) — no per-scenario JAX dispatch — so 1e4+ scenarios run in seconds.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from pokerbot.engine import jax_engine as je
from pokerbot.engine import np_engine as ne
from pokerbot.engine.nlhe_bridge import game_state_from_jax
from pokerbot.engine.side_pots_jax import SEAT_ALLIN, SEAT_FOLDED, SEAT_LIVE

CU = je.CHIP_UNIT
NP = je.NUM_PLAYERS

# One numpy-backed EnvState template; per scenario we only _replace the settlement-relevant fields.
_TEMPLATE = ne.to_numpy_state(je.init_hand(jax.random.PRNGKey(0)))


def _scenario(rng: np.random.Generator):
    """Random TERMINAL state: distinct holes+board from a shuffled deck, random per-seat committed_total,
    a random folded subset (dead money retained), the rest all-in contenders."""
    n_occ = int(rng.integers(2, NP + 1))
    occ = np.zeros(NP, dtype=bool)
    occ[rng.choice(NP, size=n_occ, replace=False)] = True
    deck = rng.permutation(52).astype(np.int32)

    holes = np.full((NP, 2), -1, dtype=np.int32)
    p = 0
    for s in range(NP):
        if occ[s]:
            holes[s] = [deck[p], deck[p + 1]]
            p += 2
    board = np.full(je.MAX_BOARD, -1, dtype=np.int32)
    board[:5] = deck[p:p + 5]

    occ_seats = [s for s in range(NP) if occ[s]]
    style = int(rng.integers(0, 4))
    committed = np.zeros(NP, dtype=np.int32)
    for s in occ_seats:
        if style == 0:
            committed[s] = 300                                  # equal -> single pot
        elif style == 1:
            committed[s] = int(rng.integers(1, 12)) * 100       # unequal all-ins -> side pots
        elif style == 2:
            committed[s] = int(rng.integers(50, 1000))          # odd totals -> remainder routing
        else:
            committed[s] = int(rng.choice([700, 700, 250, 999]))  # tied-top + short

    status = np.where(occ, int(SEAT_LIVE), 0).astype(np.int8)
    # REACHABILITY constraint: the single LARGEST contributor must stay a contender (non-folded). In
    # real poker you never fold as the top contributor; if a folded seat were the unique max, the top
    # contribution layer would have no eligible claimant and BOTH engines silently drop those chips
    # (non-zero-sum), which is meaningless to compare. Keeping the global-max seat live guarantees every
    # capped layer (capped at the 2nd-highest) has a live claimant -> zero-sum + reachable.
    max_seat = max(occ_seats, key=lambda s: committed[s])
    foldable = [s for s in occ_seats if s != max_seat]
    rng.shuffle(foldable)
    n_fold = int(rng.integers(0, len(foldable) + 1))            # 0..n-1 folds (max_seat always contends)
    folded = set(foldable[:n_fold])
    for s in occ_seats:
        status[s] = int(SEAT_FOLDED) if s in folded else int(SEAT_ALLIN)

    button = int(rng.integers(0, NP))
    state = _TEMPLATE._replace(
        holes=holes, board=board, seat_status=status, committed_total=committed,
        committed_street=committed, pot_dead=np.int32(0), stacks=np.zeros(NP, dtype=np.int32),
        street=np.int32(je.STREET_TERMINAL), current_bet=np.int32(0), button=np.int32(button),
        action_seat=np.int32(-1), last_aggressor=np.int32(-1),
    )
    n_showdown = int(np.sum((status == int(SEAT_LIVE)) | (status == int(SEAT_ALLIN))))
    return state, committed, status, n_showdown


def _settle_game(state) -> np.ndarray:
    gs = game_state_from_jax(state)
    out = np.zeros(NP, dtype=np.int64)
    for seat, bb in gs.resolve_showdown().items():
        out[seat] = int(round(bb * CU))
    return out


def test_settlement_parity_exact():
    rng = np.random.default_rng(0xB5B6)
    N = 6000
    mismatches = []
    nonzero_sum = 0
    style_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(N):
        state, committed, status, n_showdown = _scenario(rng)
        jr = np.asarray(ne.compute_returns_np(
            holes=state.holes, board=state.board, seat_status=state.seat_status,
            committed_total=committed, button=int(state.button), n_showdown=n_showdown,
        )).astype(np.int64)
        gr = _settle_game(state)
        if int(jr.sum()) != 0:
            nonzero_sum += 1
        if not np.array_equal(jr, gr):
            if len(mismatches) < 15:
                mismatches.append({
                    "row": i, "committed": committed.tolist(),
                    "status": [int(x) for x in status], "button": int(state.button),
                    "jax": jr.tolist(), "game": gr.tolist(), "delta(jax-game)": (jr - gr).tolist(),
                })
    assert nonzero_sum == 0, f"compute_returns_np not zero-sum in {nonzero_sum}/{N} scenarios"
    assert not mismatches, (
        f"{len(mismatches)}+ settlement mismatches game.py.resolve_showdown vs compute_returns "
        f"(BUG-B5/B6). Examples:\n" + "\n".join(str(m) for m in mismatches)
    )


if __name__ == "__main__":
    test_settlement_parity_exact()
    print("settlement parity OK")
