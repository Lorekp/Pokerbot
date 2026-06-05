"""Fast external-sampling MCCFR for AoF push/fold (combo deals, no OpenSpiel traversal)."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

from aof_cfr.combo_deal import FULL_DECK, bucket_from_card_ids, card_id
from aof_cfr.exploit_mc import estimate_nash_conv_mc
from aof_cfr.infoset import encode_infoset_key, history_after_action
from aof_cfr.terminal_ev import fold_terminal_returns, showdown_terminal_returns
from aof_ranges import active_seats

_STACK_BB = 8.0


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def _iteration_progress(
    *,
    current: int,
    total: int,
    started: float,
    label: str = "iteration",
) -> None:
    elapsed = time.monotonic() - started
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = total - current
    eta = remaining / rate if rate > 0 else 0.0
    pct = 100.0 * current / total if total else 100.0
    print(
        f"    {label} {current}/{total} ({pct:.0f}%) "
        f"elapsed={_fmt_duration(elapsed)} eta={_fmt_duration(eta)} "
        f"({rate:.1f} iter/s)",
        flush=True,
    )


def _run_iterations(
    solver: PushfoldMCCFRSolver,
    *,
    start: int,
    target: int,
    progress_every: int,
    label: str,
) -> None:
    if target <= start:
        return
    block_started = time.monotonic()
    for i in range(start, target):
        solver.iteration()
        done = i + 1
        if done % progress_every == 0 or done == target:
            _iteration_progress(
                current=done - start,
                total=target - start,
                started=block_started,
                label=label,
            )

def _post_blinds(n: int, seats: tuple[str, ...]) -> np.ndarray:
    inv = np.zeros(n, dtype=np.float64)
    for i, seat in enumerate(seats):
        if seat == "SB":
            inv[i] = min(0.5, _STACK_BB)
        elif seat == "BB":
            inv[i] = min(1.0, _STACK_BB)
    return inv


def _terminal(
    n: int,
    holes: list[tuple[int, int]],
    folded: set[int],
    invested: np.ndarray,
) -> list[float]:
    alive = [i for i in range(n) if i not in folded]
    if len(alive) == 1:
        return fold_terminal_returns(folded, alive, invested)
    hole_ids = [holes[i] for i in alive]
    inv = np.array([invested[i] for i in alive], dtype=np.float64)
    shares = showdown_terminal_returns(hole_ids, inv)
    rets = [-float(x) for x in invested]
    for j, seat in enumerate(alive):
        rets[seat] = shares[j]
    return rets


def _regret_matching(r: np.ndarray) -> np.ndarray:
    pos = np.maximum(r, 0.0)
    s = pos.sum()
    if s <= 1e-12:
        return np.array([0.5, 0.5], dtype=np.float64)
    return pos / s


@dataclass
class _MCCFRState:
    regrets: dict[str, np.ndarray] = field(default_factory=dict)
    strat_sum: dict[str, np.ndarray] = field(default_factory=dict)

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if key not in self.regrets:
            self.regrets[key] = np.zeros(2, dtype=np.float64)
            self.strat_sum[key] = np.zeros(2, dtype=np.float64)
        return self.regrets[key], self.strat_sum[key]


class PushfoldMCCFRSolver:
    def __init__(self, num_players: int, *, seed: int = 42) -> None:
        self.num_players = num_players
        self.seats = active_seats(num_players)
        self.rng = random.Random(seed)
        self.state = _MCCFRState()

    def _sample_deal(self) -> list[tuple[int, int]]:
        deck = list(FULL_DECK)
        self.rng.shuffle(deck)
        picked = deck[: 2 * self.num_players]
        return [(card_id(picked[2 * i]), card_id(picked[2 * i + 1])) for i in range(self.num_players)]

    def _key(self, step: int, holes: list[tuple[int, int]], history: str) -> str:
        seat = self.seats[step]
        bucket = bucket_from_card_ids(holes[step][0], holes[step][1])
        return encode_infoset_key(seat, bucket, history if history else "-")

    def _traverse(
        self,
        step: int,
        holes: list[tuple[int, int]],
        invested: np.ndarray,
        folded: set[int],
        acted: set[int],
        history: str,
        *,
        update_player: int,
        reach: np.ndarray,
    ) -> list[float]:
        n = self.num_players
        if step >= n:
            return _terminal(n, holes, folded, invested.copy())
        if step in folded or step in acted:
            return self._traverse(
                step + 1,
                holes,
                invested,
                folded,
                acted,
                history,
                update_player=update_player,
                reach=reach,
            )

        key = self._key(step, holes, history)
        regrets, strat_sum = self.state.get(key)
        sigma = _regret_matching(regrets)

        if step != update_player:
            # OpenSpiel "simple" average: accumulate at (update_player + 1) % n nodes.
            if step == (update_player + 1) % n:
                strat_sum[:] += sigma
            action = 1 if self.rng.random() < sigma[1] else 0
            return self._apply(
                step,
                action,
                holes,
                invested,
                folded,
                acted,
                history,
                update_player=update_player,
                reach=reach,
            )

        inv = invested.copy()
        u0 = self._apply(
            step, 0, holes, inv, set(folded), set(acted), history,
            update_player=update_player, reach=reach,
        )
        u1 = self._apply(
            step, 1, holes, inv, set(folded), set(acted), history,
            update_player=update_player, reach=reach,
        )
        node = [sigma[0] * u0[i] + sigma[1] * u1[i] for i in range(n)]
        for a in (0, 1):
            ua = u0 if a == 0 else u1
            regrets[a] += ua[update_player] - node[update_player]
        return node

    def _apply(
        self,
        step: int,
        action: int,
        holes: list[tuple[int, int]],
        invested: np.ndarray,
        folded: set[int],
        acted: set[int],
        history: str,
        *,
        update_player: int,
        reach: np.ndarray,
    ) -> list[float]:
        acted.add(step)
        tok = "F" if action == 0 else "A"
        history = history_after_action(history, tok)
        if action == 0:
            folded.add(step)
        else:
            invested[step] = _STACK_BB
        alive = [i for i in range(self.num_players) if i not in folded]
        if len(alive) == 1:
            return _terminal(self.num_players, holes, folded, invested)
        reach = reach.copy()
        reach[step] *= 1.0
        return self._traverse(
            step + 1,
            holes,
            invested,
            folded,
            acted,
            history,
            update_player=update_player,
            reach=reach,
        )

    def iteration(self) -> None:
        n = self.num_players
        for p in range(n):
            holes = self._sample_deal()
            inv = _post_blinds(n, self.seats)
            reach = np.ones(n, dtype=np.float64)
            self._traverse(
                0, holes, inv, set(), set(), "-",
                update_player=p, reach=reach,
            )

    def average_policy_rows(self) -> dict[str, np.ndarray]:
        rows: dict[str, np.ndarray] = {}
        for key, ssum in self.state.strat_sum.items():
            tot = float(ssum.sum())
            if tot > 1e-12:
                rows[key] = (ssum / tot).astype(np.float64)
            else:
                rows[key] = np.array([0.5, 0.5], dtype=np.float64)
        return rows


def solve_pushfold_mccfr_native(
    num_players: int,
    *,
    iterations: int = 20_000,
    seed: int = 42,
    target_mbb_g: float = 10.0,
    max_iterations: int = 200_000,
    check_every: int = 5_000,
    mc_samples: int = 1500,
):
    from aof_cfr.solve import SolveResult

    from aof_cfr.terminal_ev import _exact_nway_shares

    _exact_nway_shares([(0, 1), (2, 3), (4, 5), (6, 7)][:num_players])

    def _mbbg(bb: float) -> float:
        return float(bb) * 1000.0
    solver = PushfoldMCCFRSolver(num_players, seed=seed)
    done = iterations
    solve_started = time.monotonic()
    progress_every = 500 if iterations >= 1000 else 100
    print(
        f"  {num_players}-max native MCCFR: {iterations} initial iterations "
        f"(max={max_iterations}, target={target_mbb_g:.1f} mbb/g, "
        f"check_every={check_every}) ...",
        flush=True,
    )
    _run_iterations(
        solver,
        start=0,
        target=iterations,
        progress_every=progress_every,
        label="iteration",
    )

    def measure(done_iters: int) -> tuple[SolveResult, float]:
        rows = solver.average_policy_rows()
        print(
            f"  measuring MC exploitability ({mc_samples} deals, {len(rows)} rows) ...",
            flush=True,
        )
        nc, imps, method = estimate_nash_conv_mc(
            num_players, rows, samples=mc_samples, seed=seed + 1, show_progress=True
        )
        mbb = _mbbg(nc)
        total_elapsed = time.monotonic() - solve_started
        print(
            f"  after {done_iters} iters: {mbb:.2f} mbb/g ({len(rows)} rows) "
            f"total_elapsed={_fmt_duration(total_elapsed)}",
            flush=True,
        )
        res = SolveResult(
            num_players=num_players,
            solver_name="external_sampling_mccfr_native",
            iterations=done_iters,
            seed=seed,
            exploitability_bb_per_hand=nc,
            exploitability_mbb_g=mbb,
            br_method=method,
            player_improvements=imps,
            policy_rows=rows,
            nash_disclaimer=(
                "N>2: exploitability is MC combo-deal best-response gain "
                "(quality metric, not certified unique Nash equilibrium)."
            ),
        )
        return res, mbb

    result, mbb = measure(iterations)
    if mbb <= target_mbb_g:
        return result

    step = iterations
    while mbb > target_mbb_g and step < max_iterations:
        extra = min(check_every, max_iterations - step)
        target_step = step + extra
        print(
            f"  continuing to {target_step} iterations "
            f"(+{extra}, current={mbb:.1f} mbb/g, target={target_mbb_g:.1f}) ...",
            flush=True,
        )
        _run_iterations(
            solver,
            start=step,
            target=target_step,
            progress_every=min(500, max(100, extra // 4)),
            label="continuation",
        )
        step = target_step
        result, mbb = measure(step)
        done = step
        if mbb <= target_mbb_g:
            break
    print(
        f"  {num_players}-max MCCFR finished at {done} iterations: "
        f"{mbb:.2f} mbb/g (wall={_fmt_duration(time.monotonic() - solve_started)})",
        flush=True,
    )
    return result
