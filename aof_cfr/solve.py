"""CFR+ / MCCFR drivers and exploitability measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyspiel

from aof_cfr.hu_payoff_matrix import load_equity_matrix
from open_spiel.python.algorithms import exploitability as exp_lib
from open_spiel.python.algorithms import external_sampling_mccfr


@dataclass
class SolveResult:
    num_players: int
    solver_name: str
    iterations: int
    seed: int
    exploitability_bb_per_hand: float
    exploitability_mbb_g: float
    br_method: str
    player_improvements: list[float] | None
    policy_rows: dict[str, np.ndarray]
    nash_disclaimer: str | None = None


def _bb_to_mbbg(bb_per_hand: float) -> float:
    return float(bb_per_hand) * 1000.0


def _infoset_key(state: pyspiel.State, player: int) -> str:
    raw = state.information_state_string(player)
    if "|" in raw:
        return raw.split("-", 3)[3]
    return raw


def _collect_policy_rows(game: pyspiel.Game, policy) -> dict[str, np.ndarray]:
    """Map infoset key -> [P(FOLD), P(ALL_IN)]."""
    rows: dict[str, np.ndarray] = {}
    state = game.new_initial_state()

    def visit(s: pyspiel.State) -> None:
        if s.is_terminal():
            return
        if s.is_chance_node():
            for action, _ in s.chance_outcomes():
                visit(s.child(action))
            return
        p = s.current_player()
        key = _infoset_key(s, p)
        if key not in rows:
            probs = policy.action_probabilities(s)
            vec = np.zeros(2, dtype=np.float64)
            for action, prob in probs.items():
                vec[int(action)] = float(prob)
            ssum = vec.sum()
            if ssum > 0:
                vec /= ssum
            else:
                vec[:] = 0.5
            rows[key] = vec
        for action in s.legal_actions(p):
            visit(s.child(action))

    visit(state)
    return rows


def measure_exploitability(
    game: pyspiel.Game,
    policy,
    *,
    br_method: str = "exact",
    policy_rows: dict[str, np.ndarray] | None = None,
    mc_samples: int = 4000,
) -> tuple[float, list[float], str]:
    """Return (bb/hand BR gain, per-player improvements, method label)."""
    n = game.num_players()
    if n == 2:
        bb = float(exp_lib.exploitability(game, policy))
        return bb, [bb / 2.0, bb / 2.0], br_method
    if policy_rows is None:
        from aof_cfr.policy_collect import collect_pushfold_policy_rows

        policy_rows = collect_pushfold_policy_rows(game, policy)
    from aof_cfr.exploit_mc import estimate_nash_conv_mc

    nc, improvements, method = estimate_nash_conv_mc(
        n, policy_rows, samples=mc_samples, seed=42
    )
    return nc, improvements, method


def solve_hu_cfr_plus(
    *,
    iterations: int = 2000,
    eq_matrix_path: str = "",
    use_efg: bool = True,
) -> SolveResult:
    from pathlib import Path

    from aof_cfr.efg_hu import load_hu_efg_game, write_hu_efg
    from aof_cfr.hu_export import hu_policy_rows_from_freqs

    eq_path = Path(eq_matrix_path) if eq_matrix_path else None
    if eq_path and eq_path.is_file():
        eq = load_equity_matrix(eq_path)
    else:
        from aof_cfr.hu_payoff_matrix import build_hu_bucket_equity_matrix

        eq = build_hu_bucket_equity_matrix(progress=False)

    if use_efg:
        out_efg = Path("data/aof/cfr_base/hu_pushfold.efg")
        write_hu_efg(out_efg, eq)
        game = load_hu_efg_game(eq)
        solver = pyspiel.CFRPlusSolver(game)
        for _ in range(iterations):
            solver.evaluate_and_update_policy()
        policy = solver.average_policy()
        bb, imps, method = measure_exploitability(game, policy)
        rows = _collect_policy_rows(game, policy)
        solver_name = "cfr_plus_efg"
    else:
        from pokerbot.solver.hu_preflop_dcfr import solve_hu_pushfold

        dcfr = solve_hu_pushfold(iterations=iterations, eq_matrix=eq, seed=42)
        rows = hu_policy_rows_from_freqs(dcfr.sb_push_freq, dcfr.bb_call_freq)
        bb = dcfr.exploitability_bb
        imps = [bb / 2.0, bb / 2.0]
        method = "exact"
        solver_name = "cfr_plus_dcfr"

    return SolveResult(
        num_players=2,
        solver_name=solver_name,
        iterations=iterations,
        seed=0,
        exploitability_bb_per_hand=bb,
        exploitability_mbb_g=_bb_to_mbbg(bb),
        br_method=method,
        player_improvements=imps,
        policy_rows=rows,
    )


def solve_3max_cfr_plus(
    *,
    iterations: int = 3000,
    seed: int = 42,
    tensor_path: str = "",
) -> SolveResult:
    from pathlib import Path

    from pokerbot.solver.trio_preflop_dcfr import solve_3max_pushfold, trio_policy_rows

    path = Path(tensor_path) if tensor_path else Path("data/aof/cfr_base/trio_eq_tensor.npz")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run scripts/build_3way_eq_tensor_npz.py first"
        )
    from aof_cfr.trio_payoff_tensor import load_trio_tensor

    from aof_cfr.fp_br import nash_gap_3max
    from aof_cfr.pushfold_payoffs import build_static_payoffs

    shares, pair_eq, deal_w = load_trio_tensor(path)
    print(f"  3-max vectorized CFR+ ({iterations} iterations) ...", flush=True)
    result = solve_3max_pushfold(
        iterations=iterations,
        seed=seed,
        shares=shares,
        pair_eq=pair_eq,
        deal_w=deal_w,
    )
    rows = trio_policy_rows(result)
    bb = result.exploitability_bb
    _, imps = nash_gap_3max(
        build_static_payoffs(shares, pair_eq),
        deal_w,
        result.btn_push_freq,
        result.sb_push_freq,
        result.bb_push_freq,
    )
    print(
        f"  3-max exploitability: {bb * 1000:.3f} mbb/g ({len(rows)} rows)",
        flush=True,
    )
    return SolveResult(
        num_players=3,
        solver_name="cfr_plus_trio_dcfr",
        iterations=iterations,
        seed=seed,
        exploitability_bb_per_hand=bb,
        exploitability_mbb_g=_bb_to_mbbg(bb),
        br_method="exact",
        player_improvements=imps,
        policy_rows=rows,
        nash_disclaimer=(
            "N=3: exploitability is exact tabular best-response gain over bucket deal."
        ),
    )


def solve_3max_fp(
    *,
    max_rounds: int = 1000,
    damping: float = 1.0,
    seed: int = 42,
    tensor_path: str = "",
) -> SolveResult:
    from pathlib import Path

    from pokerbot.solver.trio_preflop_fp import solve_3max_fp as _fp, trio_fp_policy_rows

    path = Path(tensor_path) if tensor_path else Path("data/aof/cfr_base/trio_eq_tensor.npz")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    from aof_cfr.trio_payoff_tensor import load_trio_tensor

    shares, pair_eq, deal_w = load_trio_tensor(path)
    print(
        f"  3-max fictitious play (max_rounds={max_rounds}, damping={damping}) ...",
        flush=True,
    )
    result = _fp(
        max_rounds=max_rounds,
        damping=damping,
        seed=seed,
        shares=shares,
        pair_eq=pair_eq,
        deal_w=deal_w,
    )
    rows = trio_fp_policy_rows(result)
    bb = result.exploitability_bb
    print(
        f"  3-max Nash gap: {bb * 1000:.3f} mbb/g rounds={result.rounds} "
        f"converged={result.converged} ({len(rows)} rows)",
        flush=True,
    )
    print(f"  trivial baselines mbb/g: {result.trivial_baselines_mbb_g}", flush=True)
    return SolveResult(
        num_players=3,
        solver_name="fictitious_play_3max",
        iterations=result.rounds,
        seed=seed,
        exploitability_bb_per_hand=bb,
        exploitability_mbb_g=_bb_to_mbbg(bb),
        br_method="exact",
        player_improvements=result.player_improvements,
        policy_rows=rows,
        nash_disclaimer=(
            "N=3: Nash gap is exact tabular BR over bucket deal; multiplayer has no unique Nash."
        ),
    )


def solve_4max_fp(
    *,
    max_rounds: int = 30,
    br_samples: int = 4000,
    seed: int = 42,
    damping: float = 1.0,
) -> SolveResult:
    from aof_cfr.fp_4max import solve_4max_fp as _fp

    print(
        f"  4-max fictitious play (max_rounds={max_rounds}, br_samples={br_samples}) ...",
        flush=True,
    )
    result = _fp(
        max_rounds=max_rounds,
        br_samples=br_samples,
        seed=seed,
        damping=damping,
    )
    bb = result.exploitability_bb
    print(
        f"  4-max Nash gap: {bb * 1000:.3f} mbb/g rounds={result.rounds} "
        f"converged={result.converged} method={result.br_method}",
        flush=True,
    )
    return SolveResult(
        num_players=4,
        solver_name="fictitious_play_4max",
        iterations=result.rounds,
        seed=seed,
        exploitability_bb_per_hand=bb,
        exploitability_mbb_g=_bb_to_mbbg(bb),
        br_method=result.br_method,
        player_improvements=result.player_improvements,
        policy_rows=result.policy_rows,
        nash_disclaimer=(
            "N=4: Nash gap is MC combo-deal best-response estimate; not certified unique Nash."
        ),
    )


def solve_pushfold_mccfr(
    num_players: int,
    *,
    iterations: int = 50_000,
    seed: int = 42,
    target_mbb_g: float = 10.0,
    max_iterations: int = 500_000,
    check_every: int = 3_000,
    use_native: bool = True,
) -> SolveResult:
    if num_players == 3:
        raise ValueError(
            "3-max uses solve_3max_fp() or solve_3max_cfr_plus(), not MCCFR"
        )
    if use_native:
        from aof_cfr.mccfr_native import solve_pushfold_mccfr_native

        return solve_pushfold_mccfr_native(
            num_players,
            iterations=iterations,
            seed=seed,
            target_mbb_g=target_mbb_g,
            max_iterations=max_iterations,
            check_every=check_every,
        )

    from aof_cfr.policy_collect import collect_pushfold_policy_rows

    import aof_cfr.game_pushfold  # noqa: F401

    game = pyspiel.load_game(f"python_aof_pushfold_{num_players}p")
    solver = external_sampling_mccfr.ExternalSamplingSolver(game)
    done = iterations
    print(f"  {num_players}-max MCCFR: {iterations} initial iterations ...", flush=True)
    for i in range(iterations):
        solver.iteration()
        if iterations >= 5000 and (i + 1) % 5000 == 0:
            print(f"    iteration {i + 1}/{iterations}", flush=True)
    policy = solver.average_policy()
    rows = collect_pushfold_policy_rows(game, policy)
    print(f"  collected {len(rows)} policy rows", flush=True)
    print(f"  measuring exploitability ({num_players}-max, MC) ...", flush=True)
    bb, imps, method = measure_exploitability(
        game, policy, br_method="mc_combo_br", policy_rows=rows
    )
    mbb = _bb_to_mbbg(bb)
    done = iterations
    print(f"  after {iterations} iters: {mbb:.2f} mbb/g", flush=True)
    if mbb <= target_mbb_g:
        disclaimer = (
            "N>2: exploitability reported as sum of unilateral best-response gains "
            "(MC combo-deal estimate; not certified unique Nash equilibrium)."
        )
        return SolveResult(
            num_players=num_players,
            solver_name="external_sampling_mccfr",
            iterations=done,
            seed=seed,
            exploitability_bb_per_hand=bb,
            exploitability_mbb_g=mbb,
            br_method=method,
            player_improvements=imps,
            policy_rows=rows,
            nash_disclaimer=disclaimer,
        )
    for step in range(iterations + check_every, max_iterations + 1, check_every):
        print(f"  continuing to {step} iterations ...", flush=True)
        for _ in range(check_every):
            solver.iteration()
        policy = solver.average_policy()
        rows = collect_pushfold_policy_rows(game, policy)
        bb, imps, method = measure_exploitability(
            game, policy, br_method="mc_combo_br", policy_rows=rows
        )
        mbb = _bb_to_mbbg(bb)
        done = step
        print(f"  after {step} iters: {mbb:.2f} mbb/g", flush=True)
        if mbb <= target_mbb_g:
            break
    disclaimer = (
        "N>2: exploitability reported as sum of unilateral best-response gains "
        "(MC combo-deal estimate; not certified unique Nash equilibrium)."
    )
    return SolveResult(
        num_players=num_players,
        solver_name="external_sampling_mccfr",
        iterations=done,
        seed=seed,
        exploitability_bb_per_hand=bb,
        exploitability_mbb_g=mbb,
        br_method=method,
        player_improvements=imps,
        policy_rows=rows,
        nash_disclaimer=disclaimer,
    )
