"""Per-stake CoinPoker rake-floor eval for the cash 6-max game.

Plays a seat-rotated, duplicate-dealt hero-vs-field sim, captures per-hand (pot_bb, ended_preflop,
hero contribution, hero chip delta), then applies the CoinPoker rake + CoinRewards rakeback model
(`pokerbot/economics/rake.py`) to produce, FOR EVERY STAKE FROM ONE SIM:

    * gross bb/100          -- chip-EV win-rate vs the field (no rake)
    * gross rake drag       -- bb/100 of rake the hero pays (contributed method), pre-rakeback
    * net rake (floor)      -- after only the certain 15% Daily Guaranteed rakeback
    * net rake (modeled)    -- after the modeled CoinRewards promotional band
    * net bb/100            -- gross - net-rake, with 95% CI (floor & modeled)

The pot distribution is in BB and the cap is in BB, so one sim covers all 11 stakes (only the cap swaps).
The gross win-rate is a PROXY (archetype field, not the real CoinPoker population) -- the decision-relevant
artifact is the rake-drag / net-rake curve, which is style-robust. Full ring => every hand is the 5+ tier.

Usage:
  python scripts/cash_rake_floor.py --hands 6000 --hero tag
  python scripts/cash_rake_floor.py --hands 6000 --hero tag --field calling_station,lag,nit,tag,random,human_like
"""

from __future__ import annotations

import argparse
import math
import random

import numpy as np

from pokerbot.engine.game import NUM_PLAYERS, play_hand, GameState
from pokerbot.eval.bots import BOT_REGISTRY
from pokerbot.economics import rake as rk


def _ci95(series: np.ndarray) -> tuple[float, float]:
    """(mean_bb_per_100, ci95_bb_per_100) for a per-hand bb series."""
    n = series.shape[0]
    if n == 0:
        return 0.0, 0.0
    mean = float(series.mean())
    stderr = float(series.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mean * 100.0, 1.96 * stderr * 100.0


def simulate(hero_name: str, field_names: list[str], hands: int, seed: int):
    """Return per-hand arrays: hero_delta_bb, pot_bb, hero_committed_bb, ended_preflop(bool)."""
    hero = BOT_REGISTRY[hero_name]
    field = [BOT_REGISTRY[field_names[i % len(field_names)]] for i in range(NUM_PLAYERS)]

    delta, pot, contrib, pre = [], [], [], []
    groups = max(1, math.ceil(hands / NUM_PLAYERS))
    for g in range(groups):
        for hero_seat in range(NUM_PLAYERS):
            rng = random.Random(seed + g)  # duplicate dealing: same deck across the seat rotation
            policies = {s: field[s](random.Random(seed + 101 * s)) for s in range(NUM_PLAYERS)}
            policies[hero_seat] = hero(random.Random(seed ^ (hero_seat + 1)))
            state = GameState.new_hand(rng=rng)
            final_state, d = play_hand(policies, state=state, rng=rng)
            ps = final_state.players
            delta.append(float(d.get(hero_seat, 0.0)))
            pot.append(float(sum(p.committed_total_bb for p in ps)))
            contrib.append(float(ps[hero_seat].committed_total_bb))
            pre.append(len(final_state.board) == 0)
    return (
        np.asarray(delta, dtype=np.float64),
        np.asarray(pot, dtype=np.float64),
        np.asarray(contrib, dtype=np.float64),
        np.asarray(pre, dtype=bool),
    )


def _lognormal_pots(mean_bb: float, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n flopped-pot sizes (bb) from a lognormal with the given mean and coefficient of variation."""
    sigma2 = math.log(1.0 + cv * cv)
    mu = math.log(mean_bb) - 0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=math.sqrt(sigma2), size=n)


def analytic_rake_curve(flop_per_100: float, mean_pot_bb: float, hero_share: float,
                        cv: float, n_samples: int, seed: int,
                        splash_share: float = 1.0 / NUM_PLAYERS):
    """Per-stake rake+splash bb/100 from a realistic flopped-pot-size model (cap nonlinearity by sampling).

    The rake floor = (pots/100 hero contributes to that see a flop) x E[ min(5%*pot, cap_5+) * hero_share ].
    The splash floor = 0.1bb on EVERY dealt hand (100/100) x hero's symmetric contribution share, at the
    eligible low stakes only (0 elsewhere); it is flop-frequency-independent (charged even on preflop folds).
    Full ring => 5+ tier. Returns {stake: (gross_drag, net_floor, net_modeled, splash)} in bb/100.
    """
    rng = np.random.default_rng(seed)
    pots = _lognormal_pots(mean_pot_bb, cv, n_samples, rng)
    five_pct = rk.RAKE_PCT * pots
    out = {}
    for sk in rk.STAKE_ORDER:
        cap = rk.STAKES[sk].cap_bb[5]
        per_pot_hero = float(np.minimum(five_pct, cap).mean()) * hero_share
        gross = flop_per_100 * per_pot_hero
        splash = 100.0 * rk.SPLASH_FEE_BB * splash_share if rk.splash_eligible(sk) else 0.0
        out[sk] = (
            gross,
            gross * (1.0 - rk.rakeback_fraction(sk, "floor")),
            gross * (1.0 - rk.rakeback_fraction(sk, "modeled")),
            splash,
        )
    return out


def total_floor_modeled(curve: dict, sk: str) -> float:
    """The full bb/100 a winning reg must beat at a stake: net rake (modeled rakeback) + splash fee."""
    return curve[sk][2] + curve[sk][3]


def _print_analytic(label: str, curve: dict) -> None:
    print(f"\n[{label}]")
    hdr = (f"{'stake':<11}{'cap5+(bb)':>10}{'band':>6}{'rakeDrag':>10}{'netRake.f':>11}"
           f"{'netRake.m':>11}{'splash':>8}{'floor.m':>9}")
    print(hdr)
    print("-" * len(hdr))
    for sk in rk.STAKE_ORDER:
        g, nf, nm, sp = curve[sk]
        print(f"{sk:<11}{rk.STAKES[sk].cap_bb[5]:>10.2f}{rk.STAKES[sk].tier_band:>6}"
              f"{g:>10.2f}{nf:>11.2f}{nm:>11.2f}{sp:>8.2f}{nm + sp:>9.2f}")


def per_hand_rake(pot, contrib, pre, stake_key, mode):
    """Vectorized hero net rake (BB) per hand for one stake + rakeback mode (rake only, no splash)."""
    out = np.zeros_like(pot)
    rb = rk.rakeback_fraction(stake_key, mode)
    for i in range(pot.shape[0]):
        gross = rk.contributed_rake_bb(
            float(pot[i]), float(contrib[i]), NUM_PLAYERS, bool(pre[i]), stake_key
        )
        out[i] = gross * (1.0 - rb)
    return out


def per_hand_cost(pot, contrib, pre, stake_key, mode):
    """Vectorized hero full net cost (BB) per hand for one stake: net rake (after rakeback) + splash fee.

    Unlike per_hand_rake this uses the REAL per-hand contribution share for the splash fee (the sim tracks
    pot+contrib on every hand, including preflop folds), so splash is exact rather than the symmetric proxy.
    """
    out = np.zeros_like(pot)
    for i in range(pot.shape[0]):
        out[i] = rk.net_cost_bb(
            float(pot[i]), float(contrib[i]), NUM_PLAYERS, bool(pre[i]), stake_key, rakeback_mode=mode
        )
    return out


def parse_gross_bb(spec: str) -> dict[str, float]:
    """Parse 'soft=8.5,realistic=2.1,tough=-1.0' -> {field: bb_per_100}. Empty -> {}."""
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, val = part.partition("=")
        out[name.strip()] = float(val)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=6000)
    ap.add_argument("--hero", default="tag")
    ap.add_argument("--field", default="calling_station,lag,nit,tag,random,human_like")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--empirical", action="store_true",
                    help="also run the toy-bot full-hand sim (pots/flop-freq are NOT realistic; shape only)")
    # Realistic 6-max TAG-reg operating point for the analytic rake floor:
    ap.add_argument("--flop-per-100", type=float, default=26.0,
                    help="flopped pots/100 the hero contributes to (VPIP*reach-flop, ~22-30 for a reg)")
    ap.add_argument("--mean-pot", type=float, default=20.0, help="mean flopped pot in bb")
    ap.add_argument("--hero-share", type=float, default=0.45, help="hero's mean contribution share of those pots")
    ap.add_argument("--pot-cv", type=float, default=1.2, help="coeff. of variation of pot size (heavy tail)")
    ap.add_argument("--splash-share", type=float, default=1.0 / NUM_PLAYERS,
                    help="hero's all-hands contribution share for the 0.1bb splash fee (symmetric ~1/6)")
    ap.add_argument("--gross-bb", default="",
                    help="compose a stake x field NET matrix from MEASURED real-bot gross edges, e.g. "
                         "'soft=8.5,realistic=2.1,tough=-1.0,pro=-3.0' (bb/100 per field)")
    args = ap.parse_args()

    # ---- Analytic rake floor (headline): credible magnitudes from a realistic pot-size model ----
    print("\n==================  CoinPoker cash RAKE FLOOR (analytic, realistic pots)  ==================")
    print(f"operating point: flop_per_100={args.flop_per_100}  mean_pot={args.mean_pot}bb  "
          f"hero_share={args.hero_share}  pot_cv={args.pot_cv}  (full ring => 5+ tier, NLNFD)")
    base = analytic_rake_curve(args.flop_per_100, args.mean_pot, args.hero_share, args.pot_cv,
                               1 << 16, args.seed, args.splash_share)
    _print_analytic("central operating point", base)
    # sensitivity: scale the pot/volume operating point +/-50%
    lo = analytic_rake_curve(args.flop_per_100 * 0.6, args.mean_pot * 0.7, args.hero_share, args.pot_cv,
                             1 << 16, args.seed, args.splash_share)
    hi = analytic_rake_curve(args.flop_per_100 * 1.4, args.mean_pot * 1.3, args.hero_share, args.pot_cv,
                             1 << 16, args.seed, args.splash_share)
    worst = max(rk.STAKE_ORDER, key=lambda s: total_floor_modeled(base, s))
    print(f"\nfloor.modeled VALLEY (worst stake to play, fees-wise): {worst}  "
          f"({total_floor_modeled(base, worst):.2f} bb/100, incl. splash)")
    print("sensitivity (floor.modeled = netRake.m + splash, bb/100) tighter..wider operating point:")
    for sk in rk.STAKE_ORDER:
        print(f"  {sk:<11} {total_floor_modeled(lo, sk):>6.2f} .. {total_floor_modeled(hi, sk):>6.2f}")
    print("\nLegend: rakeDrag = pre-rakeback contributed rake (bb/100); netRake.f = after the certain 15% "
          "Daily Guaranteed; netRake.m = after the modeled CoinRewards band (micro~100/low~40/mid~25/high~20%);")
    print("        splash = 0.1bb/hand fee at the low stakes (NOT no-flop-no-drop); floor.m = netRake.m + splash.")
    print("Read: to net-profit at a stake the bot's TRUE gross edge vs the field must exceed floor.m there.")
    print("Caveat: the splash FEE is partly returned as Infused Splash Pots (already credited in the low band),")
    print("        so for a high-volume reg floor.m at the low stakes is a slightly CONSERVATIVE upper bound.")

    # ---- Optional: compose a real-bot gross edge per field into a stake x field NET matrix ----
    gross_by_field = parse_gross_bb(args.gross_bb)
    if gross_by_field:
        fields = list(gross_by_field)
        print("\n==================  NET bb/100 by STAKE x FIELD (gross edge - floor.modeled)  ==================")
        print("gross edges (bb/100): " + "  ".join(f"{f}={gross_by_field[f]:+.2f}" for f in fields))
        hdr = f"{'stake':<11}{'floor.m':>9}" + "".join(f"{f:>12}" for f in fields)
        print(hdr)
        print("-" * len(hdr))
        for sk in rk.STAKE_ORDER:
            floor = total_floor_modeled(base, sk)
            cells = []
            for f in fields:
                net = gross_by_field[f] - floor
                cells.append(f"{net:>+11.2f}{'*' if net > 0 else ' '}")
            print(f"{sk:<11}{floor:>9.2f}" + "".join(cells))
        print("\n('*' = net-positive: the bot's gross edge clears the modeled fee floor at that stake/field.)")
        print("Gross edges are MEASURED real-bot numbers (run_field_absolute off_aivat.adj_mean_bb); the floor")
        print("is style-robust, so this matrix is the bot's profitable operating envelope by stake and field.")

    if not args.empirical:
        return

    field_names = [s.strip() for s in args.field.split(",") if s.strip()]
    delta, pot, contrib, pre = simulate(args.hero, field_names, args.hands, args.seed)
    n = delta.shape[0]

    gross_wr, gross_ci = _ci95(delta)
    flopped = float((~pre).mean()) * 100.0
    mean_pot = float(pot[~pre].mean()) if (~pre).any() else 0.0

    print(f"\nCoinPoker cash rake-floor  (hero={args.hero} vs field={field_names})")
    print(f"hands={n}  flopped={flopped:.1f}%  mean_flopped_pot={mean_pot:.1f}bb  "
          f"gross={gross_wr:+.2f} +/- {gross_ci:.2f} bb/100  (PROXY field)\n")

    hdr = (f"{'stake':<11}{'cap5+(bb)':>9}{'gross':>8}{'rakeDrag':>9}{'splash':>8}"
           f"{'netCost.f':>10}{'netCost.m':>10}{'netWR.f':>9}{'netWR.m':>9}")
    print(hdr)
    print("-" * len(hdr))
    for sk in rk.STAKE_ORDER:
        cap5 = rk.STAKES[sk].cap_bb[5]
        gross_drag = np.array([
            rk.contributed_rake_bb(float(pot[i]), float(contrib[i]), NUM_PLAYERS, bool(pre[i]), sk)
            for i in range(n)
        ])
        splash = np.array([rk.splash_fee_bb(float(pot[i]), float(contrib[i]), sk) for i in range(n)])
        cost_f = per_hand_cost(pot, contrib, pre, sk, "floor")
        cost_m = per_hand_cost(pot, contrib, pre, sk, "modeled")
        drag_bb100 = float(gross_drag.mean()) * 100.0
        splash_bb100 = float(splash.mean()) * 100.0
        netcost_f = float(cost_f.mean()) * 100.0
        netcost_m = float(cost_m.mean()) * 100.0
        netwr_f, _ = _ci95(delta - cost_f)
        netwr_m, _ = _ci95(delta - cost_m)
        print(f"{sk:<11}{cap5:>9.2f}{gross_wr:>+8.2f}{drag_bb100:>9.2f}{splash_bb100:>8.2f}"
              f"{netcost_f:>10.2f}{netcost_m:>10.2f}{netwr_f:>+9.2f}{netwr_m:>+9.2f}")

    print("\nLegend: rakeDrag=pre-rakeback contributed rake (bb/100); splash=0.1bb/hand fee at low stakes; "
          "netCost.f/.m=net rake (after 15%-floor / modeled rakeback) + splash; netWR=gross-netCost.")
    print("gross is a PROXY vs the archetype field. To be net-positive the bot's TRUE gross edge must exceed "
          "that stake's netCost column (pass real per-field gross edges via --gross-bb for the honest matrix).")


if __name__ == "__main__":
    main()
