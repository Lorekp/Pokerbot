"""CoinPoker NLHE cash-game rake + CoinRewards rakeback model.

Pure, dependency-light, certifiable in isolation. The chip-EV engine terminal stays untouched; this
module is what eval (and later the raked-equilibrium retrain) calls to turn a chip-EV pot into a *net*
result. Everything here is in **big-blind (BB) units**, which is the key simplification:

    * CoinPoker rake is **5% flat across every stake**.
    * What changes by stake is only the **cap, expressed in BB** — and the cap is tiered by the number of
      players **dealt** into the hand (2 / 3-4 / 5+), confirmed dealt-count (not flop-seers) with
      **no-flop-no-drop** (a hand that ends preflop pays 0).

Because the harness already works in BB and the cap is in BB, ONE simulation's pot distribution yields the
rake curve for **every stake** by swapping the cap value — the stake never enters except through `cap_bb`.

The decisive structural fact this encodes: the BB cap collapses ~25x from micro (10.00 BB, 5+) to high
(0.40 BB, $25/50, 5+). So at micro the cap essentially never binds (5% on ~every flopped pot = brutal);
at high it binds at an ~8 BB pot (rake is a tiny flat tax). CoinRewards rakeback then INVERTS this again at
the bottom: micro fees are ~100% redistributed, so net micro rake is small but variance-laden. The worst
*net* rake lands in the low/mid valley where the cap is still fat AND the rich rakeback programs have ended.

Rake attribution uses the **contributed method** (industry standard, and exactly how CoinPoker's CoinRaces
counts "1 Coin = $1 rake generated"): a player's rake share = total pot rake x (their contribution / pot).
That is unbiased, lower-variance than winner-pays, and makes the rakeback computation trivial.
"""

from __future__ import annotations

from dataclasses import dataclass

RAKE_PCT = 0.05  # 5% flat, all stakes, all formats (Hold'em cash).


@dataclass(frozen=True)
class Stake:
    key: str
    sb_usd: float
    bb_usd: float
    cap_bb: dict[int, float]  # tier (2 / 3 / 5) -> cap in BB
    tier_band: str            # "micro" | "low" | "mid" | "high" (CoinRewards segmentation)


# CoinPoker Hold'em cash rake schedule (cap in BB), per the published table (2026-06).
# tier key: 2 = heads-up (<=2 dealt), 3 = 3-4 dealt, 5 = 5+ dealt.
STAKES: dict[str, Stake] = {
    s.key: s
    for s in [
        # ---- Micro (CoinRewards: 100% of fees redistributed to the micro pool) ----
        Stake("0.01/0.02", 0.01, 0.02, {2: 2.50, 3: 6.00, 5: 10.00}, "micro"),
        Stake("0.02/0.05", 0.02, 0.05, {2: 2.60, 3: 6.00, 5: 10.00}, "micro"),
        # ---- Low ----
        Stake("0.05/0.10", 0.05, 0.10, {2: 2.50, 3: 6.00, 5: 10.00}, "low"),
        Stake("0.10/0.25", 0.10, 0.25, {2: 2.00, 3: 4.80, 5: 8.00}, "low"),
        Stake("0.25/0.50", 0.25, 0.50, {2: 2.00, 3: 4.80, 5: 8.00}, "low"),
        # ---- Mid ----
        Stake("0.50/1", 0.50, 1.00, {2: 1.25, 3: 3.00, 5: 5.00}, "mid"),
        Stake("1/2", 1.00, 2.00, {2: 0.75, 3: 1.80, 5: 3.00}, "mid"),
        Stake("2/5", 2.00, 5.00, {2: 0.40, 3: 0.96, 5: 1.60}, "mid"),
        # ---- High ----
        Stake("5/10", 5.00, 10.00, {2: 0.25, 3: 0.60, 5: 1.00}, "high"),
        Stake("10/20", 10.00, 20.00, {2: 0.19, 3: 0.45, 5: 0.75}, "high"),
        Stake("25/50", 25.00, 50.00, {2: 0.10, 3: 0.24, 5: 0.40}, "high"),
    ]
}

# Ordered low->high, for reporting the curve.
STAKE_ORDER: list[str] = list(STAKES.keys())


def cap_tier(n_dealt: int) -> int:
    """Map players-dealt to the cap-tier key (2 / 3 / 5). Confirmed dealt-count, not flop-seers."""
    if n_dealt <= 2:
        return 2
    if n_dealt <= 4:
        return 3
    return 5


def coinpoker_rake_bb(pot_bb: float, n_dealt: int, ended_preflop: bool, stake_key: str) -> float:
    """Total rake removed from a pot, in BB.

    NLNFD: a hand that ends preflop (no flop dealt) pays zero. Otherwise 5% of the pot, capped at the
    stake's BB cap for the dealt-count tier.
    """
    if ended_preflop:
        return 0.0  # no flop, no drop
    cap = STAKES[stake_key].cap_bb[cap_tier(n_dealt)]
    return min(RAKE_PCT * pot_bb, cap)


def contributed_rake_bb(
    pot_bb: float, hero_committed_bb: float, n_dealt: int, ended_preflop: bool, stake_key: str
) -> float:
    """Hero's share of the pot's rake, contributed method = rake x (hero_committed / pot).

    This is the unbiased, low-variance attribution and matches CoinRaces' "Coins = rake generated".
    """
    if pot_bb <= 0.0:
        return 0.0
    rake = coinpoker_rake_bb(pot_bb, n_dealt, ended_preflop, stake_key)
    return rake * (hero_committed_bb / pot_bb)


# ---------------------------------------------------------------------------------------------------
# CoinRewards rakeback model
# ---------------------------------------------------------------------------------------------------
# CoinPoker returns a large fraction of fees through CoinRewards. Only ONE piece is contractually certain
# and per-player deterministic: the **15% Daily Guaranteed Rewards** (all stakes, automatic). The rest
# (CoinRaces $1M/wk proportional to rake, Infused Splash Pots, Bankroll Boosts, and at micro an explicit
# "100% of fees redistributed to the micro pool") is volume/population-shared and variance-laden, so it is
# an *estimate in expectation*, not a guarantee. We therefore expose TWO modes:
#
#   "floor"   -> 0.15 everywhere (only the Daily Guaranteed; the honest lower bound).
#   "modeled" -> floor PLUS a per-band promotional estimate (clearly an assumption, reported as a band).
#
# Promotional estimates (expectation, proportional-redistribution reasoning):
#   * micro  ($0.01/0.02-$0.02/0.05): official "100% of fees back to the micro pool" -> ~1.00 total.
#   * low    ($0.05/0.10-$0.25/0.50): Splash Pots active (<=$0.25/0.50) + CoinRaces share + 15% -> ~0.40.
#   * mid    ($0.50/1-$2/5):          CoinRaces share + 15% (no splash) -> ~0.25.
#   * high   ($5/10-$25/50):          CoinRaces (eligible <=$25/50) + 15%; HSPR VIP at $25/50+ is separate
#                                     and not counted here -> ~0.20.
_MODELED_RAKEBACK_BY_BAND = {"micro": 1.00, "low": 0.40, "mid": 0.25, "high": 0.20}
_GUARANTEED_RAKEBACK = 0.15


def rakeback_fraction(stake_key: str, mode: str = "modeled") -> float:
    """Fraction of a player's paid rake returned via CoinRewards. mode in {"floor","modeled"}.

    Clamped to [0, 1]. "floor" is the certain 15% Daily Guaranteed; "modeled" adds the documented but
    expectation-only promotional band for the stake's CoinRewards segment.
    """
    if mode == "floor":
        return _GUARANTEED_RAKEBACK
    if mode != "modeled":
        raise ValueError(f"mode must be 'floor' or 'modeled', got {mode!r}")
    band = STAKES[stake_key].tier_band
    return max(0.0, min(1.0, _MODELED_RAKEBACK_BY_BAND[band]))


def net_rake_bb(
    pot_bb: float,
    hero_committed_bb: float,
    n_dealt: int,
    ended_preflop: bool,
    stake_key: str,
    *,
    rakeback_mode: str = "modeled",
) -> float:
    """Hero's net rake paid after CoinRewards rakeback, in BB (>= 0). The cash 'cost' of one hand."""
    gross = contributed_rake_bb(pot_bb, hero_committed_bb, n_dealt, ended_preflop, stake_key)
    return gross * (1.0 - rakeback_fraction(stake_key, rakeback_mode))


# ---------------------------------------------------------------------------------------------------
# Infused Splash Pots fee (a SEPARATE cost from rake)
# ---------------------------------------------------------------------------------------------------
# CoinPoker takes a flat 0.1 BB "splash fee" on EVERY dealt hand at the non-micro low stakes
# ($0.05/0.10-$0.25/0.50). Two ways it differs from rake: (1) it is taken even on hands that end
# preflop (it is NOT subject to no-flop-no-drop), and (2) it is charged regardless of pot size. Micro
# ($0.01/0.02-$0.02/0.05) is "not taken from the pot, all splashes infused" -> 0; mid/high have no
# splash program -> 0. The fee funds the Splash Pool that pays back as Infused Splash Pots; that REBATE
# is already folded into the "low ~40%" modeled rakeback band above. Adding this COST is what makes the
# low band symmetric -- before this, low stakes were credited the splash-pot rebate with no matching fee.
SPLASH_FEE_BB = 0.1  # per dealt hand, eligible stakes only.


def splash_eligible(stake_key: str) -> bool:
    """True only for the non-micro low stakes ($0.05/0.10-$0.25/0.50) -- the model's 'low' band."""
    return STAKES[stake_key].tier_band == "low"


def splash_fee_bb(pot_bb: float, hero_committed_bb: float, stake_key: str) -> float:
    """Hero's share of the per-hand splash fee, contributed method (same attribution as rake).

    Charged on EVERY dealt hand at eligible stakes, INCLUDING hands that end preflop (no NLNFD). On a
    preflop fold the 'pot' is just the posted blinds, so contributed attribution gives the hero their
    blind share of the 0.1 BB. 0 at micro/mid/high (no splash fee taken).
    """
    if not splash_eligible(stake_key) or pot_bb <= 0.0:
        return 0.0
    return SPLASH_FEE_BB * (hero_committed_bb / pot_bb)


def net_cost_bb(
    pot_bb: float,
    hero_committed_bb: float,
    n_dealt: int,
    ended_preflop: bool,
    stake_key: str,
    *,
    rakeback_mode: str = "modeled",
) -> float:
    """Hero's full net cash cost of one hand, in BB: net rake (after rakeback) + splash fee.

    This is the complete per-hand 'cost' the net win-rate subtracts. net_rake_bb stays the rake-only
    figure (unchanged) for the rake-floor curve; this is the figure to subtract for true net bb/100.
    """
    rake = net_rake_bb(
        pot_bb, hero_committed_bb, n_dealt, ended_preflop, stake_key, rakeback_mode=rakeback_mode
    )
    return rake + splash_fee_bb(pot_bb, hero_committed_bb, stake_key)
