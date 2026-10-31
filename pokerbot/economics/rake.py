"""CoinPoker NLHE cash-game rake + CoinRewards rakeback model + Kelly bankroll advisory.

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

import logging
from dataclasses import dataclass

_log = logging.getLogger(__name__)

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

    Defensive (ECON-1): ``pot_bb`` MUST be >= 0 — a negative pot is a caller bug (chips-vs-BB confusion)
    that would silently return a NEGATIVE rake (a "rake credit") in production. We raise ValueError
    instead of returning a silent-negative so a deploy catches the unit-confusion bug.
    """
    if pot_bb < 0.0:
        raise ValueError(
            f"coinpoker_rake_bb: pot_bb={pot_bb} is negative (chips-vs-BB confusion?)"
        )
    if ended_preflop:
        return 0.0  # no flop, no drop
    cap = STAKES[stake_key].cap_bb[cap_tier(n_dealt)]
    return min(RAKE_PCT * pot_bb, cap)


def marginal_rake_bb(
    pot_before_bb: float, pot_after_bb: float, n_dealt: int, stake_key: str
) -> float:
    """Extra TOTAL rake incurred by growing a (postflop) pot from ``pot_before`` to ``pot_after``, in BB.

    ``= coinpoker_rake_bb(pot_after) - coinpoker_rake_bb(pot_before)`` with ``ended_preflop=False`` (a
    pot that reaches a postflop bet has, by definition, seen a flop — no-flop-no-drop does not apply).

    Direction (the whole point of the rake-aware sizing discount): rake is ``min(5%·pot, cap)``.
      * BELOW the cap, both endpoints are on the 5% ramp ⇒ marginal rake = ``5%·(after - before)`` —
        growing the pot by Δ forfeits 5% of Δ. Thin value into a small pot pays this in full.
      * AT/ABOVE the cap, rake is FLAT ⇒ marginal rake → 0 — an over-bet above the cap is rake-FREE,
        so escalation there must NOT be discounted. The ``min`` makes this fall out naturally: when both
        endpoints are above the cap the difference is 0; when ``before`` is below and ``after`` above,
        only the taxed portion up to the cap counts.

    Never negative (``pot_after >= pot_before`` for an escalation; clamped at 0 defensively)."""
    if pot_after_bb <= pot_before_bb:
        return 0.0
    after = coinpoker_rake_bb(pot_after_bb, n_dealt, ended_preflop=False, stake_key=stake_key)
    before = coinpoker_rake_bb(pot_before_bb, n_dealt, ended_preflop=False, stake_key=stake_key)
    return max(0.0, after - before)


def contributed_rake_bb(
    pot_bb: float, hero_committed_bb: float, n_dealt: int, ended_preflop: bool, stake_key: str
) -> float:
    """Hero's share of the pot's rake, contributed method = rake x (hero_committed / pot).

    This is the unbiased, low-variance attribution and matches CoinRaces' "Coins = rake generated".

    Defensive (ECON-2): ``hero_committed_bb`` MUST satisfy ``0 <= h <= pot_bb``. An over-committed hero
    (h > pot) is a caller bug that silently over-pays rake by a factor of ``h/pot``. A negative
    ``hero_committed_bb`` is also rejected — a negative share is impossible. We raise ValueError
    so the bug is caught at the call site instead of silently leaking extra rake attribution.
    """
    if pot_bb <= 0.0:
        return 0.0
    if hero_committed_bb < 0.0:
        raise ValueError(
            f"contributed_rake_bb: hero_committed_bb={hero_committed_bb} is negative"
        )
    if hero_committed_bb > pot_bb:
        raise ValueError(
            f"contributed_rake_bb: hero_committed_bb={hero_committed_bb} > pot_bb={pot_bb} "
            f"(impossible — caller bug?)"
        )
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

    Defensive (ECON-3): when ``splash_eligible(stake_key)`` is True and ``pot_bb > 0`` BUT
    ``hero_committed_bb == 0``, the caller has likely forgotten to track the posted blind. The
    splash fee is then silently ZERO (instead of the hero's BB share), which over-states net win-rate
    in the eval pipeline at low stakes. We log a WARNING so the silent zero is observable in
    operator logs.
    """
    if not splash_eligible(stake_key) or pot_bb <= 0.0:
        return 0.0
    if hero_committed_bb == 0.0:
        _log.warning(
            "splash_fee_bb: hero_committed_bb=0 at eligible stake %s (pot_bb=%s) "
            "— caller forgot to track posted blind? Splash fee silently dropped.",
            stake_key, pot_bb,
        )
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


# ---------------------------------------------------------------------------------------------------
# Kelly criterion (bankroll fraction, advisory only)
# ---------------------------------------------------------------------------------------------------
# ECONOMICS-AUDIT (Top 5 design upgrades, item 1): add a Kelly helper for the live table-select
# advisory so it can stop using the conservative 30-buyin heuristic at the extremes. Pure, dependency-
# light, certified in isolation. The Kelly fraction is *advisory only* — the live decision still
# consults the comfortable-bankroll gate (a multi-buyin minimum), but the Kelly output tells the
# player WHAT FRACTION of bankroll to put at risk at a given edge/variance.
#
# Banker (no-loss) Kelly: f* = edge / (odds^2) where odds = sigma/edge. Equivalent form f* = edge /
# sigma^2. Output is bankroll FRACTION in [0, 1] (1.0 = "all-in at this edge"). The clamp at 0 (no
# negative bet) and 1 (no leverage) is a safety: a formula blow-up (e.g. zero sigma, edge>0) is
# downgraded to 1.0 (full-Kelly, the most aggressive the math ever recommends) instead of NaN/Inf.
#
# ECONOMICS-AUDIT note: We use a slightly conservative fractional-Kelly ceiling of 1.0 (full Kelly)
# to keep the test properties clean (the bound is "0..1"); live callers can multiply by an
# additional fractional-Kelly factor (e.g. 0.5 for half-Kelly) at the call site.
KELLY_MAX = 1.0  # clamp the advisory at full-Kelly (no leverage)


def kelly_fraction_banker(edge_bb: float, sigma_bb: float) -> float:
    """Kelly criterion (banker / no-loss form) as a bankroll fraction in [0, KELLY_MAX].

    Parameters
    ----------
    edge_bb  : per-hand expected edge in BB (positive = advantage, negative = disadvantage).
    sigma_bb : per-hand standard deviation in BB (>= 0). A non-positive sigma is degenerate and is
               treated as "no variance" -> Kelly is undefined; the helper returns KELLY_MAX
               (conservatively full-Kelly) for edge > 0 and 0.0 for edge <= 0.

    Returns
    -------
    f* in [0, KELLY_MAX] : the bankroll fraction to wager. 0 means "do not play this edge".
    A negative-edge + finite-sigma game returns exactly 0 (you don't bet on a -EV game). The result
    is finite for every (edge, sigma) in the live decision space.

    Defensive: a non-finite or NaN input -> returns 0.0 (do not play). A negative sigma -> 0.0.
    """
    import math as _math
    for v in (edge_bb, sigma_bb):
        if v is None or not _math.isfinite(float(v)):
            return 0.0
    if sigma_bb < 0.0:
        return 0.0
    if sigma_bb == 0.0:
        # Degenerate: zero variance. If edge > 0, the bet is a "lock" -> max Kelly; if edge <= 0,
        # no bet. Matches the limit-from-the-right of f* as sigma -> 0+.
        return KELLY_MAX if edge_bb > 0.0 else 0.0
    if edge_bb <= 0.0:
        return 0.0  # no wagering on a -EV game (full Kelly is non-negative for a bankroll bet)
    f = edge_bb / (sigma_bb * sigma_bb)
    if f > KELLY_MAX:
        return KELLY_MAX
    return float(f)


# ---------------------------------------------------------------------------------------------------
# Play-money guard (fail-closed action override)
# ---------------------------------------------------------------------------------------------------
# The live stack's play-money guard is the LAST line of defense against sending actions to a real-
# money table. The contract: a "refuse" verdict from the guard wins over ANY bot decision, including
# an apparently +EV one. The economics layer models this as a tiny pure function that maps
# (is_practice, bot_action) -> (allowed_action) so it can be property-tested in isolation — the
# actual live send path is a different module, but the GUARANTEE is the same.
#
# Only an explicit True (verified play-money) clears the guard. Unknown or false inputs
# fail closed. Research callers can use this check on supplied game metadata.
#
# We model bot decisions as opaque strings ("FOLD" / "CALL" / "RAISE_3" / etc.). The guard's job is
# to either pass the decision through unchanged or replace it with the sentinel "REFUSE".

PLAY_MONEY_REFUSE = "REFUSE"  # sentinel: the action the live sender must NEVER transmit


def play_money_guard(is_practice, bot_action: str) -> str:
    """Pure, fail-closed play-money guard. Returns bot_action iff is_practice is True, else REFUSE.

    Parameters
    ----------
    is_practice : bool | None
        The verified coin-type verdict from the live capture. None = unknown (RPC failed) or
        False = real money; the only cleared value is True (confirmed play-money).
    bot_action : str
        The bot's chosen action (opaque string). Ignored unless the guard passes.

    Returns
    -------
    str
        bot_action verbatim if is_practice is True (play-money confirmed); the sentinel
        ``PLAY_MONEY_REFUSE`` otherwise. The caller must treat ``PLAY_MONEY_REFUSE`` as a hard
        block — i.e. do not transmit the action to the table.

    The guard is symmetric: any non-True is_practice value (False, None, the integer 0, the empty
    string) blocks. This is the documented "is_practice() is not True -> block" contract.
    """
    if is_practice is True:
        return bot_action
    return PLAY_MONEY_REFUSE
