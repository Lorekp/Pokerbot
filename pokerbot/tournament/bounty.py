"""PKO (Progressive Knockout) bounty accounting + per-stack bounty value.

A PKO splits each post-rake buy-in 50/50 into a PRIZE pool (the finishing ladder, a flatter curve --
see :mod:`pokerbot.tournament.payouts`) and a BOUNTY pool distributed progressively as knockouts
happen. The progressive rule (``progressiveBountyPercentage = 50``):

    when you knock a player out you receive 50% of that player's CURRENT bounty as immediate cash,
    and the other 50% is added to YOUR OWN bounty (growing your value as a target).

Two responsibilities:
  * :class:`BountyState` -- track each live player's bounty, realize KO cash, and finalize the
    winner's own outstanding bounty. Maintains the **conservation invariant**
    ``sum(live_bounty) + sum(realized_cash) == initial_pool`` at every step (proved by construction:
    a KO moves ``0.5*victim`` from live->realized and keeps ``0.5*victim`` in live).
  * :func:`min_equity_to_call` / :func:`pko_call_dollar_ev` -- the reason PKO is **chip-EV != $-EV**:
    covering an all-in collects ``~0.5 * villain_bounty`` in cash on a win, so the indifference
    equity to call is LOWER than the chip-EV (ICM-only) threshold -> the bot correctly calls WIDER to
    collect bounties (often +$EV at a chip-EV loss). Built on the certified ICM primitive.
"""

from __future__ import annotations

from typing import Sequence

from pokerbot.tournament.icm import icm_equity


class BountyState:
    """Live progressive-bounty ledger for a PKO. ``progressive_frac`` is the fraction of a victim's
    bounty that GROWS the eliminator's own bounty (CoinPoker = 0.5); the rest is paid as cash, so the
    full victim bounty is always redistributed (cash + growth = victim bounty)."""

    def __init__(self, bounties: Sequence[float], *, progressive_frac: float = 0.5):
        if not (0.0 <= progressive_frac <= 1.0):
            raise ValueError("progressive_frac must be in [0,1]")
        self.live: list[float] = [float(b) for b in bounties]
        self.realized: list[float] = [0.0] * len(self.live)
        self.progressive_frac = float(progressive_frac)
        self.initial_pool = float(sum(self.live))

    @classmethod
    def uniform(cls, n_players: int, entry_bounty: float, *, progressive_frac: float = 0.5) -> "BountyState":
        return cls([entry_bounty] * n_players, progressive_frac=progressive_frac)

    def award_ko(self, eliminator: int, victim: int, *, winners: Sequence[int] | None = None) -> float:
        """Apply a knockout. ``winners`` (a split-pot KO) defaults to ``[eliminator]``; the victim's
        bounty cash + growth are divided equally among the winners. Returns the TOTAL cash realized
        by the winner(s) on this KO. The victim's bounty leaves play (fully redistributed)."""
        vb = self.live[victim]
        if vb < 0:
            raise ValueError("negative bounty")
        win = list(winners) if winners else [eliminator]
        if victim in win:
            raise ValueError("victim cannot be among the winners")
        k = len(win)
        cash_total = (1.0 - self.progressive_frac) * vb
        grow_total = self.progressive_frac * vb
        for w in win:
            self.realized[w] += cash_total / k
            self.live[w] += grow_total / k
        self.live[victim] = 0.0  # victim eliminated; their bounty is now fully distributed
        return cash_total

    def finalize_winner(self, winner: int) -> float:
        """The tournament winner is never eliminated, so their remaining OWN bounty is paid out to
        them at the end. Returns the amount realized. After this the ledger is fully drained."""
        amt = self.live[winner]
        self.realized[winner] += amt
        self.live[winner] = 0.0
        return amt

    def conserved(self, tol: float = 1e-9) -> bool:
        """The invariant: live + realized always equals the initial bounty pool."""
        return abs(sum(self.live) + sum(self.realized) - self.initial_pool) <= tol


def bounty_cash_on_win(villain_bounty: float, *, progressive_frac: float = 0.5,
                       growth_value: float = 0.5) -> float:
    """The $-value to hero of eliminating a villain with bounty ``villain_bounty``: the immediate
    cash ``(1-progressive_frac)*vb`` plus a discounted value of the half that grows hero's OWN bounty
    (``growth_value`` in [0,1] discounts that future realization for hero's bust risk; default 0.5 --
    conservative). ``growth_value=0`` counts only the certain cash."""
    cash = (1.0 - progressive_frac) * villain_bounty
    growth = progressive_frac * villain_bounty * growth_value
    return cash + growth


def pko_call_dollar_ev(stacks: Sequence[float], payouts: Sequence[float], hero: int, villain: int,
                       hero_equity: float, *, villain_bounty: float = 0.0,
                       progressive_frac: float = 0.5, growth_value: float = 0.5) -> float:
    """$-EV (prize + bounty) of hero CALLING villain's all-in for the smaller stack, given hero's
    showdown equity ``hero_equity`` (P(win), chops folded in). Prize side is ICM over the win/lose
    post-stacks; the bounty side adds the bounty cash collected on a win. Reduces to the pure-ICM
    call value when ``villain_bounty == 0``."""
    s = [float(x) for x in stacks]
    h, v = s[hero], s[villain]
    risked = min(h, v)
    win = list(s); win[hero] = h + risked; win[villain] = v - risked
    lose = list(s); lose[hero] = h - risked; lose[villain] = v + risked
    icm_win = float(icm_equity(win, payouts)[hero])
    icm_lose = float(icm_equity(lose, payouts)[hero])
    bcash = bounty_cash_on_win(villain_bounty, progressive_frac=progressive_frac, growth_value=growth_value)
    p = float(hero_equity)
    return p * (icm_win + bcash) + (1.0 - p) * icm_lose


def min_equity_to_call(stacks: Sequence[float], payouts: Sequence[float], hero: int, villain: int,
                       *, villain_bounty: float = 0.0, progressive_frac: float = 0.5,
                       growth_value: float = 0.5) -> float:
    """The indifference showdown equity p* at which calling villain's all-in is break-even vs folding
    (fold = hero keeps its current ICM equity). Closed form: the call value is linear in p, so
    ``p* = (fold - icm_lose) / (icm_win + bounty_cash - icm_lose)``. A positive ``villain_bounty``
    enlarges the denominator -> LOWERS p* -> hero calls a WIDER range (the chip-EV != $-EV effect).
    Clamped to [0,1]; returns the equity threshold to call."""
    s = [float(x) for x in stacks]
    h, v = s[hero], s[villain]
    risked = min(h, v)
    win = list(s); win[hero] = h + risked; win[villain] = v - risked
    lose = list(s); lose[hero] = h - risked; lose[villain] = v + risked
    fold = float(icm_equity(s, payouts)[hero])
    icm_win = float(icm_equity(win, payouts)[hero])
    icm_lose = float(icm_equity(lose, payouts)[hero])
    bcash = bounty_cash_on_win(villain_bounty, progressive_frac=progressive_frac, growth_value=growth_value)
    denom = icm_win + bcash - icm_lose
    if denom <= 1e-12:
        return 1.0  # winning gains ~no equity (locked / covered) -> never +EV to call on equity alone
    p_star = (fold - icm_lose) / denom
    return min(1.0, max(0.0, p_star))
