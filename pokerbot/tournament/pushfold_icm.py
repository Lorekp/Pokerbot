"""ICM-optimal heads-up push/fold solver (the highest-leverage tournament component).

Short-stack play -- jam-or-fold preflop -- is where most turbo-MTT decision-EV lives and where ICM
bites hardest (bubble, pay jumps). This solves the heads-up SB-jam / BB-call game to a Nash
equilibrium **under ICM**, using the ICM terminals from :mod:`pokerbot.tournament.icm`.

Reuse (read-only) of the AoF stack:
  * ``data/aof/cfr_base/hu_eq_matrix.npz`` -- the 169x169 heads-up equity grid ``eq[i,j]`` (P(hand i
    beats hand j), chop folded into the equity). Same grid the AoF HU base was solved against.
  * 169 bucket labels + combo multiplicities (pair=6, suited=4, offsuit=12) for range weighting.

Key efficiency: the post-showdown stacks depend only on the *stacks* (the at-risk amount), not the
cards, so the all-in ICM value is **affine in eq[i,j]**:
    V_allin(i,j) = eq[i,j] * ICM(win) + (1 - eq[i,j]) * ICM(lose).
That collapses the whole CFR to matrix-vector products over the 169x169 grid -- a full solve is
milliseconds, so live on-demand solving is feasible (no precompute grid required, though one can be
cached later).

Certification: :func:`nash_gap` returns the ICM exploitability (sum of both players' best-response
gains vs the average strategy) -- this is a *real* certificate because the heads-up all-in game is
finite and well-defined under ICM. Converged solve -> gap -> ~0.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

# Resolve the HU equity grid relative to THIS module so cwd-fragility (R4 P1 §4.2) does not
# surface as a confusing "no equity grid" when the process is launched from a different dir.
_HERE = Path(__file__).resolve().parent
_HU_EQ_PATH = _HERE.parent.parent / "data" / "aof" / "cfr_base" / "hu_eq_matrix.npz"


@lru_cache(maxsize=1)
def _hu_grid() -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """(eq[169,169], labels, combo_weights[169]) for the heads-up equity grid."""
    d = np.load(_HU_EQ_PATH, allow_pickle=True)
    eq = np.asarray(d["eq"], dtype=np.float64)
    labels = tuple(str(x) for x in d["labels"])
    w = np.array([_combo_weight(l) for l in labels], dtype=np.float64)
    return eq, labels, w


def _combo_weight(label: str) -> int:
    """Number of real 4-card combos a 169 bucket represents: pair 6, suited 4, offsuit 12."""
    label = label.strip()
    if len(label) == 2:          # 'AA'
        return 6
    return 4 if label.endswith("s") else 12


def _regret_match(reg: np.ndarray) -> np.ndarray:
    """Row-wise regret matching (CFR+): positive-part regrets normalized; uniform if all <= 0."""
    pos = np.clip(reg, 0.0, None)
    s = pos.sum(axis=1, keepdims=True)
    out = np.where(s > 0, pos / np.where(s > 0, s, 1.0), 1.0 / reg.shape[1])
    return out


@dataclass
class HUPushFoldSolution:
    sb_jam: np.ndarray          # P(jam) per 169 bucket (SB / button)
    bb_call: np.ndarray         # P(call a jam) per 169 bucket (BB)
    labels: tuple[str, ...]
    weights: np.ndarray         # combo multiplicities (for range %)
    nash_gap: float             # ICM exploitability ($, prize units) -- ~0 when converged
    sb_jam_pct: float           # weighted % of hands SB jams
    bb_call_pct: float          # weighted % of hands BB calls

    def range_pct(self, freq: np.ndarray) -> float:
        return float((freq * self.weights).sum() / self.weights.sum())

    def _idx(self, hand: str) -> int:
        """169 bucket index for a hand label ('AKs') or a 4-char two-card string ('AhKd')."""
        hand = hand.strip()
        if len(hand) == 4:
            hand = hand_to_label(hand)
        try:
            return self.labels.index(hand)
        except ValueError as exc:                       # pragma: no cover - input guard
            raise KeyError(f"unknown hand label {hand!r}") from exc

    def jam_freq(self, hand: str) -> float:
        """SB equilibrium jam frequency for a hand ('AKs' / 'AhKd')."""
        return float(self.sb_jam[self._idx(hand)])

    def call_freq(self, hand: str) -> float:
        """BB equilibrium call frequency (facing a jam) for a hand."""
        return float(self.bb_call[self._idx(hand)])


_RANK_ORDER = "AKQJT98765432"


def hand_to_label(cards: str) -> str:
    """'AhKd' / 'AhKh' / 'AhAs' -> 169 bucket label ('AKo' / 'AKs' / 'AA')."""
    cards = cards.strip()
    if len(cards) != 4:
        raise ValueError(f"expected a 4-char two-card string, got {cards!r}")
    ra, sa = cards[0].upper(), cards[1].lower()
    rb, sb = cards[2].upper(), cards[3].lower()
    if ra not in _RANK_ORDER or rb not in _RANK_ORDER:
        raise ValueError(f"bad ranks in {cards!r}")
    if _RANK_ORDER.index(ra) > _RANK_ORDER.index(rb):
        ra, rb, sa, sb = rb, ra, sb, sa
    if ra == rb:
        return ra + rb
    return ra + rb + ("s" if sa == sb else "o")


def decide_hu_pushfold(stacks: Sequence[float], payouts: Sequence[float], hero: int, villain: int,
                       hero_hand: str, role: str, *, sb_post: float = 0.5, bb_post: float = 1.0,
                       ante: float = 0.0, dead: float = 0.0, iters: int = 4000) -> dict:
    """Live decision for a heads-up / blind-vs-blind short-stack spot.

    ``role`` is ``"sb"`` (hero is first-in: jam or fold) or ``"bb"`` (hero faces a jam: call or fold).
    Returns ``{"action", "freq", "nash_gap", ...}``. ``hero`` is always the SB seat and ``villain``
    the BB seat of the solve (so the roles match the solved game); pass them accordingly.
    """
    sol = solve_hu_icm_pushfold(stacks, payouts, hero, villain, sb_post=sb_post, bb_post=bb_post,
                                ante=ante, dead=dead, iters=iters)
    if role == "sb":
        freq = sol.jam_freq(hero_hand)
        action = "jam" if freq >= 0.5 else "fold"
    elif role == "bb":
        freq = sol.call_freq(hero_hand)
        action = "call" if freq >= 0.5 else "fold"
    else:
        raise ValueError(f"role must be 'sb' or 'bb', got {role!r}")
    return {"action": action, "freq": freq, "nash_gap": sol.nash_gap,
            "sb_jam_pct": sol.sb_jam_pct, "bb_call_pct": sol.bb_call_pct}


def solve_hu_icm_pushfold(stacks: Sequence[float], payouts: Sequence[float], sb: int, bb: int, *,
                          sb_post: float = 0.5, bb_post: float = 1.0, ante: float = 0.0,
                          dead: float = 0.0, iters: int = 4000) -> HUPushFoldSolution:
    """Solve the heads-up ICM push/fold game (SB jams or folds; BB calls or folds a jam).

    ``stacks`` is the full remaining-field vector (so ICM sees every player); ``sb``/``bb`` index the
    two players in the hand. ``payouts`` is the remaining prize structure (best first). ``dead`` =
    chips already dead in the pot from folded antes/blinds (collected by an uncontested winner).
    Winner-take-all ``payouts=[1, 0]`` over a 2-player ``stacks`` recovers the classic chip-EV Nash
    push/fold (ICM == chip-proportional there), which is how this is validated.
    """
    eq, labels, w = _hu_grid()
    n = len(labels)
    stacks = [float(s) for s in stacks]
    R = min(stacks[sb], stacks[bb])                          # all-in is for the smaller stack

    def _icm_at(d_sb: float, d_bb: float) -> np.ndarray:
        post = list(stacks)
        post[sb] += d_sb
        post[bb] += d_bb
        from pokerbot.tournament.icm import icm_equity
        return icm_equity(post, payouts)

    # Terminal ICM equities (per seat). All independent of the cards.
    sbfold = _icm_at(-(sb_post + ante), +(sb_post + ante))   # SB folds -> BB wins SB's posted
    bbfold = _icm_at(+(bb_post + ante) + dead, -(bb_post + ante))  # SB jam, BB folds -> SB wins BB posted + dead
    sbwin = _icm_at(+R + dead, -R)                           # showdown, SB wins (collects dead)
    sblose = _icm_at(-R, +R + dead)                          # showdown, SB loses

    sbfold_sb = sbfold[sb]
    bbfold_sb, bbfold_bb = bbfold[sb], bbfold[bb]
    win_sb, win_bb = sbwin[sb], sbwin[bb]
    lose_sb, lose_bb = sblose[sb], sblose[bb]

    pj = w / w.sum()                                         # marginal hand distribution (combo-weighted)

    rs = np.zeros((n, 2))                                    # SB regrets [fold, jam]
    rb = np.zeros((n, 2))                                    # BB regrets [fold, call]
    acc_sb = np.zeros(n)                                     # avg strategy accumulators (linear CFR+)
    acc_bb = np.zeros(n)
    acc_bb_w = 0.0

    for t in range(1, iters + 1):
        sig_sb = _regret_match(rs)
        sig_bb = _regret_match(rb)
        jam, call = sig_sb[:, 1], sig_bb[:, 1]

        # ---- SB decision (root) ----
        bbfold_mass = float((pj * (1.0 - call)).sum())       # P(BB folds | jam), marginal
        call_mass = float((pj * call).sum())
        eq_call = eq @ (pj * call)                           # sum_j eq[i,j] * pj * call[j]
        V_jam = bbfold_sb * bbfold_mass + lose_sb * call_mass + (win_sb - lose_sb) * eq_call
        V_fold = np.full(n, sbfold_sb)
        V_sb = jam * V_jam + (1.0 - jam) * V_fold
        rs[:, 0] += V_fold - V_sb
        rs[:, 1] += V_jam - V_sb

        # ---- BB decision (facing a jam) ----
        jam_mass = float((pj * jam).sum())
        eq_jam = (pj * jam) @ eq                             # sum_i pj*jam[i]*eq[i,j]
        if jam_mass > 1e-12:
            V_call = (lose_bb * jam_mass + (win_bb - lose_bb) * eq_jam) / jam_mass
        else:
            V_call = np.full(n, lose_bb)
        V_bbfold = np.full(n, bbfold_bb)
        V_bb = call * V_call + (1.0 - call) * V_bbfold
        rb[:, 0] += V_bbfold - V_bb
        rb[:, 1] += V_call - V_bb

        rs = np.clip(rs, 0.0, None)                          # CFR+
        rb = np.clip(rb, 0.0, None)
        acc_sb += t * jam                                    # linear averaging
        acc_bb += t * jam_mass * call                       # BB weighted by reach (jam_mass)
        acc_bb_w += t * jam_mass

    t_sum = iters * (iters + 1) / 2.0                       # sum of linear weights
    avg_jam = acc_sb / t_sum
    avg_call = acc_bb / acc_bb_w if acc_bb_w > 0 else np.zeros(n)

    gap = _nash_gap(avg_jam, avg_call, eq, pj, sbfold_sb, bbfold_sb, bbfold_bb,
                    win_sb, lose_sb, win_bb, lose_bb)
    sol = HUPushFoldSolution(
        sb_jam=avg_jam, bb_call=avg_call, labels=labels, weights=w, nash_gap=gap,
        sb_jam_pct=float((avg_jam * w).sum() / w.sum()),
        bb_call_pct=float((avg_call * w).sum() / w.sum()),
    )
    return sol


def _nash_gap(jam, call, eq, pj, sbfold_sb, bbfold_sb, bbfold_bb, win_sb, lose_sb, win_bb, lose_bb):
    """ICM exploitability: best-response gain of each player vs the other's average strategy."""
    # SB best response vs avg BB call
    bbfold_mass = float((pj * (1.0 - call)).sum())
    call_mass = float((pj * call).sum())
    eq_call = eq @ (pj * call)
    V_jam = bbfold_sb * bbfold_mass + lose_sb * call_mass + (win_sb - lose_sb) * eq_call
    V_fold = np.full_like(V_jam, sbfold_sb)
    br_sb = float((pj * np.maximum(V_jam, V_fold)).sum())
    cur_sb = float((pj * (jam * V_jam + (1.0 - jam) * V_fold)).sum())

    # BB best response vs avg SB jam
    jam_mass = float((pj * jam).sum())
    eq_jam = (pj * jam) @ eq
    if jam_mass > 1e-12:
        V_call = (lose_bb * jam_mass + (win_bb - lose_bb) * eq_jam) / jam_mass
    else:
        V_call = np.full(eq.shape[0], lose_bb)
    V_bbfold = np.full_like(V_call, bbfold_bb)
    # BB only acts when SB jams (reach = pj*jam mass); weight the gap by that reach
    br_bb = float((pj * np.maximum(V_call, V_bbfold)).sum()) * 1.0
    cur_bb = float((pj * (call * V_call + (1.0 - call) * V_bbfold)).sum())
    return (br_sb - cur_sb) + (br_bb - cur_bb)
