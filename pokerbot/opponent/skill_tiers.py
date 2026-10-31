"""Synthetic NLHE opponent baselines for offline evaluation.

Models vary strategic mistakes such as overfolding, calling too often and poor sizing.
Tier names are configuration labels, not independently established strength claims.
"""
from __future__ import annotations
from dataclasses import dataclass, fields, replace
import numpy as np
from pokerbot.engine.actions import NUM_ACTIONS
FOLD = 0
CHECK = 1
CALL = 2
BET_LO = 3
ALLIN = NUM_ACTIONS - 1
AGGR = tuple(range(BET_LO, NUM_ACTIONS))
BETS = tuple(range(BET_LO, NUM_ACTIONS - 1))
_REP_BET = 5
TIER_ORDER = ('beginner', 'intermediate', 'advanced', 'expert', 'semi_pro', 'professional', 'superhuman')
_TIERS_JSON = 'data/player_profiles/nlhe_skill_tiers.json'

@dataclass(frozen=True)
class NLHELeakProfile:
    """Structural deviations for synthetic opponents in offline evaluation."""
    overfold: float = 0.0
    station: float = 0.0
    passivity: float = 0.0
    underbluff: float = 0.0
    loose_preflop: float = 0.0
    sizing_blind: float = 0.0
    scale: float = 1.0

    def scaled(self) -> 'NLHELeakProfile':
        s = float(self.scale)
        return NLHELeakProfile(overfold=min(0.98, self.overfold * s), station=min(0.98, self.station * s), passivity=min(0.98, self.passivity * s), underbluff=min(0.98, self.underbluff * s), loose_preflop=min(0.98, self.loose_preflop * s), sizing_blind=min(1.0, self.sizing_blind * s), scale=1.0)

    def is_flawless(self) -> bool:
        """Whether the model applies no structural deviations from its base policy."""
        return self == NLHELeakProfile(scale=self.scale)
DEFAULT_PROFILES: dict[str, NLHELeakProfile] = {'beginner': NLHELeakProfile(station=0.55, loose_preflop=0.45, passivity=0.45, underbluff=0.6, sizing_blind=1.0, scale=1.0), 'intermediate': NLHELeakProfile(overfold=0.3, passivity=0.25, underbluff=0.3, loose_preflop=0.1, sizing_blind=0.6, scale=1.0), 'advanced': NLHELeakProfile(overfold=0.12, passivity=0.12, underbluff=0.12, sizing_blind=0.25, scale=1.0), 'expert': NLHELeakProfile(overfold=0.05, passivity=0.05, underbluff=0.05, scale=1.0), 'semi_pro': NLHELeakProfile(passivity=0.02, underbluff=0.02, scale=1.0), 'professional': NLHELeakProfile(), 'superhuman': NLHELeakProfile()}

def load_calibrated_profiles() -> dict[str, NLHELeakProfile]:
    """``DEFAULT_PROFILES`` with calibrated per-tier ``scale`` from ``nlhe_skill_tiers.json`` if present
    (mirrors AoF's ``practice_tiers.json``). Missing/unreadable file -> hand-set defaults."""
    import json
    from pathlib import Path
    profiles = dict(DEFAULT_PROFILES)
    path = Path(_TIERS_JSON)
    if not path.is_file():
        return profiles
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return profiles
    for tier, rec in (data.get('tiers') or {}).items():
        if tier in profiles and isinstance(rec, dict) and ('scale' in rec):
            try:
                profiles[tier] = replace(profiles[tier], scale=float(rec['scale']))
            except (TypeError, ValueError):
                continue
    return profiles
ADAPT: dict[str, tuple[float, float]] = {'advanced': (0.3, 0.5), 'expert': (0.5, 0.4), 'semi_pro': (0.7, 0.32), 'professional': (0.88, 0.27), 'superhuman': (1.0, 0.25)}

def is_adaptive(tier: str) -> bool:
    """True iff this tier exploits profiled opponents (advanced and up)."""
    return tier in ADAPT

def resolve(*, tier: str | None=None, level: float | None=None):
    """Resolve a difficulty selection to ``(profile, exploit_strength, argmax_only)``. ``level``
    (continuous, beginner..professional) beats ``tier`` (default ``intermediate``). For the automatic
    mode the caller owns the ``AutoDifficulty`` and passes ``level=auto.level`` here."""
    if level is not None:
        lvl = float(level)
        return (profile_for_level(lvl), exploit_strength_for_level(lvl), False)
    t = tier or 'intermediate'
    if t not in TIER_ORDER:
        raise ValueError(f'unknown tier {t!r} (one of {list(TIER_ORDER)})')
    return (tier_profile(t), ADAPT.get(t, (0.0, 0.0))[0], t == 'superhuman')

def tier_profile(tier: str, profiles: dict[str, NLHELeakProfile] | None=None) -> NLHELeakProfile:
    if tier not in TIER_ORDER:
        raise ValueError(f'unknown tier {tier!r} (one of {list(TIER_ORDER)})')
    return (profiles or load_calibrated_profiles())[tier]

def _lerp_profile(a: NLHELeakProfile, b: NLHELeakProfile, t: float) -> NLHELeakProfile:
    t = min(1.0, max(0.0, float(t)))
    a, b = (a.scaled(), b.scaled())

    def m(x, y):
        return (1.0 - t) * x + t * y
    return NLHELeakProfile(overfold=m(a.overfold, b.overfold), station=m(a.station, b.station), passivity=m(a.passivity, b.passivity), underbluff=m(a.underbluff, b.underbluff), loose_preflop=m(a.loose_preflop, b.loose_preflop), sizing_blind=m(a.sizing_blind, b.sizing_blind), scale=1.0)

def profile_for_level(level: float, profiles: dict[str, NLHELeakProfile] | None=None) -> NLHELeakProfile:
    """Map a continuous skill ``level`` in [0,1] (0=beginner .. 1=professional) to a LeakProfile by
    interpolating between adjacent calibrated tiers. ``superhuman`` is excluded from the ladder."""
    profiles = profiles or load_calibrated_profiles()
    ladder = [t for t in TIER_ORDER if t != 'superhuman']
    x = min(1.0, max(0.0, float(level))) * (len(ladder) - 1)
    lo = int(np.floor(x))
    hi = min(lo + 1, len(ladder) - 1)
    return _lerp_profile(profiles[ladder[lo]], profiles[ladder[hi]], x - lo)

def exploit_strength_for_level(level: float) -> float:
    """Continuous exploitation magnitude for a ``level`` in [0,1] — interpolates the per-tier
    ADAPT exploit_strength along the beginner..professional ladder (beginner/intermediate=0)."""
    ladder = [t for t in TIER_ORDER if t != 'superhuman']
    strengths = [ADAPT.get(t, (0.0, 0.0))[0] for t in ladder]
    x = min(1.0, max(0.0, float(level))) * (len(ladder) - 1)
    lo = int(np.floor(x))
    hi = min(lo + 1, len(ladder) - 1)
    return float((1.0 - (x - lo)) * strengths[lo] + (x - lo) * strengths[hi])

def _legal(lm: np.ndarray, i: int) -> bool:
    return bool(lm[i])

def _collapse_sizing(p: np.ndarray, lm: np.ndarray, strength: float) -> np.ndarray:
    """Shift ``strength`` of the legal sized-bet mass onto a single representative bucket (no sizing
    skill). all-in is left untouched. Total aggression frequency is preserved."""
    bets = [i for i in BETS if lm[i]]
    if len(bets) <= 1:
        return p
    total = float(sum((p[i] for i in bets)))
    if total <= 0:
        return p
    rep = min(bets, key=lambda i: abs(i - _REP_BET))
    for i in bets:
        p[i] *= 1.0 - strength
    p[rep] += total * strength
    return p

def apply_leaks(probs, legal_mask, *, profile: NLHELeakProfile, facing_bet: bool, hand_strength: float | None=None, street: str='preflop') -> np.ndarray:
    """Degrade a legal-masked 11-action policy by a tier's ``profile`` and return a renormalized
    distribution over legal actions. Pure function (no I/O). ``facing_bet`` = the hero faces a bet
    (cannot check); ``hand_strength`` is the hero's equity proxy in [0,1] (None -> 0.5)."""
    p = np.asarray(probs, dtype=np.float64).copy()
    lm = np.asarray(legal_mask, dtype=bool)
    p = p * lm
    s = p.sum()
    if s <= 0:
        u = lm.astype(np.float64)
        return u / max(u.sum(), 1e-12)
    p /= s
    lp = profile.scaled()
    if lp.is_flawless():
        return p
    hs = 0.5 if hand_strength is None else float(np.clip(hand_strength, 0.0, 1.0))
    aggr = [i for i in AGGR if lm[i]]
    if lp.sizing_blind > 0:
        p = _collapse_sizing(p, lm, lp.sizing_blind)
    if facing_bet:
        if lp.overfold > 0 and lm[FOLD]:
            shrink = lp.overfold * (1.0 - hs)
            moved = float(p[CALL] + sum((p[i] for i in aggr))) * shrink
            p[CALL] *= 1.0 - shrink
            for i in aggr:
                p[i] *= 1.0 - shrink
            p[FOLD] += moved
        if lp.station > 0 and lm[CALL]:
            mf = lp.station * (1.0 - hs) * p[FOLD]
            p[FOLD] -= mf
            p[CALL] += mf
            for i in aggr:
                m = lp.station * 0.5 * p[i]
                p[i] -= m
                p[CALL] += m
    elif lp.loose_preflop > 0 and street == 'preflop' and lm[FOLD]:
        move = lp.loose_preflop * p[FOLD] * (0.4 + 0.6 * hs)
        p[FOLD] -= move
        rem = move
        if lm[CALL]:
            p[CALL] += move * 0.8
            rem = move * 0.2
        sb = next((i for i in BETS if lm[i]), None)
        if sb is not None:
            p[sb] += rem
        elif lm[CALL]:
            p[CALL] += rem
        else:
            p[FOLD] += rem
    if lp.passivity > 0 and aggr:
        sink = CHECK if lm[CHECK] else CALL if lm[CALL] else None
        if sink is not None:
            for i in aggr:
                m = lp.passivity * p[i]
                p[i] -= m
                p[sink] += m
    if lp.underbluff > 0 and aggr and (hs < 0.5):
        sink = CHECK if lm[CHECK] else CALL if lm[CALL] else None
        if sink is not None:
            cut = lp.underbluff * (0.5 - hs) * 2.0
            for i in aggr:
                m = cut * p[i]
                p[i] -= m
                p[sink] += m
    p = np.clip(p, 0.0, None) * lm
    p /= max(p.sum(), 1e-12)
    return p

def sample_action(probs, legal_mask, seed: int) -> int:
    """Deterministic seeded draw from a legal-masked distribution (CRN-stable across runs). Robust to
    degenerate input (tiny/zero/non-finite legal mass) — falls back to uniform-over-legal and always
    yields a normalized ``p`` so numpy's strict ``choice`` tolerance never trips."""
    lm = np.asarray(legal_mask, dtype=bool)
    p = np.asarray(probs, dtype=np.float64) * lm
    s = p.sum()
    if not np.isfinite(s) or s <= 1e-12:
        p = lm.astype(np.float64)
        if p.sum() <= 0:
            p = np.ones(len(p), dtype=np.float64)
        s = p.sum()
    p = p / s
    p = p / p.sum()
    rng = np.random.default_rng(int(seed) & 4294967295)
    return int(rng.choice(len(p), p=p))

@dataclass
class AutoDifficulty:
    """Closed-loop difficulty: drive the BOT's bb/100 toward a SIGNED target.

    ``target_bot_bb_per_100`` > 0 -> bot wins by X; < 0 -> human wins by |X|. Holds a continuous
    ``level`` in [0,1] (0=beginner..1=professional) interpolating leak strength. ``update`` is fed ONE
    hand's bot pnl (bb) and smooths it with an EMA (``half_life`` hands) into a bb/100 estimate before
    proportional control — so per-hand variance (±100bb swings) doesn't whipsaw the level. Mirrors
    AoF's AutoDifficulty (which is fed a windowed rate); here the windowing is built in.
    """
    target_bot_bb_per_100: float = 0.0
    level: float = 0.6
    gain: float = 0.0008
    level_min: float = 0.0
    level_max: float = 1.0
    half_life: float = 200.0
    _ema_pnl_bb: float = 0.0
    _n: int = 0

    def current_profile(self, profiles: dict[str, NLHELeakProfile] | None=None) -> NLHELeakProfile:
        return profile_for_level(self.level, profiles)

    def update(self, hand_pnl_bb: float) -> float:
        """Feed one hand's BOT pnl (bb); returns the new level. The pnl is EMA-smoothed to a bb/100
        estimate (mean per-hand pnl x 100), then ``level`` moves to close the gap to the target — bot
        running below target (too weak) raises level, above target lowers it."""
        a = 1.0 - 2.0 ** (-1.0 / max(self.half_life, 1e-09))
        self._n += 1
        self._ema_pnl_bb = float(hand_pnl_bb) if self._n == 1 else (1.0 - a) * self._ema_pnl_bb + a * float(hand_pnl_bb)
        bb_per_100 = self._ema_pnl_bb * 100.0
        err = float(self.target_bot_bb_per_100) - bb_per_100
        self.level = float(min(self.level_max, max(self.level_min, self.level + self.gain * err)))
        return self.level

    @classmethod
    def from_dict(cls, d: dict) -> 'AutoDifficulty':
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
