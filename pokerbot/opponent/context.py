"""Per-context opponent-stat keying — the SINGLE locked spec shared by ingestion (offline hand
derivation) and serving (live search / mixer). One module so train- and serve-time bucket a decision
*identically* (the train/serve-skew firewall). ``context_spec_hash()`` is asserted in tests so any
silent change to the spec is caught.

A context = ``street × board_texture × facing_action``:
  - street:        preflop / flop / turn / river
  - board_texture: "-" preflop; postflop one of dry / wet / paired
  - facing_action: preflop FULL depth (unopened / facing_bet / facing_raise / facing_3bet, by # prior
                   raises, blinds excluded); postflop BINARY (unopened / facing_bet). The split is the
                   skew firewall — see the facing section for why postflop raise-depth is collapsed.

22 live contexts/player (4 preflop + 3×3×2 postflop), coarse enough that the population-prior shrinkage
keeps each robust, fine enough to separate fold-to-3bet (preflop|-|facing_raise) and 4bet spots
(preflop|-|facing_3bet) from fold-to-cbet (flop|*|facing_bet) and river stations (river|*|facing_bet).

Opponent actions are folded into K=4 OUTCOME BINS for dense counts that map directly onto the
exploit directions the search/mixer already use:
  bin 0 = fold
  bin 1 = passive (check / call)
  bin 2 = aggressive small/medium (bet/raise ≲ pot)
  bin 3 = aggressive large + all-in (≳ pot, overbets, jams)
"""

from __future__ import annotations

import hashlib
from collections import Counter

from pokerbot.engine.actions import NUM_ACTIONS

# ---- action -> outcome bin (the 11-index NLHE action space folded to K=4) ----
# NLHE indices: 0=fold, 1=check, 2=call, 3..NUM_ACTIONS-2 = bet/raise size buckets (ascending),
# NUM_ACTIONS-1 = all-in. Bet buckets are pot-fractions (1/3,1/2,2/3,3/4,1,1.5,2); the small/medium
# vs large split is at "pot" (index where the size reaches ~1x pot).
NUM_BINS = 4
FOLD_BIN, PASSIVE_BIN, AGG_SMALL_BIN, AGG_LARGE_BIN = 0, 1, 2, 3

_FIRST_BET = 3                          # first bet/raise bucket index
_ALL_IN = NUM_ACTIONS - 1              # all-in index
# bet buckets [_FIRST_BET .. _ALL_IN-1]; split the lower ~half (small/med) from the upper (≳pot).
_N_BET_BUCKETS = _ALL_IN - _FIRST_BET   # e.g. 7 for NUM_ACTIONS=11
_LARGE_BET_START = _FIRST_BET + _N_BET_BUCKETS - 4  # top 4 sizes (~pot, 1.5, 2, ...) -> large


def action_bin(action_index: int) -> int:
    """Map an NLHE action index (0..NUM_ACTIONS-1) to one of K=4 outcome bins."""
    a = int(action_index)
    if a <= 0:
        return FOLD_BIN
    if a in (1, 2):  # check / call
        return PASSIVE_BIN
    if a == _ALL_IN:
        return AGG_LARGE_BIN
    # a bet/raise size bucket
    return AGG_LARGE_BIN if a >= _LARGE_BET_START else AGG_SMALL_BIN


# String label of an outcome bin from a recorded action TYPE (hand_recorder schema), used by the
# offline derivation which has action types, not engine indices. Same K=4 partition.
def action_type_bin(action_type: str, *, amount_is_large: bool = False) -> int | None:
    """Bin a recorded action type ('fold'/'check'/'call'/'bet'/'raise'/'all_in'). Returns None for
    types that aren't a voluntary decision outcome (e.g. blinds, which derive_hand_stats filters)."""
    t = (action_type or "").lower()
    if t == "fold":
        return FOLD_BIN
    if t in ("check", "call"):
        return PASSIVE_BIN
    if t == "all_in":
        return AGG_LARGE_BIN
    if t in ("bet", "raise"):
        return AGG_LARGE_BIN if amount_is_large else AGG_SMALL_BIN
    return None


# ---- street ----
STREET_NAMES = ("preflop", "flop", "turn", "river")


def street_name(street: int | str) -> str:
    if isinstance(street, str):
        s = street.lower()
        return s if s in STREET_NAMES else "preflop"
    i = int(street)
    return STREET_NAMES[i] if 0 <= i < len(STREET_NAMES) else "preflop"


# ---- board texture ----
TEXTURES = ("dry", "wet", "paired")
_PREFLOP_TEXTURE = "-"
_RANKVAL = {r: i for i, r in enumerate("23456789TJQKA", start=2)}


def board_texture_rs(ranks: list[int], suits: list[int]) -> str:
    """Texture from rank values (2..14) + suit codes. The shared core both call sites route through."""
    n = len(ranks)
    if n < 3:
        return _PREFLOP_TEXTURE
    if len(set(ranks)) < n:
        return "paired"
    if max(Counter(suits).values()) >= 3:  # flush-heavy
        return "wet"
    rs = sorted(set(ranks))
    gaps = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    # straight-heavy: tight cluster (<=4 span) or >=2 close connections
    tight = (rs[-1] - rs[0]) <= 4
    if tight or sum(1 for g in gaps if g <= 2) >= 2:
        return "wet"
    return "dry"


def board_texture(board: list) -> str:
    """Texture from a list of card STRINGS ('Jh','4c',...) — the hand_recorder/offline format.
    Robust to None / short boards (preflop => '-')."""
    cards = [c for c in (board or []) if c and len(str(c)) >= 2]
    if len(cards) < 3:
        return _PREFLOP_TEXTURE
    ranks, suits = [], []
    for c in cards:
        c = str(c)
        rv = _RANKVAL.get(c[0].upper())
        if rv is None:
            continue
        ranks.append(rv)
        suits.append(c[1].lower())
    return board_texture_rs(ranks, suits)


# ---- facing action ----
# The facing depth = number of aggressive actions (bet/raise/all-in; preflop excludes blinds) live on
# the CURRENT street BEFORE this decision. The train/serve-skew firewall requires every site to bucket
# a decision IDENTICALLY, so the spec is split by what each site can compute WITHOUT replaying the hand:
#   - PREFLOP: full depth (unopened=open / facing_bet=vs open / facing_raise=vs 3bet / facing_3bet=vs
#     4bet+). Reliable everywhere: preflop is the first street, so the entire flat history IS the
#     preflop action sequence — offline (street-tagged list), the live search env (hist_action), the
#     agent and the battery all count the same raises. This is the high-value granularity (3bet/4bet
#     fold and squeeze tendencies).
#   - POSTFLOP: binary (unopened / facing_bet). Raise-depth (check-raise = facing_raise) can't be
#     recovered identically from the live flat env (no per-entry street tag) without re-replaying the
#     betting, so it is deliberately collapsed to keep zero skew. ``facing_token`` enforces the cap
#     CENTRALLY: every site passes its raw per-street aggression count and the postflop cap lives here.
FACINGS = ("unopened", "facing_bet", "facing_raise", "facing_3bet")  # preflop realizes all four
_POSTFLOP_FACINGS = ("unopened", "facing_bet")                        # postflop realizes only these


def facing_bucket(n_prior_aggressions: int) -> str:
    """Full-depth bucket: 0 ⇒ unopened, 1 ⇒ facing_bet, 2 ⇒ facing_raise, ≥3 ⇒ facing_3bet."""
    n = max(0, int(n_prior_aggressions))
    if n == 0:
        return "unopened"
    if n == 1:
        return "facing_bet"
    if n == 2:
        return "facing_raise"
    return "facing_3bet"


def facing_token(street: int | str, n_prior_aggressions: int) -> str:
    """The skew-firewalled facing label. Full depth preflop; binary postflop (cap at 1). The single
    place the postflop cap is applied, so all sites agree as long as they pass the raw count."""
    n = max(0, int(n_prior_aggressions))
    if street_name(street) != "preflop":
        n = min(n, 1)
    return facing_bucket(n)


def is_aggressive_action(action_index: int) -> bool:
    """Shared predicate: does an engine action index COUNT toward facing depth (bet/raise/all-in)?
    Used by the live serve side over ``hist_action`` so its count matches offline ingestion."""
    return action_bin(action_index) in (AGG_SMALL_BIN, AGG_LARGE_BIN)


def is_aggressive_type(action_type: str) -> bool:
    """Shared predicate over a recorded/enum action-type name ('bet'/'raise'/'all_in' ⇒ True;
    blinds/fold/check/call ⇒ False). Offline ingest and the python-engine sites use this."""
    return action_type_bin(action_type) in (AGG_SMALL_BIN, AGG_LARGE_BIN)


# ---- the context key ----
def context_key(street: int | str, board: list, n_prior_aggressions: int) -> str:
    """``"<street>|<texture>|<facing>"`` — the locked per-context key. ``board`` is a list of card
    strings (or empty preflop); texture is '-' until the flop. ``n_prior_aggressions`` is the RAW
    per-street aggression count; the preflop-full/postflop-binary cap is applied in ``facing_token``."""
    return f"{street_name(street)}|{board_texture(board)}|{facing_token(street, n_prior_aggressions)}"


def context_key_rs(street: int | str, ranks: list[int], suits: list[int], n_prior_aggressions: int) -> str:
    """Same key from rank/suit lists — the serve-side entry (engine card ids decoded to rank/suit)."""
    tex = board_texture_rs(ranks, suits) if len(ranks) >= 3 else _PREFLOP_TEXTURE
    return f"{street_name(street)}|{tex}|{facing_token(street, n_prior_aggressions)}"


def all_contexts() -> list[str]:
    """Every REALIZABLE context key: preflop has only the '-' texture but full 4-level facing;
    postflop has 3 textures × binary facing. 4 + 3×3×2 = 22 keys; seeds the population prior."""
    out: list[str] = []
    for st in STREET_NAMES:
        textures = (_PREFLOP_TEXTURE,) if st == "preflop" else TEXTURES
        facings = FACINGS if st == "preflop" else _POSTFLOP_FACINGS
        for tex in textures:
            for fac in facings:
                out.append(f"{st}|{tex}|{fac}")
    return out


def context_spec_hash() -> str:
    """Stable digest of the entire context+bin spec. Asserted in tests so any silent change to the
    keying (which would split a player's history across incompatible keys = train/serve skew) fails
    loudly and forces an intentional migration."""
    spec = {
        "num_actions": NUM_ACTIONS,
        "num_bins": NUM_BINS,
        "action_bins": [action_bin(a) for a in range(NUM_ACTIONS)],
        "streets": STREET_NAMES,
        "textures": (_PREFLOP_TEXTURE, *TEXTURES),
        "facings": FACINGS,
        "postflop_facings": _POSTFLOP_FACINGS,
        "contexts": all_contexts(),
    }
    return hashlib.blake2b(repr(spec).encode("utf-8"), digest_size=8).hexdigest()
