"""Persistent per-player opponent profiles, keyed by userId, for cross-session adaptation.

Bridges recorded hands (``pokerbot/live/hand_recorder``) to the in-memory exploit model
(``OpponentTracker``/``OpponentStats`` in ``pokerbot/opponent/stats``): each completed hand
is folded into per-player aggregate stats (VPIP, PFR, 3-bet, aggression, fold-to-cbet, …),
persisted to a single atomic JSON keyed by userId so the live AI can load a known player's
tendencies before the first hand of a new session.

Storage is a flat JSON dict ``{user_id: stats}`` — fine for thousands of players; if it
grows past ~10k, migrate the same accessors to sqlite (one row per player). The per-hand
derivation lives here so both live recording and offline batch backfill use identical math.

Play-money / own-account evaluation use only.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from pokerbot.opponent.context import (
    NUM_BINS,
    action_type_bin,
    context_key as _context_key,
)
from pokerbot.opponent.population_prior import context_prior as _context_prior
from pokerbot.opponent.stats import OpponentStats

DEFAULT_PATH = "data/player_profiles/profiles.json"
# v1 = scalar-only; v2 adds per-context Dirichlet counts; v3 adds dual-timescale discounted counts
# (ctx_fast/ctx_slow/ctx_last) for non-stationarity; v4 adds the BOCPD run-length (ctx_runlen);
# v5 adds the showdown-conditioned bluff-ratio Beta counts (ctx_bluff/ctx_bluff_last); v6 adds the
# server HUD-stat prior (hud); v7 adds the per-format hand split (hands_cash/hands_tourney) so cash and
# MTT hands share one unified profile yet stay segmentable. All reads are back-compatible (older files
# load with newer fields empty/zero).
SCHEMA_VERSION = 7
_STAT_FIELDS = (
    "hands", "vpip", "pfr", "three_bet", "fold_to_cbet", "cbet_faced",
    "aggression_bets_raises", "aggression_calls", "wtsd", "saw_flop",
    "hands_cash", "hands_tourney",
)

_BLIND_TYPES = frozenset({"sb", "bb", "post", "ante", "straddle"})
_AGGRESSIVE_TYPES = frozenset({"bet", "raise", "all_in"})
_DECISION_TYPES = frozenset({"fold", "check", "call", "bet", "raise", "all_in"})
_STREET_BOARD_LEN = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def _derive_contexts(actions: list, board: list | None) -> dict[str, dict[str, list[int]]]:
    """Per-player per-context outcome-bin increments {player_key: {context_key: [k0..k3]}} from one
    hand. Walks the GLOBAL ordered action sequence tracking, per street, the number of prior
    aggressive actions (the facing bucket) and a running pot (for the small/large bet split). Uses
    the SAME ``pokerbot.opponent.context`` spec as the live serve side (train/serve-skew firewall)."""
    out: dict[str, dict[str, list[int]]] = {}
    cur_street = None
    aggr_count = 0
    pot = 0.0
    for a in actions or []:
        st = a.get("street")
        if st != cur_street:
            cur_street, aggr_count = st, 0
        atype = (a.get("type") or "").lower()
        amount = float(a.get("amount") or 0.0)
        if atype in _BLIND_TYPES:
            pot += amount
            continue
        if atype in _DECISION_TYPES:
            key = _player_key(a.get("user_id"), a.get("name"))
            if key is not None:
                board_slice = list((board or [])[: _STREET_BOARD_LEN.get(st, 0)])
                ck = _context_key(st, board_slice, aggr_count)
                is_large = atype == "all_in" or (
                    atype in ("bet", "raise") and pot > 0.0 and amount >= 0.75 * pot
                )
                b = action_type_bin(atype, amount_is_large=is_large)
                if b is not None:
                    row = out.setdefault(key, {}).setdefault(ck, [0] * NUM_BINS)
                    row[b] += 1
        if atype in _AGGRESSIVE_TYPES:
            aggr_count += 1
        pot += amount
    return out


def _static_hand_strength(hole, board_slice) -> float:
    """Hand strength for REVEALED cards on a street-truncated board — the record-replay mirror of
    ``pokerbot.agent._hero_hand_strength`` (weva preflop proxy; the rank heuristic postflop), so live
    bluff classification matches the battery's sim-side classification bit-for-bit."""
    if not board_slice:
        from pokerbot.encoding.weva_encoder import _preflop_strength

        return float(_preflop_strength(list(hole)))
    r1, r2 = int(hole[0].rank), int(hole[1].rank)
    hi, lo = max(r1, r2), min(r1, r2)
    suited = int(hole[0].suit) == int(hole[1].suit)
    pair = r1 == r2
    base = 0.55 + (hi - 2) / 30.0 if pair else (hi + lo) / 30.0 + (0.05 if suited else 0.0)
    board_ranks = [int(c.rank) for c in board_slice]
    if pair and hi in board_ranks:
        base += 0.15
    if hi == max(board_ranks, default=0):
        base += 0.08
    return float(min(0.95, max(0.05, base)))


def _derive_bluffs(actions: list, board: list | None, shown: list | None) -> dict[str, dict[str, list[float]]]:
    """Showdown-conditioned bluff/value increments {player_key: {context_key: [n_bluff, n_value]}}.

    Only players in the ``shown`` list (hand_collector v2 reveals) contribute: their aggressive
    actions are re-scored with the now-known holes on the street-truncated board — hand strength
    < 0.45 ⇒ bluff, > 0.60 ⇒ value, middle skipped (thin-value ambiguity stays out). Walks the
    global sequence with the SAME street/facing/pot tracking as ``_derive_contexts`` so the keys
    land in the identical (locked) context keyspace."""
    if not shown:
        return {}
    from pokerbot.engine.cards import Card

    holes: dict[str, list] = {}
    for srec in shown:
        key = _player_key(srec.get("user_id"), srec.get("name"))
        cards = [c for c in (srec.get("cards") or []) if c]
        if key is None or len(cards) < 2:
            continue
        try:
            holes[key] = [Card.from_str(cards[0]), Card.from_str(cards[1])]
        except Exception:  # noqa: BLE001
            continue
    if not holes:
        return {}
    board_cards: list = []
    for code in board or []:
        try:
            board_cards.append(Card.from_str(code))
        except Exception:  # noqa: BLE001
            board_cards.append(None)
    out: dict[str, dict[str, list[float]]] = {}
    cur_street = None
    aggr_count = 0
    for a in actions or []:
        st = a.get("street")
        if st != cur_street:
            cur_street, aggr_count = st, 0
        atype = (a.get("type") or "").lower()
        if atype in _AGGRESSIVE_TYPES:
            key = _player_key(a.get("user_id"), a.get("name"))
            hole = holes.get(key)
            if hole is not None:
                n_board = _STREET_BOARD_LEN.get(st, 0)
                bslice = [c for c in board_cards[:n_board] if c is not None]
                if len(bslice) == n_board:  # skip if any board card failed to parse
                    hs = _static_hand_strength(hole, bslice)
                    ck = _context_key(st, [str(c) for c in bslice], aggr_count)
                    if hs < 0.45:
                        out.setdefault(key, {}).setdefault(ck, [0.0, 0.0])[0] += 1.0
                    elif hs > 0.60:
                        out.setdefault(key, {}).setdefault(ck, [0.0, 0.0])[1] += 1.0
            aggr_count += 1
    return out


def _aggressive_idxs(actions: list[dict]) -> set[int]:
    """Indices of actions that are GENUINE aggression (a bet, or a raise/all-in that INCREASES the
    wager). An all-in is aggressive only when it exceeds the current street bet — a short-stack all-in
    CALL just matches the bet and must NOT inflate PFR / 3-bet / aggression-factor for a passive player.
    Blinds/antes/calls advance the running street bet but are not aggression."""
    out: set[int] = set()
    street_max: dict[str, float] = {}
    for i, a in enumerate(actions):
        t, s = a.get("type"), a.get("street")
        amt = float(a.get("amount") or 0.0)
        prev = street_max.get(s, 0.0)
        if t in ("bet", "raise"):
            out.add(i)
            street_max[s] = max(prev, amt)
        elif t == "all_in":
            if amt > prev + 1e-9:      # all-in RAISE (exceeds the standing bet) -> aggression
                out.add(i)
                street_max[s] = amt
            # else: all-in CALL -> matches the bet, not aggression
        elif t in ("call", "sb", "bb", "ante", "post"):
            street_max[s] = max(prev, amt)
    return out


def derive_hand_stats(record: dict) -> dict[str, dict]:
    """Per-player per-hand stat increments from a recorded hand (hand_recorder schema).

    Returns ``{player_key: {field: increment}}`` matching OpponentStats fields. player_key
    is the str(user_id) when present, else ``name:<username>`` so anonymous tables still
    aggregate per seat-name. Hero is included too (harmless; the mixer only reads opponent-seat keys).
    """
    actions = record.get("actions") or []
    aggr = _aggressive_idxs(actions)     # genuine-aggression action indices (all-in CALL excluded)
    # group actions (with their global index) by player key, preserving order
    by_player: dict[str, list[tuple[int, dict]]] = {}
    keys_seen: dict[str, str] = {}  # player_key -> display name
    for i, a in enumerate(actions):
        key = _player_key(a.get("user_id"), a.get("name"))
        if key is None:
            continue
        by_player.setdefault(key, []).append((i, a))
        keys_seen[key] = a.get("name") or keys_seen.get(key)

    # 3-bet per player: their first preflop AGGRESSION happened after >=1 prior (re)raise.
    three_bettors: set[str] = set()
    pre_raises_so_far = 0
    for i, a in enumerate(actions):
        if a.get("street") != "preflop":
            continue
        key = _player_key(a.get("user_id"), a.get("name"))
        if i in aggr:                    # genuine raise (not an all-in call)
            if pre_raises_so_far >= 1 and key is not None:
                three_bettors.add(key)
            pre_raises_so_far += 1

    flop_seen = len(record.get("board") or []) >= 3
    # First flop AGGRESSION = the c-bet, for fold-to-cbet attribution (who bet, who folded to it).
    flop_cbet = next(((i, actions[i]) for i in sorted(aggr) if actions[i].get("street") == "flop"), None)
    cbet_idx = flop_cbet[0] if flop_cbet else None
    cbet_key = _player_key(flop_cbet[1].get("user_id"), flop_cbet[1].get("name")) if flop_cbet else None
    # Per-format hand split (v7). Legacy records (no table.format) predate the tag and were cash/AoF;
    # AoF routes to a separate store, so an untagged NLHE record is treated as cash.
    is_tourney = (record.get("table") or {}).get("format") == "tournament"
    contexts = _derive_contexts(actions, record.get("board"))
    bluffs = _derive_bluffs(actions, record.get("board"), record.get("shown"))
    out: dict[str, dict] = {}
    for key, iacts in by_player.items():
        acts = [a for _, a in iacts]
        idxs = {i for i, _ in iacts}
        pre = [a for _, a in iacts if a.get("street") == "preflop"]
        postflop = [a for _, a in iacts if a.get("street") in ("flop", "turn", "river")]
        atypes = [a.get("type") for a in acts]
        pre_idxs = [i for i, a in iacts if a.get("street") == "preflop"]

        vpip = any(t in ("call", "bet", "raise", "all_in") for t in (a.get("type") for a in pre))
        pfr = any(i in aggr for i in pre_idxs)                  # genuine preflop aggression (not all-in call)
        three_bet = key in three_bettors
        saw_flop = bool(postflop) or (flop_seen and not _folded_preflop(pre))
        # WTSD: of hands where the player saw the flop, did they reach showdown (board complete, not
        # folded)? Keyed on saw_flop (the standard denominator), NOT on the winners-capture succeeding.
        wtsd = saw_flop and len(record.get("board") or []) >= 5 and not _folded(atypes)
        bet_or_raise = sum(1 for i in idxs if i in aggr)        # aggression-factor numerator (true raises)
        calls = sum(1 for t in atypes if t == "call")
        # fold-to-cbet, aggressor-attributed: did this player face the flop c-bet (a flop aggression by
        # SOMEONE ELSE, before their flop action) and fold to it? Excludes the c-bettor and check-throughs.
        cbet_faced = False
        fold_to_cbet = False
        if cbet_idx is not None and cbet_key != key:
            post_cbet = [a for i, a in iacts if a.get("street") == "flop" and i > cbet_idx]
            # ingestion audit #6: a SERVER AUTO (timeout / sit-out / disconnect) fold facing the c-bet is
            # NOT a fold-to-c-bet DECISION — counting it would mis-read a timed-out player as a sticky-
            # avoiding over-folder (and wrongly tighten the toughness read). Drop the whole observation
            # when the first post-c-bet action was an auto-fold; a deliberate fold/call still counts.
            if post_cbet and not (post_cbet[0].get("type") == "fold" and post_cbet[0].get("auto")):
                cbet_faced = True
                fold_to_cbet = any(a.get("type") == "fold" and not a.get("auto") for a in post_cbet)

        out[key] = {
            "hands": 1,
            "hands_cash": int(not is_tourney),
            "hands_tourney": int(is_tourney),
            "vpip": int(vpip),
            "pfr": int(pfr),
            "three_bet": int(three_bet),
            "saw_flop": int(saw_flop),
            "wtsd": int(wtsd),
            "aggression_bets_raises": bet_or_raise,
            "aggression_calls": calls,
            "cbet_faced": int(cbet_faced),
            "fold_to_cbet": int(fold_to_cbet),
            "_name": keys_seen.get(key),
            "_ctx": contexts.get(key, {}),
            "_ctx_bluff": bluffs.get(key, {}),
        }
    return out


def _player_key(user_id, name) -> str | None:
    if user_id:
        return str(user_id)
    if name:
        return f"name:{name}"
    return None


def _folded(atypes: list) -> bool:
    return "fold" in atypes


def _folded_preflop(pre: list[dict]) -> bool:
    return any(a.get("type") == "fold" for a in pre)


SEEN_HANDS_CAP = 50_000  # idempotency ledger cap (lexicographic ≈ chronological for live hand ids)


class PlayerProfileStore:
    def __init__(self, path: str | Path = DEFAULT_PATH, *, wal: bool = True) -> None:
        self.path = Path(path)
        self.profiles: dict[str, OpponentStats] = {}
        self.names: dict[str, str] = {}  # key -> last seen display name
        # O4 name<->uid reconciliation: name:<name> -> canonical str(user_id), learned whenever a seat
        # shows BOTH a uid and a name in one hand. A later hand that carries only the name (the feed
        # sometimes drops the uid) then folds into the SAME profile instead of a split-brain `name:` twin.
        self.alias: dict[str, str] = {}
        # Idempotency ledger as an INSERTION-ORDERED set (dict keys): the cap prune keeps the most
        # RECENTLY folded-in ids, not the lexicographically-largest (which, with per-room-namespaced
        # `room:hid` keys, would drop a whole room's ids regardless of recency and re-count them).
        self.seen_hands: dict[str, None] = {}
        # Reentrant lock: in live MULTITABLING each table runs on its own Frida message thread and they
        # all share ONE store. Without this, a get()/update_from_hand() mutating self.profiles while
        # save() iterates it raises "dict changed size during iteration" / writes a torn snapshot. RLock
        # (not Lock) because the mutators nest — update_from_hand -> get/seed_from_hud, save_throttled ->
        # save. Single-thread callers pay an uncontended lock (negligible).
        self._lock = threading.RLock()
        self._dirty = False
        # G6 crash journal: every update_from_hand APPENDS the raw record to <path>.wal.jsonl
        # (cheap, immediate); load() replays anything newer than the snapshot (idempotent by
        # hand_id); save() truncates it. A kill between saves loses nothing.
        self._wal_enabled = wal
        self._wal_replaying = False
        # WAL appends use their OWN lock (not the main RLock) so a disk append/flush never blocks a
        # decision-thread read of self.profiles under self._lock.
        self._wal_lock = threading.Lock()
        self._updates_since_save = 0
        self._last_save_ts = 0.0
        self.load()

    @property
    def wal_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".wal.jsonl")

    def load(self) -> None:
        self._load_snapshot()
        self._replay_wal()

    def _replay_wal(self) -> None:
        if not (self._wal_enabled and self.wal_path.exists()):
            return
        self._wal_replaying = True
        try:
            for line in self.wal_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self.update_from_hand(json.loads(line))  # idempotent; no re-append (flag)
                except (ValueError, TypeError):
                    continue  # torn tail line from a mid-write kill — ignore
        except OSError:
            pass
        finally:
            self._wal_replaying = False

    def _load_snapshot(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Corrupt snapshot (kill mid-replace is prevented by os.replace, but disk faults
            # happen): fall back to the .bak rotation if present.
            bak = self.path.with_suffix(self.path.suffix + ".bak")
            try:
                raw = json.loads(bak.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
        for key, d in (raw.get("profiles") or {}).items():
            s = OpponentStats()
            for f in _STAT_FIELDS:
                setattr(s, f, int(d.get(f, 0)))
            # Per-context counts (absent in v1 files ⇒ empty). Coerce to K-length int lists.
            for ck, row in (d.get("ctx") or {}).items():
                s.ctx_counts[ck] = [int(x) for x in (list(row) + [0] * NUM_BINS)[:NUM_BINS]]
            # Dual-timescale discounted counts (absent in v1/v2 ⇒ empty ⇒ bayes uses static ctx).
            for ck, row in (d.get("ctx_fast") or {}).items():
                s.ctx_fast[ck] = [float(x) for x in (list(row) + [0.0] * NUM_BINS)[:NUM_BINS]]
            for ck, row in (d.get("ctx_slow") or {}).items():
                s.ctx_slow[ck] = [float(x) for x in (list(row) + [0.0] * NUM_BINS)[:NUM_BINS]]
            s.ctx_last = {ck: int(v) for ck, v in (d.get("ctx_last") or {}).items()}
            # R5 BOCPD run-length (absent in v1–v3 ⇒ empty ⇒ bayes uses the binary fast/slow switch).
            s.ctx_runlen = {ck: int(v) for ck, v in (d.get("ctx_runlen") or {}).items()}
            # v5 showdown bluff-ratio Beta counts (absent in v1–v4 ⇒ empty ⇒ bluff_read None).
            for ck, row in (d.get("ctx_bluff") or {}).items():
                s.ctx_bluff[ck] = [float(x) for x in (list(row) + [0.0, 0.0])[:2]]
            s.ctx_bluff_last = {ck: int(v) for ck, v in (d.get("ctx_bluff_last") or {}).items()}
            # v6 server HUD-stat prior (absent in v1–v5 ⇒ empty ⇒ no HUD prior).
            s.hud = dict(d.get("hud") or {})
            self.profiles[key] = s
            if d.get("name"):
                self.names[key] = d["name"]
        self.seen_hands = dict.fromkeys(raw.get("seen_hands") or [])  # insertion-ordered set
        self.alias = dict(raw.get("alias") or {})  # name:<name> -> canonical uid key (O4; absent => empty)

    def save(self) -> None:
        with self._lock:
            self._save_impl()

    def _save_impl(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.seen_hands) > SEEN_HANDS_CAP:  # cap the ledger; keep the most-recently-added ids
            self.seen_hands = dict.fromkeys(list(self.seen_hands)[-SEEN_HANDS_CAP:])
        payload = {
            "schema_version": SCHEMA_VERSION,
            "profiles": {
                key: {
                    **{f: getattr(s, f) for f in _STAT_FIELDS},
                    "name": self.names.get(key),
                    # Only emit non-empty maps to keep legacy/cold profiles compact.
                    **({"ctx": s.ctx_counts} if s.ctx_counts else {}),
                    **({"ctx_fast": {k: [round(x, 4) for x in v] for k, v in s.ctx_fast.items()}}
                       if s.ctx_fast else {}),
                    **({"ctx_slow": {k: [round(x, 4) for x in v] for k, v in s.ctx_slow.items()}}
                       if s.ctx_slow else {}),
                    **({"ctx_last": s.ctx_last} if s.ctx_last else {}),
                    **({"ctx_runlen": s.ctx_runlen} if s.ctx_runlen else {}),
                    **({"ctx_bluff": {k: [round(x, 4) for x in v] for k, v in s.ctx_bluff.items()}}
                       if s.ctx_bluff else {}),
                    **({"ctx_bluff_last": s.ctx_bluff_last} if s.ctx_bluff_last else {}),
                    **({"hud": s.hud} if s.hud else {}),
                }
                for key, s in self.profiles.items()
            },
            # idempotency ledger in INSERTION order (recency) so the cap prune across restarts is
            # chronological, not lexicographic; grows with hands -> migrate to sqlite at large scale
            "seen_hands": list(self.seen_hands),
            # O4 name->uid alias map so reconciliation survives a restart (a later name-only hand keeps
            # resolving to the uid profile); only emitted when non-empty to keep cold files compact.
            **({"alias": self.alias} if self.alias else {}),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8")
        if self.path.exists():  # rotate a recovery copy before the swap (disk-fault insurance)
            try:
                import shutil

                shutil.copy2(self.path, self.path.with_suffix(self.path.suffix + ".bak"))
            except OSError:
                pass
        os.replace(tmp, self.path)  # atomic
        if self._wal_enabled:  # everything in the WAL is now in the snapshot
            try:
                self.wal_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._dirty = False
        self._updates_since_save = 0
        import time

        self._last_save_ts = time.monotonic()

    def save_throttled(self, *, min_hands: int = 20, min_seconds: float = 60.0) -> bool:
        """Snapshot only every ``min_hands`` updates or ``min_seconds`` — the per-hand cost drops
        to one WAL append (the journal makes the throttle lossless). Returns True if it saved."""
        import time

        with self._lock:
            if not self._dirty:
                return False
            if (self._updates_since_save >= min_hands
                    or time.monotonic() - self._last_save_ts >= min_seconds):
                self.save()
                return True
            return False

    def get(self, key: str) -> OpponentStats:
        with self._lock:
            if key not in self.profiles:
                self.profiles[key] = OpponentStats()
            return self.profiles[key]

    def snapshot(self, key: str) -> OpponentStats:
        """A DEEP copy of a player's stats taken UNDER the lock, for cross-thread readers. The live mixer
        reads a villain's profile on a decision thread while another table's recorder thread may be
        mutating that same object's ctx_*/hud dicts in place; a shallow ``copy.copy(get(...))`` outside the
        lock both races the mutation (can raise 'dict changed size') and aliases the still-mutating dicts.
        ``deepcopy`` under the lock is the safe, self-contained snapshot."""
        import copy as _copy
        with self._lock:
            if key not in self.profiles:
                self.profiles[key] = OpponentStats()
            return _copy.deepcopy(self.profiles[key])

    def _canonical_key(self, key: str) -> str:
        """Resolve a name-only key to its canonical uid key when an alias is known (O4)."""
        return self.alias.get(key, key) if key.startswith("name:") else key

    def _learn_aliases(self, record: dict) -> None:
        """O4: remember name:<name> -> str(user_id) for every seat that shows BOTH this hand, and fold
        any name-only profile that accreted BEFORE the uid was first seen into the uid profile once."""
        for a in record.get("actions") or []:
            uid, nm = a.get("user_id"), a.get("name")
            if not (uid and nm):
                continue
            ukey, nkey = str(uid), f"name:{nm}"
            if self.alias.get(nkey) == ukey:
                continue
            self.alias[nkey] = ukey
            self.names.setdefault(ukey, nm)
            orphan = self.profiles.pop(nkey, None)  # pre-uid name-only history -> merge into the uid
            if orphan is not None:
                self._merge_stats(self.get(ukey), orphan)
                self.names.pop(nkey, None)

    @staticmethod
    def _merge_stats(dst: OpponentStats, src: OpponentStats) -> None:
        """Fold one profile's aggregates into another (O4 orphan reconciliation). Cumulative counts add;
        discounted/last-hand markers take the max (they self-heal over subsequent hands); the HUD prior
        fills only if the destination has none. A rare, one-shot event, so approximate discounting is fine."""
        for f in _STAT_FIELDS:
            setattr(dst, f, int(getattr(dst, f, 0)) + int(getattr(src, f, 0)))
        for ck, row in src.ctx_counts.items():
            dst.add_ctx(ck, row)
        for attr in ("ctx_fast", "ctx_slow", "ctx_bluff"):
            d, sm = getattr(dst, attr), getattr(src, attr)
            for ck, row in sm.items():
                cur = d.get(ck)
                if cur is None:
                    d[ck] = list(row)
                else:
                    for i in range(min(len(cur), len(row))):
                        cur[i] += row[i]
        for attr in ("ctx_last", "ctx_runlen", "ctx_bluff_last"):
            d, sm = getattr(dst, attr), getattr(src, attr)
            for ck, v in sm.items():
                d[ck] = max(d.get(ck, 0), v)
        if not dst.hud and src.hud:
            dst.hud = dict(src.hud)

    def confidence(self, key: str) -> float:
        s = self.get(key)
        base = min(1.0, s.hands / 50.0)
        # A server-HUD station (whole-history stats) raises confidence even with few locally observed
        # hands. FIX B2: the old FLAT 0.6 floor reached full HUD trust on hand ONE, so a single play-money
        # HUD lit the whole exploit. Ramp it: 0.6 is only reached after ~5 corroborating LOCAL hands
        # (0.1 + 0.1*hands), so a 1-hand HUD read sits at ~0.2, not 0.6. (Matches mixer._stats_confidence.)
        if getattr(s, "hud", None) and len(s.hud) >= 3:
            return max(base, min(0.6, 0.1 + 0.1 * s.hands))
        return base

    def update_from_hand(self, record: dict) -> int:
        # Crash-journal the raw record FIRST, under a DEDICATED lock (not the main RLock) so the disk
        # append/flush never stalls a decision-thread read of self.profiles. A duplicate that slips
        # through is harmless — WAL replay dedups by hand_id. Skipped during replay (reading, not extending).
        if self._wal_enabled and not self._wal_replaying:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps(record, ensure_ascii=False) + "\n"
                with self._wal_lock, open(self.wal_path, "a", encoding="utf-8") as wf:
                    wf.write(line)
            except (OSError, TypeError):
                pass
        with self._lock:
            return self._update_from_hand_impl(record)

    def _update_from_hand_impl(self, record: dict) -> int:
        """Fold one recorded hand into the per-player aggregates. Idempotent by hand_id —
        re-folding the same hand (restart, log reprocessing) is a no-op. Returns players updated."""
        hid = record.get("hand_id")
        if hid:
            # Per-table dedup key: in live multitabling two tables can share a server hand id, so
            # namespace by the stable room key when the record carries one (live recorder stamps it) so
            # table B's hand isn't dropped as a dup of table A's. Records with NO room_key (legacy /
            # pre-upgrade WAL) keep the raw id -> idempotent across the upgrade and on WAL replay (the
            # room_key is persisted in the record, so a replayed hand recomputes the same key).
            room_key = (record.get("table") or {}).get("room_key")
            dkey = f"{room_key}:{hid}" if room_key else str(hid)
            if dkey in self.seen_hands:
                return 0
            self.seen_hands[dkey] = None     # insertion-ordered set add
        self._learn_aliases(record)  # O4: map name:<name> -> uid (+ merge any pre-uid orphan) first
        increments = derive_hand_stats(record)
        for key, inc in increments.items():
            key = self._canonical_key(key)  # fold a name-only seat into its uid profile when known
            s = self.get(key)
            for f in _STAT_FIELDS:
                setattr(s, f, getattr(s, f) + int(inc.get(f, 0)))
            for ck, bins in (inc.get("_ctx") or {}).items():
                s.add_ctx(ck, bins)  # lifetime cumulative counts (archetype / static posterior)
                # dual-timescale discounted counts (non-stationarity); s.hands is now this hand's index
                s.decay_and_add(ck, bins, _context_prior(ck), s.hands)
            # Showdown-ground-truthed bluff/value evidence (v5). Same-hand second call decays by
            # Δ=0 ⇒ no double discount. Capture-confidence weighting (floor 0.5): the Mono-bridge
            # collector is exact (no field ⇒ 1.0); an OCR-sourced record carrying "confidence"
            # contributes proportionally less ground truth.
            wconf = max(0.5, float(record.get("confidence") or 1.0))
            for ck, (nb, nv) in (inc.get("_ctx_bluff") or {}).items():
                if nb:
                    s.bluff_decay_and_add(ck, True, s.hands, weight=float(nb) * wconf)
                if nv:
                    s.bluff_decay_and_add(ck, False, s.hands, weight=float(nv) * wconf)
            if inc.get("_name"):
                self.names[key] = inc["_name"]
        # Server HUD snapshot (whole-history VPIP/PFR/WTSD/WSD/foldToCbet/...) carried on the record by
        # the live recorder (hand_recorder `hud`, keyed by str(user_id) == our player_key). Fold it into
        # the SAME high-confidence prior the live RPC uses (seed_from_hud) so the value-sizing / exploit
        # path keys on it for EVERY opponent we have history on — not only those seeded live this session.
        # Idempotent (seed_from_hud no-ops when unchanged) ⇒ WAL replay / log reprocessing is safe.
        hud_block = record.get("hud")
        if isinstance(hud_block, dict):
            for uid, hud_stats in hud_block.items():
                if isinstance(hud_stats, dict) and hud_stats:
                    key = str(uid)
                    # _count=False: the per-HAND increment below drives the save throttle; without this
                    # a 6-HUD hand bumped _updates_since_save by 7, snapshotting ~7x too often (inline I/O).
                    self.seed_from_hud(key, hud_stats, name=self.names.get(key), _count=False)
        if increments:
            self._dirty = True
            self._updates_since_save += 1
        return len(increments)

    def seed_from_hud(self, key: str, stats: dict, *, name: str | None = None, _count: bool = True) -> bool:
        """Store the server's HUD stats (VPIP/PFR/WTSD/3bet/cbet/...) for a player as a high-confidence
        prior. Authoritative over the player's whole history — kept verbatim and SEPARATE from the
        count-based fields, so it never distorts per-hand derivation. Idempotent: only marks dirty when
        the payload actually changes. ``_count`` advances the save-throttle counter (default True for the
        standalone live-RPC seed path; the in-hand fold-in passes False so a multi-HUD hand counts once).
        Returns True if stored/updated."""
        if not key or not isinstance(stats, dict) or not stats:
            return False
        with self._lock:
            s = self.get(key)
            changed = s.hud != stats
            if changed:
                s.hud = dict(stats)
            if name:
                self.names[key] = name
            # FIX D.2: the decision/overlay path looks a seat up via _player_key(seat), which returns
            # `name:<name>` whenever the seat exposes NO user_id (the live seats in this session are
            # name-only). HUD is seeded here by user_id (key=str(uid)), and get()/snapshot() do NOT apply
            # _canonical_key, so a name-only lookup hit an EMPTY profile -> station_payoff_score ~0 ->
            # the overlay fired but escalated ~0. Mirror the HUD onto the `name:<name>` profile so the
            # SAME key the decision path uses carries it. Both keys hold the identical dict (no desync);
            # the alias map still folds the name profile into the uid one once a hand shows both. Skip
            # when key already IS the name key (no self-mirror) or no name is known.
            mirrored = False
            if name and not key.startswith("name:"):
                nkey = f"name:{name}"
                ns = self.get(nkey)
                if ns.hud != stats:
                    ns.hud = dict(stats)
                    self.names[nkey] = name
                    mirrored = True
            if not (changed or mirrored):
                return False
            self._dirty = True
            if _count:
                self._updates_since_save += 1
            return True

    @property
    def dirty(self) -> bool:
        return self._dirty
