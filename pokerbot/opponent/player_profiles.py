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
# v5 adds the showdown-conditioned bluff-ratio Beta counts (ctx_bluff/ctx_bluff_last). All reads are
# back-compatible (older files load with the newer fields empty).
SCHEMA_VERSION = 5
_STAT_FIELDS = (
    "hands", "vpip", "pfr", "three_bet", "fold_to_cbet", "cbet_faced",
    "aggression_bets_raises", "aggression_calls", "wtsd", "saw_flop",
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


def derive_hand_stats(record: dict) -> dict[str, dict]:
    """Per-player per-hand stat increments from a recorded hand (hand_recorder schema).

    Returns ``{player_key: {field: increment}}`` matching OpponentStats fields. player_key
    is the str(user_id) when present, else ``name:<username>`` so anonymous tables still
    aggregate per seat-name. Hero is included too (harmless; lets us self-track).
    """
    actions = record.get("actions") or []
    # group actions by player key, preserving order
    by_player: dict[str, list[dict]] = {}
    keys_seen: dict[str, str] = {}  # player_key -> display name
    for a in actions:
        key = _player_key(a.get("user_id"), a.get("name"))
        if key is None:
            continue
        by_player.setdefault(key, []).append(a)
        keys_seen[key] = a.get("name") or keys_seen.get(key)

    # 3-bet per player: their first preflop raise happened after >=1 prior (re)raise.
    # Walk the global preflop sequence once, counting voluntary raises before each.
    three_bettors: set[str] = set()
    pre_raises_so_far = 0
    for a in actions:
        if a.get("street") != "preflop":
            continue
        key = _player_key(a.get("user_id"), a.get("name"))
        if a.get("type") in ("raise", "all_in"):
            if pre_raises_so_far >= 1 and key is not None:
                three_bettors.add(key)
            pre_raises_so_far += 1

    flop_seen = len(record.get("board") or []) >= 3
    contexts = _derive_contexts(actions, record.get("board"))
    bluffs = _derive_bluffs(actions, record.get("board"), record.get("shown"))
    out: dict[str, dict] = {}
    for key, acts in by_player.items():
        pre = [a for a in acts if a.get("street") == "preflop"]
        postflop = [a for a in acts if a.get("street") in ("flop", "turn", "river")]
        atypes = [a.get("type") for a in acts]

        vpip = any(t in ("call", "bet", "raise", "all_in") for t in (a.get("type") for a in pre))
        pfr = any(t in ("raise", "bet", "all_in") for t in (a.get("type") for a in pre))
        three_bet = key in three_bettors
        saw_flop = bool(postflop) or (flop_seen and not _folded_preflop(pre))
        wtsd = bool(record.get("winners")) and not _folded(atypes) and len(record.get("board") or []) >= 5
        bet_or_raise = sum(1 for t in atypes if t in ("bet", "raise", "all_in"))
        calls = sum(1 for t in atypes if t == "call")
        # fold-to-cbet: folded on the flop (proxy; true cbet attribution needs aggressor tracking)
        flop_acts = [a for a in acts if a.get("street") == "flop"]
        cbet_faced = bool(flop_acts)
        fold_to_cbet = any(a.get("type") == "fold" for a in flop_acts)

        out[key] = {
            "hands": 1,
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
        self.seen_hands: set[str] = set()  # hand_ids already folded in (idempotency)
        self._dirty = False
        # G6 crash journal: every update_from_hand APPENDS the raw record to <path>.wal.jsonl
        # (cheap, immediate); load() replays anything newer than the snapshot (idempotent by
        # hand_id); save() truncates it. A kill between saves loses nothing.
        self._wal_enabled = wal
        self._wal_replaying = False
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
            self.profiles[key] = s
            if d.get("name"):
                self.names[key] = d["name"]
        self.seen_hands = set(raw.get("seen_hands") or [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.seen_hands) > SEEN_HANDS_CAP:  # cap the ledger; keep the newest ids
            self.seen_hands = set(sorted(self.seen_hands)[-SEEN_HANDS_CAP:])
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
                }
                for key, s in self.profiles.items()
            },
            # idempotency ledger; grows with hands — prune or migrate to sqlite at large scale
            "seen_hands": sorted(self.seen_hands),
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

        if not self._dirty:
            return False
        if (self._updates_since_save >= min_hands
                or time.monotonic() - self._last_save_ts >= min_seconds):
            self.save()
            return True
        return False

    def get(self, key: str) -> OpponentStats:
        if key not in self.profiles:
            self.profiles[key] = OpponentStats()
        return self.profiles[key]

    def confidence(self, key: str) -> float:
        return min(1.0, self.get(key).hands / 50.0)

    def update_from_hand(self, record: dict) -> int:
        """Fold one recorded hand into the per-player aggregates. Idempotent by hand_id —
        re-folding the same hand (restart, log reprocessing) is a no-op. Returns players updated."""
        hid = record.get("hand_id")
        if hid:
            if hid in self.seen_hands:
                return 0
            self.seen_hands.add(hid)
        if self._wal_enabled and not self._wal_replaying:
            try:  # crash journal first — the cheap append makes the throttled snapshot lossless
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.wal_path, "a", encoding="utf-8") as wf:
                    wf.write(json.dumps(record, ensure_ascii=False) + "\n")
            except (OSError, TypeError):
                pass
        increments = derive_hand_stats(record)
        for key, inc in increments.items():
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
        if increments:
            self._dirty = True
            self._updates_since_save += 1
        return len(increments)

    @property
    def dirty(self) -> bool:
        return self._dirty
