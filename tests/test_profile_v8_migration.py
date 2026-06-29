"""Profile schema v7 → v8 migration: lazy init in PlayerProfileStore, idempotent migration script,
ingest-side per-format fold, and round-trip persistence.

F4 (HIGH #3 from the bug-hunt audit): the live data file is missing the v8 per-format context
sub-dicts (ctx_fmt / ctx_fast_fmt / ctx_slow_fmt / ctx_bluff_fmt / ctx_last_fmt /
ctx_runlen_fmt / ctx_bluff_last_fmt). Without this fix the cross-format hierarchical shrinkage
(planned in bayes_exploit LEVEL 1) silently falls back to the merged view on every read.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

from pokerbot.opponent.player_profiles import (  # noqa: E402
    FORMAT_LABELS,
    SCHEMA_VERSION,
    PlayerProfileStore,
    derive_hand_stats,
)


# ---------- fixtures ----------

def _v7_hand(hand_id="H1", *, fmt=None):
    """A fish-vs-reg hand matching the pre-v8 test fixtures. Optional ``fmt`` stamps the table
    format on the record (None ⇒ legacy pre-tag, defaults to cash in derive_hand_stats)."""
    rec = {
        "hand_id": hand_id,
        "board": ["2c", "7d", "Qh"],
        "winners": [{"seat": 3, "user_id": 333}],
        "actions": [
            {"street": "preflop", "seat": 1, "user_id": 111, "name": "hero", "type": "sb", "amount": 50},
            {"street": "preflop", "seat": 2, "user_id": 222, "name": "villain", "type": "bb", "amount": 100},
            {"street": "preflop", "seat": 4, "user_id": 444, "name": "reg", "type": "raise", "amount": 250},
            {"street": "preflop", "seat": 3, "user_id": 333, "name": "fish", "type": "raise", "amount": 800},
            {"street": "preflop", "seat": 1, "user_id": 111, "name": "hero", "type": "call", "amount": 800},
            {"street": "preflop", "seat": 2, "user_id": 222, "name": "villain", "type": "fold", "amount": 0},
            {"street": "preflop", "seat": 4, "user_id": 444, "name": "reg", "type": "fold", "amount": 0},
            {"street": "flop", "seat": 3, "user_id": 333, "name": "fish", "type": "bet", "amount": 600},
            {"street": "flop", "seat": 1, "user_id": 111, "name": "hero", "type": "fold", "amount": 0},
        ],
    }
    if fmt is not None:
        rec["table"] = {"format": fmt, "room_key": "r1"}
    else:
        rec["table"] = {"room_key": "r1"}  # legacy: no format tag
    return rec


# ---------- load-path: lazy init v8 fields on v7 input ----------

def test_load_v7_file_initializes_v8_subdicts(tmp_path):
    """A v7-shaped file (no ctx_fmt / ctx_fast_fmt / etc.) loads cleanly with the v8 sub-dicts
    lazy-initialized to empty dicts — the store does not raise on the missing fields."""
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps({
        "schema_version": 7,
        "profiles": {
            "333": {
                "hands": 5, "vpip": 3, "pfr": 2, "three_bet": 1,
                "fold_to_cbet": 1, "cbet_faced": 2, "aggression_bets_raises": 4,
                "aggression_calls": 1, "wtsd": 2, "saw_flop": 3,
                "hands_cash": 1, "hands_tourney": 4,
                "name": "fish",
                "ctx": {"preflop|-|facing_bet": [0, 0, 0, 3]},
                "ctx_fast": {"preflop|-|facing_bet": [0.0, 0.0, 0.0, 3.0]},
                "ctx_slow": {"preflop|-|facing_bet": [0.0, 0.0, 0.0, 3.0]},
                "ctx_last": {"preflop|-|facing_bet": 5},
                "ctx_bluff": {"preflop|-|facing_bet": [0.0, 1.0]},
                "ctx_bluff_last": {"preflop|-|facing_bet": 5},
            }
        },
    }), encoding="utf-8")
    store = PlayerProfileStore(p)
    s = store.get("333")
    # merged view survived the round-trip
    assert s.hands == 5 and s.hands_cash == 1 and s.hands_tourney == 4
    assert s.ctx_counts["preflop|-|facing_bet"] == [0, 0, 0, 3]
    # v8 sub-dicts STAY ABSENT (lazy init: a v7 file with no per-format data carries no per-format
    # entries; the save contract below preserves this so we don't bloat the file with empty maps).
    assert s.ctx_fmt == {}
    assert s.ctx_fast_fmt == {}
    assert s.ctx_slow_fmt == {}
    assert s.ctx_last_fmt == {}
    assert s.ctx_runlen_fmt == {}
    assert s.ctx_bluff_fmt == {}
    assert s.ctx_bluff_last_fmt == {}


def test_load_v1_legacy_file_lazy_inits_v8(tmp_path):
    """The oldest legacy file (pre-v7, no per-format split, no ctx) loads cleanly with v8 fields
    absent — back-compat for the entire schema history."""
    p = tmp_path / "profiles.json"
    p.write_text('{"profiles": {"999": {"hands": 5, "vpip": 3, "name": "old"}}, "seen_hands": ["X"]}',
                 encoding="utf-8")
    store = PlayerProfileStore(p)
    s = store.get("999")
    assert s.hands == 5 and s.vpip == 3
    assert s.ctx_fmt == {}


def test_load_v8_file_round_trips_per_format(tmp_path):
    """A v8 file carrying the per-format sub-dicts loads them verbatim so a restart preserves the
    migration result."""
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps({
        "schema_version": 8,
        "profiles": {
            "555": {
                "hands": 10, "vpip": 4, "pfr": 3, "three_bet": 1,
                "fold_to_cbet": 0, "cbet_faced": 1, "aggression_bets_raises": 3,
                "aggression_calls": 1, "wtsd": 1, "saw_flop": 2,
                "hands_cash": 3, "hands_tourney": 7,
                "name": "tourney_reg",
                "ctx": {"preflop|-|facing_bet": [0, 0, 0, 4]},
                "ctx_fast": {"preflop|-|facing_bet": [0.0, 0.0, 0.0, 4.0]},
                "ctx_fmt": {
                    "cash": {"preflop|-|facing_bet": [0, 0, 0, 1]},
                    "tournament": {"preflop|-|facing_bet": [0, 0, 0, 3]},
                },
                "ctx_fast_fmt": {
                    "cash": {"preflop|-|facing_bet": [0.0, 0.0, 0.0, 1.0]},
                    "tournament": {"preflop|-|facing_bet": [0.0, 0.0, 0.0, 3.0]},
                },
                "ctx_bluff_fmt": {
                    "cash": {"preflop|-|facing_bet": [0.0, 0.5]},
                    "tournament": {"preflop|-|facing_bet": [0.0, 1.5]},
                },
            }
        },
    }), encoding="utf-8")
    store = PlayerProfileStore(p)
    s = store.get("555")
    # per-format sub-dicts survived the load
    assert s.ctx_fmt["cash"]["preflop|-|facing_bet"] == [0, 0, 0, 1]
    assert s.ctx_fmt["tournament"]["preflop|-|facing_bet"] == [0, 0, 0, 3]
    assert s.ctx_fast_fmt["cash"]["preflop|-|facing_bet"] == [0.0, 0.0, 0.0, 1.0]
    assert s.ctx_bluff_fmt["tournament"]["preflop|-|facing_bet"] == [0.0, 1.5]


# ---------- ingest path: re-fold records into per-format sub-dicts ----------

def test_ingest_cash_record_populates_cash_subdict(tmp_path):
    """A legacy (no table.format) record is treated as cash and feeds the 'cash' per-format
    sub-dict. The 'tournament' sub-dict stays absent until a tagged record arrives (lazy-init
    contract: only the touched format gets an entry)."""
    from pokerbot.opponent.population_prior import context_prior
    from pokerbot.opponent.context import context_key as _ck_fn
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)
    assert store.update_from_hand(_v7_hand("H1")) > 0
    s = store.get("333")
    # merged view populated
    ck = "preflop|-|facing_bet"
    assert s.ctx_counts[ck] == [0, 0, 0, 1]
    # 'cash' per-format sub-dict populated; 'tournament' not yet touched
    assert s.ctx_fmt["cash"][ck] == [0, 0, 0, 1]
    assert "tournament" not in s.ctx_fmt
    # dual-timescale sub-dicts: the first-hand per-format row matches the merged row exactly
    # (same clock, same prior, same bins — the per-format sub-dict is a parallel write).
    assert s.ctx_fast_fmt["cash"][ck] == s.ctx_fast[ck]
    assert s.ctx_slow_fmt["cash"][ck] == s.ctx_slow[ck]
    assert s.ctx_last_fmt["cash"][ck] == s.ctx_last[ck] == 1


def test_ingest_tourney_record_populates_tournament_subdict(tmp_path):
    """A tournament-tagged record feeds the 'tournament' per-format sub-dict; the 'cash' sub-dict
    stays absent until a cash record arrives."""
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)
    assert store.update_from_hand(_v7_hand("T1", fmt="tournament")) > 0
    s = store.get("333")
    assert s.ctx_fmt["tournament"]["preflop|-|facing_bet"] == [0, 0, 0, 1]
    assert "cash" not in s.ctx_fmt
    assert s.hands_tourney == 1 and s.hands_cash == 0


def test_ingest_mixed_records_segregate_per_format(tmp_path):
    """A mix of cash + tournament hands accumulates per-format independently — the merged view
    sums, the per-format sub-dicts each see only their own hand."""
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)
    store.update_from_hand(_v7_hand("C1", fmt="cash"))
    store.update_from_hand(_v7_hand("C2", fmt="cash"))
    store.update_from_hand(_v7_hand("T1", fmt="tournament"))
    s = store.get("333")
    # merged view: 3 hands, all 3 folded a preflop-3-bet
    assert s.hands == 3
    assert s.ctx_counts["preflop|-|facing_bet"] == [0, 0, 0, 3]
    # per-format split: 2 cash + 1 tourney
    assert s.ctx_fmt["cash"]["preflop|-|facing_bet"] == [0, 0, 0, 2]
    assert s.ctx_fmt["tournament"]["preflop|-|facing_bet"] == [0, 0, 0, 1]


def test_ingest_bluff_evidence_populates_per_format_bluff(tmp_path):
    """A showdown-ground-truthed bluff reveal on a tournament hand feeds the tournament
    per-format bluff sub-dict; the cash bluff sub-dict stays absent until a cash reveal arrives."""
    rec = _v7_hand("T1", fmt="tournament")
    # fish has hole cards that hand-strength < 0.45 ⇒ classified as a bluff
    rec["shown"] = [{"user_id": 333, "name": "fish", "cards": ["2c", "3d"]}]
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)
    store.update_from_hand(rec)
    s = store.get("333")
    assert sum(s.ctx_bluff_fmt["tournament"].get(k, [0.0])[0] for k in s.ctx_bluff_fmt["tournament"]) > 0
    assert "cash" not in s.ctx_bluff_fmt


# ---------- migration script ----------

def test_migrate_v7_payload_idempotent_and_proportional():
    """The migration script splits merged counts proportionally by hands_cash/hands_tourney and
    re-running it is a no-op (already_v8 short-circuits)."""
    from scripts.migrate_profile_v7_to_v8 import migrate_payload
    payload = {
        "schema_version": 7,
        "profiles": {
            "333": {
                "hands": 10, "hands_cash": 2, "hands_tourney": 8,
                "ctx": {"preflop|-|facing_bet": [0, 0, 0, 10]},
                "ctx_fast": {"preflop|-|facing_bet": [0.0, 0.0, 0.0, 10.0]},
                "ctx_slow": {"preflop|-|facing_bet": [0.0, 0.0, 0.0, 10.0]},
                "ctx_last": {"preflop|-|facing_bet": 10},
                "ctx_runlen": {"preflop|-|facing_bet": 5},
                "ctx_bluff": {"preflop|-|facing_bet": [0.0, 5.0]},
                "ctx_bluff_last": {"preflop|-|facing_bet": 10},
            }
        },
    }
    migrated, n = migrate_payload(payload)
    assert n == 1
    p = migrated["profiles"]["333"]
    # ratio = 2/10 cash, 8/10 tournament
    assert p["ctx_fmt"]["cash"]["preflop|-|facing_bet"] == [0, 0, 0, 2]
    assert p["ctx_fmt"]["tournament"]["preflop|-|facing_bet"] == [0, 0, 0, 8]
    assert p["ctx_fast_fmt"]["cash"]["preflop|-|facing_bet"] == [0.0, 0.0, 0.0, 2.0]
    assert p["ctx_slow_fmt"]["tournament"]["preflop|-|facing_bet"] == [0.0, 0.0, 0.0, 8.0]
    assert p["ctx_last_fmt"]["cash"]["preflop|-|facing_bet"] == 2
    assert p["ctx_last_fmt"]["tournament"]["preflop|-|facing_bet"] == 8
    assert p["ctx_bluff_fmt"]["cash"]["preflop|-|facing_bet"] == [0.0, 1.0]
    assert p["ctx_bluff_fmt"]["tournament"]["preflop|-|facing_bet"] == [0.0, 4.0]
    assert p["ctx_bluff_last_fmt"]["tournament"]["preflop|-|facing_bet"] == 8
    # schema_version bumped to SCHEMA_VERSION
    assert migrated["schema_version"] == SCHEMA_VERSION

    # idempotent: re-run on the migrated payload is a no-op
    _, n2 = migrate_payload(migrated)
    assert n2 == 0


def test_migrate_v7_no_format_split_falls_back_to_50_50():
    """A v1–v6 file with NO hands_cash/hands_tourney falls back to a 50/50 split (the migration is
    honest about not knowing the format mix rather than assuming cash)."""
    from scripts.migrate_profile_v7_to_v8 import migrate_payload
    payload = {
        "schema_version": 5,
        "profiles": {
            "999": {
                "hands": 4,
                "ctx": {"preflop|-|facing_bet": [0, 0, 0, 4]},
                "ctx_fast": {"preflop|-|facing_bet": [0.0, 0.0, 0.0, 4.0]},
            }
        },
    }
    migrated, n = migrate_payload(payload)
    assert n == 1
    p = migrated["profiles"]["999"]
    assert p["ctx_fmt"]["cash"]["preflop|-|facing_bet"] == [0, 0, 0, 2]
    assert p["ctx_fmt"]["tournament"]["preflop|-|facing_bet"] == [0, 0, 0, 2]


def test_migrate_v7_single_format_player_assigns_full_weight():
    """A player who has only ever played tournament hands (hands_cash=0, hands_tourney=N) gets
    the FULL merged count into the tournament sub-dict (cash stays empty)."""
    from scripts.migrate_profile_v7_to_v8 import migrate_payload
    payload = {
        "schema_version": 7,
        "profiles": {
            "777": {
                "hands": 5, "hands_cash": 0, "hands_tourney": 5,
                "ctx_fast": {"preflop|-|facing_bet": [0.0, 0.0, 0.0, 5.0]},
            }
        },
    }
    migrated, n = migrate_payload(payload)
    p = migrated["profiles"]["777"]
    assert p["ctx_fast_fmt"]["tournament"]["preflop|-|facing_bet"] == [0.0, 0.0, 0.0, 5.0]
    # cash sub-dict may carry all-zero rows from the proportional split; what matters is the
    # signal lives in the tournament sub-dict.
    cash_row = p["ctx_fast_fmt"].get("cash", {}).get("preflop|-|facing_bet", [0, 0, 0, 0])
    assert sum(cash_row) == 0.0


# ---------- end-to-end: load v7, ingest a new tagged hand, save, reload ----------

def test_load_v7_then_ingest_then_round_trip(tmp_path):
    """End-to-end: a v7 file loads with lazy v8 init, an ingest of a tagged tournament record
    populates the tournament per-format sub-dict, save+reload preserves everything."""
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps({
        "schema_version": 7,
        "profiles": {
            "333": {
                "hands": 5, "hands_cash": 5, "hands_tourney": 0,
                "ctx": {"preflop|-|facing_bet": [0, 0, 0, 5]},
                "ctx_fast": {"preflop|-|facing_bet": [0.0, 0.0, 0.0, 5.0]},
            }
        },
    }), encoding="utf-8")

    store = PlayerProfileStore(p)
    # ingest a NEW tournament record (different hand_id)
    store.update_from_hand(_v7_hand("T-new", fmt="tournament"))
    store.save()

    # reload: merged view + per-format sub-dicts survived
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    p333 = raw["profiles"]["333"]
    # merged view: 5 cash hands + 1 tourney hand = 6 hands, ctx incremented by the new hand
    assert p333["hands"] == 6
    assert p333["hands_cash"] == 5
    assert p333["hands_tourney"] == 1
    # v8 per-format sub-dicts emitted (only the touched 'tournament' key is present)
    assert "ctx_fmt" in p333
    assert p333["ctx_fmt"]["tournament"]["preflop|-|facing_bet"] == [0, 0, 0, 1]
    # cash sub-dict absent — no cash hand ingested, no migration applied
    assert "cash" not in p333["ctx_fmt"]
    assert "ctx_fast_fmt" in p333
    assert "ctx_slow_fmt" in p333
    assert "ctx_last_fmt" in p333

    store2 = PlayerProfileStore(p)
    s = store2.get("333")
    assert s.ctx_fmt["tournament"]["preflop|-|facing_bet"] == [0, 0, 0, 1]
    # cash sub-dict absent — no cash hand ingested, no migration applied
    assert "cash" not in s.ctx_fmt


def test_migration_then_ingest_populates_both_formats(tmp_path):
    """End-to-end: load a v7 file, run the migration script in-process, then ingest a tagged
    hand. Both the migrated cash sub-dict (from proportional split) AND the new tournament
    sub-dict (from the live ingest) survive the round-trip."""
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps({
        "schema_version": 7,
        "profiles": {
            "333": {
                "hands": 4, "hands_cash": 1, "hands_tourney": 3,
                "ctx": {"preflop|-|facing_bet": [0, 0, 0, 4]},
                "ctx_fast": {"preflop|-|facing_bet": [0.0, 0.0, 0.0, 4.0]},
            }
        },
    }), encoding="utf-8")

    from scripts.migrate_profile_v7_to_v8 import migrate_payload
    raw = json.loads(p.read_text(encoding="utf-8"))
    migrated, _ = migrate_payload(raw)
    p.write_text(json.dumps(migrated), encoding="utf-8")

    store = PlayerProfileStore(p)
    store.update_from_hand(_v7_hand("T-new", fmt="tournament"))
    store.save()

    p333 = json.loads(p.read_text(encoding="utf-8"))["profiles"]["333"]
    # migration: 1/4 to cash (round-down ⇒ 1), 3/4 to tournament (round-down ⇒ 3); the new
    # hand then ADDS to tournament.
    assert p333["ctx_fmt"]["cash"]["preflop|-|facing_bet"] == [0, 0, 0, 1]
    assert p333["ctx_fmt"]["tournament"]["preflop|-|facing_bet"] == [0, 0, 0, 4]  # 3 + 1

    store2 = PlayerProfileStore(p)
    s = store2.get("333")
    assert s.ctx_fmt["cash"]["preflop|-|facing_bet"] == [0, 0, 0, 1]
    assert s.ctx_fmt["tournament"]["preflop|-|facing_bet"] == [0, 0, 0, 4]


def test_v8_save_omits_empty_per_format_subdicts(tmp_path):
    """A profile with NO per-format data emits no per-format keys (compactness contract)."""
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)
    store.update_from_hand(_v7_hand("H1"))  # legacy = cash, fills 'cash' sub-dict
    store.save()
    raw = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    p333 = raw["profiles"]["333"]
    # 'cash' is non-empty so ctx_fmt is emitted; 'tournament' sub-dict is empty so it is
    # skipped (the outer dict has the cash key only because tournament is empty)
    assert "ctx_fmt" in p333
    assert "tournament" not in p333["ctx_fmt"]
    assert p333["ctx_fmt"]["cash"]["preflop|-|facing_bet"] == [0, 0, 0, 1]
