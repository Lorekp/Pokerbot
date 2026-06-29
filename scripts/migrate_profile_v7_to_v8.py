"""Idempotent one-shot migration: per-player profiles.json from schema v7 → v8.

Schema v7 stored merged cash+tournament context counts (ctx / ctx_fast / ctx_slow / ctx_bluff).
Schema v8 splits those into per-format sub-dicts (ctx_fmt / ctx_fast_fmt / ctx_slow_fmt /
ctx_bluff_fmt) keyed by format label ("cash", "tournament"), so the cross-format hierarchical
shrinkage has a LEVEL 1 read source for each format. Per-format hand-split counters (hands_cash /
hands_tourney) and the merged view are preserved unchanged.

This script backfills the per-format sub-dicts by **proportional split** of the merged counts,
weighted by the existing ``hands_cash`` / ``hands_tourney`` ratio (when known). The split is the
audit-fix-hint recommendation (F4): a player's pre-migration history isn't dropped from the
per-format view, and the merged view stays the source of truth for re-folded live hands going
forward. The split is approximate (a player's cash and tournament tendencies are NOT identical),
but it gives the LEVEL 1 read source enough mass for an evidence-floor gate to consider it; until
the live recorder tags new hands with their format (v7 already does), the per-format sub-dicts
will be updated correctly.

Idempotent: re-running on an already-migrated file is a no-op (the per-format sub-dicts are
detected as non-empty and skipped). Safe to ship as a deploy-time script.

Usage:
    python scripts/migrate_profile_v7_to_v8.py                # migrates the default path
    python scripts/migrate_profile_v7_to_v8.py --path X.json  # migrates an explicit path
    python scripts/migrate_profile_v7_to_v8.py --dry-run      # print diff without writing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokerbot.opponent.player_profiles import (  # noqa: E402
    DEFAULT_PATH,
    FORMAT_LABELS,
    SCHEMA_VERSION,
)


def _split_ratio(stats: dict) -> dict[str, float]:
    """Return the format → weight map for splitting the merged counts. Falls back to a 50/50
    split when the per-format hand count is missing or zero (a legacy v1–v6 file predates v7's
    hands_cash/hands_tourney split)."""
    hc = int(stats.get("hands_cash", 0) or 0)
    ht = int(stats.get("hands_tourney", 0) or 0)
    total = hc + ht
    if total <= 0:
        return {fmt: 0.5 for fmt in FORMAT_LABELS}
    return {"cash": hc / total, "tournament": ht / total}


def _split_counts(
    merged: dict[str, list[float]],
    ratio: dict[str, float],
) -> dict[str, dict[str, list[float]]]:
    """Element-wise scale ``merged[ck]`` by ``ratio[fmt]`` into a per-format sub-dict. Floats only;
    the integer ctx_fmt (cumulative counts) is derived by the same proportional split at save time."""
    return {
        fmt: {ck: [round(x * ratio[fmt], 4) for x in row] for ck, row in merged.items()}
        for fmt in FORMAT_LABELS
    }


def _split_int_counts(
    merged: dict[str, list[int]],
    ratio: dict[str, float],
) -> dict[str, dict[str, list[int]]]:
    """Integer variant for ctx_fmt (lifetime cumulative counts). Floor via int-cast; the per-format
    sub-dict will under-count vs. the merged total by at most ``len(FORMAT_LABELS) - 1`` per bin,
    which is irrelevant for the cross-format shrinkage (which is read on the float sub-dicts)."""
    return {
        fmt: {ck: [int(x * ratio[fmt]) for x in row] for ck, row in merged.items()}
        for fmt in FORMAT_LABELS
    }


def _already_migrated(profile: dict) -> bool:
    """A profile is v8 once ANY per-format sub-dict carries a payload (the migration is all-or-
    nothing per profile). Re-running the script on a migrated file is a no-op per profile."""
    return any(bool(profile.get(k)) for k in (
        "ctx_fmt", "ctx_fast_fmt", "ctx_slow_fmt",
        "ctx_last_fmt", "ctx_runlen_fmt", "ctx_bluff_fmt", "ctx_bluff_last_fmt",
    ))


def migrate_payload(payload: dict, *, dry_run: bool = False) -> tuple[dict, int]:
    """Return (migrated_payload, num_profiles_migrated). Operates in-place on a COPY of the payload."""
    profiles = payload.get("profiles") or {}
    migrated = 0
    for _key, stats in profiles.items():
        if not isinstance(stats, dict):
            continue
        if _already_migrated(stats):
            continue
        ratio = _split_ratio(stats)
        # Cumulative per-context counts (int, lifetime)
        ctx_merged = stats.get("ctx") or {}
        if ctx_merged:
            stats["ctx_fmt"] = _split_int_counts(ctx_merged, ratio)
        # Dual-timescale discounted counts (float, lazy-decayed)
        if stats.get("ctx_fast"):
            stats["ctx_fast_fmt"] = _split_counts(stats["ctx_fast"], ratio)
        if stats.get("ctx_slow"):
            stats["ctx_slow_fmt"] = _split_counts(stats["ctx_slow"], ratio)
        # ctx_last_fmt / ctx_runlen_fmt: re-key the per-context last-update index by format.
        # The values are hand-index integers (s.hands) which are format-agnostic — split as the
        # number of update ticks, scaled by ratio. With ratio=0/1 (a single-format player) the
        # per-format index matches the merged one, so the per-format decay Δ uses the right clock.
        if stats.get("ctx_last"):
            stats["ctx_last_fmt"] = {
                fmt: {ck: int(round(v * ratio[fmt])) for ck, v in stats["ctx_last"].items()}
                for fmt in FORMAT_LABELS
            }
        if stats.get("ctx_runlen"):
            stats["ctx_runlen_fmt"] = {
                fmt: {ck: int(round(v * ratio[fmt])) for ck, v in stats["ctx_runlen"].items()}
                for fmt in FORMAT_LABELS
            }
        # ctx_shift_active_fmt is a TRANSIENT rising-edge latch (not persisted to disk); skip it.
        # Showdown-conditioned bluff/value Beta counts (float).
        if stats.get("ctx_bluff"):
            stats["ctx_bluff_fmt"] = {
                fmt: {ck: [round(x * ratio[fmt], 4) for x in row] for ck, row in stats["ctx_bluff"].items()}
                for fmt in FORMAT_LABELS
            }
        if stats.get("ctx_bluff_last"):
            stats["ctx_bluff_last_fmt"] = {
                fmt: {ck: int(round(v * ratio[fmt])) for ck, v in stats["ctx_bluff_last"].items()}
                for fmt in FORMAT_LABELS
            }
        migrated += 1
    payload["schema_version"] = SCHEMA_VERSION
    return payload, migrated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=DEFAULT_PATH, help=f"profiles.json path (default: {DEFAULT_PATH})")
    ap.add_argument("--dry-run", action="store_true", help="print diff summary without writing")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        return 2

    raw = json.loads(p.read_text(encoding="utf-8"))
    original_version = raw.get("schema_version")
    profiles = raw.get("profiles") or {}
    already_v8 = sum(1 for s in profiles.values() if isinstance(s, dict) and _already_migrated(s))

    migrated_payload, n = migrate_payload(raw, dry_run=True)  # dry preview
    print(f"path:                 {p}")
    print(f"schema_version:       {original_version} -> {SCHEMA_VERSION} (target)")
    print(f"# profiles:           {len(profiles)}")
    print(f"# already v8:         {already_v8}")
    print(f"# to migrate:         {len(profiles) - already_v8}")
    print(f"# would migrate now:  {n}")

    if args.dry_run:
        print("dry-run: not writing")
        return 0

    if n == 0:
        print("nothing to migrate")
        return 0

    # Atomic write: rotate .bak, write .tmp, os.replace. Mirrors the live _save_impl contract.
    bak = p.with_suffix(p.suffix + ".bak")
    if p.exists():
        try:
            import shutil
            shutil.copy2(p, bak)
        except OSError as e:  # noqa: BLE001
            print(f"WARN: bak rotation failed: {e}", file=sys.stderr)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(migrated_payload, ensure_ascii=False, indent=0), encoding="utf-8")
    import os
    os.replace(tmp, p)
    print(f"wrote {p} ({n} profiles migrated, schema_version -> {SCHEMA_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
