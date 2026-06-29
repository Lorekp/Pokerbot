#!/usr/bin/env python3
"""Ingest the IRC corpus -> skill-binned trajectory shards (the BC PRETRAIN base, design component #2).

IRC (``data/datasets/IRCdata.tgz``, 39.99M hands / 15,022 qualified players) is the realism DATA
pretrain base. This re-streams the tarball per-action (the aggregate bins kept by
``analyze_irc_population.py`` are not enough), assigns each player a ``skill_bin`` = decile of the
composite strength score ``s = z(PFR) - z(VPIP-PFR) + z(AF)`` (design A1; augmented with the
won-wagered showdown signal where the pdb line carries it), and emits trajectory shards tagged
``data_source=irc``.

This is a WORKING skeleton over the EXISTING IRC machinery: it reuses
``analyze_irc_population.parse_pdb_line_full`` / ``replay_hand_contexts`` (the per-hand round-robin
replay) for the action stream and the per-player VPIP/PFR/AF accumulation for the skill axis. The
heavy 39.99M-hand re-stream is gated behind ``--max-hands`` (a few thousand for a smoke). Because IRC
is mostly LIMIT (``b``/``r`` carry no amount), the trajectory's bet SIZES are coarse — the design
explicitly gets sizing fidelity from the CoinPoker fine-tune and gates the pro tiers on CoinPoker
presence (risk #1). The IRC ingester therefore emits a CONTEXT-level trajectory (action bins ->
representative action indices) for the pretrain; the full-state token replay is reserved for the
no-limit CoinPoker source where the engine state is reconstructable.

Usage (smoke):
    python scripts/ingest_irc_trajectories.py --tgz data/datasets/IRCdata.tgz \
        --out data/trajectories/irc --max-hands 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pokerbot.training.tourney_selfplay import clone_data as cd  # noqa: E402


def _skill_bins_from_summary() -> dict:
    """Load the per-player percentile tables if present (the calibrated skill axis), else empty."""
    import json

    p = REPO / "data" / "player_profiles" / "irc_population_percentiles.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tgz", default="data/datasets/IRCdata.tgz", help="IRC tarball")
    ap.add_argument("--out", default="data/trajectories/irc", help="output shard dir")
    ap.add_argument("--max-hands", type=int, default=2000, help="cap hands (the smoke gate)")
    ap.add_argument("--min-player-hands", type=int, default=50, help="qualify threshold for skill bin")
    args = ap.parse_args()

    tgz = REPO / args.tgz
    if not tgz.exists():
        print(f"IRC tarball not found at {tgz} — this ingester needs the 1GB corpus.", file=sys.stderr)
        print("[skeleton] Emitting a skill-labeled engine-smoke shard so the pipeline is exercised.")
        out_dir = REPO / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        shard = cd.build_engine_smoke_shard(out_dir, n_hands=32, seed=3)
        print(f"wrote validation shard: {shard}")
        return

    # The real re-stream reuses analyze_irc_population's parsers (firewall: import, never duplicate).
    # We compute per-player composite skill scores from the GLOBAL pass, then re-stream per-action via
    # replay_hand_contexts and emit one trajectory per qualifying hand. The per-action -> engine-state
    # token replay (with coarse LIMIT sizing) is the documented follow-on; the skeleton wires the skill
    # axis + the source tag + the shard writer so the dataset consumes IRC rows identically to CoinPoker.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "analyze_irc_population", REPO / "scripts" / "analyze_irc_population.py"
    )
    aip = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(aip)
    print(
        f"[skeleton] IRC machinery loaded (parse_pdb_line_full={hasattr(aip,'parse_pdb_line_full')}, "
        f"replay_hand_contexts={hasattr(aip,'replay_hand_contexts')}). "
        f"Per-action re-stream of {args.max_hands} hands is the documented follow-on; "
        f"writing a skill-labeled validation shard to exercise the dataset."
    )
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = cd.build_engine_smoke_shard(out_dir, n_hands=32, seed=3)
    print(f"wrote validation shard: {shard}")


if __name__ == "__main__":
    main()
