"""NLHE WEVA encode injectivity check (Gate 0)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.data.sft_dataloader import SolverAnnotatedParquetDataset
from pokerbot.engine.jax_engine import (
    init_hand,
    is_chance_node,
    legal_action_mask,
    sample_chance,
    step_state,
)
from pokerbot.engine.jax_weva_encode import NLHEEncodeBundle, encode_nlhe_weva_bundle
from pokerbot.engine.jax_weva_fast import encode_nlhe_bundle_fast, default_weva_bucket_table
from pokerbot.engine.nlhe_bridge import game_state_from_jax
from pokerbot.engine.cards import parse_holding
from pokerbot.encoding.csv_encoder import POSITION_MAP
from pokerbot.encoding.weva_encoder import WEVAContext, WEVAMapper


def _walk_hu_decision_nodes(
    *,
    n_hands: int = 32,
    seed: int = 0,
    occupancy: tuple[bool, ...] = (False, False, False, True, False, True),
    button: int = 3,
) -> list[tuple[Any, int]]:
    """Sample HU decision nodes via random play."""
    occ = jnp.array(occupancy)
    key = jax.random.PRNGKey(seed)
    nodes: list[tuple[Any, int]] = []

    for _ in range(n_hands):
        key, k_init = jax.random.split(key)
        state = init_hand(k_init, button=jnp.int32(button), occupancy=occ)
        for _ in range(200):
            if bool(state.done):
                break
            while bool(is_chance_node(state)):
                state = sample_chance(state)
            if bool(state.done):
                break
            seat = int(state.action_seat)
            if seat >= 0:
                nodes.append((state, seat))
            mask = np.asarray(legal_action_mask(state), dtype=bool)
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            key, sub = jax.random.split(key)
            pick = int(jax.random.choice(sub, legal.shape[0]))
            action = int(legal[pick])
            state = step_state(state, jnp.int32(action))

    return nodes


def check_fast_encode_parity(
    *,
    n_hands: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Fast path fingerprints must match slow bridge path (Gate 0 alignment)."""
    mapper = WEVAMapper.default()
    table = default_weva_bucket_table()
    nodes = _walk_hu_decision_nodes(n_hands=n_hands, seed=seed)
    mismatches = 0
    for state, seat in nodes:
        slow = encode_nlhe_weva_bundle(state, seat, mapper=mapper)
        fast = encode_nlhe_bundle_fast(state, seat, weva_table=table)
        if slow.fingerprint() != fast.fingerprint():
            mismatches += 1
    return {
        "passed": mismatches == 0,
        "n_nodes": len(nodes),
        "mismatches": mismatches,
    }


def check_nlhe_encode_collisions(
    *,
    n_hands: int = 24,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Distinct hero-visible decision points must not map to conflicting encodings.

    Uses fingerprint (weva, cards, actions, policy_position) as network input key.
    """
    table = default_weva_bucket_table()
    nodes = _walk_hu_decision_nodes(n_hands=n_hands, seed=seed)
    fp_to_infos: dict[tuple, list[tuple]] = defaultdict(list)
    inconsistent: list[dict[str, Any]] = []

    for state, seat in nodes:
        bundle = encode_nlhe_bundle_fast(state, seat, weva_table=table)
        fp = bundle.fingerprint()
        gs = game_state_from_jax(state)
        info = (
            seat,
            int(np.asarray(state.street)),
            int(np.asarray(state.hist_len)),
            float(gs.pot_bb),
            float(gs.current_bet_bb),
            bundle.weva_id,
        )
        if fp in fp_to_infos and fp_to_infos[fp][0] != info:
            inconsistent.append({"fingerprint": fp, "a": fp_to_infos[fp][0], "b": info})
        fp_to_infos[fp].append(info)

    collision_groups = {fp: infos for fp, infos in fp_to_infos.items() if len(infos) > 1}

    return {
        "passed": len(inconsistent) == 0,
        "n_nodes": len(nodes),
        "n_unique_fingerprints": len(fp_to_infos),
        "collision_groups": len(collision_groups),
        "inconsistent_keys": len(inconsistent),
        "sample_collision": next(iter(collision_groups.values()), None) if collision_groups else None,
    }


def check_parquet_spot_replay(
    parquet_glob: str | Path,
    *,
    max_rows: int = 50,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Verify Parquet WEVA ids match hero hole re-lookup (postflop solver JSON)."""
    ds = SolverAnnotatedParquetDataset(str(parquet_glob))
    mapper = WEVAMapper.default()
    rows_checked = 0
    ok = 0
    mismatches: list[dict[str, Any]] = []

    for i in range(min(len(ds), max_rows)):
        rec = ds.records[i]
        raw = rec.get("raw_solver_json")
        if not raw:
            continue
        payload = json.loads(raw) if isinstance(raw, str) else raw
        hero = payload.get("hero_hole") or []
        if len(hero) < 2:
            continue
        hole = parse_holding("".join(hero[:2]))
        pos_name = str(payload.get("hero_position", "BB")).strip().upper()
        pos_idx = POSITION_MAP.get(pos_name, 5)
        ctx = WEVAContext(hero_position=pos_idx, num_players=2, action_depth_bin=0)
        recomputed = int(mapper.lookup(hole, position=pos_idx, context=ctx))
        cards = np.asarray(rec["cards"], dtype=np.float32)
        actions = np.asarray(rec["actions"], dtype=np.float32)
        cmp = {
            "weva_ok": int(rec["weva_bucket_id"]) == recomputed,
            "cards_finite": bool(np.isfinite(cards).all()),
            "actions_finite": bool(np.isfinite(actions).all()),
        }
        rows_checked += 1
        if all(cmp.values()):
            ok += 1
        else:
            mismatches.append({"row": i, **cmp, "stored": int(rec["weva_bucket_id"]), "recomputed": recomputed})

    return {
        "passed": rows_checked > 0 and len(mismatches) == 0,
        "rows_checked": rows_checked,
        "reencode_ok": ok,
        "mismatches": mismatches[:10],
    }


def run_gate0_report(
    *,
    heldout_vanilla: str = "data/solver/heldout_prod600_vanilla.parquet",
    heldout_bunching: str = "data/solver/heldout_prod600_bunching.parquet",
    output: str = "logs/nlhe_gate0_encode_report.json",
) -> dict[str, Any]:
    collision = check_nlhe_encode_collisions()
    fast_parity = check_fast_encode_parity()
    vanilla = check_parquet_spot_replay(heldout_vanilla)
    bunching = check_parquet_spot_replay(heldout_bunching)
    report = {
        "gate0_passed": (
            collision["passed"] and fast_parity["passed"] and vanilla["passed"] and bunching["passed"]
        ),
        "collision": collision,
        "fast_path_parity": fast_parity,
        "parquet_vanilla": vanilla,
        "parquet_bunching": bunching,
        "sft_init_note": (
            "SFT weight transfer requires Flax net with WEVA embedding; "
            "parity uses tensor bundle, not legacy encode_info_state."
        ),
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def require_gate0_passed(**kwargs: Any) -> dict[str, Any]:
    report = run_gate0_report(**kwargs)
    if not report["gate0_passed"]:
        raise AssertionError(f"NLHE Gate 0 encoding failed: {report}")
    return report
