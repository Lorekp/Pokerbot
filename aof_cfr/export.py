"""Export CFR base artifacts and manifest."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aof_cfr.infoset import infoset_key_spec_hash
from aof_cfr.solve import SolveResult


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def export_policy_npz(path: Path, result: SolveResult) -> None:
    from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default

    keys = sorted(result.policy_rows.keys())
    probs = np.stack([result.policy_rows[k] for k in keys], axis=0).astype(np.float32)
    # Stamp the game model / fee / key-spec the policy was solved under, mirroring the 4-max batch
    # stamp (exploit_router._router_batch). load_cfr_base hard-rejects a stamped NPZ that disagrees
    # with the runtime model, closing the stale-base poisoning vector on the GTO blueprint path.
    np.savez_compressed(
        path,
        infoset_keys=np.array(keys, dtype=object),
        action_probs=probs,
        num_players=np.int32(result.num_players),
        game_model=np.int64(game_model_default()),
        flat_fee=np.float64(flat_fee_default()),
        infoset_key_spec_hash=np.str_(infoset_key_spec_hash()),
    )


def export_manifest(
    path: Path,
    results: list[SolveResult],
    *,
    hu_validation: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
    merge_existing: bool = False,
) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "git_hash": _git_hash(),
        "infoset_key_spec_hash": infoset_key_spec_hash(),
        "tables": [],
    }
    if merge_existing and path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        # Carry over EVERYTHING from the existing manifest, not just tables — otherwise a
        # later per-table re-solve (e.g. the 4-max script) silently drops the other solvers'
        # extras (depth_grid, stack_depth_buckets, ...). Fresh core fields + extra overwrite.
        for k, v in old.items():
            if k not in ("schema_version", "git_hash", "infoset_key_spec_hash"):
                payload[k] = v
        payload["tables"] = list(old.get("tables", []))
    if hu_validation:
        payload["hu_validation"] = hu_validation
    if extra:
        payload.update(extra)

    by_n = {int(t["num_players"]): t for t in payload["tables"]}
    for r in results:
        by_n[r.num_players] = {
            "num_players": r.num_players,
            "solver": r.solver_name,
            "iterations": r.iterations,
            "seed": r.seed,
            "exploitability_bb_per_hand": r.exploitability_bb_per_hand,
            "exploitability_mbb_g": r.exploitability_mbb_g,
            "br_method": r.br_method,
            "player_improvements": r.player_improvements,
            "nash_disclaimer": r.nash_disclaimer,
            "n_infosets": len(r.policy_rows),
        }
    payload["tables"] = [by_n[k] for k in sorted(by_n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
