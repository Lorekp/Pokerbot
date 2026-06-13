"""Stamp existing AoF ``cfr_base_{n}p.npz`` base policies with the game-model / fee / infoset-key
provenance, and refresh the manifest's ``infoset_key_spec_hash`` — WITHOUT re-solving.

Why no re-solve: the shipped policies were already solved under the manifest's declared model
(``game_model_version`` + ``flat_fee_bb``) via the fee-correct paths; the arrays are valid. What was
missing is the per-NPZ stamp that ``query.load_cfr_base`` now verifies (P0-1), and the spec-hash value
changed once when it was made content-sensitive (P0-2). This script adds the stamps and re-emits the
manifest hash. Originals are backed up to ``*_prestamp.npz``.

Safety: if the manifest's declared model disagrees with the runtime defaults
(``game_model_default`` / ``flat_fee_default`` — e.g. an env override, or a genuinely different
solve), the script STOPS and tells you to re-solve rather than mis-stamping a wrong provenance.

    python scripts/stamp_aof_artifacts.py            # stamp in place + re-emit manifest hash
    python scripts/stamp_aof_artifacts.py --dry-run  # report what would change, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aof_cfr.infoset import infoset_key_spec_hash  # noqa: E402
from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default  # noqa: E402

DEFAULT_DIR = ROOT / "data/aof/cfr_base"


def _stamp_npz(path: Path, gm: int, fee: float, spec: str, *, dry_run: bool) -> str:
    data = np.load(path, allow_pickle=True)
    files = set(data.files)
    if not {"infoset_keys", "action_probs", "num_players"} <= files:
        return f"SKIP {path.name}: not a base-policy NPZ (keys={sorted(files)})"
    already = {"game_model", "flat_fee", "infoset_key_spec_hash"} <= files
    if dry_run:
        return f"would stamp {path.name} (game_model={gm} flat_fee={fee} spec={spec})" + (
            " [re-stamp]" if already else "")
    backup = path.with_name(path.stem + "_prestamp.npz")
    if not backup.exists():                       # keep the first pre-stamp original only
        backup.write_bytes(path.read_bytes())
    np.savez_compressed(
        path,
        infoset_keys=data["infoset_keys"],
        action_probs=np.asarray(data["action_probs"], dtype=np.float32),
        num_players=np.int32(int(data["num_players"])),
        game_model=np.int64(gm),
        flat_fee=np.float64(fee),
        infoset_key_spec_hash=np.str_(spec),
    )
    return f"stamped {path.name} (backup {backup.name})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stamp AoF cfr_base NPZs + refresh manifest spec hash")
    ap.add_argument("--base-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = Path(args.base_dir)
    manifest_path = base / "cfr_base.manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    declared_gm = int(manifest.get("game_model_version", 1))
    declared_fee = float(manifest.get("flat_fee_bb", 0.0))
    runtime_gm, runtime_fee = game_model_default(), flat_fee_default()
    if declared_gm != runtime_gm or abs(declared_fee - runtime_fee) > 1e-9:
        sys.exit(
            f"ABORT: manifest declares game_model={declared_gm} fee={declared_fee} but runtime is "
            f"game_model={runtime_gm} fee={runtime_fee}. The bases were solved under a DIFFERENT "
            "model than the runtime prices — RE-SOLVE (scripts/solve_aof_cfr_base.py / "
            "solve_4max_base.py / aof/precompute_4max_batches.py), don't stamp. (Or align "
            "AOF_GAME_MODEL/AOF_FLAT_FEE_BB to the manifest if the env is the odd one out.)")

    spec = infoset_key_spec_hash()
    print(f"stamping with game_model={declared_gm} flat_fee={declared_fee} spec={spec} "
          f"(dry_run={args.dry_run})")
    for n in (2, 3, 4):
        p = base / f"cfr_base_{n}p.npz"
        if not p.is_file():
            print(f"  -- cfr_base_{n}p.npz absent, skipping")
            continue
        print("  " + _stamp_npz(p, declared_gm, declared_fee, spec, dry_run=args.dry_run))

    old_spec = manifest.get("infoset_key_spec_hash")
    if old_spec != spec:
        print(f"  manifest infoset_key_spec_hash {old_spec} -> {spec}")
        if not args.dry_run:
            manifest["infoset_key_spec_hash"] = spec
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        print("  manifest spec hash already current")
    print("done." + ("" if not args.dry_run else " (dry run — nothing written)"))


if __name__ == "__main__":
    main()
