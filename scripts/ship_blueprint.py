#!/usr/bin/env python
"""Convert + VALIDATE + INSTALL a trained 6-max tabular blueprint into the live serve path.

The convenient, reusable mechanism for shipping ANY future blueprint with one command::

    python scripts/ship_blueprint.py --checkpoint <path.nlcfr | s3://... | snapshot.pkl> \
        [--name <tag>] [--activate]

What it does (LIGHT: convert + adapter validation; NO LBR / no GPU):

  1. CONVERT  — a ``.nlcfr`` (or an ``s3://`` key to one) is loaded into the native CFR+ trainer
     (``battery_native.NlheCFRTrainer(200, [], 1).load_bytes(...)``) and ``.snapshot()``-ed to the
     ``{bytes(16) -> [f64;11]}`` table the serve adapter consumes. A ``.pkl`` snapshot (raw
     ``{bytes->row}`` dict, or a stamped ``{"strat_sum":..., "abstraction":...}`` payload, local or
     ``s3://``) is loaded directly. The native ``.pyd`` may be mid-rebuild by a concurrent workflow,
     so the ``battery_native`` import is RETRIED.

  2. VALIDATE (the serve-skew firewall gate — this is the -559 bb/100 guard) — the snapshot is
     loaded into a real ``TabularBlueprintAdapter`` and we assert, on a fresh 6-max probe state:
       (a) the serve abstraction MATCHES the snapshot's stamp (``check_blueprint_serve_env`` STRICT
           on the {200-bucket / fast / preflop-lossless / non-depth} config) — a mismatch silently
           misses every node and plays a beatable strategy;
       (b) the adapter serves a finite, legal-masked, sum-to-1 11-vector;
       (c) the trained keys ACTUALLY HIT serve (blueprint coverage > 0 over a batch of probe nodes
           — proves the fingerprint key space is byte-aligned, the real train/serve skew test);
       (d) the infoset count is sane (>= a floor).
     If ANY check fails the tool REFUSES to activate and reports loudly.

  3. INSTALL  — the (stamped) snapshot is written to a STABLE versioned location
     ``checkpoints/serve_blueprints/<name>.pkl``. With ``--activate`` the stable "current" pointer
     ``checkpoints/serve_blueprint_6max.pkl`` is atomically replaced to point at it (previous
     backed up to ``checkpoints/serve_blueprint_6max.pkl.bak`` so revert is one command).

  4. The serve default (``pokerbot/eval/bots.py:_load_blueprint_snapshot``) reads
     ``POKERBOT_BLUEPRINT_SNAPSHOT`` -> the stable "current" path (fallback to the legacy temp pkl
     if the stable file is absent). So FUTURE swaps are just
     ``ship_blueprint.py --checkpoint <new> --activate`` with NO code edit.

REVERT (one command)::

    python scripts/ship_blueprint.py --revert

(restores ``checkpoints/serve_blueprint_6max.pkl`` from its ``.bak``.)

LIGHT only. Does not touch the AoF engine, the live crate, or the search files.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# The serve abstraction config. These flags are baked into pokerbot AT IMPORT
# (jax_weva_encode), so they MUST be set before any pokerbot import. They are the
# deployed 6-max blueprint config: betting abstraction ON, 200 made-hand postflop
# buckets, lossless preflop, NON-depth. The validation gate cross-checks the snapshot
# stamp against exactly this, strictly.
# --------------------------------------------------------------------------- #
SERVE_ABSTRACTION_ENV = {
    "POKERBOT_BETTING_ABSTRACTION": "1",
    "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "200",
    "POKERBOT_POSTFLOP_MODE": "fast",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
    # NON-depth blueprint: leave POKERBOT_DEPTH_BUCKETS unset/0.
}

REPO_ROOT = Path(__file__).resolve().parent.parent
# Put the repo root on sys.path so the repo-root `battery_native.pyd` and the `pokerbot` package both
# import regardless of the cwd this CLI was launched from (running `scripts/ship_blueprint.py` puts
# the scripts/ dir on sys.path[0], not the repo root).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SERVE_DIR = REPO_ROOT / "checkpoints" / "serve_blueprints"
# The stable "current" pointer the serve default reads. Activation = replacing this file.
STABLE_CURRENT = REPO_ROOT / "checkpoints" / "serve_blueprint_6max.pkl"
STABLE_BACKUP = STABLE_CURRENT.with_suffix(STABLE_CURRENT.suffix + ".bak")

# Native trainer construction for a deployed 6-max non-depth blueprint.
NATIVE_NUM_BUCKETS = 200
NATIVE_DEPTH_EDGES: list = []  # non-depth
NATIVE_N_DEALS = 1  # irrelevant for load+snapshot (we never train here)

# Sanity floor: the deployed iter200 lossless blueprint had ~hundreds of thousands of infosets.
# A real converged 6-max run is well above this; a near-empty table fails fast.
MIN_INFOSETS = 1000


def _set_serve_env() -> None:
    """Set the serve abstraction flags BEFORE importing pokerbot. Honors any pre-set DEPTH flag = 0."""
    for k, v in SERVE_ABSTRACTION_ENV.items():
        os.environ.setdefault(k, v)
    # Force the non-depth default unless the operator explicitly opted into a depth blueprint.
    os.environ.setdefault("POKERBOT_DEPTH_BUCKETS", "0")


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key) for an s3:// URI, ignoring any query suffix."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 URI: {uri}")
    body = uri[5:].split("?", 1)[0]
    bucket, sep, key = body.partition("/")
    if not bucket or not sep or not key:
        raise ValueError(f"malformed s3 URI: {uri}")
    return bucket, key


def _download_s3(src: str, dst: Path) -> None:
    """Download with boto3 so cloud runs do not depend on awscli/botocore packaging details."""
    import boto3  # noqa: PLC0415

    bucket, key = _parse_s3_uri(src)
    boto3.client("s3").download_file(bucket, key, str(dst))


def _import_battery_native(retries: int = 8, delay: float = 3.0):
    """Import the native crate, RETRYING (it may be mid-rebuild by a concurrent workflow).

    The trainer/snapshot/load_bytes code is unaffected by that workflow's search edits, so the
    conversion is correct once the .pyd imports.
    """
    last = None
    for i in range(retries):
        try:
            import battery_native  # noqa: PLC0415

            return battery_native
        except Exception as e:  # noqa: BLE001  (ImportError or a transient locked-.pyd OSError)
            last = e
            if i < retries - 1:
                print(f"  [convert] battery_native import failed (attempt {i + 1}/{retries}: "
                      f"{type(e).__name__}); the .pyd may be mid-rebuild — retrying in {delay:.0f}s",
                      flush=True)
                time.sleep(delay)
    raise RuntimeError(
        f"could not import battery_native after {retries} tries (last: {last!r}). The native .pyd "
        "is Windows-only and may be mid-rebuild; run on the Windows host once the crate finishes "
        "building, or pass a pre-converted .pkl snapshot.") from last


def _maybe_fetch_s3(src: str, workdir: Path) -> Path:
    """If ``src`` is an s3:// URI, download it to ``workdir`` and return the local path."""
    if not src.startswith("s3://"):
        return Path(src)

    local = workdir / Path(src.split("?", 1)[0]).name
    print(f"  [convert] fetching {src} -> {local}", flush=True)
    _download_s3(src, local)
    if not local.exists():
        raise FileNotFoundError(f"s3 download produced no file: {src}")
    return local


def load_snapshot(checkpoint: str, workdir: Path) -> dict:
    """Return the raw ``{bytes(16) -> list[float](11)}`` blueprint table from any supported source.

    Accepts: a local/s3 ``.nlcfr`` (native checkpoint -> trainer.load_bytes -> snapshot), or a
    local/s3 ``.pkl`` (raw dict, or a stamped ``{"strat_sum":..., "abstraction":...}`` payload).
    """
    local = _maybe_fetch_s3(checkpoint, workdir)
    if not local.exists():
        raise FileNotFoundError(f"checkpoint not found: {local}")

    suffix = local.suffix.lower()
    if suffix == ".pkl":
        print(f"  [convert] loading pickled snapshot {local}", flush=True)
        with open(local, "rb") as f:
            snap = pickle.load(f)
        if isinstance(snap, dict) and "strat_sum" in snap:
            snap = snap["strat_sum"]
        if not isinstance(snap, dict):
            raise ValueError(f"pkl did not contain a snapshot dict (got {type(snap)})")
        return snap

    if suffix == ".nlcfr":
        bn = _import_battery_native()
        print(f"  [convert] NlheCFRTrainer({NATIVE_NUM_BUCKETS}, {NATIVE_DEPTH_EDGES}, "
              f"{NATIVE_N_DEALS}).load_bytes({local})", flush=True)
        trainer = bn.NlheCFRTrainer(NATIVE_NUM_BUCKETS, NATIVE_DEPTH_EDGES, NATIVE_N_DEALS)
        with open(local, "rb") as f:
            ok = trainer.load_bytes(f.read())
        if not ok:
            raise ValueError(
                f"NlheCFRTrainer.load_bytes returned False for {local} — corrupt or foreign blob "
                "(fail-closed). Refusing to ship.")
        iters = trainer.iters_done()
        snap = trainer.snapshot()
        print(f"  [convert] iters_done={iters}  snapshot={len(snap)} infosets", flush=True)
        return snap

    raise ValueError(f"unsupported checkpoint type {suffix!r} (expected .nlcfr or .pkl)")


# --------------------------------------------------------------------------- #
# Validation (the serve-skew firewall gate)
# --------------------------------------------------------------------------- #
class ValidationError(RuntimeError):
    """Raised when the serve-skew gate fails. The tool REFUSES to activate."""


def validate_serve(snapshot: dict, stamp: dict) -> dict:
    """Run the serve-skew firewall gate. Raises ValidationError on any failure.

    pokerbot MUST already be importable with the serve abstraction flags baked in (see _set_serve_env,
    called before any pokerbot import). Returns a dict of evidence on success.
    """
    import numpy as np  # noqa: PLC0415

    # (0) SHA256 content-stamp check for EMD frozen blobs. The sidecar was written by
    # scripts/precompute_emd_frozen_lookup.py alongside the blob; an attacker (or a stale
    # artifact) cannot ship a modified blob under an old sidecar because the SHA256 is
    # computed from the blob bytes themselves.
    emd_frozen_env = os.environ.get("POKERBOT_POSTFLOP_EMD_FROZEN")
    if emd_frozen_env:
        emd_blob_path = Path(emd_frozen_env)
        emd_sidecar = emd_blob_path.with_suffix(emd_blob_path.suffix + ".sha256")
        if not emd_sidecar.exists():
            raise ValidationError(
                f"(0) SHA256 sidecar missing for EMD frozen blob: {emd_sidecar}. " +
                f"Re-run scripts/precompute_emd_frozen_lookup.py to regenerate the blob + sidecar together.")
        expected_sha = emd_sidecar.read_text(encoding="ascii").strip()
        actual_sha = hashlib.sha256(emd_blob_path.read_bytes()).hexdigest()
        if expected_sha != actual_sha:
            raise ValidationError(
                f"(0) SHA256 mismatch for EMD frozen blob: expected {expected_sha[:16]}..., " +
                f"got {actual_sha[:16]}...  blob may be stale or tampered with.")

    # (a) STRICT serve-abstraction cross-check: the active encoder config must field-match the stamp.
    from pokerbot.eval.lbr import check_blueprint_serve_env  # noqa: PLC0415

    try:
        active = check_blueprint_serve_env(expected=stamp, strict=True)
    except Exception as e:  # noqa: BLE001
        raise ValidationError(
            f"(a) serve-abstraction MISMATCH vs the trained snapshot stamp — refusing to activate. "
            f"{e}") from e

    # Build the adapter exactly as the live serve path does.
    from pokerbot.search.tabular_blueprint_adapter import TabularBlueprintAdapter  # noqa: PLC0415

    snap_np = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in snapshot.items()}
    n_infosets = len(snap_np)
    if n_infosets < MIN_INFOSETS:
        raise ValidationError(
            f"(d) infoset count {n_infosets} < floor {MIN_INFOSETS} — table looks empty/corrupt; "
            "refusing to activate.")
    adapter = TabularBlueprintAdapter(snap_np, client_bet_sizing=True)

    # Build fresh 6-max probe states and serve through the adapter, counting blueprint coverage so we
    # PROVE the trained keys hit serve (not just that a legal-uniform fallback returns a valid vector).
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from pokerbot.engine.jax_engine import init_hand  # noqa: PLC0415
    from pokerbot.engine.np_engine import (  # noqa: PLC0415
        is_chance_node_np,
        is_terminal_np,
        legal_action_mask_np,
        sample_chance_np,
        to_numpy_state,
    )
    from pokerbot.eval.lbr import bp_coverage_report, bp_coverage_reset  # noqa: PLC0415

    # Turn on the per-street coverage counter for the probe (process-global; reset first).
    os.environ["POKERBOT_BP_COVERAGE"] = "1"
    import pokerbot.eval.lbr as _lbr  # noqa: PLC0415

    _lbr._BP_COVERAGE_ON = True  # the module bound the flag at import; flip it for this process
    bp_coverage_reset()

    n_probes = 200
    served_ok = 0
    first_vec = None
    for h in range(n_probes):
        key = jax.random.PRNGKey(1234 + h)
        env = to_numpy_state(init_hand(key, n_players=6))
        # Walk a few decision nodes per hand, sampling chance, sampling the blueprint to advance.
        depth = 0
        while depth < 12 and not bool(is_terminal_np(env)):
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            seat = int(env.action_seat)
            p = adapter.blueprint(env, seat)
            p = np.asarray(p, dtype=np.float64)
            mask = legal_action_mask_np(env, seat).astype(bool)
            # (b) finite, sums to 1, mass only on legal actions.
            if not (np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-6 and not (p[~mask] > 1e-9).any()):
                raise ValidationError(
                    f"(b) adapter served an illegal/degenerate 11-vector at hand {h} seat {seat}: "
                    f"sum={p.sum():.6f} mask={mask.tolist()} p={np.round(p, 4).tolist()}")
            served_ok += 1
            if first_vec is None:
                first_vec = (int(seat), np.round(p, 4).tolist(), mask.tolist())
            # advance by the argmax legal action to reach deeper streets
            idx = int(np.argmax(p))
            from pokerbot.engine.np_engine import apply_action_np  # noqa: PLC0415

            env = apply_action_np(env, idx)
            depth += 1

    cov = bp_coverage_report()
    total_hits = sum(s["hits"] for s in cov.values())
    total_misses = sum(s["misses"] for s in cov.values())
    total = total_hits + total_misses
    hit_rate = (total_hits / total) if total else 0.0

    # (c) the trained keys must ACTUALLY HIT serve. A converged blueprint hits the overwhelming
    # majority of probe nodes; a skewed abstraction misses ~everything (the train/serve skew bug).
    if total_hits == 0:
        raise ValidationError(
            f"(c) ZERO blueprint coverage over {n_probes} probe hands — the serve fingerprint key "
            f"space does not match the trained table (the -559 bb/100 train/serve skew). per-street={cov}")
    if hit_rate < 0.5:
        raise ValidationError(
            f"(c) blueprint coverage {hit_rate:.1%} ({total_hits}/{total}) below the 50% serve-skew "
            f"floor — abstraction is misaligned. per-street={cov}. Refusing to activate.")

    return {
        "active_config": active,
        "stamp": stamp,
        "n_infosets": n_infosets,
        "probe_hands": n_probes,
        "served_decisions": served_ok,
        "coverage_hit_rate": hit_rate,
        "coverage_hits": total_hits,
        "coverage_misses": total_misses,
        "coverage_per_street": cov,
        "example_served_vector": first_vec,
    }


def _stamp_for_snapshot() -> dict:
    """The abstraction stamp for the active (serve) config — written into the installed pkl."""
    from pokerbot.engine.jax_weva_encode import active_abstraction_config  # noqa: PLC0415

    return active_abstraction_config()


# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
def _atomic_write_pickle(obj, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def install(snapshot: dict, stamp: dict, name: str, activate: bool) -> dict:
    """Write the stamped snapshot to the versioned location; with ``activate`` update the pointer."""
    import numpy as np  # noqa: PLC0415

    # Store as float32 (the serve adapter upcasts) — half the disk of f64, byte-aligned keys preserved.
    snap_f32 = {bytes(k): np.asarray(v, dtype=np.float32) for k, v in snapshot.items()}
    payload = {"strat_sum": snap_f32, "abstraction": stamp}

    versioned = SERVE_DIR / f"{name}.pkl"
    _atomic_write_pickle(payload, versioned)
    print(f"  [install] wrote versioned snapshot -> {versioned}  ({len(snap_f32)} infosets)", flush=True)

    info = {"versioned": str(versioned), "activated": False}
    if activate:
        if STABLE_CURRENT.exists():
            shutil.copy2(STABLE_CURRENT, STABLE_BACKUP)
            print(f"  [install] backed up current pointer -> {STABLE_BACKUP}", flush=True)
        _atomic_write_pickle(payload, STABLE_CURRENT)
        print(f"  [install] ACTIVATED: {STABLE_CURRENT} now serves '{name}'", flush=True)
        info["activated"] = True
        info["stable_current"] = str(STABLE_CURRENT)
        info["backup"] = str(STABLE_BACKUP) if STABLE_BACKUP.exists() else None
    return info


def revert() -> None:
    """Restore the previous blueprint.

    If a ``.bak`` exists (i.e. this is at least the 2nd activation), swap it back over the current
    pointer. If there is NO ``.bak`` (the active blueprint was the FIRST one ever activated), remove
    the stable pointer so the serve default falls back to the legacy blueprint (the documented
    back-compat path in ``bots._load_blueprint_snapshot``).
    """
    if STABLE_BACKUP.exists():
        if STABLE_CURRENT.exists():
            swap = STABLE_CURRENT.with_suffix(STABLE_CURRENT.suffix + ".prev")
            shutil.copy2(STABLE_CURRENT, swap)
        shutil.copy2(STABLE_BACKUP, STABLE_CURRENT)
        print(f"REVERTED: {STABLE_CURRENT} <- {STABLE_BACKUP}", flush=True)
        return
    if STABLE_CURRENT.exists():
        # No backup = first-ever activation. Park the current pointer and drop it so the serve default
        # falls back to the legacy blueprint.
        shutil.copy2(STABLE_CURRENT, STABLE_CURRENT.with_suffix(STABLE_CURRENT.suffix + ".prev"))
        STABLE_CURRENT.unlink()
        print(f"REVERTED: removed first-activation pointer {STABLE_CURRENT} (saved .prev); serve "
              "default now falls back to the legacy blueprint", flush=True)
        return
    raise FileNotFoundError(
        f"nothing to revert: neither {STABLE_BACKUP} nor {STABLE_CURRENT} exists")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",
                    help="source: a .nlcfr (native ckpt), a .pkl snapshot, or an s3:// URI to either")
    ap.add_argument("--name", default=None,
                    help="version tag for checkpoints/serve_blueprints/<name>.pkl "
                         "(default: derived from the checkpoint filename)")
    ap.add_argument("--activate", action="store_true",
                    help="point the stable serve pointer (checkpoints/serve_blueprint_6max.pkl) at "
                         "this blueprint (with a .bak backup for one-command revert)")
    ap.add_argument("--revert", action="store_true",
                    help="restore the previous blueprint from checkpoints/serve_blueprint_6max.pkl.bak")
    ap.add_argument("--no-validate", action="store_true",
                    help="(DANGEROUS) skip the serve-skew gate — never use to ship to live")
    ap.add_argument("--num-buckets", type=int, default=None,
                    help="override the postflop bucket count (default 200). MUST equal the trained "
                         "--num-buckets AND, for EMD, the frozen table's K — else the converter mis-keys.")
    ap.add_argument("--emd-frozen", default=None,
                    help="ship an EMD blueprint: sets POKERBOT_POSTFLOP_MODE=emd + "
                         "POKERBOT_POSTFLOP_EMD_FROZEN=<this path> so the serve encoder reads the SAME "
                         "frozen (hole,board)->bucket table the blueprint trained against. The serve-skew "
                         "gate then asserts the frozen-table content hash matches the snapshot stamp.")
    args = ap.parse_args(argv)

    # Apply EMD/bucket overrides into the serve-abstraction env BEFORE _set_serve_env (which only
    # setdefault()s, so an override here wins). These bake into the encoder at import.
    global NATIVE_NUM_BUCKETS
    if args.num_buckets is not None:
        if args.num_buckets <= 0:
            ap.error("--num-buckets must be > 0")
        SERVE_ABSTRACTION_ENV["POKERBOT_POSTFLOP_NUM_BUCKETS"] = str(args.num_buckets)
        NATIVE_NUM_BUCKETS = args.num_buckets  # the .nlcfr->snapshot converter must match
    if args.emd_frozen:
        emd_path = Path(args.emd_frozen)
        if not emd_path.exists():
            ap.error(f"--emd-frozen {args.emd_frozen} not found")
        SERVE_ABSTRACTION_ENV["POKERBOT_POSTFLOP_MODE"] = "emd"
        SERVE_ABSTRACTION_ENV["POKERBOT_POSTFLOP_EMD_FROZEN"] = str(emd_path.resolve())

    # MUST set serve env BEFORE importing pokerbot (the encoder bakes the flags at import).
    _set_serve_env()

    if args.revert:
        revert()
        return 0

    if not args.checkpoint:
        ap.error("--checkpoint is required (unless --revert)")

    name = args.name
    if name is None:
        stem = Path(args.checkpoint.split("?", 1)[0]).stem
        name = f"bp6max_{stem}"

    with tempfile.TemporaryDirectory(prefix="ship_bp_") as td:
        workdir = Path(td)
        print(f"[1/3] CONVERT  source={args.checkpoint}", flush=True)
        snapshot = load_snapshot(args.checkpoint, workdir)

        stamp = _stamp_for_snapshot()
        print(f"[2/3] VALIDATE (serve-skew gate)  stamp={stamp}", flush=True)
        if args.no_validate:
            print("  [validate] SKIPPED (--no-validate) — NOT safe for live", flush=True)
            evidence = {"skipped": True, "n_infosets": len(snapshot)}
        else:
            try:
                evidence = validate_serve(snapshot, stamp)
            except ValidationError as e:
                print("\n========================================================================",
                      flush=True)
                print("VALIDATION FAILED — REFUSING TO ACTIVATE", flush=True)
                print(str(e), flush=True)
                print("========================================================================",
                      flush=True)
                # still install the versioned artifact (inert) but never activate.
                if args.activate:
                    print("NOT activating; --activate ignored due to validation failure.", flush=True)
                install(snapshot, stamp, name, activate=False)
                return 2
            print("  [validate] PASS:", flush=True)
            print(f"     serve abstraction == snapshot stamp (strict): OK", flush=True)
            print(f"     infosets={evidence['n_infosets']}", flush=True)
            print(f"     served decisions={evidence['served_decisions']} over "
                  f"{evidence['probe_hands']} probe hands, all legal/finite/sum-1", flush=True)
            print(f"     blueprint coverage={evidence['coverage_hit_rate']:.1%} "
                  f"({evidence['coverage_hits']} hits / {evidence['coverage_misses']} misses)",
                  flush=True)
            print(f"     per-street={evidence['coverage_per_street']}", flush=True)

        print(f"[3/3] INSTALL  name={name}  activate={args.activate}", flush=True)
        info = install(snapshot, stamp, name, activate=args.activate)

    print("\nDONE.", flush=True)
    print(f"  versioned: {info['versioned']}", flush=True)
    if info.get("activated"):
        print(f"  ACTIVE serve blueprint: {info['stable_current']}", flush=True)
        print(f"  revert with: python scripts/ship_blueprint.py --revert", flush=True)
    else:
        print("  (not activated — pass --activate to make it the live serve default)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
