#!/usr/bin/env python3
"""PHASE 5 of the EMD blueprint pipeline -- the VERIFIED-BETTER gate.

Harshly LBR-evaluates the NEW EMD blueprint AND the shipped iter62000 baseline on the SAME seeds
(common random numbers) and decides whether the EMD blueprint is BETTER, using the per-seat BLIND
LBR as the acceptance metric (NOT the noise-wide pooled CI).

WHY per-seat, why CRN
---------------------
The pooled value-only LBR CI is too wide to certify a small global improvement
(iter62000: value-only -3.5 bb/100 CI [-97,+90]; fold-eq-ON +41.5 CI [-62,+145]). But the residual
exploitability of iter62000 CONCENTRATES in the BLINDS: with ``button_seat=0`` fixed every hand the
per-seat index IS the button-offset (seat 0 = BTN, seat 1 = SB, seat 2 = BB; 3-5 = UTG/HJ/CO), and
iter62000's leak is seat 1 (+95) / seat 2 (+154) with the button & late seats ~0
(``artifacts/better_blueprint_setup_v2_blinds.md``). The EMD (draw-aware) abstraction's whole thesis is
that it pulls those blind seats TOWARD the button's ~0. Because both blueprints are evaluated on the
SAME ``LBR6Config(hands, seed)`` schedule, the per-hand deals are SHARED, and each seat's estimate is a
CRN slice (``lbr6.py``: ``results[s::n_seats]``). The deal noise CANCELS in the per-seat DELTA even
though the absolute per-seat CIs are wide. Because per-seat LBR can be negative,
the acceptance delta is ``abs(iter62000_per_seat) - abs(emd_per_seat)``:
positive means exploitability magnitude moved DOWN toward the button's ~0. This
is a low-variance read of "did EMD fix the blinds leak."

WHY native is allowed for the EMD side
--------------------------------------
The native ``Lbr6Harness`` now has the same frozen EMD lookup seam the Python encoder uses:
flop/turn buckets come from the shared ``EMDFRZN1`` table, while river/preflop/misses fall back to the
made-hand bucket exactly like ``jax_weva_encode.compute_postflop_bucket``. The dictionary key is therefore
bit-identical where it must be (blueprint lookup); the Monte Carlo LBR stream is statistical-equivalence
by design (native xoshiro / Rayon), just like ``pokerbot.eval.lbr_native``'s existing made-hand path.
If the native extension is unavailable, this script loudly falls back to the Python reference harness.

ANCHOR CALIBRATION (trust gate)
-------------------------------
Before any blueprint number is trusted, the always-fold policy must reproduce the documented
~ +125 bb/100 6-max anchor (the harness self-test: a table that always folds donates the blinds to a
greedy best-responder). always-fold ignores postflop buckets, so the anchor is HARNESS-level (env- and
blueprint-independent) and is the cheap, sharp "is the harness sane" check. If it lands outside the band
the eval is UNTRUSTWORTHY and we ABORT (``anchor_ok=false``, ``emd_better=false``) -- never ship on a
miscalibrated harness.

THREE MODES (one clean single-env process each; the cloud-init calls all three)
-------------------------------------------------------------------------------
  --mode emd   : set POKERBOT_POSTFLOP_MODE=emd + POKERBOT_POSTFLOP_EMD_FROZEN before import; LBR the EMD
                 blueprint + the always-fold anchor; write a partial JSON.
  --mode fast  : default made-hand env; LBR iter62000 + the always-fold anchor; write a partial JSON.
  --combine    : read both partials, compute the per-seat blind deltas + the verdict, write the final
                 verdict JSON (and upload to S3 if --s3-out is given).

Splitting emd/fast into separate processes is REQUIRED: the abstraction flags bake into the encoder AT
IMPORT (``jax_weva_encode``), so one process cannot evaluate a mode=emd and a mode=fast blueprint.

MANDATORY PROGRESS LOGGING (per the standing rule): each seed prints a percent-done / throughput / ETA
line and a heartbeat; the anchor calibration line is printed FIRST and loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Sign-correct verdict printer (single source of truth for the per-seat
# IMPROVEMENT/REGRESSION label). Lives at scripts/_emd_verdict_sign.py --
# sibling of scripts/eval/ -- intentionally invokable from anywhere with
# just stdlib + json. Insert its directory onto sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _emd_verdict_sign import (  # noqa: E402  (intentional path injection)
    format_seat_delta, format_pooled, sign_correct_verified, SIGN_CONVENTION,
)
from _emd_selection_gate import (  # noqa: E402
    accepted_survivor_ks,
    build_selection_report,
    stamp_primary_action_set,
)

# Atomic JSON write helper (SCRIPTS-EVAL-AUDIT CRIT). The partial + verdict
# JSONs are uploaded to S3 by the babysitter; a torn tail would publish a
# partial verdict. Use ``scripts._atomic_io.write_json_atomic`` (tmp + fsync +
# os.replace + parent-dir fsync; matches label_completion.write_atomic).
# Inject the scripts dir on sys.path so the import works regardless of cwd
# (the onbox script runs `python3 scripts/eval/eval_...py` from CODE_DIR; without
# this injection the script's own dir (scripts/eval/) would be on sys.path[0],
# not the parent scripts/ where _atomic_io lives).
import sys as _sys_atomic
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in _sys_atomic.path:
    _sys_atomic.path.insert(0, str(_SCRIPTS_DIR))
from _atomic_io import write_json_atomic  # noqa: E402  (intentional path injection)

# Anchor band: the documented 6-max always-fold LBR exploitability is ~ +125-128 bb/100 (this Python
# harness and the native one are CI-overlap-equivalent on always-fold -- tests/test_lbr_native_equiv.py).
ANCHOR_LO = 110.0
ANCHOR_HI = 145.0
ANCHOR_TARGET = 125.0

# Named probe menus for the selection ladder. The LBR harness can only play legal
# engine action indexes, so off-tree fractions are resolved through the same
# pseudo-harmonic bucket mapping used by the live action-translation path and the
# original requested fractions are preserved in the partial metadata.
ACTION_SET_BET_PRESETS = {
    "in_tree": "0.5,1.0,all_in",
    "pot_fraction": "0.3333333333333333,0.5,0.75,1.0,all_in",
    "off_tree_adversarial": "0.25,0.9,1.25,2.5,all_in",
}

# CRN reseed per replication. Per artifacts/pluribus_emd_retrain_report.md hypothesis B, the previous
# `seeds = list(range(args.seeds))` lets the table_id + deck sequence collide across replications that
# share seed_idx. A stable XOR with a fixed base seed gives every replication a different starting
# point while keeping the per-hand deal schedule reproducible (the per-hand LBR6Config(hands, seed)
# drives the schedule). The base itself is the documented value so all readers agree on the stream.
BASE_SEED = 0x5EEDBAB0  # 1592263344; "BABY" mnemonic preserved (note: 0x5EEDBABY is invalid Python hex,
                       # so the closest valid literal that preserves the mnemonic is used here)


def seed_for_replication(replication_id: int) -> int:
    """XOR with a stable base so each replication gets a different stream.

    Per artifacts/pluribus_emd_retrain_report.md hypothesis B: list(range(N)) lets the table_id
    and deck sequence collide across replications sharing the same seed_idx. A stable XOR with
    BASE_SEED gives every replication a different starting point while keeping the per-hand
    deal schedule reproducible (the per-hand LBR6Config(hands, seed) drives the schedule).

    Edge case: id=0 is the documented xor-identity (BASE_SEED ^ 0 == BASE_SEED). The reseed only
    diverges from the base for non-zero replication_id; the call site may freely pass `range(N)`.

    Asserts: (a) non-zero replication_id produces a stream != BASE_SEED, (b) result is non-negative.
    """
    s = BASE_SEED ^ int(replication_id)
    # XOR with 0 is the identity (BASE_SEED ^ 0 == BASE_SEED); this is the documented reproduction-id=0
    # case. Only assert != BASE_SEED for non-zero replication_id where the reseed must diverge.
    if replication_id != 0:
        assert s != BASE_SEED, f"CRN reseed must differ from base for non-zero replication_id (got {s})"
    assert s >= 0, "XOR of non-negative ints must be non-negative"
    return s

# A meaningful blind improvement: the EMD thesis is that the +95/+154 blind leak moves materially toward
# the button's ~0. Require the blinds to drop by at least this many bb/100 (CRN delta, low-variance) AND
# the pooled global mean to be no worse than iter62000 by more than a small slack.
DEFAULT_MIN_BLIND_GAIN = 15.0     # bb/100 each blind seat must improve (drop) by, summed across s1+s2 / 2
DEFAULT_GLOBAL_SLACK = 10.0       # bb/100: EMD's pooled mean may be at most this much worse than iter62000
EVAL_PROGRESS_EVERY_SEC = 60.0


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
    """Download using boto3 so the EC2 role path does not depend on awscli/botocore packaging."""
    import boto3  # noqa: PLC0415

    bucket, key = _parse_s3_uri(src)
    boto3.client("s3").download_file(bucket, key, str(dst))


def _upload_s3(src: Path, dst: str) -> None:
    """Upload using boto3 so eval verdict publishing matches the on-box marker path."""
    import boto3  # noqa: PLC0415

    bucket, key = _parse_s3_uri(dst)
    boto3.client("s3").upload_file(str(src), bucket, key)


def _make_hand_progress(label: str, seed: int, seed_idx: int, seed_count: int, hands: int, run_started: float):
    """Return a throttled per-hand progress callback for the pure-Python LBR loop.

    The evaluator's unit of work is a seed, but each seed can run for a long time. This callback keeps the
    cloud console and periodically-uploaded S3 logs alive without touching LBR decisions or ranges.
    """
    seed_started = time.time()
    last_emit = 0.0
    hand_stride = max(1, hands // 20)  # at least every 5% even when a seed is quick

    def progress(done: int, total: int, lbr_seat: int, hand_ev_bb: float, lbr_decisions: int) -> None:
        nonlocal last_emit
        now = time.time()
        if done not in (1, total) and done % hand_stride and (now - last_emit) < EVAL_PROGRESS_EVERY_SEC:
            return
        last_emit = now
        seed_elapsed = max(now - seed_started, 1e-9)
        run_elapsed = max(now - run_started, 1e-9)
        seed_rate = done / seed_elapsed
        total_done = (seed_idx - 1) * hands + done
        total_hands = seed_count * hands
        total_rate = total_done / run_elapsed
        seed_eta = (total - done) / seed_rate if seed_rate > 1e-9 else -1.0
        total_eta = (total_hands - total_done) / total_rate if total_rate > 1e-9 else -1.0
        print(
            f"  [{label}] HAND_PROGRESS seed={seed} seed_idx={seed_idx}/{seed_count} "
            f"hand={done}/{total} ({100.0 * done / max(total, 1):.1f}%) "
            f"total_hands={total_done}/{total_hands} "
            f"elapsed_seed={seed_elapsed:.0f}s hand_rate={seed_rate:.2f}/s "
            f"eta_seed={seed_eta:.0f}s eta_all={total_eta:.0f}s "
            f"lbr_seat={lbr_seat} last_ev_bb={hand_ev_bb:+.3f} "
            f"lbr_decisions={lbr_decisions} HEARTBEAT",
            flush=True,
        )

    return progress


def _run_python_lbr6(policy_fn, cfg, *, progress_fn=None):
    """Call Python lbr6 with progress on both old and new harness signatures."""
    from pokerbot.eval.lbr6 import lbr6_exploitability  # noqa: PLC0415

    if progress_fn is None:
        return lbr6_exploitability(policy_fn, cfg)

    try:
        import inspect  # noqa: PLC0415

        if "progress_fn" in inspect.signature(lbr6_exploitability).parameters:
            return lbr6_exploitability(policy_fn, cfg, progress_fn=progress_fn)
    except Exception:  # noqa: BLE001 - fall through to hook-based compatibility
        pass

    state = {"done": 0, "lbr_seat": -1, "lbr_decisions": 0}

    def on_hand_start(lbr_seat: int) -> None:
        state["lbr_seat"] = int(lbr_seat)

    def hand_sink(info: dict) -> None:
        state["done"] += 1
        state["lbr_decisions"] += int(info.get("lbr_decisions", 0))
        progress_fn(
            int(state["done"]),
            int(cfg.hands),
            int(state["lbr_seat"]),
            float(info.get("reward", 0.0)),
            int(state["lbr_decisions"]),
        )

    return lbr6_exploitability(policy_fn, cfg, on_hand_start=on_hand_start, hand_sink=hand_sink)


def _parse_fraction_token(token: str) -> float:
    if "/" in token:
        num, den = token.split("/", 1)
        return float(num) / float(den)
    return float(token)


def resolve_bet_candidates(spec: str | None, *, action_set_name: str | None = None) -> dict:
    """Resolve a user-facing bet menu into legal LBR action indexes.

    Numeric tokens are pot fractions and are translated to engine bet buckets
    with the live serve path's pseudo-harmonic mapping. Use ``action:N`` for an
    explicit action index when auditing an exact menu.
    """
    from pokerbot.engine.actions import BET_SIZE_BUCKETS, index_to_bucket, pseudo_harmonic_bucket  # noqa: PLC0415
    from pokerbot.engine.jax_engine import ALL_IN, BET_BUCKET_START  # noqa: PLC0415

    name = str(action_set_name or "in_tree")
    text = (spec or "").strip()
    if not text:
        text = ACTION_SET_BET_PRESETS.get(name, ACTION_SET_BET_PRESETS["in_tree"])
    elif text in ACTION_SET_BET_PRESETS:
        if action_set_name is None or name == "in_tree":
            name = text
        text = ACTION_SET_BET_PRESETS[text]

    actions: list[int] = []
    mapping: list[dict] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        low = token.lower().replace("-", "_")
        if low in ("a", "ai", "allin", "all_in", "jam"):
            action = int(ALL_IN)
            row = {
                "token": token,
                "kind": "all_in",
                "action": action,
                "bucket_fraction": None,
            }
        elif low.startswith("action:"):
            action = int(low.split(":", 1)[1])
            if action < BET_BUCKET_START or action > ALL_IN:
                raise argparse.ArgumentTypeError(
                    f"action index {action} outside LBR bet-action range [{BET_BUCKET_START},{ALL_IN}]"
                )
            row = {
                "token": token,
                "kind": "action_index",
                "action": action,
                "bucket_fraction": index_to_bucket(action),
            }
        else:
            try:
                frac = _parse_fraction_token(token)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid bet candidate {token!r}; use a pot fraction, all_in, or action:N"
                ) from exc
            bucket = int(pseudo_harmonic_bucket(frac, BET_SIZE_BUCKETS))
            action = int(BET_BUCKET_START + bucket)
            row = {
                "token": token,
                "kind": "pot_fraction",
                "requested_fraction": frac,
                "action": action,
                "bucket_fraction": float(BET_SIZE_BUCKETS[bucket]),
            }
        if action not in actions:
            actions.append(action)
        mapping.append(row)

    if not actions:
        raise argparse.ArgumentTypeError("bet candidate menu must resolve to at least one LBR bet action")

    return {
        "action_set_name": name,
        "spec": text,
        "actions": tuple(actions),
        "mapping": mapping,
    }


def _set_emd_env(frozen_path: str, num_buckets: int) -> None:
    """Bake the EMD serve abstraction BEFORE importing pokerbot (the encoder reads these at import).
    Mirrors ship_blueprint.py's --emd-frozen env exactly so the eval keys postflop through the SAME
    frozen (hole,board)->bucket table the EMD blueprint trained against."""
    os.environ["POKERBOT_BETTING_ABSTRACTION"] = "1"
    os.environ["POKERBOT_POSTFLOP_BUCKETS"] = "1"
    os.environ["POKERBOT_POSTFLOP_NUM_BUCKETS"] = str(int(num_buckets))
    os.environ["POKERBOT_PREFLOP_LOSSLESS"] = "1"
    os.environ["POKERBOT_POSTFLOP_MODE"] = "emd"
    os.environ["POKERBOT_POSTFLOP_EMD_FROZEN"] = str(Path(frozen_path).resolve())
    os.environ.setdefault("POKERBOT_DEPTH_BUCKETS", "0")


def _set_fast_env(num_buckets: int) -> None:
    """Bake the made-hand serve abstraction (iter62000's training config) BEFORE importing pokerbot."""
    os.environ["POKERBOT_BETTING_ABSTRACTION"] = "1"
    os.environ["POKERBOT_POSTFLOP_BUCKETS"] = "1"
    os.environ["POKERBOT_POSTFLOP_NUM_BUCKETS"] = str(int(num_buckets))
    os.environ["POKERBOT_PREFLOP_LOSSLESS"] = "1"
    os.environ["POKERBOT_POSTFLOP_MODE"] = "fast"
    os.environ.pop("POKERBOT_POSTFLOP_EMD_FROZEN", None)
    os.environ.setdefault("POKERBOT_DEPTH_BUCKETS", "0")


def _load_snapshot(path: str) -> dict:
    """Return {bytes16 -> ndarray(11)} from a .pkl (raw dict OR a stamped {strat_sum,abstraction}) or a
    .nlcfr native checkpoint. s3:// is fetched first."""
    import numpy as np  # noqa: PLC0415

    local = path
    if path.startswith("s3://"):
        import tempfile  # noqa: PLC0415

        dst = Path(tempfile.gettempdir()) / Path(path.split("?", 1)[0]).name
        print(f"  [load] fetching {path} -> {dst}", flush=True)
        _download_s3(path, dst)
        local = str(dst)

    lp = Path(local)
    if not lp.exists():
        raise FileNotFoundError(f"snapshot not found: {lp}")
    if lp.suffix.lower() == ".nlcfr":
        import battery_native as bn  # noqa: PLC0415

        nb = int(os.environ.get("POKERBOT_POSTFLOP_NUM_BUCKETS", "200"))
        t = bn.NlheCFRTrainer(nb, [], 1)
        with open(lp, "rb") as f:
            if not t.load_bytes(f.read()):
                raise ValueError(f"NlheCFRTrainer.load_bytes False for {lp} (corrupt/foreign)")
        snap = t.snapshot()
    else:
        with open(lp, "rb") as f:
            obj = pickle.load(f)
        snap = obj["strat_sum"] if (isinstance(obj, dict) and "strat_sum" in obj) else obj
    return {bytes(k): np.asarray(v, dtype=np.float64) for k, v in snap.items()}


def _native_available() -> bool:
    try:
        from pokerbot.eval import lbr_native as L  # noqa: PLC0415

        ok = bool(L.available())
        if os.environ.get("POKERBOT_NATIVE_DEBUG"):
            ext_file = None
            ext_has_lbr6 = None
            try:
                ext = L.native_solver._try_load()
                ext_file = getattr(ext, "__file__", None) if ext is not None else None
                ext_has_lbr6 = bool(ext is not None and hasattr(ext, "Lbr6Harness"))
            except Exception as _diag_e:  # noqa: BLE001
                ext_file = f"<diag failed: {_diag_e!r}>"
            print(
                f"[eval] _native_available() = {ok} "
                f"ext={ext_file} has_Lbr6Harness={ext_has_lbr6}",
                flush=True,
            )
        return ok
    except Exception as _e:  # noqa: BLE001
        if os.environ.get("POKERBOT_NATIVE_DEBUG"):
            import traceback as _tb
            print(f"[eval] _native_available() import failed: {_e!r}\n{_tb.format_exc()}", flush=True)
        return False


def _run_one_blueprint(snapshot: dict, *, label: str, seeds: list[int], hands: int, n_seats: int,
                       value_only: bool, use_native: bool, emd_frozen_path: str | None,
                       bet_candidates: tuple[int, ...] | None = None,
                       action_set_meta: dict | None = None,
                       require_native: bool = False) -> dict:
    """Pool the Python 6-max LBR over ``seeds`` and return the pooled mean + the per-seat (CRN-slice)
    breakdown. The seeded JAX deal schedule is env-independent, so a fast-env run and an emd-env run on
    the SAME seeds/hands share the per-hand deals -> the per-seat DELTA is CRN (low-variance)."""
    import numpy as np  # noqa: PLC0415

    from pokerbot.eval.lbr6 import LBR6Config, lbr6_exploitability  # noqa: PLC0415

    policy = None
    method = "native_lbr6_emd_frozen" if use_native and emd_frozen_path else (
        "native_lbr6_madehand" if use_native else "python_lbr6_reference")
    pooled_vals: list[float] = []
    per_seed_per_seat: list[dict[str, float]] = []
    # accumulate per-seat means across seeds (each seat is already a CRN slice within a seed)
    per_seat_acc: dict[int, list[float]] = {s: [] for s in range(n_seats)}
    t0 = time.time()
    for i, seed in enumerate(seeds):
        print(f"  [{label}] SEED_START seed={seed} seed_idx={i + 1}/{len(seeds)} hands={hands}", flush=True)
        cfg_kwargs = {
            "hands": hands,
            "seed": seed,
            "n_seats": n_seats,
            "value_only_raises": value_only,
            "paired": False,
        }
        if bet_candidates is not None:
            cfg_kwargs["bet_candidates"] = tuple(int(a) for a in bet_candidates)
        cfg = LBR6Config(**cfg_kwargs)
        if use_native:
            try:
                from pokerbot.eval import lbr_native as L  # noqa: PLC0415

                r = L.lbr6_exploitability_native(
                    snapshot,
                    cfg,
                    policy="blueprint",
                    emd_frozen_path=emd_frozen_path,
                )
            except AttributeError as e:
                if require_native:
                    raise RuntimeError(f"native LBR required for [{label}], but native seam is missing: {e}") from e
                print(f"  [{label}] native EMD seam missing ({e}); falling back to Python reference",
                      flush=True)
                use_native = False
                method = "python_lbr6_reference"
            except RuntimeError:
                raise

        if not use_native:
            if policy is None:
                from pokerbot.eval.lbr import make_blueprint_policy  # noqa: PLC0415

                policy = make_blueprint_policy(snapshot)
            r = _run_python_lbr6(
                policy,
                cfg,
                progress_fn=_make_hand_progress(label, seed, i + 1, len(seeds), hands, t0),
            )
        pooled_vals.append(r.bb_per_100)
        seed_seats: dict[str, float] = {}
        for s, d in r.detail["per_seat"].items():
            bb100 = float(d["bb_per_100"])
            per_seat_acc.setdefault(int(s), []).append(bb100)
            seed_seats[str(int(s))] = bb100
        per_seed_per_seat.append(seed_seats)
        el = time.time() - t0
        done = i + 1
        rate = done / el if el > 0 else 0.0
        eta = (len(seeds) - done) / rate if rate > 1e-9 else -1
        print(f"  [{label}] PROGRESS seed {seed} ({done}/{len(seeds)}, {100.0*done/len(seeds):.0f}%) "
              f"pooled_so_far={np.mean(pooled_vals):+.1f} bb/100  "
              f"elapsed={el:.0f}s rate={rate:.3f} seed/s eta={eta:.0f}s  HEARTBEAT", flush=True)

    pv = np.asarray(pooled_vals, dtype=np.float64)
    se = float(pv.std(ddof=1) / np.sqrt(pv.size)) if pv.size > 1 else 0.0
    per_seat = {}
    for s in range(n_seats):
        vals = per_seat_acc.get(s, [])
        if vals:
            a = np.asarray(vals, dtype=np.float64)
            sse = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
            per_seat[s] = {"bb_per_100": float(a.mean()), "stderr": sse, "n_seeds": int(a.size)}
    out = {
        "label": label, "infosets": len(snapshot), "seeds": seeds, "hands_per_seed": hands,
        "n_seats": n_seats, "value_only": value_only, "method": method,
        "pooled_bb_per_100": float(pv.mean()),
        "pooled_ci95": [float(pv.mean() - 1.96 * se), float(pv.mean() + 1.96 * se)],
        "per_seed": [float(x) for x in pooled_vals],
        "per_seed_per_seat": per_seed_per_seat,
        "per_seat": {str(k): v for k, v in per_seat.items()},
        "wall_s": time.time() - t0,
    }
    if action_set_meta is not None:
        out["action_set_name"] = action_set_meta.get("action_set_name")
        out["bet_candidate_spec"] = action_set_meta.get("spec")
        out["bet_candidates"] = [int(a) for a in action_set_meta.get("actions", ())]
        out["bet_candidate_mapping"] = action_set_meta.get("mapping", [])
    elif bet_candidates is not None:
        out["bet_candidates"] = [int(a) for a in bet_candidates]
    return out


def _anchor_check(seeds: list[int], hands: int, n_seats: int, *, use_native: bool) -> dict:
    """Run the always-fold calibration anchor (env-independent). The harness must reproduce ~ +125
    bb/100 or the whole eval is untrustworthy. Uses the FULL (fold-equity-ON) config -- the anchor is
    defined for full LBR, independent of value-only used for the real measurement."""
    import numpy as np  # noqa: PLC0415

    from pokerbot.eval.lbr6 import LBR6Config, lbr6_exploitability  # noqa: PLC0415

    # A single seed at a healthy hand count is enough -- always-fold is near-deterministic (tiny CI).
    vals = []
    t0 = time.time()
    anchor_seeds = seeds[: max(1, min(3, len(seeds)))]
    for seed in anchor_seeds:
        print(f"  [anchor] SEED_START seed={seed} seed_idx={len(vals) + 1}/{len(anchor_seeds)} hands={hands}",
              flush=True)
        if use_native:
            from pokerbot.eval import lbr_native as L  # noqa: PLC0415

            r = L.lbr6_exploitability_native(
                None, LBR6Config(hands=hands, seed=seed, n_seats=n_seats), policy="fold")
        else:
            from pokerbot.eval.lbr import always_fold_policy  # noqa: PLC0415

            r = _run_python_lbr6(
                always_fold_policy,
                LBR6Config(hands=hands, seed=seed, n_seats=n_seats),
                progress_fn=_make_hand_progress("anchor", seed, len(vals) + 1, len(anchor_seeds), hands, t0),
            )
        vals.append(r.bb_per_100)
    bb = float(np.mean(vals))
    ok = ANCHOR_LO <= bb <= ANCHOR_HI
    print(f"\n  *** ANCHOR CALIBRATION: always-fold = {bb:+.1f} bb/100  "
          f"(target ~+{ANCHOR_TARGET:.0f}; band [{ANCHOR_LO:.0f},{ANCHOR_HI:.0f}]) -> "
          f"{'OK' if ok else 'OFF -- EVAL UNTRUSTWORTHY, ABORT'}  ({time.time()-t0:.0f}s) ***\n",
          flush=True)
    return {"always_fold_bb_per_100": bb, "band": [ANCHOR_LO, ANCHOR_HI], "anchor_ok": bool(ok),
            "method": "native_lbr6" if use_native else "python_lbr6_reference",
            "anchor_seeds": anchor_seeds, "hands": hands}


def _mode_run(args, *, label: str) -> int:
    """Shared body for --mode emd / --mode fast: anchor + blueprint LBR -> partial JSON."""
    print(f"=== EMD-pipeline PHASE 5 EVAL [{label}] ===", flush=True)
    seeds = [seed_for_replication(i) for i in range(args.seeds)]
    assert len(set(seeds)) == args.seeds, f"CRN reseed produced collisions: {seeds}"
    # CRITICAL: dump seeds.json next to the partial JSON so the babysitter can audit the CRN was applied
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Atomic JSON write (SCRIPTS-EVAL-AUDIT CRIT) — torn-tail-safe.
    write_json_atomic(Path(args.out).with_name("seeds.json"),
                      {"base_seed": BASE_SEED, "seeds": seeds}, indent=2)
    snap = _load_snapshot(args.checkpoint)
    print(f"  [{label}] blueprint: {len(snap):,} infosets  ckpt={args.checkpoint}", flush=True)
    if args.require_native and not args.native:
        raise RuntimeError("--require-native conflicts with --no-native")
    use_native = bool(args.native and _native_available())
    print(f"  [{label}] harness: {'native Rust LBR' if use_native else 'Python reference LBR'}  "
          f"num_buckets={args.num_buckets}", flush=True)
    if args.require_native and not use_native:
        raise RuntimeError(
            f"--require-native was set for [{label}], but lbr_native.available() is false. "
            "Set POKERBOT_NATIVE_DEBUG=1 for import/symbol diagnostics."
        )
    if not use_native and args.native:
        import sys as _sys_native_diag
        _sys_native_diag.stderr.write(
            f"[eval] WARNING: --native requested for [{label}] but lbr_native.available()=False -- "
            f"falling back to Python reference. Set POKERBOT_NATIVE_DEBUG=1 for the underlying error.\n")
        _sys_native_diag.stderr.flush()
    action_set_meta = resolve_bet_candidates(
        getattr(args, "bet_candidates", None),
        action_set_name=getattr(args, "action_set_name", None),
    )
    print(
        f"  [{label}] action_set={action_set_meta['action_set_name']} "
        f"bet_actions={list(action_set_meta['actions'])} spec={action_set_meta['spec']}",
        flush=True,
    )
    anchor = _anchor_check(seeds, args.hands, args.n_seats, use_native=use_native)
    res = _run_one_blueprint(snap, label=label, seeds=seeds, hands=args.hands, n_seats=args.n_seats,
                             value_only=args.value_only, use_native=use_native,
                             emd_frozen_path=getattr(args, "emd_frozen", None),
                             bet_candidates=action_set_meta["actions"],
                             action_set_meta=action_set_meta,
                             require_native=args.require_native)
    out = {
        "label": label,
        "action_set_name": action_set_meta["action_set_name"],
        "bet_candidate_spec": action_set_meta["spec"],
        "bet_candidates": [int(a) for a in action_set_meta["actions"]],
        "bet_candidate_mapping": action_set_meta["mapping"],
        "anchor": anchor,
        "result": res,
    }
    # Atomic JSON write (SCRIPTS-EVAL-AUDIT CRIT) — torn-tail-safe.
    write_json_atomic(args.out, out, indent=2)
    per_seat_summary = ", ".join(
        f"s{k}={v['bb_per_100']:+.0f}"
        for k, v in sorted(res["per_seat"].items(), key=lambda kv: int(kv[0]))
    )
    print(f"  [{label}] wrote partial -> {args.out}  pooled={res['pooled_bb_per_100']:+.1f} bb/100  "
          f"per_seat={{ {per_seat_summary} }}",
          flush=True)
    return 0


def _combine(args) -> int:
    """Read the emd + fast partials, compute the per-seat blind deltas + the verdict, write + (opt) upload."""
    with open(args.emd_json, "r", encoding="utf-8") as f:
        emd = json.load(f)
    with open(args.iter62000_json, "r", encoding="utf-8") as f:
        base = json.load(f)

    # The anchor must pass on BOTH harness runs (they are the same harness, env-independent for fold).
    anchor_ok = bool(emd["anchor"]["anchor_ok"] and base["anchor"]["anchor_ok"])

    emd_ps = {int(k): v["bb_per_100"] for k, v in emd["result"]["per_seat"].items()}
    base_ps = {int(k): v["bb_per_100"] for k, v in base["result"]["per_seat"].items()}

    # seat 1 = SB, seat 2 = BB (button_seat=0 fixed -> seat index == button offset). The leak EMD must fix.
    # Per-seat LBR can be negative, so production acceptance is based on
    # exploitability magnitude moving toward zero, not the raw base-emd sign.
    def _magnitude_gain(seat: int) -> float | None:
        if seat in emd_ps and seat in base_ps:
            # POSITIVE gain = improvement (absolute exploitability moved toward zero).
            return abs(base_ps[seat]) - abs(emd_ps[seat])
        return None

    blind_delta_s1 = _magnitude_gain(1)
    blind_delta_s2 = _magnitude_gain(2)
    deltas = [d for d in (blind_delta_s1, blind_delta_s2) if d is not None]
    mean_blind_gain = (sum(deltas) / len(deltas)) if deltas else 0.0

    emd_global = emd["result"]["pooled_bb_per_100"]
    base_global = base["result"]["pooled_bb_per_100"]
    # Positive = EMD is more exploitable globally (absolute magnitude moved away from zero).
    global_regression = abs(emd_global) - abs(base_global)
    k_centroids = int(getattr(args, "num_buckets", 0) or emd["result"].get("num_buckets", 0) or 0)

    blinds_improved = mean_blind_gain >= args.min_blind_gain
    global_ok = global_regression <= args.global_slack
    emd_better = bool(anchor_ok and blinds_improved and global_ok)

    import hashlib as _hashlib
    emd_frozen_for_sha = getattr(args, "emd_frozen", None) or os.environ.get("POKERBOT_POSTFLOP_EMD_FROZEN")
    frozen_blob_p = Path(emd_frozen_for_sha) if emd_frozen_for_sha else None
    frozen_sidecar_p = Path(str(frozen_blob_p) + ".sha256") if frozen_blob_p else None
    frozen_sha_expected = ""
    frozen_sha_actual = ""
    sha256_stamped = False
    if frozen_blob_p and frozen_sidecar_p and frozen_blob_p.exists() and frozen_sidecar_p.exists():
        frozen_sha_expected = frozen_sidecar_p.read_text(encoding="ascii").strip()
        frozen_sha_actual = _hashlib.sha256(frozen_blob_p.read_bytes()).hexdigest()
        sha256_stamped = (frozen_sha_expected == frozen_sha_actual)

    serve_match = False
    serve_match_reason = "serve artifact not provided"
    serve_path = getattr(args, "serve_blueprint", "") or ""
    if serve_path:
        try:
            with open(serve_path, "rb") as f:
                serve_payload = pickle.load(f)
            stamp = serve_payload.get("abstraction", {}) if isinstance(serve_payload, dict) else {}
            expected_table_id = str(getattr(args, "emd_table_id", "") or "")
            expected_s3 = str(getattr(args, "emd_frozen_s3", "") or "")
            serve_sha = str(stamp.get("postflop_emd_frozen_sha256", ""))
            serve_table_id = str(stamp.get("postflop_emd_table_id", ""))
            serve_s3 = str(stamp.get("postflop_emd_frozen_s3", ""))
            serve_match = bool(
                sha256_stamped
                and serve_sha == frozen_sha_actual
                and (not expected_table_id or serve_table_id == expected_table_id)
                and (not expected_s3 or serve_s3 == expected_s3)
            )
            if serve_match:
                serve_match_reason = "ok"
            else:
                serve_match_reason = (
                    "stamp mismatch "
                    f"(serve_sha={serve_sha[:16]} actual_sha={frozen_sha_actual[:16]} "
                    f"serve_table_id={serve_table_id!r} expected_table_id={expected_table_id!r} "
                    f"serve_s3={serve_s3!r} expected_s3={expected_s3!r})"
                )
        except Exception as exc:  # noqa: BLE001
            serve_match = False
            serve_match_reason = f"{type(exc).__name__}: {exc}"

    verdict = {
        "anchor_ok": anchor_ok,
        "emd_anchor_bb_per_100": emd["anchor"]["always_fold_bb_per_100"],
        "iter62000_anchor_bb_per_100": base["anchor"]["always_fold_bb_per_100"],
        "anchor_target": ANCHOR_TARGET,
        "emd_per_seat": {str(k): emd_ps[k] for k in sorted(emd_ps)},
        "iter62000_per_seat": {str(k): base_ps[k] for k in sorted(base_ps)},
        "emd_pooled_bb_per_100": emd_global,
        "iter62000_pooled_bb_per_100": base_global,
        "blind_delta_s1": blind_delta_s1,   # SB magnitude gain; positive = closer to zero
        "blind_delta_s2": blind_delta_s2,   # BB magnitude gain
        "mean_blind_gain": mean_blind_gain,
        "global_regression": global_regression,
        "min_blind_gain_required": args.min_blind_gain,
        "global_slack_allowed": args.global_slack,
        "blinds_improved": blinds_improved,
        "global_ok": global_ok,
        "emd_better": emd_better,
        "seeds": emd["result"]["seeds"],
        "hands_per_seed": emd["result"]["hands_per_seed"],
        "value_only": emd["result"]["value_only"],
        "crn": True,
        "harness": "native_lbr6_emd_frozen" if (
            str(emd["result"].get("method", "")).startswith("native")
            and str(base["result"].get("method", "")).startswith("native")
        ) else "mixed_or_python_lbr6",
        "emd_method": emd["result"].get("method"),
        "iter62000_method": base["result"].get("method"),
        "k_centroids": k_centroids,
        "emd_frozen_path": str(frozen_blob_p) if frozen_blob_p else "",
        "emd_frozen_s3": str(getattr(args, "emd_frozen_s3", "") or ""),
        "emd_frozen_table_id": str(getattr(args, "emd_table_id", "") or ""),
        "emd_frozen_sha256": frozen_sha_actual,
        "emd_frozen_sha256_expected": frozen_sha_expected,
        "emd_frozen_sha256_sidecar": str(frozen_sidecar_p) if frozen_sidecar_p else "",
        "sha256_stamped": sha256_stamped,
        "serve_blueprint_path": str(serve_path),
        "serve_blueprint_s3": str(getattr(args, "serve_blueprint_s3", "") or ""),
        "serve_artifact_match": serve_match,
        "serve_artifact_match_reason": serve_match_reason,
        "note": ("per-seat blind LBR magnitude (CRN slices) is the acceptance metric, NOT the noise-wide pooled CI. "
                 "seat 1=SB, seat 2=BB (button_seat=0 fixed -> seat index == button offset). emd_better "
                 "requires: anchor in band AND both blind magnitudes improve (gain >= min_blind_gain) AND no "
                 "global exploitability-magnitude regression beyond global_slack."),
    }
    # Stamp the sign-correct gate fields so external readers (the watcher /
    # babysitter / downstream dashboards) can verify the verdict is finite
    # and uses the locked delta=base-emd convention before trusting any
    # IMPROVEMENT/REGRESSION label. Both flags default to False; only
    # True when the anchor passed AND every value is finite + convention
    # matches. sign_correct_verified() requires the flag itself to be the
    # literal True, so we temporarily stamp it, run the verification, then
    # overwrite with the actual boolean (preserves the gate semantics).
    verdict["sign_convention"] = SIGN_CONVENTION
    verdict["sign_correct_verified"] = True  # provisional; sign_correct_verified requires this
    verdict["sign_correct_verified"] = bool(anchor_ok and sign_correct_verified(verdict))

    verdict = stamp_primary_action_set(verdict)
    action_sets_json = getattr(args, "action_sets_json", "") or ""
    if action_sets_json:
        with open(action_sets_json, "r", encoding="utf-8") as f:
            extra_sets = json.load(f)
        merged = dict(verdict.get("lbr_action_sets") or {})
        merged.update(extra_sets)
        verdict["lbr_action_sets"] = merged

    allowed_k_centroids = None
    abstraction_sweep = getattr(args, "abstraction_sweep", "") or ""
    expected_k_centroids = int(getattr(args, "expected_k_centroids", 0) or 0)
    if abstraction_sweep:
        with open(abstraction_sweep, "r", encoding="utf-8") as f:
            sweep = json.load(f)
        survivor_ks = accepted_survivor_ks(sweep)
        allowed_k_centroids = (
            {expected_k_centroids}
            if expected_k_centroids and expected_k_centroids in survivor_ks
            else (set() if expected_k_centroids else survivor_ks)
        )
        verdict["abstraction_sweep_path"] = str(abstraction_sweep)
        verdict["accepted_survivor_ks"] = sorted(survivor_ks)
        verdict["expected_k_centroids"] = expected_k_centroids or None
    elif expected_k_centroids:
        verdict["expected_k_centroids"] = expected_k_centroids
        verdict["k_policy_error"] = "expected_k_requires_abstraction_sweep"

    print("\n=== EMD-vs-iter62000 VERDICT (per-seat blind LBR, CRN) ===", flush=True)
    print(f"  anchor: emd={verdict['emd_anchor_bb_per_100']:+.1f} "
          f"iter62000={verdict['iter62000_anchor_bb_per_100']:+.1f}  "
          f"(target ~+{ANCHOR_TARGET:.0f}) -> anchor_ok={anchor_ok}", flush=True)
    if not anchor_ok:
        print("  !! ANCHOR OFF -> eval UNTRUSTWORTHY; emd_better forced False (never ship on this).",
              flush=True)
    # Sign-correct per-seat rendering delegated to scripts/_emd_verdict_sign.py
    # (single source of truth -- prevents the literal 'improvement' sign-flip bug
    # that previously labeled the k500 SB regression as an improvement).
    for seat in range(6):
        if seat in emd_ps and seat in base_ps:
            label = "SB" if seat == 1 else "BB" if seat == 2 else "BTN" if seat == 0 else f"S{seat}"
            print("  " + format_seat_delta(f"{label}(s{seat})", base_ps[seat], emd_ps[seat]),
                  flush=True)
    print("  " + format_pooled(base_global, emd_global), flush=True)
    print(f"  mean blind magnitude gain = {mean_blind_gain:+.1f} bb/100  (need >= {args.min_blind_gain:.0f}) -> "
          f"blinds_improved={blinds_improved}", flush=True)
    print(f"  global: iter62000={base_global:+.1f} -> emd={emd_global:+.1f}  "
          f"(magnitude regression {global_regression:+.1f}, allowed <= {args.global_slack:.0f}) -> global_ok={global_ok}",
          flush=True)
    print(f"  ===> emd_better = {emd_better}", flush=True)

    # Atomic JSON write (SCRIPTS-EVAL-AUDIT CRIT) — torn-tail-safe.
    write_json_atomic(args.out, verdict, indent=2)
    print(f"\n  wrote verdict -> {args.out}", flush=True)

    selection_tier = getattr(args, "selection_tier", "cheap") or "cheap"
    selection_report = build_selection_report(
        verdict,
        tier=selection_tier,
        emd_partial=emd,
        base_partial=base,
        allowed_k_centroids=allowed_k_centroids,
    )
    selection_out = Path(str(args.out).replace(".json", "_selection_report.json"))
    if str(selection_out) == str(args.out):
        selection_out = Path(str(args.out) + ".selection_report.json")
    write_json_atomic(selection_out, selection_report, indent=2)
    print(
        f"  selection_pass={selection_report['selection_pass']} "
        f"acceptance_pass={selection_report['acceptance_pass']} "
        f"tier={selection_tier} -> {selection_out}",
        flush=True,
    )

    print(f"  sha256_stamped={sha256_stamped}  (EMD frozen blob sidecar check)", flush=True)
    print(f"  serve_artifact_match={serve_match}  ({serve_match_reason})", flush=True)

    if args.s3_out:
        _upload_s3(Path(args.out), args.s3_out)
        print(f"  uploaded verdict -> {args.s3_out}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", required=True, help=".pkl/.nlcfr/s3:// blueprint snapshot to LBR")
    common.add_argument("--out", required=True, help="partial result JSON path")
    common.add_argument("--seeds", type=int, default=12, help="number of seeds to pool (CRN across modes)")
    common.add_argument("--hands", type=int, default=2000, help="hands per seed")
    common.add_argument("--n-seats", type=int, default=6)
    common.add_argument("--num-buckets", type=int, default=200,
                        help="postflop bucket count for this checkpoint (EMD K for emd mode; NB=200 for iter62000)")
    common.add_argument("--native", dest="native", action="store_true", default=True,
                        help="use native Rust LBR when battery_native is available (default)")
    common.add_argument("--no-native", dest="native", action="store_false",
                        help="force the Python reference LBR")
    common.add_argument("--require-native", action="store_true",
                        help="fail immediately instead of falling back when native Rust LBR is unavailable")
    common.add_argument("--value-only", action="store_true", default=True,
                        help="value-only raise valuation (default ON; lower-variance, conservative)")
    common.add_argument("--no-value-only", dest="value_only", action="store_false",
                        help="fold-equity-ON raise valuation (the +41.5 strongest-adversary arm)")
    common.add_argument(
        "--action-set-name",
        default="in_tree",
        choices=tuple(ACTION_SET_BET_PRESETS),
        help="name stamped into the partial; also selects the default --bet-candidates preset",
    )
    common.add_argument(
        "--bet-candidates",
        default="",
        help=(
            "comma-separated LBR bet menu. Tokens are pot fractions, all_in, action:N, "
            "or a preset name: in_tree, pot_fraction, off_tree_adversarial"
        ),
    )

    pe = sub.add_parser("emd", parents=[common], help="LBR the EMD blueprint (mode=emd env)")
    pe.add_argument("--emd-frozen", required=True, help="the frozen (hole,board)->bucket lookup the EMD bp trained on")

    sub.add_parser("fast", parents=[common], help="LBR iter62000 (made-hand mode=fast env)")

    pc = sub.add_parser("combine", help="combine the emd+fast partials into the verdict JSON")
    pc.add_argument("--emd-json", required=True)
    pc.add_argument("--iter62000-json", required=True)
    pc.add_argument("--out", required=True, help="verdict JSON path")
    pc.add_argument("--s3-out", default=None, help="optional s3:// destination to upload the verdict to")
    pc.add_argument("--num-buckets", type=int, default=0, help="EMD K used by the evaluated blueprint")
    pc.add_argument("--emd-frozen", default=None, help="frozen EMD lookup path for SHA256 sidecar verification")
    pc.add_argument("--emd-frozen-s3", default="", help="exact s3:// frozen EMD lookup key evaluated")
    pc.add_argument("--emd-table-id", default="", help="deterministic frozen table id used by train/serve")
    pc.add_argument("--serve-blueprint", default="", help="local staged serve blueprint .pkl to compare frozen identity")
    pc.add_argument("--serve-blueprint-s3", default="", help="exact s3:// staged serve blueprint key")
    pc.add_argument("--min-blind-gain", type=float, default=DEFAULT_MIN_BLIND_GAIN,
                    help="bb/100 mean blind (s1,s2) improvement required for emd_better")
    pc.add_argument("--global-slack", type=float, default=DEFAULT_GLOBAL_SLACK,
                    help="bb/100 the EMD pooled mean may be worse than iter62000 by")
    pc.add_argument(
        "--selection-tier",
        default="cheap",
        choices=("heartbeat", "cheap", "medium", "final"),
        help="tier for selection_report.json (looser than activation gate)",
    )
    pc.add_argument(
        "--action-sets-json",
        default="",
        help="optional JSON with pot_fraction/off_tree_adversarial lbr_action_sets partials",
    )
    pc.add_argument(
        "--abstraction-sweep",
        default="",
        help="optional abstraction_sweep.json; restricts selection K to accepted survivors",
    )
    pc.add_argument(
        "--expected-k-centroids",
        type=int,
        default=0,
        help="optional exact candidate K; with --abstraction-sweep it must be an accepted survivor",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "emd":
        _set_emd_env(args.emd_frozen, args.num_buckets)
        return _mode_run(args, label="emd")
    if args.mode == "fast":
        _set_fast_env(args.num_buckets)
        return _mode_run(args, label="iter62000")
    if args.mode == "combine":
        return _combine(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
