#!/usr/bin/env python3
"""Eval-based marginal-improvement stop for the EMD blueprint pipeline.

This helper is intentionally separate from ``blueprint_convergence_stop.py``.
The strategy-delta plateau says "the average strategy stopped moving"; the K200
incident showed that this is not the same as "the blueprint is good enough".

Here we measure the same CRN, per-seat blind verdict used by the production
activation gate, at periodic checkpoints. We stop only when that gate-oriented
score has not improved by more than ``--min-improvement-bb`` for ``--k``
evaluated checkpoints after a minimum training floor.

Exit codes:
  0 = CONVERGED  marginal-improvement stop should fire
  1 = CONTINUING evaluated or already evaluated, but not stopped
  2 = UNKNOWN    insufficient checkpoints / below floors / skipped by interval
  3 = ERROR      tool or eval failure
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval" / "eval_emd_vs_iter62000.py"
_ITER_RE = re.compile(r"/iter_(\d+)\.nlcfr$")


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _load_state(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return _load_json(p)
    except Exception:
        return {}


def _native_pythonpath() -> str:
    """Prefer the isolated EMD crate so .nlcfr checkpoints can deserialize."""
    candidates = [
        ROOT / "runs" / "bp_crate_emd" / "battery_native" / "target" / "release",
        ROOT / "runs" / "bp_crate_frozen" / "battery_native" / "target" / "release",
        ROOT / "tools" / "battery_native" / "target" / "release",
        ROOT,
    ]
    parts = [str(p) for p in candidates if p.exists()]
    old = os.environ.get("PYTHONPATH")
    if old:
        parts.append(old)
    return os.pathsep.join(parts)


def _run(cmd: list[str], *, env: dict[str, str], log_path: Path) -> None:
    t0 = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode} elapsed={dt:.1f}s; see {log_path}")


def _parse_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 URI: {uri}")
    body = uri[5:]
    bucket, sep, key = body.partition("/")
    if not bucket or not sep or not key:
        raise ValueError(f"bad s3 URI: {uri}")
    return bucket, key


def _download_s3(s3, uri: str, dst: Path) -> Path:
    bucket, key = _parse_s3(uri)
    dst.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(dst))
    return dst


def _list_versioned_checkpoints(s3, bucket: str, run_prefix: str) -> list[tuple[int, str]]:
    prefix = run_prefix.rstrip("/") + "/checkpoints/"
    out: list[tuple[int, str]] = []
    token = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = str(obj.get("Key", ""))
            m = _ITER_RE.search("/" + key if not key.startswith("/") else key)
            if m:
                out.append((int(m.group(1)), key))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    out.sort(key=lambda x: x[0])
    return out


def _find_frozen_s3(s3, bucket: str, run_prefix: str) -> str:
    prefix = run_prefix.rstrip("/") + "/frozen/"
    best_key = ""
    best_size = -1
    token = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = str(obj.get("Key", ""))
            if key.endswith(".emdfrzn"):
                size = int(obj.get("Size", 0))
                if size > best_size:
                    best_key = key
                    best_size = size
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    if not best_key:
        raise FileNotFoundError(f"no .emdfrzn under s3://{bucket}/{prefix}")
    return f"s3://{bucket}/{best_key}"


def _score_verdict(v: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    """Gate-oriented scalar score; higher is better.

    We score relative to the production acceptance thresholds. The minimum margin
    is deliberately strict: if any critical dimension stops improving, the
    checkpoint is not treated as meaningful progress.
    """
    emd_ps = v.get("emd_per_seat", {}) or {}
    base_ps = v.get("iter62000_per_seat", {}) or {}

    def _seat_gain(seat: int) -> float:
        sk = str(seat)
        if sk in emd_ps and sk in base_ps:
            return abs(float(base_ps[sk])) - abs(float(emd_ps[sk]))
        if seat in emd_ps and seat in base_ps:
            return abs(float(base_ps[seat])) - abs(float(emd_ps[seat]))
        return float("nan")

    sb = _seat_gain(1)
    bb = _seat_gain(2)
    mean_blind = (sb + bb) / 2.0
    emd_global = v.get("emd_pooled_bb_per_100")
    base_global = v.get("iter62000_pooled_bb_per_100")
    if emd_global is not None and base_global is not None:
        global_reg = abs(float(emd_global)) - abs(float(base_global))
    else:
        global_reg = float("nan")
    global_margin = float(args.global_reg_max) - global_reg
    gate_margin = min(
        sb - float(args.sb_min),
        bb - float(args.bb_min),
        mean_blind - float(args.blind_mean_min),
        global_margin,
    )
    # A smoother companion is useful in reports, but the stop decision uses
    # gate_margin so one blind cannot hide behind a pooled/global improvement.
    blended = min(sb, bb, mean_blind) + 0.25 * (-global_reg)
    return {
        "gate_margin": gate_margin,
        "blended_score": blended,
        "sb_gain": sb,
        "bb_gain": bb,
        "mean_blind_gain": mean_blind,
        "global_gain": -global_reg,
    }


def _params_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seeds": int(args.seeds),
        "hands": int(args.hands),
        "num_buckets": int(args.num_buckets),
        "min_improvement_bb": float(args.min_improvement_bb),
        "k": int(args.k),
        "score": "gate_margin_v1",
        "sb_min": float(args.sb_min),
        "bb_min": float(args.bb_min),
        "blind_mean_min": float(args.blind_mean_min),
        "global_reg_max": float(args.global_reg_max),
    }


def evaluate(args: argparse.Namespace) -> int:
    import boto3  # local import keeps --help dependency-light

    s3 = boto3.client("s3", region_name=args.region)
    run_prefix = args.run_prefix.rstrip("/")
    ckpts = _list_versioned_checkpoints(s3, args.bucket, run_prefix)
    if not ckpts:
        print("STATUS=UNKNOWN reason=no-versioned-checkpoints")
        return 2
    latest_iter, latest_key = ckpts[-1]
    if latest_iter < args.min_iters:
        print(
            f"STATUS=UNKNOWN reason=below-min-iters latest_iter={latest_iter} "
            f"min_iters={args.min_iters}"
        )
        return 2

    state_path = Path(args.state_file)
    state = _load_state(state_path)
    params = _params_signature(args)
    if state.get("params") != params:
        state = {"params": params}

    last_eval_iter = int(state.get("last_eval_iter", -1))
    if last_eval_iter == latest_iter:
        streak = int(state.get("marginal_streak", 0))
        status = "CONVERGED" if streak >= args.k else "CONTINUING"
        print(
            f"STATUS={status} reason=already-evaluated latest_iter={latest_iter} "
            f"score={state.get('last_score')} best={state.get('best_score')} "
            f"streak={streak}/{args.k}"
        )
        return 0 if status == "CONVERGED" else 1
    if last_eval_iter >= 0 and (latest_iter - last_eval_iter) < args.eval_every_iters:
        print(
            f"STATUS=UNKNOWN reason=eval-interval latest_iter={latest_iter} "
            f"last_eval_iter={last_eval_iter} every={args.eval_every_iters}"
        )
        return 2

    run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_prefix.split("/")[-1])
    work = Path(args.workdir) if args.workdir else Path(tempfile.gettempdir()) / f"emd_eval_conv_{run_id}"
    work.mkdir(parents=True, exist_ok=True)
    frozen_s3 = args.emd_frozen_s3 or _find_frozen_s3(s3, args.bucket, run_prefix)
    frozen_local = work / Path(_parse_s3(frozen_s3)[1]).name
    if not frozen_local.exists() or frozen_local.stat().st_size == 0:
        _download_s3(s3, frozen_s3, frozen_local)
    # Sidecar is not required for a trend probe, but if present keep it beside
    # the blob so combine can stamp sha256_stamped in diagnostic verdicts.
    try:
        _download_s3(s3, frozen_s3 + ".sha256", Path(str(frozen_local) + ".sha256"))
    except Exception:
        pass

    py = args.python or sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = _native_pythonpath()
    env.setdefault("POKERBOT_NATIVE_DEBUG", "1")
    baseline_json = work / f"iter62000_s{args.seeds}_h{args.hands}.json"
    if not baseline_json.exists() or baseline_json.stat().st_size == 0:
        _run(
            [
                py,
                str(EVAL_SCRIPT),
                "fast",
                "--checkpoint",
                args.iter62000_snapshot_s3,
                "--out",
                str(baseline_json),
                "--seeds",
                str(args.seeds),
                "--hands",
                str(args.hands),
                "--num-buckets",
                "200",
                "--value-only",
                "--native",
                "--require-native",
            ],
            env=env,
            log_path=work / "iter62000.log",
        )

    ckpt_uri = f"s3://{args.bucket}/{latest_key}"
    emd_json = work / f"emd_iter_{latest_iter:08d}.json"
    verdict_json = work / f"verdict_iter_{latest_iter:08d}.json"
    _run(
        [
            py,
            str(EVAL_SCRIPT),
            "emd",
            "--checkpoint",
            ckpt_uri,
            "--emd-frozen",
            str(frozen_local),
            "--out",
            str(emd_json),
            "--seeds",
            str(args.seeds),
            "--hands",
            str(args.hands),
            "--num-buckets",
            str(args.num_buckets),
            "--value-only",
            "--native",
            "--require-native",
        ],
        env=env,
        log_path=work / f"emd_iter_{latest_iter:08d}.log",
    )
    _run(
        [
            py,
            str(EVAL_SCRIPT),
            "combine",
            "--emd-json",
            str(emd_json),
            "--iter62000-json",
            str(baseline_json),
            "--out",
            str(verdict_json),
            "--num-buckets",
            str(args.num_buckets),
            "--emd-frozen",
            str(frozen_local),
        ],
        env=env,
        log_path=work / f"combine_iter_{latest_iter:08d}.log",
    )

    verdict = _load_json(verdict_json)
    if not bool(verdict.get("anchor_ok")):
        raise RuntimeError(f"anchor failed in eval convergence probe: {verdict_json}")
    scores = _score_verdict(verdict, args)
    score = float(scores["gate_margin"])
    if not math.isfinite(score):
        raise RuntimeError(f"non-finite eval convergence score: {scores}")

    eval_count = int(state.get("eval_count", 0)) + 1
    best_score_raw = state.get("best_score")
    best_score = float(best_score_raw) if best_score_raw is not None else None
    if best_score is None:
        improvement = float("inf")
        best_score = score
        best_iter = latest_iter
        streak = 0
        reason = "first-eval"
    else:
        improvement = score - best_score
        if improvement > args.min_improvement_bb:
            best_score = score
            best_iter = latest_iter
            streak = 0
            reason = "meaningful-improvement"
        else:
            best_iter = int(state.get("best_iter", latest_iter))
            streak = int(state.get("marginal_streak", 0)) + 1
            reason = "marginal-improvement"

    converged = eval_count >= args.min_evals and streak >= args.k
    state.update(
        {
            "params": params,
            "last_eval_iter": latest_iter,
            "last_checkpoint_key": latest_key,
            "last_score": score,
            "last_scores": scores,
            "last_verdict": str(verdict_json),
            "last_improvement_vs_best": improvement if math.isfinite(improvement) else None,
            "best_score": best_score,
            "best_iter": best_iter,
            "marginal_streak": streak,
            "eval_count": eval_count,
            "updated": time.time(),
        }
    )
    _write_json(state_path, state)

    if args.json_out:
        _write_json(args.json_out, state)

    status = "CONVERGED" if converged else "CONTINUING"
    imp_txt = "first" if not math.isfinite(improvement) else f"{improvement:+.2f}"
    print(
        f"STATUS={status} reason={reason} iter={latest_iter} "
        f"score={score:+.2f} best={best_score:+.2f}@{best_iter} "
        f"improvement_vs_best={imp_txt} min_improvement={args.min_improvement_bb:.2f} "
        f"streak={streak}/{args.k} eval_count={eval_count}/{args.min_evals} "
        f"sb={scores['sb_gain']:+.1f} bb={scores['bb_gain']:+.1f} "
        f"blind_mean={scores['mean_blind_gain']:+.1f} global_gain={scores['global_gain']:+.1f} "
        f"verdict={verdict_json}"
    )
    return 0 if converged else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--bucket", default="video-processing-6789")
    p.add_argument("--region", default="eu-central-1")
    p.add_argument("--num-buckets", type=int, default=200)
    p.add_argument("--iter62000-snapshot-s3", required=True)
    p.add_argument("--emd-frozen-s3", default="", help="optional explicit s3://...emdfrzn; otherwise auto-detect")
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--hands", type=int, default=4000)
    p.add_argument("--min-iters", type=int, default=300000)
    p.add_argument("--eval-every-iters", type=int, default=50000)
    p.add_argument("--min-evals", type=int, default=4)
    p.add_argument("--min-improvement-bb", type=float, default=3.0)
    p.add_argument("--k", type=int, default=3, help="consecutive marginal eval probes required to stop")
    p.add_argument("--state-file", required=True)
    p.add_argument("--json-out", default="")
    p.add_argument("--workdir", default="")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--sb-min", type=float, default=15.0)
    p.add_argument("--bb-min", type=float, default=15.0)
    p.add_argument("--blind-mean-min", type=float, default=15.0)
    p.add_argument("--global-reg-max", type=float, default=10.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return evaluate(args)
    except Exception as exc:  # noqa: BLE001
        _eprint(f"ERROR: {type(exc).__name__}: {exc}")
        print(f"STATUS=ERROR reason={type(exc).__name__}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
