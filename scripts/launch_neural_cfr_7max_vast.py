#!/usr/bin/env python3
"""Launch the 7-max neural-CFR RESUME training on a vast.ai INTERRUPTIBLE GPU box, on a $71 budget.

This is the vast.ai sibling of scripts/launch_neural_cfr_spot.py (which is AWS/CPU and 6-max-only).
It mirrors that launcher's conventions -- S3 prefixes under pokerbot/prod6000/*, an allow-list code
tarball, an onstart provisioning script, checkpoint/resume via S3 -- but swaps the EC2-isms for the
vast.ai model (Docker image instead of AMI, bid instead of spot, injected S3 keys instead of an IAM
role, self-destroy instead of shutdown-terminate).

WHAT IT DOES
  1. (optional) builds + uploads an allow-list code tarball to s3://{bucket}/{prefix}/code/ (the
     "s3" repo transport). Skip with --repo-transport git to clone a private repo on-box instead.
  2. queries `vastai search offers` for interruptible single-GPU >=24GB offers under a price ceiling,
     sorted by DLPerf/$/hr, and builds a capacity-diverse fallback chain (vast_offers.chain).
  3. computes the budget-capped max runtime: MAX_RUNTIME_SEC = floor($71 / bid * 3600), so the box
     auto-destroys before the budget is spent (the onstart budget watchdog enforces it).
  4. creates the instance on the best offer (iterating the chain on capacity failure) with the chosen
     Docker image + the onstart script + all env (S3 creds, run params) injected via --env.
  5. prints the instance id + ssh + log-tail + destroy commands, and the S3 run location.

SECRETS ARE ENV-DRIVEN (nothing baked): export before running --
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   scoped to s3:Get/Put/List on the bucket prefix
  VAST_API_KEY                                (or run `vastai set api-key` once)
  GITHUB_TOKEN + GIT_REPO_URL                 only for --repo-transport git
The launcher passes them to the box via `vastai create --env -e KEY=VAL`; they never hit git.

BUDGET MATH (RECON): RTX 4090 interruptible ~$0.29-0.35/hr -> $71 buys ~200-245 GPU-h. At 711 ms/iter
that is ~1.0-1.2M additional iters -- far beyond a useful extension; the LBR gate (or --max-budget)
stops it first.
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "vast"))
import vast_offers  # noqa: E402  (local module next to this launcher)

DEFAULT_BUCKET = "video-processing-6789"
DEFAULT_PREFIX = "pokerbot/prod6000/neural_cfr/vast_7max"
DEFAULT_REGION = "eu-central-1"
DEFAULT_IMAGE = "nvcr.io/nvidia/pytorch:25.04-py3"  # CUDA12 + python; do NOT pip a 2nd torch
DEFAULT_BUDGET = 71.0

# Allow-list (mirror launch_neural_cfr_spot.py _INCLUDE): the training import chain + the tournament
# eval suite (the gate runs it on-box) + the vast scripts + the WEVA index the trainer may gate on.
_INCLUDE = [
    "pokerbot",
    "scripts/train_neural_cfr.py",
    "scripts/neural_cfr",
    "scripts/tournament",
    "scripts/vast",
    "data/weva",
    "requirements.txt",
]


def _tar_filter(ti: tarfile.TarInfo):
    parts = ti.name.split("/")
    if any(p in ("__pycache__", ".git", "target", ".pytest_cache") for p in parts):
        return None
    if ti.name.endswith((".pyc", ".pyo")):
        return None
    return ti


def build_and_upload_code(bucket: str, prefix: str, region: str) -> str:
    """Tarball the allow-list and upload to S3 (boto3). Returns the s3 key."""
    import boto3  # imported lazily so `git` transport users need no boto3

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in _INCLUDE:
            p = ROOT / rel
            if p.exists():
                tar.add(p, arcname=rel, filter=_tar_filter)
            else:
                print(f"  (skip absent {rel})", flush=True)
    buf.seek(0)
    key = f"{prefix}/code/pokerbot_code.tar.gz"
    s3 = boto3.client("s3", region_name=region)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"Uploaded code tarball -> s3://{bucket}/{key} ({buf.tell()} bytes)", flush=True)
    return key


def maybe_seed_checkpoints(bucket: str, prefix: str, region: str) -> bool:
    """Upload the local 7-max checkpoint pickles to S3 if S3 has none yet (warm-resume seed).

    Returns True if a usable _full.pkl is present in S3 after this call."""
    import boto3

    s3 = boto3.client("s3", region_name=region)
    ckpt_prefix = f"{prefix}/checkpoints/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=ckpt_prefix)
    keys = {o["Key"].split("/")[-1] for o in resp.get("Contents", [])}
    if "seven_v1_full.pkl" in keys:
        print(f"S3 already has a resumable checkpoint at s3://{bucket}/{ckpt_prefix}", flush=True)
        return True
    local_dir = ROOT / "checkpoints" / "sd_cfr_7max_neural"
    seeded = False
    for name in ("seven_v1.pkl", "seven_v1_full.pkl", "seven_v1_buffers.pkl"):
        p = local_dir / name
        if p.exists():
            print(f"Seeding s3://{bucket}/{ckpt_prefix}{name} ({p.stat().st_size} bytes)...", flush=True)
            s3.upload_file(str(p), bucket, ckpt_prefix + name)
            seeded = True
    if not seeded:
        print("WARNING: no local 7-max checkpoint to seed AND none in S3 -- the box will start COLD.",
              flush=True)
    return seeded and "seven_v1_full.pkl" in {p.name for p in local_dir.glob("seven_v1_full.pkl")}


def read_onstart() -> str:
    return (ROOT / "scripts" / "vast" / "onstart_7max.sh").read_text(encoding="utf-8")


# Env var names that MUST be redacted from any stdout/stderr capture.
# Adding a secret here is the LAST line of defense; the primary fix is
# build_env_flags() writing them to a 0600 env file (not argv).
_SECRET_ENV_NAMES = frozenset(
    {
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "VAST_API_KEY",
        "GITHUB_TOKEN",
    }
)


def _redact_secrets(text: str) -> str:
    """Replace values of any known-secret env var with '***REDACTED***'."""
    if not text:
        return text
    for name in _SECRET_ENV_NAMES:
        v = os.environ.get(name)
        if not v:
            continue
        # Be greedy: substitute any occurrence of the literal value
        text = text.replace(v, "***REDACTED***")
    return text


def build_env_pairs(args, run_prefix: str) -> list[tuple[str, str]]:
    """Build the env pairs (KEY, VALUE) to inject into the box.

    Secrets come from THE OPERATOR'S ENV only. Values are NEVER put on argv
    (R4 CRITICAL 2): they go to a 0600 env file consumed by `vastai create
    --env -`. Non-secret config goes to argv `--env` flags.
    """
    pairs: list[tuple[str, str]] = []

    def need(k: str) -> str:
        v = os.environ.get(k)
        if not v:
            raise SystemExit(f"FATAL: env var {k} is required but not set (export it before launching).")
        return v

    # Secrets (written to 0600 env file, never argv):
    pairs.append(("AWS_ACCESS_KEY_ID", need("AWS_ACCESS_KEY_ID")))
    pairs.append(("AWS_SECRET_ACCESS_KEY", need("AWS_SECRET_ACCESS_KEY")))
    if os.environ.get("AWS_SESSION_TOKEN"):
        pairs.append(("AWS_SESSION_TOKEN", os.environ["AWS_SESSION_TOKEN"]))
    if args.repo_transport == "git":
        pairs.append(("GIT_REPO_URL", need("GIT_REPO_URL")))
        if os.environ.get("GITHUB_TOKEN"):
            pairs.append(("GITHUB_TOKEN", os.environ["GITHUB_TOKEN"]))
    if os.environ.get("VAST_API_KEY"):
        pairs.append(("VAST_API_KEY", os.environ["VAST_API_KEY"]))

    # Non-secret config (argv-safe, exposed in `ps` on the launcher host):
    pairs.append(("AWS_DEFAULT_REGION", args.region))
    pairs.append(("S3_BUCKET", args.bucket))
    pairs.append(("S3_PREFIX", run_prefix))
    pairs.append(("REPO_TRANSPORT", args.repo_transport))
    pairs.append(("GIT_BRANCH", os.environ.get("GIT_BRANCH", "aof-exploit-engine")))
    pairs.append(("ADDITIONAL_ITERS", str(args.iters)))
    pairs.append(("PUSH_EVERY_SEC", str(args.push_every)))
    pairs.append(("GATE_EVERY_SEC", str(args.gate_every)))
    pairs.append(("GATE_TIER", str(args.gate_tier)))
    pairs.append(("MAX_RUNTIME_SEC", str(args.max_runtime_sec)))
    pairs.append(("AVG_STEPS", str(args.avg_steps)))
    pairs.append(("HIST_STEPS", str(args.hist_steps)))
    pairs.append(("DEPTH_BB", args.depth_bb))
    pairs.append(("LR", args.lr))
    return pairs


def _write_env_file(pairs: list[tuple[str, str]], *, path: Path) -> None:
    """Write the env file with 0600 perms on POSIX; on Windows chmod is a no-op
    but the file is still scoped to the current user via the standard ACL.
    Format: KEY=VALUE per line, no quoting needed (vastai treats each line as
    a raw KEY=VALUE token)."""
    lines = [f"{k}={v}" for k, v in pairs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows: chmod is a partial no-op for bits not in the standard ACL.
        # The file is still created in the user's profile with default ACL.
        pass


def create_instance(offer: vast_offers.Offer, *, image: str, disk: int, bid: float,
                    env_argv: str, env_file: str | None,
                    onstart_path: str, label: str, vastai_bin: str) -> tuple[bool, str]:
    """Attempt `vastai create instance`. Returns (ok, redacted stdout/err).

    `env_argv` = non-secret `--env KEY=VAL ...` flags (argv-safe).
    `env_file` = path to a 0600 file of KEY=VAL lines for secrets (vastai
    supports reading env from a file with `--env -` style, but the precise
    flag is `--envfile`; we fall back to merging if unsupported).
    """
    cmd = [
        vastai_bin, "create", "instance", str(offer.offer_id),
        "--image", image,
        "--disk", str(disk),
        "--bid_price", f"{bid:.4f}",
        "--onstart", onstart_path,
        "--label", label,
        "--ssh", "--direct",
        "--raw",
    ]
    # Non-secret env goes to argv `--env` (preserves prior behavior for
    # the config block; nothing secret is in this list).
    if env_argv:
        cmd += ["--env", env_argv]
    # Secret env goes to a 0600 file (vastai `--envfile`).
    if env_file:
        cmd += ["--envfile", env_file]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    # Last-line defense: redact any secret VALUES that may have leaked into
    # stdout (e.g. via vastai echoing back its received env on error).
    out = _redact_secrets(out)
    return proc.returncode == 0, out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # offer selection
    p.add_argument("--gpu-name", default=None, help="pin a GPU model e.g. RTX_4090 (default: any >=24GB)")
    p.add_argument("--gpu-ram-min-mb", type=int, default=24000)
    p.add_argument("--price-ceiling", type=float, default=0.40, help="hard dph_total ceiling for search")
    p.add_argument("--reliability-min", type=float, default=0.90)
    p.add_argument("--min-inet-up", type=float, default=300.0, help="Mbps; fast S3 sync (RECON gotcha)")
    p.add_argument("--markup", type=float, default=0.15, help="bid = min_bid * (1+markup)")
    p.add_argument("--chain-n", type=int, default=6, help="capacity-diverse fallback chain length")
    p.add_argument("--limit", type=int, default=40, help="offers to fetch before ranking")
    # budget / runtime
    p.add_argument("--max-budget", type=float, default=DEFAULT_BUDGET, help="USD cap (default 71)")
    p.add_argument("--max-runtime-sec", type=int, default=None,
                   help="override the budget-derived runtime cap (default = budget/bid hours)")
    # box
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--disk", type=int, default=40, help="GB local NVMe (>=40 for image+316MB buffers)")
    # repo + run
    p.add_argument("--repo-transport", choices=["s3", "git"], default="s3")
    p.add_argument("--iters", type=int, default=50000, help="ADDITIONAL iters (start+this; warm resume)")
    p.add_argument("--avg-steps", type=int, default=512, help="served-net SGD steps (free convergence lever)")
    p.add_argument("--hist-steps", type=int, default=64)
    p.add_argument("--depth-bb", default="8,120", help="POKERBOT_7MAX_DEPTH_BB (8,40 = bubble-weighted leg)")
    p.add_argument("--lr", default="1e-4")
    p.add_argument("--push-every", type=int, default=900, help="S3 checkpoint-push cadence (sec)")
    p.add_argument("--gate-every", type=int, default=5400, help="LBR-gate eval cadence (sec)")
    p.add_argument("--gate-tier", type=int, default=1, choices=[1, 2])
    # infra
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--run-subdir", default=None, help="run subdir under --prefix (default run_<ts>)")
    p.add_argument("--vastai-bin", default="vastai")
    p.add_argument("--dry-run", action="store_true", help="rank offers + print the plan; do NOT create")
    p.add_argument("--no-seed-checkpoint", action="store_true",
                   help="do NOT upload the local checkpoint to S3 (assume S3 already seeded)")
    args = p.parse_args()

    run_subdir = args.run_subdir or f"run_{int(time.time())}"
    run_prefix = f"{args.prefix}/{run_subdir}"
    print(f"=== vast.ai 7-max neural-CFR launcher ===  run={run_subdir}", flush=True)
    print(f"S3 run: s3://{args.bucket}/{run_prefix}/", flush=True)

    # 1. rank interruptible offers by DLPerf/$ and build a diverse fallback chain.
    print("Querying vast.ai interruptible offers (sorted DLPerf/$/hr)...", flush=True)
    try:
        offers = vast_offers.search_offers(
            gpu_ram_min_mb=args.gpu_ram_min_mb, price_ceiling=args.price_ceiling,
            reliability_min=args.reliability_min, gpu_name=args.gpu_name,
            min_inet_up=args.min_inet_up, limit=args.limit, vastai_bin=args.vastai_bin,
        )
    except RuntimeError as exc:
        raise SystemExit(f"offer search failed: {exc}\n"
                         f"Is the vastai CLI installed + `vastai set api-key` done? (pip install vastai)")
    ch = vast_offers.chain(offers, n=args.chain_n)
    if not ch:
        raise SystemExit("No matching offers. Loosen --price-ceiling / --reliability-min / --min-inet-up, "
                         "or drop --gpu-name.")
    print(f"Top {len(ch)} capacity-diverse offers (best DLPerf/$ first):", flush=True)
    for i, o in enumerate(ch):
        print(f"  [{i}] {vast_offers.describe(o, args.markup)}", flush=True)

    best = ch[0]
    bid = best.bid_for(args.markup)
    if args.max_runtime_sec is not None:
        max_runtime = args.max_runtime_sec
    else:
        # budget cap: hours = budget / bid; convert to seconds. Storage adds ~$0.01/hr -> bid+0.011.
        eff_rate = bid + 0.011
        max_runtime = int((args.max_budget / eff_rate) * 3600)
    args.max_runtime_sec = max_runtime
    hours = max_runtime / 3600.0
    print(f"\nBudget plan: bid ${bid:.4f}/hr (+~$0.011 storage) -> "
          f"max runtime {hours:.1f}h to stay under ${args.max_budget:.0f}", flush=True)
    print(f"At ~711 ms/iter that is ~{int(hours*3600/0.711):,} additional iters (LBR gate may stop earlier).",
          flush=True)

    if args.dry_run:
        print("\n--dry-run: not creating an instance.", flush=True)
        return

    # 2. repo + checkpoint to S3 (s3 transport seeds both; git transport seeds only the checkpoint).
    if args.repo_transport == "s3":
        print("\nUploading code tarball (s3 transport)...", flush=True)
        build_and_upload_code(args.bucket, run_prefix, args.region)
    if not args.no_seed_checkpoint:
        print("Seeding 7-max checkpoint to S3 (warm resume)...", flush=True)
        maybe_seed_checkpoints(args.bucket, run_prefix, args.region)

    # 3. write the onstart script to a temp path vast can upload (--onstart <file>).
    onstart_path = str(ROOT / "scripts" / "vast" / "onstart_7max.sh")
    # R4 CRITICAL 2: split env into argv-safe (non-secret) and env-file
    # (secrets) halves. The launcher process's argv never contains any
    # AWS_SECRET_ACCESS_KEY / VAST_API_KEY / GITHUB_TOKEN value.
    all_pairs = build_env_pairs(args, run_prefix)
    secret_keys = _SECRET_ENV_NAMES | {"AWS_ACCESS_KEY_ID"}  # access key id is non-secret by AWS design
    secret_pairs = [(k, v) for k, v in all_pairs if k in secret_keys or k in _SECRET_ENV_NAMES]
    public_pairs = [(k, v) for k, v in all_pairs if (k, v) not in secret_pairs]
    env_argv = " ".join(f"-e {k}={v}" for k, v in public_pairs)
    env_file_path: str | None = None
    if secret_pairs:
        env_file_path = str(ROOT / "logs" / f"vast_7max_env_{run_subdir}.env")
        Path(env_file_path).parent.mkdir(parents=True, exist_ok=True)
        _write_env_file(secret_pairs, path=Path(env_file_path))
        print(f"Secrets written to 0600 env file: {env_file_path}", flush=True)

    # 4. create the instance, iterating the chain on capacity failure.
    label = f"pokerbot-7max-{run_subdir}"
    created = None
    for i, o in enumerate(ch):
        ob = o.bid_for(args.markup)
        print(f"\nCreating on offer {o.offer_id} ({o.gpu_name}) bid ${ob:.4f} ...", flush=True)
        ok, out = create_instance(
            o, image=args.image, disk=args.disk, bid=ob, env_argv=env_argv,
            env_file=env_file_path, onstart_path=onstart_path, label=label,
            vastai_bin=args.vastai_bin,
        )
        # out is already redacted; one more pass for paranoia
        print(_redact_secrets(out.strip()), flush=True)
        if ok and ("success" in out.lower() or "new_contract" in out.lower() or '"success": true' in out.lower()):
            created = (o, ob, out)
            break
        print(f"  offer {o.offer_id} failed/declined -> trying next in chain", flush=True)
        time.sleep(2)
    if created is None:
        raise SystemExit("All offers in the chain failed to create. Re-run (supply may have shifted) "
                         "or loosen the search filters.")

    o, ob, out = created
    print("\n=== LAUNCHED ===", flush=True)
    print(f"Offer {o.offer_id} ({o.gpu_name}, {o.gpu_ram_mb/1000:.0f}GB) bid ${ob:.4f}/hr  "
          f"runtime cap {hours:.1f}h  budget ${args.max_budget:.0f}", flush=True)
    print(f"Run S3: s3://{args.bucket}/{run_prefix}/", flush=True)
    print(f"Find instance id:   {args.vastai_bin} show instances", flush=True)
    print(f"Tail logs:          {args.vastai_bin} logs <INSTANCE_ID> --tail 200", flush=True)
    print(f"SSH:                {args.vastai_bin} ssh-url <INSTANCE_ID>", flush=True)
    print(f"Destroy:            {args.vastai_bin} destroy instance <INSTANCE_ID>", flush=True)
    print(f"Babysit (relaunch on preempt):  powershell -ExecutionPolicy Bypass -File "
          f"scripts/neural_cfr_7max_vast_babysitter.ps1 -Run {run_subdir}", flush=True)
    print(f"Watch:              powershell -ExecutionPolicy Bypass -File "
          f"scripts/watch_neural_cfr_7max_vast.ps1 -Run {run_subdir} -MaxBudget {args.max_budget:.0f}",
          flush=True)


if __name__ == "__main__":
    main()
