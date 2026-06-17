#!/usr/bin/env python3
"""Spot-fleet coordinator for the single-machine SD-CFR convergence run.

The training is single-machine (one box at a time), but spot capacity is reclaimed by AWS at will,
so a long run must survive a SEQUENCE of boxes. This coordinator keeps EXACTLY ONE spot instance
training the run until it reaches the target iteration, automatically replacing any box that is
reclaimed/terminated. Each replacement resumes bit-exactly from the latest S3 checkpoint (the trainer
already does ``latest_checkpoint_iteration`` -> ``checkpoint_from_store``), so no completed iteration
is ever lost — at most the single in-flight iteration of a reclaimed box.

Design / safety:
  * SINGLE WRITER. Only one instance per run is ever allowed to run (two boxes resuming the same
    checkpoint would double-work and clobber checkpoints). Before launching a replacement the
    coordinator terminates any lingering (non-terminated) box for the run.
  * CAPACITY FALLBACK. Spot launches can fail with InsufficientInstanceCapacity / price; the
    coordinator rotates through a candidate instance-type list to find available capacity.
  * QUOTA-LAG RETRY. A just-terminated box holds its spot vCPU quota for a few minutes
    (MaxSpotInstanceCountExceeded); the coordinator waits and retries.
  * STALL DETECTION. A box that runs but produces no new checkpoint for ``--stall-minutes`` is
    assumed wedged (e.g. a thundering-herd hang) and is killed + replaced.
  * BUDGET CAP. A rough $/hr-per-type estimate accumulates running time; the coordinator stops
    launching once ``--max-cost`` is reached (a running box is left to finish its current work).
  * IDEMPOTENT / RESUMABLE. Safe to Ctrl-C and re-run: it adopts the live box (by ``Run`` tag) and
    reads progress from S3. Ctrl-C leaves the box running by default (``--terminate-on-exit`` to kill).

This is NOT a multi-box parallel fleet — the trainer is single-machine; "fleet" here means the
time-sequence of spot boxes that carry one run to completion.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Reuse the (tested) launcher building blocks.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_sd_cfr_training_spot import (  # noqa: E402
    DEFAULT_BUCKET,
    DEFAULT_PREFIX,
    DEFAULT_PROFILE,
    DEFAULT_REGION,
    build_and_upload_code,
    resolve_ubuntu_ami,
    user_data,
)

# Preferred spot types, in fallback order (type, arch, approx $/hr spot eu-central-1, vCPU). The
# bounded-abstraction runs are SMALL-RAM now (saturated tables ~200 MB; the OOM era was the unbounded
# lossless run), so COMPUTE-optimized Graviton (c-class, 2 GB/vCPU) is cheapest+fastest; m/r are RAM
# fallbacks. Broad family/generation coverage so one type being capacity-full never stalls the run:
# acquire_instance() rotates to the next on InsufficientInstanceCapacity / quota lag. 32-vCPU 8xlarge
# first (fits the default 24 workers, stays well under the ~64-vCPU spot quota → room for headroom);
# 48-vCPU 12xlarge for more parallelism; 16xlarge (64 vCPU = whole quota) last.
# FULL 64-vCPU spot quota first (that's the point of the cloud — max parallelism), dropping to 48-
# then 32-vCPU only when no 64-vCPU capacity is available. Within a size, cheapest compute-optimized
# Graviton first; m/r are RAM fallbacks. Worker count auto-clamps to vCPU-2 per box (try_launch).
# (type, arch, ~$/hr spot, vCPU, GB). FULL 64-vCPU spot quota first — but ~2 GB/worker means 60
# workers needs ≥256 GB (a 128 GB box OOMs on the compile spike), so the RAM-heavy 64-vCPU boxes lead
# the tier; 128 GB 64-vCPU boxes follow (workers auto-clamp down by RAM in try_launch). Then 48- and
# 32-vCPU as capacity fallbacks. acquire_instance() rotates on capacity/quota failure.
CANDIDATES: list[tuple[str, str, float, int, int]] = [
    ("m7g.16xlarge", "arm64", 1.30, 64, 256),   # 64 vCPU — holds 60 workers safely
    ("r7g.16xlarge", "arm64", 1.45, 64, 512),   # 64 vCPU — most RAM
    ("c7g.16xlarge", "arm64", 1.10, 64, 128),   # 64 vCPU — cheaper, workers RAM-clamped (~48)
    ("c6g.16xlarge", "arm64", 0.95, 64, 128),   # 64 vCPU — prev gen
    ("m7g.12xlarge", "arm64", 0.93, 48, 192),   # 48 vCPU fallback
    ("c7g.12xlarge", "arm64", 0.83, 48, 96),    # 48 vCPU
    ("c6g.12xlarge", "arm64", 0.70, 48, 96),    # 48 vCPU
    ("c7g.8xlarge", "arm64", 0.55, 32, 64),     # 32 vCPU last-resort
    ("r7g.8xlarge", "arm64", 0.62, 32, 256),    # 32 vCPU
]
_PRICE = {t: p for t, _a, p, _v, _g in CANDIDATES}
_VCPU = {t: v for t, _a, _p, v, _g in CANDIDATES}
_RAM_GB = {t: g for t, _a, _p, _v, g in CANDIDATES}

_CAPACITY_ERRORS = (
    "InsufficientInstanceCapacity",
    "SpotMaxPriceTooLow",
    "Unsupported",
    "InsufficientHostCapacity",
)


def latest_ckpt_iter(s3, bucket: str, run_prefix: str) -> int:
    """Highest completed iteration N from ``.../iter_NNNN/`` checkpoint dirs in S3 (0 if none)."""
    best = 0
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=f"{run_prefix}/")
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            rel = o["Key"][len(run_prefix) + 1 :]
            if rel.startswith("iter_"):
                frag = rel.split("/")[0].replace("iter_", "")
                if frag.isdigit():
                    best = max(best, int(frag))
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return best


def run_instances_for(ec2, run_subdir: str, states: tuple[str, ...]):
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Run", "Values": [run_subdir]},
            {"Name": "instance-state-name", "Values": list(states)},
        ]
    )
    out = []
    for r in resp["Reservations"]:
        out.extend(r["Instances"])
    return out


def terminate_all(ec2, instances) -> None:
    ids = [i["InstanceId"] for i in instances]
    if ids:
        ec2.terminate_instances(InstanceIds=ids)
        print(f"  terminated stragglers: {ids}", flush=True)


def try_launch(ec2, ami_by_arch: dict, code_key: str, *, cfg, instance_type: str, arch: str) -> str | None:
    """Attempt one spot RunInstances; return instance id, or None on capacity/quota failure."""
    ami = ami_by_arch.get(arch) or resolve_ubuntu_ami(ec2, arch)
    ami_by_arch[arch] = ami
    wheels_dir = "wheels_py310_x86" if arch == "x86_64" else "wheels_py310_arm64"
    ud = user_data(
        # Clamp workers to the chosen box's vCPUs (leave 2 for the main proc/OS) so a smaller fallback
        # type never oversubscribes; the pool stays bit-identical regardless of worker count.
        bucket=cfg["bucket"], prefix=cfg["prefix"], region=cfg["region"], code_key=code_key,
        run_subdir=cfg["run_subdir"],
        # Clamp to BOTH (vCPU - 2) and a ~2 GB/worker RAM budget (reserve 16 GB for parent+OS), so a
        # 128 GB 64-vCPU box runs ~56 workers (not 60→OOM) while a 256 GB box runs the full 60.
        workers=min(
            cfg["workers"],
            max(1, _VCPU.get(instance_type, cfg["workers"]) - 2),
            max(1, (_RAM_GB.get(instance_type, 256) - 16) // 2),
        ),
        iterations=cfg["iterations"],
        traversals=cfg["traversals"], deal_sweeps=cfg["deal_sweeps"], max_depth=cfg["max_depth"],
        no_sft=False, wheels_dir=wheels_dir, keep_alive=False, bet_buckets=cfg.get("bet_buckets"),
        postflop_buckets=cfg.get("postflop_buckets"), betting_abstraction=cfg.get("betting_abstraction", False),
        client_bet_sizing=cfg.get("client_bet_sizing", False),
        trainer=cfg.get("trainer", "pool"), hogwild_capacity=cfg.get("hogwild_capacity"), arch=arch,
        occupancy=cfg.get("occupancy", "hu"),
        cfr_variant=cfg.get("cfr_variant", "cfr_plus"), vr_baseline=cfg.get("vr_baseline", False),
    )
    run_kwargs = dict(
        ImageId=ami,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": cfg["profile"]},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"sd-cfr-hu-train-{cfg['run_subdir']}"},
                {"Key": "Project", "Value": "pokerbot-sd-cfr"},
                {"Key": "Run", "Value": cfg["run_subdir"]},
            ],
        }],
        InstanceInitiatedShutdownBehavior="terminate",
        InstanceMarketOptions={
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"},
        },
    )
    try:
        resp = ec2.run_instances(**run_kwargs)
        return resp["Instances"][0]["InstanceId"]
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "MaxSpotInstanceCountExceeded":
            print(f"  [{instance_type}] quota lag (MaxSpotInstanceCountExceeded)", flush=True)
            return None
        if code in _CAPACITY_ERRORS:
            print(f"  [{instance_type}] no spot capacity ({code})", flush=True)
            return None
        raise


def acquire_instance(ec2, ami_by_arch, code_key, *, cfg, deadline_cost: float, spent: float) -> str | None:
    """Rotate candidate types until one launches; ride out quota lag. Return iid or None if budget hit."""
    attempt = 0
    while True:
        if spent >= deadline_cost:
            print(f"BUDGET reached (${spent:.2f} >= ${deadline_cost:.2f}); not launching.", flush=True)
            return None
        itype, arch, _price, _vcpu, _ram = CANDIDATES[attempt % len(CANDIDATES)]
        print(f"launch attempt {attempt + 1}: {itype} ({arch}, {_vcpu} vCPU / {_ram} GB)", flush=True)
        iid = try_launch(ec2, ami_by_arch, code_key, cfg=cfg, instance_type=itype, arch=arch)
        if iid:
            print(f"  launched {iid} [{itype}]", flush=True)
            return iid
        attempt += 1
        # After cycling all candidates once, back off (quota lag / broad capacity crunch).
        wait = 45 if attempt % len(CANDIDATES) else 90
        time.sleep(wait)


class _Tee:
    """Mirror stdout to a persistent logfile, timestamping each line — so an unattended coordinator
    run has a durable, greppable record (not just the ephemeral terminal). Wraps existing print()s,
    so no call sites change."""

    def __init__(self, logpath: Path):
        self._term = sys.__stdout__
        logpath.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(logpath, "a", encoding="utf-8", buffering=1)
        self._at_line_start = True

    def write(self, s: str) -> None:
        for part in s.splitlines(keepends=True):
            if self._at_line_start and part.strip():
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S ")
                self._term.write(ts)
                self._f.write(ts)
            self._term.write(part)
            self._f.write(part)
            self._at_line_start = part.endswith("\n")

    def flush(self) -> None:
        self._term.flush()
        self._f.flush()


def _tail_training_line(ec2, iid: str) -> str | None:
    """Latest 'iter N/M loss=... wall=...s collisions=...' line from the box's console (the actual
    training progress, which checkpoint counts alone don't show). Also surfaces OOM/death markers."""
    try:
        out = ec2.get_console_output(InstanceId=iid, Latest=True).get("Output", "") or ""
    except Exception:  # noqa: BLE001 — console is best-effort telemetry
        return None
    last_iter = None
    for ln in out.splitlines():
        s = ln.split("]: ")[-1].strip()
        if s.startswith("iter ") and "loss=" in s:
            last_iter = s
        elif any(k in s for k in ("Out of memory", "pool worker(s) died", "Traceback (most")):
            return f"WARN box log: {s[:90]}"
    return last_iter


def coordinate(cfg: dict) -> None:
    sys.stdout = _Tee(Path("runs") / f"coordinator_{cfg['run_subdir']}.log")
    print(f"=== coordinator start {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
          f"run={cfg['run_subdir']} log=runs/coordinator_{cfg['run_subdir']}.log ===", flush=True)
    ec2 = boto3.client("ec2", region_name=cfg["region"])
    s3 = boto3.client("s3", region_name=cfg["region"])
    run_prefix = f"{cfg['prefix']}/{cfg['run_subdir']}"
    ami_by_arch: dict[str, str] = {}

    code_key = build_and_upload_code(s3, cfg["bucket"], cfg["prefix"])
    print(f"Coordinating run '{cfg['run_subdir']}' -> {cfg['iterations']} iters "
          f"(budget ${cfg['max_cost']:.0f}); poll {cfg['poll']}s, stall {cfg['stall_minutes']}m", flush=True)

    spent = 0.0
    launches = 0
    reclaims = 0
    cur_iid: str | None = None
    cur_type: str | None = None
    cur_since = time.time()
    last_ckpt_iter = latest_ckpt_iter(s3, cfg["bucket"], run_prefix)
    last_ckpt_change = time.time()
    poll_i = 0
    print(f"start: latest checkpoint = iter {last_ckpt_iter}", flush=True)

    try:
        while True:
            done = latest_ckpt_iter(s3, cfg["bucket"], run_prefix)
            if done != last_ckpt_iter:
                print(f"progress: checkpoint advanced {last_ckpt_iter} -> {done}", flush=True)
                last_ckpt_iter = done
                last_ckpt_change = time.time()
            if done >= cfg["iterations"]:
                print(f"DONE: reached iter {done} >= target {cfg['iterations']}.", flush=True)
                break

            live = run_instances_for(ec2, cfg["run_subdir"], ("pending", "running"))
            if len(live) > 1:
                # Safety: never allow >1 writer. Keep the oldest, kill the rest.
                live.sort(key=lambda i: i["LaunchTime"])
                terminate_all(ec2, live[1:])
                live = live[:1]

            if live:
                inst = live[0]
                iid = inst["InstanceId"]
                itype = inst["InstanceType"]
                if iid != cur_iid:
                    cur_iid, cur_type, cur_since = iid, itype, time.time()
                spent_now = spent + (time.time() - cur_since) / 3600.0 * _PRICE.get(itype, 0.8)
                # Stall guard: running but no checkpoint progress for too long => wedged. Give a
                # generous grace for the first checkpoint (boot+compile+iter1).
                stall_s = (time.time() - last_ckpt_change)
                grace = cfg["stall_minutes"] * 60 * (2 if last_ckpt_iter == 0 else 1)
                if stall_s > grace:
                    print(f"STALL: {iid} no checkpoint progress in {stall_s/60:.0f}m (>{grace/60:.0f}m) "
                          "— killing + replacing.", flush=True)
                    terminate_all(ec2, [inst])
                    spent = spent_now
                    cur_iid = None
                    last_ckpt_change = time.time()  # reset so the replacement gets its own grace
                    time.sleep(10)
                    continue
                # Surface the box's actual training progress (loss/wall/collisions), throttled to
                # avoid console-API spam — checkpoint counts alone don't show per-iter health.
                train = _tail_training_line(ec2, iid) if poll_i % 3 == 0 else None
                print(f"healthy: {iid} [{itype}] iter {done}/{cfg['iterations']} "
                      f"~${spent_now:.2f} stall={stall_s/60:.0f}m"
                      + (f" | {train}" if train else ""), flush=True)
            else:
                # No live box -> account the dead one's runtime, then respawn (resumes from S3).
                if cur_iid is not None:
                    spent += (time.time() - cur_since) / 3600.0 * _PRICE.get(cur_type or "", 0.8)
                    reclaims += 1
                    print(f"instance gone (reclaim/term #{reclaims}); cumulative ~${spent:.2f}.", flush=True)
                    cur_iid = None
                # Defensive: clear any non-terminated straggler before relaunch (single writer).
                terminate_all(ec2, run_instances_for(ec2, cfg["run_subdir"], ("stopping", "shutting-down", "stopped")))
                iid = acquire_instance(ec2, ami_by_arch, code_key, cfg=cfg, deadline_cost=cfg["max_cost"], spent=spent)
                if iid is None:
                    print("No instance acquired (budget exhausted). Stopping coordinator.", flush=True)
                    break
                launches += 1
                cur_iid, cur_since = iid, time.time()
                live = run_instances_for(ec2, cfg["run_subdir"], ("pending", "running"))
                cur_type = live[0]["InstanceType"] if live else CANDIDATES[0][0]
                last_ckpt_change = time.time()  # fresh grace window for the new box
            poll_i += 1
            time.sleep(cfg["poll"])
    except KeyboardInterrupt:
        print("\nCoordinator interrupted.", flush=True)
        if cfg["terminate_on_exit"] and cur_iid:
            terminate_all(ec2, run_instances_for(ec2, cfg["run_subdir"], ("pending", "running")))
            print("Terminated running box (--terminate-on-exit).", flush=True)
        else:
            print("Left running box alive (it checkpoints each iter; re-run coordinator to resume).", flush=True)
    print(f"summary: launches={launches} reclaims={reclaims} est_cost=${spent:.2f} "
          f"final_ckpt=iter {latest_ckpt_iter(s3, cfg['bucket'], run_prefix)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-subdir", required=True, help="fixed S3 run subdir (resume key), e.g. run_conv_v2")
    p.add_argument("--iterations", type=int, default=200, help="target iteration count (completion gate)")
    p.add_argument("--workers", type=int, default=16,
                   help="pool workers; auto-clamped to (instance vCPU-2) and a ~2 GB/worker RAM budget. "
                        "Default 16: this workload is SERIAL-BOUND past ~24 workers (per-iter merge/"
                        "checkpoint don't parallelize), and the first iteration after a RESUME recompiles "
                        "across all workers + loads the saturated base — 60 workers thundering-herd-stall, "
                        ">24 on <256 GB OOMs. 16 on a 256 GB Graviton3 box (candidate #1 m7g.16xlarge) "
                        "clears the resume spike reliably.")
    p.add_argument("--traversals", type=int, default=90)
    p.add_argument("--deal-sweeps", type=int, default=6)
    p.add_argument("--max-depth", type=int, default=64)
    p.add_argument("--bet-buckets", default=None,
                   help="action abstraction indices into (0.33,0.50,0.66,0.75,1.0,1.5,2.0), e.g. "
                        "'1,4' = half-pot+pot (+fold/check/call/all-in). Omit = exact (infeasible at depth).")
    p.add_argument("--postflop-buckets", type=int, default=None,
                   help="card abstraction: N postflop made-hand buckets (e.g. 200). Bounds the card "
                        "dimension so tables saturate. Omit = raw cards (grows ~unbounded postflop).")
    p.add_argument("--betting-abstraction", action="store_true",
                   help="bounded betting-STATE key (street/pot/facing-size/SPR/position) instead of the "
                        "lossless action sequence — REQUIRED for the blueprint to saturate+converge.")
    p.add_argument("--client-bet-sizing", action="store_true",
                   help="CoinPoker client-matched street-aware bet sizing (preflop 2x/2.75x/3.5x, "
                        "postflop 1/3,1/2,3/4 pot + all-in) — realizable + indistinguishable sizes.")
    p.add_argument("--trainer", choices=["pool", "hogwild"], default="pool",
                   help="pool=bulk-sync reference (serial-bound past ~24 workers); hogwild=shared-memory "
                        "lock-free continuous CFR (H5 — drives the 64-vCPU box to near-linear scaling).")
    p.add_argument("--hogwild-capacity", type=int, default=None,
                   help="hogwild shared-table slot capacity (default in trainer; RuntimeError-on-full canary).")
    p.add_argument("--occupancy", choices=["hu", "6max"], default="hu",
                   help="table layout: hu (default) or 6max (Pluribus foundation target).")
    p.add_argument("--cfr-variant", choices=["cfr_plus", "dcfr"], default="cfr_plus",
                   help="cfr_plus (default) or dcfr (Discounted CFR); hogwild only.")
    p.add_argument("--vr-baseline", action="store_true", help="VR-MCCFR baseline variance reduction (hogwild).")
    p.add_argument("--max-cost", type=float, default=90.0, help="approx USD budget cap for launches")
    p.add_argument("--poll", type=int, default=60, help="seconds between health polls")
    p.add_argument("--stall-minutes", type=int, default=90,
                   help="kill+replace a box with no checkpoint progress for this long (2x grace "
                        "pre-iter1). MUST exceed the real per-iteration wall or healthy boxes get "
                        "killed in a no-progress respawn loop — the pool's own worker-death guard "
                        "already fails fast on a true hang, so this can be generous.")
    p.add_argument("--terminate-on-exit", action="store_true", help="kill the box on Ctrl-C (default: leave it)")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--profile", default=DEFAULT_PROFILE)
    args = p.parse_args()
    coordinate(vars(args))


if __name__ == "__main__":
    main()
