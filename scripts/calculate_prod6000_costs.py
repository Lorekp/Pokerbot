#!/usr/bin/env python3
"""Estimate prod6000 EC2 Spot costs from instance pricing CSV + throughput assumptions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

UNIQUE = 3000
REMAINING = 530
INTERRUPT_PCT = 0.075

INSTANCE = "c6i.xlarge"
VCPU_PER_INST = 4

THROUGHPUT = {
    "baseline": 5.5,
    "conservative": 4.5,
    "sb_heavy_worst": 3.8,
}

VCPU_TIERS = [32, 64, 128]


def load_pricing(csv_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r["Name"]
            spot_raw = (r.get("Historical Avg Spot Hourly Cost") or "").strip()
            rows[name] = {
                "vcpus": int(r["vCPUs"]),
                "od_hourly": float(r["On-Demand Hourly Cost"]),
                "od_per_vcpu": float(r["On-Demand Cost per vCPU Hour"]),
                "spot_hourly": float(spot_raw) if spot_raw else 0.0,
                "spot_discount": (r.get("Historical Average Discount") or "").strip(),
            }
    return rows


def wall_hours(shards: int, vcpus: int, shards_per_worker_hr: float) -> float:
    workers = vcpus // VCPU_PER_INST
    return shards / (workers * shards_per_worker_hr)


def cost_for(
    shards: int,
    vcpus: int,
    shards_per_worker_hr: float,
    pricing: dict,
    *,
    lifecycle: str = "spot",
    interrupt: bool = True,
) -> dict:
    workers = vcpus // VCPU_PER_INST
    wh = wall_hours(shards, vcpus, shards_per_worker_hr)
    if interrupt and lifecycle == "spot":
        wh *= 1 + INTERRUPT_PCT
    inst_hrs = workers * wh
    rate = pricing["spot_hourly"] if lifecycle == "spot" else pricing["od_hourly"]
    total = inst_hrs * rate
    return {
        "workers": workers,
        "wall_hours": wh,
        "wall_days": wh / 24,
        "instance_hours": inst_hrs,
        "vcpu_hours": vcpus * wh,
        "rate_per_instance_hr": rate,
        "total_usd": total,
        "usd_per_shard": total / shards,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "csv",
        type=Path,
        default=Path(r"data/external/ec2_instances_1084_unique_with_cost_per_vcpu.csv"),
        nargs="?",
    )
    args = p.parse_args()
    rows = load_pricing(args.csv)
    px = rows[INSTANCE]
    px_large = rows["c6i.large"]

    print("=" * 72)
    print("PROD6000 CLOUD COST ESTIMATE")
    print("=" * 72)
    print(f"Instance: {INSTANCE} ({VCPU_PER_INST} vCPU, 1 worker/inst)")
    print(
        f"Spot: ${px['spot_hourly']:.4f}/inst-hr  "
        f"(${px['spot_hourly'] / VCPU_PER_INST:.4f}/vcpu-hr, {px['spot_discount']} off OD)"
    )
    print(
        f"On-Demand: ${px['od_hourly']:.4f}/inst-hr  "
        f"(${px['od_per_vcpu']:.4f}/vcpu-hr)"
    )
    print(f"Spot interruption buffer: +{INTERRUPT_PCT * 100:.1f}% wall time")
    print()

    for label, shards in [
        ("Full prod6000 (3000 shards)", UNIQUE),
        ("Remaining only (~530 keys)", REMAINING),
    ]:
        print("-" * 72)
        print(label)
        print("-" * 72)
        print(
            f"{'vCPUs':>6} {'Wkr':>4} {'Scenario':<22} {'Wall':>8} "
            f"{'Spot $':>9} {'OD $':>9} {'$/shard':>8}"
        )
        print("-" * 72)
        for v in VCPU_TIERS:
            for scen, rate in THROUGHPUT.items():
                s = cost_for(shards, v, rate, px, lifecycle="spot")
                o = cost_for(shards, v, rate, px, lifecycle="on-demand", interrupt=False)
                wh = s["wall_hours"]
                wh_str = f"{wh:.1f}h" if wh < 48 else f"{wh / 24:.1f}d"
                print(
                    f"{v:>6} {s['workers']:>4} {scen:<22} {wh_str:>8} "
                    f"{s['total_usd']:>9.2f} {o['total_usd']:>9.2f} {s['usd_per_shard']:>8.4f}"
                )
            print()

    print("=" * 72)
    print("PLANNING NUMBER — baseline (5.5 shards/worker/hr), Spot + 7.5% interrupt")
    print("=" * 72)
    print(f"{'Scope':<28} {'32 vCPU':>12} {'64 vCPU':>12} {'128 vCPU':>12}")
    for label, shards in [("Full 3000", UNIQUE), ("Remaining 530", REMAINING)]:
        costs = [
            f"${cost_for(shards, v, THROUGHPUT['baseline'], px)['total_usd']:.2f}"
            for v in VCPU_TIERS
        ]
        print(f"{label:<28} {costs[0]:>12} {costs[1]:>12} {costs[2]:>12}")

    print()
    print("Cost formula:")
    print("  spot_usd = (vCPUs/4) * wall_hours * spot_hourly_per_instance")
    print("  wall_hours = shards / (vCPUs/4 * shards_per_worker_hr) * (1 + interrupt_pct)")

    print()
    print("=" * 72)
    print("c6i.large comparison @ 32 vCPU (16 workers, sb-heavy 3.8 shards/worker/hr)")
    print("=" * 72)
    for shards, lbl in [(UNIQUE, "full"), (REMAINING, "remaining")]:
        wh = shards / (16 * 3.8) * (1 + INTERRUPT_PCT)
        spot = 16 * wh * px_large["spot_hourly"]
        print(f"  {lbl}: {wh:.1f}h wall, spot ${spot:.2f}")

    print()
    print("=" * 72)
    print("Reference: cheapest c/m-series Spot $/vcpu-hr (2–16 vCPU instances)")
    print("=" * 72)
    candidates = []
    for name, r in rows.items():
        if r["spot_hourly"] <= 0:
            continue
        if not name.startswith(("c5", "c6", "c7", "m5", "m6", "m7")):
            continue
        if r["vcpus"] not in (2, 4, 8, 16):
            continue
        candidates.append((r["spot_hourly"] / r["vcpus"], name, r))
    candidates.sort()
    for spot_vcpu, name, r in candidates[:8]:
        print(
            f"  {name:18s} {r['vcpus']:>2} vCPU  "
            f"spot ${r['spot_hourly']:.4f}/hr  ${spot_vcpu:.5f}/vcpu-hr"
        )

    all_spot = sorted(
        (r["spot_hourly"] / r["vcpus"], n) for n, r in rows.items() if r["spot_hourly"] > 0
    )
    rank = next(i + 1 for i, (_, n) in enumerate(all_spot) if n == INSTANCE)
    print(f"  c6i.xlarge ranks #{rank} of {len(all_spot)} with spot data by $/vcpu-hr")


if __name__ == "__main__":
    main()
