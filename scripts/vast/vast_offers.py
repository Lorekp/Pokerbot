#!/usr/bin/env python3
"""vast_offers - the spotpick-equivalent for vast.ai GPU offers (pure-stdlib + the `vastai` CLI).

spotpick.py cannot rank vast.ai offers (its EC2 catalog has no GPU column), so this is the GPU
analogue: it shells out to ``vastai search offers``, parses the JSON rows, and builds a
CAPACITY-DIVERSE fallback chain sorted by DLPerf/$/hr (best value first) -- mirroring
spotpick.chain()'s "one cheapest per family, iterate down the list on capacity failure" idea, but
diversified across HOSTS (machine_id) so one flaky/over-subscribed host does not sink the whole
chain.

The vast query language (validated forms in the RECON):
  type=bid                     interruptible (bid) offers
  gpu_ram>=24000               >=24 GB VRAM (field unit MB)
  num_gpus=1                   single-GPU listings
  reliability>0.90             host uptime score (lower to 0.80 if supply thin)
  rentable=true                currently available
  dph_total<=0.50              hard price ceiling $/hr
  -o dlperf_per_dphtotal-      sort DESC by DLPerf/$ (best value first)

Defensive notes (flagged in the runbook):
  * `vastai search offers --raw` emits a JSON list of offer dicts. Field names used here
    (``id``, ``dph_total``, ``min_bid``, ``dlperf``, ``dlperf_per_dphtotal``, ``gpu_name``,
    ``num_gpus``, ``machine_id``, ``reliability2``/``reliability``, ``inet_up``,
    ``cuda_max_good``) match the documented schema, but vast changes columns occasionally -- every
    access is guarded with .get() and the picker degrades gracefully (skips a row it cannot score).
  * Sorting is done CLIENT-SIDE on dlperf_per_dphtotal so we do not depend on the server -o flag
    landing; the server -o is passed too as a hint.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Offer:
    offer_id: int
    machine_id: int
    gpu_name: str
    num_gpus: int
    gpu_ram_mb: float
    dph_total: float          # current market $/hr (the "on-demand-ish" reference)
    min_bid: float            # interruptible floor the host will accept
    dlperf: float
    dlperf_per_dphtotal: float
    reliability: float
    inet_up_mbps: float
    cuda_max_good: float

    def bid_for(self, markup: float) -> float:
        """Suggested interruptible bid: min_bid * (1+markup), floored at a tiny positive number.

        Falls back to dph_total when min_bid is missing/zero (some rows omit it)."""
        base = self.min_bid if self.min_bid and self.min_bid > 0 else self.dph_total
        return round(max(0.01, base * (1.0 + markup)), 4)


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_rows(rows: list[dict]) -> list[Offer]:
    offers: list[Offer] = []
    for r in rows:
        try:
            oid = int(r.get("id"))
        except (TypeError, ValueError):
            continue
        dph = _to_float(r.get("dph_total"))
        # vast exposes reliability as "reliability2" (0..1) on newer schemas; fall back to either.
        rel = _to_float(r.get("reliability2", r.get("reliability")))
        dlp = _to_float(r.get("dlperf"))
        dpd = _to_float(r.get("dlperf_per_dphtotal"))
        # If the server did not provide the ratio, derive it (best-effort) so the row is still scorable.
        if dpd <= 0 and dph > 0 and dlp > 0:
            dpd = dlp / dph
        offers.append(Offer(
            offer_id=oid,
            machine_id=int(_to_float(r.get("machine_id"))),
            gpu_name=str(r.get("gpu_name", "?")),
            num_gpus=int(_to_float(r.get("num_gpus"), 1)),
            gpu_ram_mb=_to_float(r.get("gpu_ram")),
            dph_total=dph,
            min_bid=_to_float(r.get("min_bid")),
            dlperf=dlp,
            dlperf_per_dphtotal=dpd,
            reliability=rel,
            inet_up_mbps=_to_float(r.get("inet_up")),
            cuda_max_good=_to_float(r.get("cuda_max_good")),
        ))
    return offers


def search_offers(
    *,
    gpu_ram_min_mb: int = 24000,
    price_ceiling: float = 0.50,
    reliability_min: float = 0.90,
    gpu_name: str | None = None,
    min_inet_up: float = 0.0,
    min_cuda: float = 0.0,
    limit: int = 40,
    vastai_bin: str = "vastai",
    extra_query: str = "",
) -> list[Offer]:
    """Query vast.ai for interruptible single-GPU offers and return parsed Offers (unsorted)."""
    clauses = [
        "type=bid",
        f"gpu_ram>={gpu_ram_min_mb}",
        "num_gpus=1",
        f"reliability>{reliability_min}",
        "rentable=true",
        f"dph_total<={price_ceiling}",
    ]
    if gpu_name:
        clauses.append(f"gpu_name={gpu_name}")
    if extra_query:
        clauses.append(extra_query)
    query = " ".join(clauses)
    cmd = [vastai_bin, "search", "offers", query,
           "-o", "dlperf_per_dphtotal-", "--limit", str(limit), "--raw"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    try:
        rows = json.loads(proc.stdout)
    except ValueError as exc:
        raise RuntimeError(f"could not parse `vastai search offers --raw` JSON: {exc}\n{proc.stdout[:400]}")
    if not isinstance(rows, list):
        raise RuntimeError(f"unexpected vastai offers payload (not a list): {type(rows)}")
    offers = _parse_rows(rows)
    # Post-filters the query language may not express on every vast version.
    if min_inet_up > 0:
        offers = [o for o in offers if o.inet_up_mbps >= min_inet_up]
    if min_cuda > 0:
        offers = [o for o in offers if o.cuda_max_good >= min_cuda or o.cuda_max_good == 0]
    return offers


def chain(offers: list[Offer], n: int = 5) -> list[Offer]:
    """Capacity-diverse fallback chain: best DLPerf/$ first, at most ONE offer per host machine_id.

    Mirrors spotpick.chain()'s diversity idea (there: one type per instance family; here: one
    offer per physical host) so iterating launch attempts down the chain does not repeatedly hit
    the same over-subscribed / flaky machine.
    """
    ranked = sorted(offers, key=lambda o: o.dlperf_per_dphtotal, reverse=True)
    seen_hosts: set[int] = set()
    out: list[Offer] = []
    for o in ranked:
        if o.machine_id and o.machine_id in seen_hosts:
            continue
        seen_hosts.add(o.machine_id)
        out.append(o)
        if len(out) >= n:
            break
    return out


def describe(o: Offer, markup: float) -> str:
    return (
        f"offer {o.offer_id} host {o.machine_id}  {o.gpu_name} {o.gpu_ram_mb/1000:.0f}GB  "
        f"dlperf/$ {o.dlperf_per_dphtotal:.0f}  dph ${o.dph_total:.3f}  min_bid ${o.min_bid:.3f}  "
        f"-> bid ${o.bid_for(markup):.3f}  rel {o.reliability:.2f}  inet_up {o.inet_up_mbps:.0f}Mbps"
    )


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="rank vast.ai interruptible GPU offers by DLPerf/$")
    ap.add_argument("--gpu-ram-min-mb", type=int, default=24000)
    ap.add_argument("--price-ceiling", type=float, default=0.50)
    ap.add_argument("--reliability-min", type=float, default=0.90)
    ap.add_argument("--gpu-name", default=None, help="e.g. RTX_4090 (omit to allow any >=24GB GPU)")
    ap.add_argument("--min-inet-up", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--markup", type=float, default=0.15)
    ap.add_argument("--vastai-bin", default="vastai")
    args = ap.parse_args(argv)

    offers = search_offers(
        gpu_ram_min_mb=args.gpu_ram_min_mb, price_ceiling=args.price_ceiling,
        reliability_min=args.reliability_min, gpu_name=args.gpu_name,
        min_inet_up=args.min_inet_up, limit=args.limit, vastai_bin=args.vastai_bin,
    )
    ch = chain(offers, n=args.n)
    if not ch:
        print("no matching offers", flush=True)
        return 1
    for o in ch:
        print(describe(o, args.markup), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
