#!/usr/bin/env python3
"""LEG A (decision-quality audit), step 2: compare the served blueprint preflop policy against the
published-solver consensus (artifacts/published_preflop_ranges.json, 4 independent sources).

Metrics per position (RFI):
  * combo-weighted L1 deviation mass: sum_h w_h * |f_bot(h) - f_consensus(h)| / 1326
    (the fraction of dealt hands played differently from the published consensus)
  * directional split: under-raise mass (consensus opens, we don't) vs over-raise mass
  * limp mass (published charts are raise-or-fold; any limp mass is a deviation by construction,
    reported separately since SB limping is legitimately mixed in some solver outputs)
  * inter-source disagreement (mean pairwise L1) as the noise floor of the anchor itself
  * worst per-hand deviations

vs-RFI (BB vs BTN 2.5x, single source src2): 3bet/call/fold per-hand comparison.

Output: artifacts/audit_preflop_vs_published.json (+ stdout summary)
"""
from __future__ import annotations

import itertools
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_RANKS = "23456789TJQKA"


def combos_of(h: str) -> int:
    if len(h) == 2:
        return 6
    return 4 if h[2] == "s" else 12


def all_hands():
    out = []
    for i, r1 in enumerate(_RANKS[::-1]):
        for j, r2 in enumerate(_RANKS[::-1]):
            if i == j:
                out.append(r1 + r1)
            elif i < j:
                out.append(r1 + _RANKS[::-1][j] + "s")
    # offsuit
    for i in range(13):
        for j in range(i + 1, 13):
            hi, lo = _RANKS[::-1][i], _RANKS[::-1][j]
            out.append(hi + lo + "o")
    # dedupe preserving order
    seen, res = set(), []
    for h in out:
        if h not in seen:
            seen.add(h)
            res.append(h)
    return res


HANDS = all_hands()
assert len(HANDS) == 169, len(HANDS)


def main() -> int:
    pub = json.load(open(os.path.join(REPO, "artifacts", "published_preflop_ranges.json")))
    bot = json.load(open(os.path.join(REPO, "artifacts", "audit_preflop_blueprint.json")))

    out = {"meta": {
        "sources": pub["sources"],
        "consensus": "unweighted mean over sources present per position",
        "bot_freqs": "served raise freq (all raise buckets + all-in) from the deployed "
                     "blueprint-preflop serve path; limp = served CALL mass",
    }, "rfi": {}, "vs_rfi": {}}

    print("=== LEG A: blueprint preflop vs published GTO consensus (RFI) ===")
    for pos in ("UTG", "HJ", "CO", "BTN", "SB"):
        srcs = pub["rfi"].get(pos, {})
        if not srcs:
            continue
        bot_rows = bot["nodes"]["rfi"][pos]["classes"]
        # consensus per hand
        cons, l1_pairs = {}, []
        src_ids = sorted(srcs.keys())
        for h in HANDS:
            vals = [min(1.0, float(srcs[s].get(h, 0.0))) for s in src_ids]
            cons[h] = sum(vals) / len(vals)
        # inter-source noise floor (combo-weighted mean pairwise L1)
        for a, b in itertools.combinations(src_ids, 2):
            d = sum(combos_of(h) * abs(min(1.0, float(srcs[a].get(h, 0.0)))
                                       - min(1.0, float(srcs[b].get(h, 0.0)))) for h in HANDS)
            l1_pairs.append(d / 1326.0)
        noise_floor = sum(l1_pairs) / len(l1_pairs) if l1_pairs else 0.0

        dev = under = over = limp_mass = bot_open = cons_open = 0.0
        per_hand = []
        for h in HANDS:
            w = combos_of(h) / 1326.0
            row = bot_rows.get(h)
            f_bot = float(row["raise"]) if row else 0.0
            f_limp = float(row["call"]) if row else 0.0
            f_c = cons[h]
            delta = f_bot - f_c
            dev += w * abs(delta)
            if delta < 0:
                under += w * (-delta)
            else:
                over += w * delta
            limp_mass += w * f_limp
            bot_open += w * f_bot
            cons_open += w * f_c
            per_hand.append((h, round(f_bot, 3), round(f_c, 3), round(w * abs(delta) * 100, 3),
                             round(f_limp, 3)))
        per_hand.sort(key=lambda x: -x[3])
        out["rfi"][pos] = {
            "bot_open_pct": round(100 * bot_open, 1),
            "consensus_open_pct": round(100 * cons_open, 1),
            "bot_limp_pct": round(100 * limp_mass, 1),
            "l1_deviation_mass_pct": round(100 * dev, 1),
            "under_raise_pct": round(100 * under, 1),
            "over_raise_pct": round(100 * over, 1),
            "source_noise_floor_pct": round(100 * noise_floor, 1),
            "n_sources": len(src_ids),
            "worst_hands": [
                {"hand": h, "bot_raise": b_, "consensus": c_, "wL1_pct": d_, "bot_limp": lm}
                for h, b_, c_, d_, lm in per_hand[:15]
            ],
        }
        r = out["rfi"][pos]
        print(f"  {pos}: bot opens {r['bot_open_pct']}% (+{r['bot_limp_pct']}% limp) vs consensus "
              f"{r['consensus_open_pct']}% | L1 deviation {r['l1_deviation_mass_pct']}% "
              f"(under {r['under_raise_pct']} / over {r['over_raise_pct']}) | "
              f"source noise floor {r['source_noise_floor_pct']}%")

    # ---- vs-RFI: BB vs BTN (src2 only) ----
    vs = pub.get("vs_rfi", {}).get("BB_vs_BTN")
    if vs:
        src_id = sorted(vs.keys())[0]
        chart = vs[src_id]
        bot_rows = bot["nodes"]["vs_BTN"]["BB"]["classes"]
        dev3 = devc = devf = 0.0
        bot3 = botc = botf = c3 = cc = cf = 0.0
        per_hand = []
        for h in HANDS:
            w = combos_of(h) / 1326.0
            row = bot_rows.get(h)
            b3 = float(row["raise"]) if row else 0.0
            bc = float(row["call"]) + float(row["check"]) if row else 0.0
            bf = float(row["fold"]) if row else 1.0
            ch = chart.get(h, {})
            s3 = min(1.0, float(ch.get("3bet", 0.0)))
            sc = min(1.0, float(ch.get("call", 0.0)))
            sf = max(0.0, 1.0 - s3 - sc)
            dev3 += w * abs(b3 - s3)
            devc += w * abs(bc - sc)
            devf += w * abs(bf - sf)
            bot3 += w * b3; botc += w * bc; botf += w * bf
            c3 += w * s3; cc += w * sc; cf += w * sf
            per_hand.append((h, round(w * (abs(b3 - s3) + abs(bc - sc) + abs(bf - sf)) * 100, 3),
                             {"bot": [round(b3, 2), round(bc, 2), round(bf, 2)],
                              "chart": [round(s3, 2), round(sc, 2), round(sf, 2)]}))
        per_hand.sort(key=lambda x: -x[1])
        out["vs_rfi"]["BB_vs_BTN"] = {
            "source": src_id,
            "bot": {"3bet_pct": round(100 * bot3, 1), "call_pct": round(100 * botc, 1),
                    "fold_pct": round(100 * botf, 1)},
            "chart": {"3bet_pct": round(100 * c3, 1), "call_pct": round(100 * cc, 1),
                      "fold_pct": round(100 * cf, 1)},
            "l1_3bet_pct": round(100 * dev3, 1), "l1_call_pct": round(100 * devc, 1),
            "l1_fold_pct": round(100 * devf, 1),
            "worst_hands": [{"hand": h, "wL1_pct": d, **det} for h, d, det in per_hand[:12]],
        }
        v = out["vs_rfi"]["BB_vs_BTN"]
        print(f"\n  BB vs BTN 2.5x: bot 3bet/call/fold = {v['bot']['3bet_pct']}/"
              f"{v['bot']['call_pct']}/{v['bot']['fold_pct']}% vs chart "
              f"{v['chart']['3bet_pct']}/{v['chart']['call_pct']}/{v['chart']['fold_pct']}% "
              f"(single source)")

    path = os.path.join(REPO, "artifacts", "audit_preflop_vs_published.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
