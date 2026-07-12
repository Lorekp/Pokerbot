#!/usr/bin/env python3
"""cert7: assemble a single VERDICT_FINAL.json from whatever section result files exist in --out-dir.

Designed to be re-run after EVERY section so a cap-stop / spot-reclaim always leaves the freshest
partial verdict on S3. Every section is optional; missing files are recorded as "pending". Nothing
here spends money or loads the blueprint -- it only reads the JSON/text the eval steps dropped.

Sections (file conventions the bootstrap writes into --out-dir):
  serve_gate_selfplay.json   serve-miss gate (total.miss_rate)
  serve_index_meta.json      index meta (n_strat rows)
  VERDICT.json               cert_verdict_merge output: RAW h2h (chips/deck+CI) + pool bb/100
  search_h2h_seed*_shard*.json  SEARCH-arm paired h2h shards (per_deck_nets) vs incumbent
  search_pool_new.json       SEARCH-arm bb/100 vs field
  lbr_new_raw.json / lbr_new_search.json / lbr_incumbent.json  LBR6 exploitability (bb_per_100)
  preflop_new_raw.json / preflop_new_vs_pub.json / preflop_new_search.json  preflop shape + limp
  ksweep_selfplay.json / ksweep_budget.json   search-K convergence
  ladder_plateau.json        rung strategy-movement proxy
"""
from __future__ import annotations

import glob
import json
import math
import os
import time


def _load(od, name):
    p = os.path.join(od, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        return {"_load_error": str(e), "_path": p}


def _paired_from_shards(od, pattern):
    """Pool per_deck_nets across engine_ab_h2h shards -> mean chips/deck + normal-approx 95% CI."""
    nets = []
    settled = None
    nfiles = 0
    for f in sorted(glob.glob(os.path.join(od, pattern))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        pdn = d.get("per_deck_nets")
        if not pdn:
            continue
        nets.extend(float(x) for x in pdn)
        nfiles += 1
        if settled is None:
            settled = d.get("showdown_money_settled")
    if not nets:
        return None
    n = len(nets)
    mean = sum(nets) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in nets) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = 0.0
    ci = [mean - 1.96 * se, mean + 1.96 * se]
    # chips/deck -> bb/100: BB=100 chips, and per-deck == per-hand here; bb/100 = chips/deck.
    return {"n_decks": n, "shards": nfiles, "mean_chips_per_deck": mean,
            "ci95_chips_per_deck": ci, "bb_per_100": mean, "ci95_bb_per_100": ci,
            "se": se, "showdown_money_settled": settled,
            "excludes_zero_favorable": bool(ci[0] > 0),
            "excludes_zero_unfavorable": bool(ci[1] < 0)}


def _lbr_num(d):
    if not d or "_load_error" in (d or {}):
        return None
    bb100 = d.get("bb_per_100")
    evh = d.get("ev_bb_per_hand")
    if bb100 is None and evh is not None:
        bb100 = evh * 100.0
    if bb100 is None:
        return None
    return {"bb_per_100": bb100, "mbb_per_g": bb100 * 10.0,
            "ev_bb_per_hand": evh, "hands": d.get("hands"), "seed": d.get("seed"),
            "target": d.get("target"), "stderr_bb_per_hand": d.get("stderr_bb_per_hand")}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/opt/pokerbot/cert_out")
    ap.add_argument("--run-id", default="cert7")
    args = ap.parse_args()
    od = args.out_dir

    V = {"run_id": args.run_id, "assembled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "object": "bpv4_tier1_faithful_1 iter_126000 (preflop) + snap_avg(postflop)",
         "sections": {}}
    S = V["sections"]

    # ---- serve object + miss gate ----
    sg = _load(od, "serve_gate_selfplay.json")
    idx = _load(od, "serve_index_meta.json")
    if sg or idx:
        mr = (sg or {}).get("total", {}).get("miss_rate") if sg else None
        S["serve_object"] = {
            "index_n_strat_rows": (idx or {}).get("n_strat"),
            "index_iters": (idx or {}).get("iters"),
            "selfplay_miss_rate": mr,
            "miss_gate_pass": (mr is not None and mr < 0.005),
            "per_street": (sg or {}).get("per_street") or (sg or {}).get("streets"),
        }
    else:
        S["serve_object"] = "pending"

    # ---- Section 1 RAW strength ----
    raw = _load(od, "VERDICT.json")
    if raw and "_load_error" not in raw:
        S["strength_raw"] = {
            "h2h_new_vs_incumbent": raw.get("h2h_new_vs_incumbent"),
            "pool_combined_bb100": raw.get("pool_combined_bb100"),
            "pool_per_archetype": raw.get("pool_per_archetype"),
            "verdict": raw.get("verdict"),
        }
    else:
        S["strength_raw"] = "pending"

    # ---- Section 1 SEARCH strength ----
    sh = _paired_from_shards(od, "search_h2h_seed*_shard*.json")
    spn = _load(od, "search_pool_new.json")
    if sh or spn:
        S["strength_search"] = {
            "h2h_new_search_vs_incumbent": sh,
            "pool_new_search_combined_bb100": (spn or {}).get("combined_bb100"),
            "pool_new_search_per_archetype": (spn or {}).get("per_archetype"),
        }
    else:
        S["strength_search"] = "pending"

    # ---- Section 2 LBR exploitability ----
    lr = _lbr_num(_load(od, "lbr_new_raw.json"))
    ls = _lbr_num(_load(od, "lbr_new_search.json"))
    li = _lbr_num(_load(od, "lbr_incumbent.json"))
    if lr or ls or li:
        S["lbr_exploitability_mbb_g"] = {
            "new_raw": lr, "new_search": ls, "incumbent": li,
            "note": "LBR6 lower bound; leaf-settle ON all arms; menu fc; lower=less exploitable",
        }
    else:
        S["lbr_exploitability_mbb_g"] = "pending"

    # ---- Section 3 preflop shape ----
    pr = _load(od, "preflop_new_raw.json")
    pvp = _load(od, "preflop_new_vs_pub.json")
    ps = _load(od, "preflop_new_search.json")

    def _btn_aa_limp(d):
        try:
            return d["nodes"]["rfi"]["BTN"]["classes"]["AA"]["call"]
        except Exception:  # noqa: BLE001
            return None
    if pr or pvp or ps:
        S["preflop_shape"] = {
            "raw_btn_aa_limp": _btn_aa_limp(pr) if pr else None,
            "raw_btn_limp_pct_vs_pub": (pvp or {}).get("rfi", {}).get("BTN", {}).get("bot_limp_pct"),
            "search_btn_aa_limp": _btn_aa_limp(ps) if ps else None,
            "raw_rfi_present": bool(pr and pr.get("nodes")),
            "search_rfi_present": bool(ps and ps.get("nodes")),
            "note": "call mass at BTN RFI AA node = limp share; compare raw vs search-resolved",
        }
    else:
        S["preflop_shape"] = "pending"

    # ---- Section 4 K-sweep ----
    ks = _load(od, "ksweep_selfplay.json")
    kb = _load(od, "ksweep_budget.json")
    if ks or kb:
        S["search_k_convergence"] = {
            "selfplay_agg": (ks or {}).get("agg"),
            "selfplay_Ks": (ks or {}).get("Ks"),
            "budget_rows": (kb or {}).get("rows"),
            "note": "TV=L1/2 vs K=4000 reference; find knee where TV stabilizes",
        }
    else:
        S["search_k_convergence"] = "pending"

    # ---- Section 5 ladder plateau ----
    lp = _load(od, "ladder_plateau.json")
    S["ladder_plateau"] = lp if lp else "pending"

    # ---- HEADLINE ----
    headline = {}
    sr = S.get("strength_raw")
    if isinstance(sr, dict):
        h2h = sr.get("h2h_new_vs_incumbent") or {}
        pool = sr.get("pool_combined_bb100") or {}
        ci = h2h.get("ci95") or h2h.get("ci95_chips_per_deck")
        diff = h2h.get("new_minus_inc_per_deck")
        if diff is None:
            diff = h2h.get("mean_chips_per_deck")
        pool_new = pool.get("new")
        pool_inc = pool.get("incumbent")
        ci_fav = bool(ci and ci[0] > 0)
        pool_ge = (pool_new is not None and pool_inc is not None and pool_new >= pool_inc)
        headline["raw_ge_incumbent"] = bool(ci_fav or pool_ge)
        headline["raw_h2h_diff_chips_per_deck"] = diff
        headline["raw_h2h_ci95"] = ci
        headline["raw_pool_new_bb100"] = pool_new
        headline["raw_pool_inc_bb100"] = pool_inc
    ss = S.get("strength_search")
    if isinstance(ss, dict):
        sh2 = ss.get("h2h_new_search_vs_incumbent") or {}
        headline["search_h2h_diff_chips_per_deck"] = sh2.get("mean_chips_per_deck")
        headline["search_h2h_ci95"] = sh2.get("ci95_chips_per_deck")
        headline["search_ge_incumbent"] = sh2.get("excludes_zero_favorable")
        headline["search_pool_new_bb100"] = ss.get("pool_new_search_combined_bb100")
    le = S.get("lbr_exploitability_mbb_g")
    if isinstance(le, dict):
        headline["lbr_mbb_g_new_raw"] = (le.get("new_raw") or {}).get("mbb_per_g") if le.get("new_raw") else None
        headline["lbr_mbb_g_new_search"] = (le.get("new_search") or {}).get("mbb_per_g") if le.get("new_search") else None
        headline["lbr_mbb_g_incumbent"] = (le.get("incumbent") or {}).get("mbb_per_g") if le.get("incumbent") else None
    V["headline"] = headline

    outp = os.path.join(od, "VERDICT_FINAL.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(V, f, indent=1)
    print(f"[assemble] wrote {outp}")
    print(json.dumps(headline, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
