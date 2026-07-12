"""Assemble the cert8 LBR exploitability verdict from sharded per-hand JSONL dumps.

Pools each arm (new-raw / new-search / incumbent) into one large-N estimate with a proper CI, forms the
CRN paired contrasts (new-vs-incumbent, search-vs-raw) by matching (seed, hand_idx) across arms, runs the
per-street / per-position profit attribution, folds in the stage-0 instrument reconciliation, and writes
VERDICT_FINAL.json. All non-fatal: whatever shards exist get pooled.

Units: bb/100 throughout (mbb/g = bb/100 * 10). A POSITIVE LBR number = the best-responder WINS that much
against the arm => MORE exploitable. new-vs-incumbent diff < 0 => new is LESS exploitable than incumbent.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

_POS = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO"}
_STREET = {-1: "no-postflop-call", 0: "preflop", 1: "flop", 2: "turn", 3: "river"}


def _load_hands(pattern):
    """Return list of per-hand records from all JSONL shards matching pattern."""
    rows = []
    for fp in sorted(glob.glob(pattern)):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _ci(vals):
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = 0.0
    return {"n": n, "bb_per_100": mean * 100.0, "mbb_per_g": mean * 1000.0,
            "stderr_bb_per_100": se * 100.0,
            "ci95_bb_per_100": [(mean - 1.96 * se) * 100.0, (mean + 1.96 * se) * 100.0]}


def _attribution(rows):
    n = len(rows)
    if n == 0:
        return {}
    by_pos, by_street, by_outcome = {}, {}, {"folded": [], "saw_through": []}
    for r in rows:
        by_pos.setdefault(r["seat"], []).append(r["reward"])
        by_street.setdefault(r.get("last_call_street", -1), []).append(r["reward"])
        (by_outcome["folded"] if r.get("folded") else by_outcome["saw_through"]).append(r["reward"])
    return {
        "by_position": {_POS.get(s, str(s)): {"n": len(v), "bb_per_100": sum(v) / n * 100.0}
                        for s, v in sorted(by_pos.items())},
        "by_last_call_street": {_STREET.get(s, str(s)):
                                {"n": len(v), "freq": len(v) / n,
                                 "mean_bb_per_hand": sum(v) / len(v),
                                 "contrib_bb_per_100": sum(v) / n * 100.0}
                                for s, v in sorted(by_street.items())},
        "by_outcome": {k: {"n": len(v), "contrib_bb_per_100": sum(v) / n * 100.0}
                       for k, v in by_outcome.items()},
    }


def _paired(rows_a, rows_c):
    """CRN paired contrast: match (seed, idx) -> per-hand difference reward_a - reward_c. Shared deck
    luck cancels. Returns exploit_a - exploit_c in bb/100 with CI."""
    ca = {(r["seed"], r["idx"]): r["reward"] for r in rows_a}
    cc = {(r["seed"], r["idx"]): r["reward"] for r in rows_c}
    keys = sorted(set(ca) & set(cc))
    diffs = [ca[k] - cc[k] for k in keys]
    out = _ci(diffs)
    out["matched_hands"] = len(keys)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="cert8_lbr")
    args = ap.parse_args()
    D = args.out_dir

    rows = {
        "new_raw": _load_hands(os.path.join(D, "hands_new_raw_*.jsonl")),
        "new_search": _load_hands(os.path.join(D, "hands_new_search_*.jsonl")),
        "incumbent": _load_hands(os.path.join(D, "hands_incumbent_*.jsonl")),
    }
    arms = {name: {**_ci([r["reward"] for r in rs]), "attribution": _attribution(rs)}
            for name, rs in rows.items()}

    # CRN contrasts
    contrasts = {}
    if rows["new_raw"] and rows["incumbent"]:
        contrasts["new_raw_minus_incumbent"] = _paired(rows["new_raw"], rows["incumbent"])
    if rows["new_search"] and rows["new_raw"]:
        contrasts["new_search_minus_new_raw"] = _paired(rows["new_search"], rows["new_raw"])

    # Stage-0 instrument reconciliation
    recon = {"note": ("lbr6_audit --unpaired and lbr6_attribute --raw at the SAME seed/hands/menu report "
                      "the RAW LBR EV via identical code (paired=False); agreement proves the cert7 sign "
                      "disagreement was the paired estimator + noise, NOT a settle/sign bug.")}
    try:
        aud = json.load(open(os.path.join(D, "recon_audit.json")))
        recon["audit_unpaired_bb_per_100"] = aud.get("bb_per_100")
        recon["audit_raw_bb_per_100"] = aud.get("detail", {}).get("br_raw_bb_per_100")
    except Exception as e:  # noqa: BLE001
        recon["audit_error"] = repr(e)
    try:
        txt = open(os.path.join(D, "recon_attribute.log")).read()
        m = re.search(r"overall:\s*([+-]?[0-9.]+)\s*bb/100", txt)
        recon["attribute_raw_bb_per_100"] = float(m.group(1)) if m else None
    except Exception as e:  # noqa: BLE001
        recon["attribute_error"] = repr(e)
    a, b = recon.get("audit_unpaired_bb_per_100"), recon.get("attribute_raw_bb_per_100")
    if a is not None and b is not None:
        recon["abs_diff_bb_per_100"] = abs(a - b)
        recon["reconciled"] = abs(a - b) < 1.0  # bit-identical expected; <1 bb/100 = agree

    # Verdicts
    def _sep(c):
        if not c or c.get("n", 0) == 0:
            return None
        lo, hi = c["ci95_bb_per_100"]
        return {"diff_bb_per_100": c["bb_per_100"], "ci95": [lo, hi],
                "less_exploitable_if_below_0": hi < 0.0, "worse_if_above_0": lo > 0.0}

    nvi = contrasts.get("new_raw_minus_incumbent")
    svr = contrasts.get("new_search_minus_new_raw")
    verdict_sentence = []
    if nvi:
        d = _sep(nvi)
        if d["less_exploitable_if_below_0"]:
            verdict_sentence.append("new-126k RAW is LESS exploitable than the re-baselined incumbent "
                                    f"(CRN diff {d['diff_bb_per_100']:+.1f} bb/100, CI95 {d['ci95']} < 0).")
        elif d["worse_if_above_0"]:
            verdict_sentence.append("new-126k RAW is MORE exploitable than incumbent "
                                    f"(CRN diff {d['diff_bb_per_100']:+.1f} bb/100, CI95 {d['ci95']} > 0).")
        else:
            verdict_sentence.append("new-126k RAW vs incumbent exploitability: CI straddles 0 "
                                    f"(CRN diff {d['diff_bb_per_100']:+.1f} bb/100, CI95 {d['ci95']}); "
                                    "not separated.")
    if svr:
        d = _sep(svr)
        if d["less_exploitable_if_below_0"]:
            verdict_sentence.append("search arm is LESS exploitable than raw (S4 form proven: CRN diff "
                                    f"{d['diff_bb_per_100']:+.1f} bb/100, CI95 {d['ci95']} < 0).")
        elif d["worse_if_above_0"]:
            verdict_sentence.append("search arm is MORE exploitable than raw (S4 form REFUTED: CRN diff "
                                    f"{d['diff_bb_per_100']:+.1f} bb/100, CI95 {d['ci95']} > 0).")
        else:
            verdict_sentence.append("search-vs-raw exploitability CI straddles 0 (CRN diff "
                                    f"{d['diff_bb_per_100']:+.1f} bb/100, CI95 {d['ci95']}); consistent with "
                                    "not-more-exploitable but not separated.")

    verdict = {
        "run_id": args.run_id,
        "assembled_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "instrument_reconciliation": recon,
        "lbr_config": {"menu": "full (half-pot+pot+all-in; strongest standard graded set)",
                       "estimator": "unpaired raw (paired estimator discarded: variance-inflating in 6-max)",
                       "settle_showdown_leaf": True,
                       "crn": "across-arms by matched (seed, hand_idx); seat rotation built-in"},
        "arms": arms,
        "crn_contrasts": contrasts,
        "verdict": verdict_sentence,
        "headline": {
            "new_raw_bb_per_100": arms["new_raw"].get("bb_per_100"),
            "new_search_bb_per_100": arms["new_search"].get("bb_per_100"),
            "incumbent_bb_per_100": arms["incumbent"].get("bb_per_100"),
            "new_minus_incumbent_bb_per_100": (nvi or {}).get("bb_per_100"),
            "search_minus_raw_bb_per_100": (svr or {}).get("bb_per_100"),
        },
    }
    with open(os.path.join(D, "VERDICT_FINAL.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict["headline"], indent=2))
    print("VERDICT:", " ".join(verdict_sentence) or "(insufficient data)")


if __name__ == "__main__":
    main()
