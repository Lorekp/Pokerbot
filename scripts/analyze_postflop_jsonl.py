"""Aggregate the per-node JSONL from measure_postflop_search_sanity.py (works on partial output)."""
import json, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/postflop_search_sanity.jsonl"
recs = [json.loads(l) for l in open(path) if l.strip()]
STREET = {1: "flop", 2: "turn", 3: "river"}
Ks = sorted(int(k) for k in recs[0]["search"].keys())
print(f"n nodes: {len(recs)}   iters: {Ks}")

def m(sub, key, K=None):
    if K is None:
        return np.mean([r["bp"][key] for r in sub])
    return np.mean([r["search"][str(K)][key] for r in sub])

print("\n=== AGGREGATE all_in / bet_mass (search - blueprint) ===")
for st in (1, 2, 3, None):
    sub = recs if st is None else [r for r in recs if r["street"] == st]
    if not sub:
        continue
    lbl = "ALL POSTFLOP" if st is None else STREET[st].upper()
    print(f"\n[{lbl}] n={len(sub)}")
    print(f"  BP    : all_in={m(sub,'all_in'):.3f}  bet={m(sub,'bet'):.3f}")
    for K in Ks:
        ai, bet = m(sub, "all_in", K), m(sub, "bet", K)
        # argmax shift & to_allin
        shift = np.mean([r["search"][str(K)]["argmax"] != r["bp"]["argmax"] for r in sub])
        to_ai = sum(1 for r in sub if r["search"][str(K)]["argmax"] == 10 and r["bp"]["argmax"] != 10)
        print(f"  K{K:<5}: all_in={ai:.3f} ({ai-m(sub,'all_in'):+.3f})  "
              f"bet={bet:.3f} ({bet-m(sub,'bet'):+.3f})  argmax_shift={shift:.2f}  argmax->all_in={to_ai}")

print("\n=== GROWTH SIGNATURE (does aggression RISE with K? preflop-degeneracy=YES) ===")
lo, hi = Ks[0], Ks[-1]
ai_rise = sum(1 for r in recs if r["search"][str(hi)]["all_in"] > r["search"][str(lo)]["all_in"] + 0.02)
ai_fall = sum(1 for r in recs if r["search"][str(hi)]["all_in"] < r["search"][str(lo)]["all_in"] - 0.02)
bet_rise = sum(1 for r in recs if r["search"][str(hi)]["bet"] > r["search"][str(lo)]["bet"] + 0.02)
bet_fall = sum(1 for r in recs if r["search"][str(hi)]["bet"] < r["search"][str(lo)]["bet"] - 0.02)
print(f"  all_in  K{lo}->K{hi}: RISE(+>0.02)={ai_rise}  FALL(<-0.02)={ai_fall}  flat={len(recs)-ai_rise-ai_fall}")
print(f"  bet_mass K{lo}->K{hi}: RISE={bet_rise}  FALL={bet_fall}  flat={len(recs)-bet_rise-bet_fall}")
print(f"  mean all_in: K{lo}={m(recs,'all_in',lo):.3f} -> K{hi}={m(recs,'all_in',hi):.3f}")
print(f"  mean bet   : K{lo}={m(recs,'bet',lo):.3f} -> K{hi}={m(recs,'bet',hi):.3f}")

print("\n=== over-bet count: search bet_mass exceeds blueprint by >0.20 ===")
for K in Ks:
    over = sum(1 for r in recs if r["search"][str(K)]["bet"] - r["bp"]["bet"] > 0.20)
    under = sum(1 for r in recs if r["bp"]["bet"] - r["search"][str(K)]["bet"] > 0.20)
    print(f"  K{K}: over-bet(+>0.2)={over}/{len(recs)}  under-bet(-<0.2)={under}/{len(recs)}")

print("\n=== all_in tail: how many nodes have search all_in > 0.5 (near-degenerate jam) ===")
for K in Ks:
    deg = sum(1 for r in recs if r["search"][str(K)]["all_in"] > 0.5)
    print(f"  K{K}: nodes with all_in>0.5 = {deg}/{len(recs)}")
