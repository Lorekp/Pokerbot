#!/usr/bin/env bash
# Root-cause probe: EV(open AA) - EV(limp AA) under the TRAINED postflop field vs the CHECK-CALL
# (full-realization) field, served from the SAME snapshot (matched current-iterate preflop).
# If trained flips to limp>=open while checkcall says open>>limp => POSTFLOP root. Else preflop.
set -e
cd .
SNAP=artifacts/bpv4_ladder/arm3_snap10k.nlsnap
SNAPFILE=artifacts/bpv4_ladder/arm3_snap10k.nlsnap
for field in blueprint checkcall; do
  pids=""
  for s in 0 1 2 3; do
    POKERBOT_BP_COVERAGE=1 python -u scripts/blueprint2/bp3_forced_response_probe.py rollout \
      --blueprint "$SNAP" --hand AA --deals 1500 --start $((s*4500)) --seed 13 \
      --key-mode fullhist --postflop-mode emd --num-buckets 160 \
      --mrr --pre-keep 2,3,4,5,6 --pre-rr-keep 4,5,6 --flop-rr-keep 1,4 \
      --postflop-field $field --snapshot-nlsnap-file "$SNAPFILE" \
      --out artifacts/_arm3_root_${field}_AA.shard$s.json > artifacts/_arm3_root_${field}_AA.shard$s.out 2>&1 &
    pids="$pids $!"
  done
  wait $pids
  python scripts/blueprint2/bp3_forced_response_probe.py merge \
    --out artifacts/_arm3_root_${field}_AA_merged.json \
    artifacts/_arm3_root_${field}_AA.shard{0,1,2,3}.json > /dev/null 2>&1
  echo "[$field] merged"
done
echo "=== ROOT-CAUSE COMPARISON (AA, bb/100) ==="
python - <<'PY'
import json
for field in ("blueprint","checkcall"):
    d=json.load(open(f"artifacts/_arm3_root_{field}_AA_merged.json"))
    h=d["hands"]["AA"]
    pf=h.get("postflop_miss_rate",0.0)
    o=h["paired_vs_limp"].get("open325_minus_limp",{})
    limp=h["arms"]["limp"]["ev_bb100"]; opn=h["arms"]["open325"]["ev_bb100"]
    lab="TRAINED postflop" if field=="blueprint" else "CHECK-CALL (full realization)"
    print(f"{lab:32s}: open325-limp = {o.get('delta_bb100',0):+7.1f} +/- {o.get('se_bb100',0):.1f}"
          f"   [limp EV {limp:+.0f}, open EV {opn:+.0f}]  postflop_miss={pf:.3f}")
PY
