#!/usr/bin/env bash
# GATE POLLER for the overnight 7-MAX (tournament) neural-CFR run. ASCII only.
#
# WHAT: every POLL_SECS, run the already-built tournament net ICM-LBR suite (CPU-only, so it does NOT
# contend the GPU the trainer holds) against the LIVE checkpoint on a cheap 3-cell subset
# {NORMAL:deep, NORMAL:shallow, NORMAL:bubble}, decide the convergence GATE from the JSON, and on PASS
# stop the training cleanly (kill the train pid + write babysit.status=GATE_MET so the supervisor stops
# instead of resuming). This is the "run-to-gate" wiring the trainer does not do itself.
#
# WHY a poller (not an in-loop hook): the eval imports CPU-JAX + numba; running it in-process would
# perturb the trainer's CUDA/XLA state. A sibling CPU process reuses 100% of the test-green suite infra
# and respects the read-only-firewall (it never touches training files).
#
# GATE TIERS (pick via GATE_TIER):
#   1 (default, hard stop-gate) = less_exploitable_than_always_fold == true on EVERY polled cell.
#   2 (quality target)          = tier 1 AND chip-LBR < GATE_MARGIN * same-cell anchor floor on every
#                                 polled cell AND no cell's $-LBR verdict == POSITIVE.
#
# PROGRESS LOGGING (per the standing rule): every poll prints chip-LBR vs floor per cell + the gate
# decision + a throughput/heartbeat line; a STALL line if the checkpoint mtime has not advanced between
# polls (training is hung or already stopped). ETA-to-gate is reported as the trainer's own ETA (read
# from the live train log) since the gate is a quality threshold, not an iteration count.
#
# Run alongside the babysitter (after start_7max_overnight.ps1):
#   wsl -d Debian -- bash -lc 'nohup bash ./scripts/neural_cfr/gate_7max_poll.sh >/dev/null 2>&1 &'
# Watch:  tail -f runs/sevenmax_neural/gate.log
# Stop:   kill $(cat runs/sevenmax_neural/gate.pid)
set -uo pipefail

REPO=.
VENV="$HOME/jaxgpu-venv"
RUNDIR="$REPO/runs/sevenmax_neural"
mkdir -p "$RUNDIR"

CKPT=${CKPT:-checkpoints/sd_cfr_7max_neural/seven_v1.pkl}
POLL_SECS=${POLL_SECS:-1800}        # how often to run the gate eval (default 30 min)
SPOTS=${SPOTS:-400}                 # net-LBR spots per cell (cheap: ~10-30s CPU for the 3-cell subset)
GATE_TIER=${GATE_TIER:-1}           # 1 = below-anchor on every cell; 2 = quality target (see header)
GATE_MARGIN=${GATE_MARGIN:-0.5}     # tier-2: chip-LBR must be < MARGIN * same-cell anchor floor
CELLS=${CELLS:-"NORMAL:deep NORMAL:shallow NORMAL:bubble"}   # cheap decisive subset
STOP_ON_GATE=${STOP_ON_GATE:-1}     # 1 = kill training on PASS; 0 = log-only (advisory)
GATE_OUT="$RUNDIR/gate_eval.json"
GLOG="$RUNDIR/gate.log"

export PYTHONPATH=.
export POKERBOT_INFO_SUITS=1        # MANDATORY: checkpoint is stamped info_suits=True; bridge fails closed on mismatch
export JAX_PLATFORMS=cpu            # do NOT contend the GPU the trainer holds

echo $$ > "$RUNDIR/gate.pid"
cd "$REPO"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

glog() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$GLOG"; }

# Build the per-cell --format/--regime arg pairs from CELLS, plus a "decisive cells" list for the gate
# decider so it only inspects the cells we actually evaluated.
FORMATS=""; REGIMES=""
for c in $CELLS; do
  f="${c%%:*}"; r="${c##*:}"
  case " $FORMATS " in *" $f "*) ;; *) FORMATS="$FORMATS $f" ;; esac
  case " $REGIMES " in *" $r "*) ;; *) REGIMES="$REGIMES $r" ;; esac
done

# Gate decider: read the suite JSON, inspect ONLY the polled cells, return rc=0 on PASS. Pure stdlib.
gate_decide() {
  python - "$GATE_OUT" "$GATE_TIER" "$GATE_MARGIN" "$CELLS" <<'PY'
import json, sys
path, tier, margin, cells_spec = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
try:
    with open(path) as f:
        doc = json.load(f)
except Exception as e:  # no/partial JSON this round -> not a pass, keep training
    print(f"GATE: eval JSON unreadable ({e!r})"); sys.exit(2)
cells = doc.get("cells", {})
want = [c for c in cells_spec.split() if c]
all_pass = True
lines = []
for key in want:
    cell = cells.get(key)
    if cell is None:
        lines.append(f"  {key}: MISSING from eval"); all_pass = False; continue
    pil = (cell.get("pillars") or {}).get("E1_E2_lbr_icm") or {}
    res = pil.get("result") if pil.get("status") == "built" else None
    if not res:
        lines.append(f"  {key}: lbr_icm {pil.get('status','?')} ({pil.get('reason','')})"); all_pass = False; continue
    chip = res.get("chip_exploit_bb_per_100")
    floor = res.get("anchor_chip_exploit_at_cell_bb_per_100")
    below = bool(res.get("less_exploitable_than_always_fold"))
    verdict = res.get("verdict", "?")
    ok = below
    note = ""
    if tier >= 2:
        margin_ok = (chip is not None and floor is not None and chip < margin * floor)
        verdict_ok = (verdict != "POSITIVE")
        ok = ok and margin_ok and verdict_ok
        note = f" margin_ok={margin_ok} verdict_ok={verdict_ok}"
    all_pass = all_pass and ok
    lines.append(f"  {key}: chip_lbr={chip} floor={floor} below_anchor={below} "
                 f"verdict={verdict} cell_pass={ok}{note}")
print("GATE: tier %d margin %.2f" % (tier, margin))
for ln in lines:
    print(ln)
print("GATE: ALL_PASS" if all_pass else "GATE: NOT_YET")
sys.exit(0 if all_pass else 1)
PY
}

glog "gate poller up (PID $$). ckpt=$CKPT poll=${POLL_SECS}s tier=$GATE_TIER margin=$GATE_MARGIN cells=[$CELLS] spots=$SPOTS stop_on_gate=$STOP_ON_GATE"
echo "POLLING" > "$RUNDIR/gate.status"

poll=0
prev_ckpt_mtime=0
while true; do
  poll=$((poll + 1))
  T0=$(date +%s)
  # Heartbeat + stall detect: the live checkpoint should advance while training runs.
  CK="$REPO/${CKPT%.pkl}_full.pkl"
  CK_MTIME=$(stat -c %Y "$CK" 2>/dev/null || echo 0)
  if [ "$poll" -gt 1 ] && [ "$CK_MTIME" = "$prev_ckpt_mtime" ]; then
    glog "STALL? checkpoint $(basename "$CK") mtime unchanged since last poll ($CK_MTIME) -- training hung or already stopped."
  fi
  prev_ckpt_mtime=$CK_MTIME

  # ETA-to-gate proxy: surface the trainer's own ETA from the current train log (the gate is a quality
  # threshold, not a fixed iter count -- this tells the operator how far the underlying run has to go).
  CUR=$(cat "$RUNDIR/seven_train.curlog" 2>/dev/null || echo "")
  ETA_LINE=$(grep -E "iter [0-9]+/" "$CUR" 2>/dev/null | tail -1 || true)
  [ -n "$ETA_LINE" ] && glog "poll #$poll heartbeat: trainer @ ${ETA_LINE}"

  glog "poll #$poll: running net ICM-LBR on [$CELLS] (spots=$SPOTS, CPU)..."
  python scripts/tournament/eval_tournament_suite.py --policy net \
    --format $FORMATS --regime $REGIMES \
    --ckpt "$CKPT" --spots "$SPOTS" --out "$GATE_OUT" >>"$GLOG" 2>&1 || \
    glog "poll #$poll: suite returned non-zero (partial eval kept; will retry next poll)"

  DEC=$(gate_decide); rc=$?
  echo "$DEC" | tee -a "$GLOG"
  DT=$(( $(date +%s) - T0 ))
  glog "poll #$poll done in ${DT}s (eval throughput) -> $([ "$rc" -eq 0 ] && echo PASS || echo CONTINUE)"

  if [ "$rc" -eq 0 ]; then
    glog "GATE MET on poll #$poll."
    echo "GATE_MET" > "$RUNDIR/gate.status"
    if [ "$STOP_ON_GATE" = "1" ]; then
      echo "GATE_MET" > "$RUNDIR/babysit.status"
      TPID=$(cat "$RUNDIR/seven_train.pid" 2>/dev/null || echo "")
      BPID=$(cat "$RUNDIR/babysit.pid" 2>/dev/null || echo "")
      # stop the babysitter FIRST so it does not resume the run we are about to kill.
      [ -n "$BPID" ] && kill "$BPID" 2>/dev/null && glog "stopped babysitter PID $BPID"
      [ -n "$TPID" ] && kill "$TPID" 2>/dev/null && glog "stopped training PID $TPID (gate met -- converged)"
    else
      glog "STOP_ON_GATE=0: advisory only, leaving training running."
    fi
    break
  fi

  sleep "$POLL_SECS"
done
glog "gate poller exiting."
