#!/usr/bin/env bash
# Overnight CPU validation battery -- high-N, CI-grounded strength numbers (the GPU stays on the 7-max
# neural run). Runs the turnkey measurement scripts SEQUENTIALLY (so each gets the full CPU) with
# timestamped logging per the standing "log non-trivial jobs" rule; each job also writes its own artifact.
# Launch:  nohup bash scripts/overnight_validation.sh > runs/overnight_validation/runner.log 2>&1 &
set -u
cd "$(dirname "$0")/.." 2>/dev/null || cd .
LOG=runs/overnight_validation
mkdir -p "$LOG"
export POKERBOT_SIM_PUSHFOLD_ITERS=300        # >0.5-threshold push/fold converges far sooner; default-identical decisions

ts() { date '+%Y-%m-%d %H:%M:%S'; }
run() {
  local name="$1"; shift
  echo "[$(ts)] START $name :: $*"
  python "$@" > "$LOG/$name.log" 2>&1
  echo "[$(ts)] DONE  $name (exit $?)"
}

echo "================ OVERNIGHT VALIDATION BATTERY $(ts) ================"

# 1. CASH value-extraction lever at high N (tight CI on the +bb/100 sizing gain + counter/lag safety cells).
run value_sizing scripts/validate_value_sizing.py --hands 2000 --seeds 12 \
    --archetypes calling_station,counter_exploiter,lag

# 2. TOURNAMENT PKO chip-EV != $-EV edge at high N.
run pko_edge scripts/tournament/pko_bounty_edge.py 6000

# 3. TOURNAMENT exploit-over-Nash ROI (CI-guarded) at high N.
run roi_vs_irc scripts/tournament/roi_vs_irc.py 6000 2000

# 4. SATELLITE seat-lock edge at high N (equal-start scope; asymmetric-start lock follow-up noted).
run satellite scripts/tournament/satellite_seat_edge.py 6000

echo "================ BATTERY COMPLETE $(ts) ================"
echo "artifacts: artifacts/{roi_vs_irc_result,pko_bounty_edge,satellite_seat_edge}.json + runs/overnight_validation/value_sizing.log"
