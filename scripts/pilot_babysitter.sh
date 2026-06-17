#!/usr/bin/env bash
# Autonomous babysitter for the H5 hogwild pilot: keeps ONE spot box alive across spot reclaims/crashes
# (resuming from S3 each time), with HARD caps so it can never run away. Self-contained so it survives
# across agent turns (unlike ScheduleWakeup, which did not re-invoke and left a 9.5h dead gap).
set -uo pipefail
REGION=eu-central-1
RUN=run_h5_hogwild_pilot
PREFIX=pokerbot/prod6000/sd_cfr_hu/$RUN
BUCKET=video-processing-6789
MAXSEC=$((14*3600))     # hard wall-clock cap ~14h (~$18 spot) — well under the $40 budget
TARGET_ITER=200         # stop once this many 180s checkpoints exist (~10h continuous)
MAXLAUNCH=10            # safety: never launch more than this many boxes total
LAUNCH='python scripts/launch_sd_cfr_training_spot.py --instance-type m7g.16xlarge --arch arm64 --workers 56 --iterations 600 --traversals 20 --deal-sweeps 1 --max-depth 64 --betting-abstraction --postflop-buckets 200 --client-bet-sizing --trainer hogwild --hogwild-capacity 4194304 --no-sft --run-subdir '"$RUN"

start=$(date +%s); launches=0
terminate_run() { ids=$(aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=running,pending,stopping" --query "Reservations[].Instances[].InstanceId" --output text 2>/dev/null); [ -n "$ids" ] && aws ec2 terminate-instances --instance-ids $ids --region $REGION >/dev/null 2>&1 && echo "terminated: $ids"; }

while true; do
  el=$(( $(date +%s) - start ))
  iter=$(aws s3 ls s3://$BUCKET/$PREFIX/ --recursive --region $REGION 2>/dev/null | grep -oE "iter_[0-9]+" | sort -u | tail -1 | grep -oE "[0-9]+$" | sed 's/^0*//'); iter=${iter:-0}
  live=$(aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=running,pending" --query "Reservations[].Instances[].InstanceId" --output text 2>/dev/null)
  dying=$(aws ec2 describe-instances --region $REGION --filters "Name=tag:Run,Values=$RUN" "Name=instance-state-name,Values=shutting-down,stopping" --query "Reservations[].Instances[].InstanceId" --output text 2>/dev/null)
  echo "[t+${el}s] iter=$iter live='$live' dying='$dying' launches=$launches"
  if [ "$iter" -ge "$TARGET_ITER" ]; then echo "DONE: target iter $TARGET_ITER reached"; terminate_run; break; fi
  if [ "$el" -ge "$MAXSEC" ]; then echo "DONE: time cap reached"; terminate_run; break; fi
  if [ -n "$live" ]; then sleep 150; continue; fi          # healthy → just wait
  if [ -n "$dying" ]; then echo "  box terminating; waiting for quota…"; sleep 45; continue; fi
  if [ "$launches" -ge "$MAXLAUNCH" ]; then echo "STOP: max launches ($MAXLAUNCH) hit"; break; fi
  echo "  no live box → relaunch (resume from S3)"; eval "$LAUNCH" 2>&1 | grep -E "Launched" || echo "  launch failed (quota?) — retry next poll"
  launches=$((launches+1)); sleep 150
done
echo "BABYSITTER EXIT after ${el}s, $launches launches, latest iter=$iter"
