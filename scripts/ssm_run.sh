#!/usr/bin/env bash
# Drive a command on the metal box via SSM (no SSH key needed).
# usage: ssm_run.sh <script-file> [timeout-sec]
# Sends the script base64-encoded (quoting-proof), polls to completion, prints stdout+stderr.
set -uo pipefail
IID="${SSM_IID:-i-0d960af56bf481c9f}"
REGION="${SSM_REGION:-eu-central-1}"
SCRIPT="$1"
TIMEOUT="${2:-600}"
B64=$(base64 -w0 "$SCRIPT")
CID=$(aws ssm send-command --region "$REGION" --instance-ids "$IID" \
  --document-name AWS-RunShellCommand \
  --parameters "commands=echo $B64 | base64 -d | bash" \
  --timeout-seconds "$TIMEOUT" --query 'Command.CommandId' --output text 2>&1)
if [[ -z "$CID" || "$CID" == *rror* ]]; then echo "SEND FAILED: $CID"; exit 1; fi
echo "CommandId: $CID  (script=$SCRIPT, send-timeout=${TIMEOUT}s)"
START=$(date +%s)
ST=Pending
while true; do
  ST=$(aws ssm get-command-invocation --region "$REGION" --command-id "$CID" --instance-id "$IID" --query 'Status' --output text 2>/dev/null || echo Pending)
  case "$ST" in Success|Failed|Cancelled|TimedOut) break;; esac
  EL=$(( $(date +%s) - START ))
  if (( EL > TIMEOUT + 90 )); then echo "POLL GAVE UP after ${EL}s (last status=$ST)"; break; fi
  sleep 6
done
EL=$(( $(date +%s) - START ))
echo "=== STATUS: $ST  (elapsed ${EL}s) ==="
echo "--- STDOUT ---"
aws ssm get-command-invocation --region "$REGION" --command-id "$CID" --instance-id "$IID" --query 'StandardOutputContent' --output text
echo "--- STDERR ---"
aws ssm get-command-invocation --region "$REGION" --command-id "$CID" --instance-id "$IID" --query 'StandardErrorContent' --output text
