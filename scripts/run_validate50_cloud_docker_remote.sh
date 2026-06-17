#!/usr/bin/env bash
# Run validate50 label batch in Docker on homeserver (cloud-proxy worker).
set -euo pipefail

ROOT="."
cd "$ROOT"

IMAGE="${LABELGEN_IMAGE:-pokerbot-labelgen:validate50}"
OUT="$ROOT/data/validate50/cloud_docker"
MANIFEST="$ROOT/data/validate50/work_manifest.json"

mkdir -p data/validate50
if [[ ! -f "$MANIFEST" ]]; then
  PYTHONPATH=. python3 scripts/plan_label_work.py \
    --config config/solver_labels_validate50.yaml \
    --output "$MANIFEST" \
    --max-spots 50 \
    --run-id validate50_gate
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -f infra/labelgen/Dockerfile -t "$IMAGE" .
fi

rm -rf "$OUT"
mkdir -p "$OUT/shards"

docker run --rm \
  -v "$MANIFEST:/manifest.json:ro" \
  -v "$OUT:/output" \
  --entrypoint python \
  "$IMAGE" \
  scripts/build_solver_labels.py \
  --config config/solver_labels_validate50.yaml \
  --work-manifest /manifest.json \
  --output /output/train.parquet \
  --manifest /output/manifest.json \
  --report /output/run.md \
  --weva-index data/weva \
  --solver-backend subprocess \
  --num-workers 1 \
  --shard-dir /output/shards

echo "cloud_docker complete: $OUT"
