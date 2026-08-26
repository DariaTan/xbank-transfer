#!/usr/bin/env bash
# Start the xbank-transfer container as a persistent background service
# (kept alive via `tail -f /dev/null`, not an interactive shell) so a
# training run survives an SSH disconnect. Use dexec.sh to get a shell in it.
#
# Mounts:
#   repo root                               -> /app                    (project code, editable live)
#   /mnt/storage/d.tanyushkina/transactions -> /app/data               (read-only, avoids clobbering the raw parquet)
#   /mnt/storage/d.tanyushkina/hf_cache     -> /root/.cache/huggingface (read-write, survives container recreation)
#
# The HF cache mount matters for Chronos-2: it's a real pretrained
# checkpoint pulled from the Hub (Apache-2.0, public weights, not
# proprietary data), and without this mount it silently re-downloads
# every time the container is recreated.
#
# PYTHONPATH=/app/src makes `xbank.*` importable from anywhere in the
# container (no per-script sys.path hack needed) -- src/xbank/training/
# holds the smoke_*.py / train_*.py entry-point scripts directly inside
# the package now, run as e.g. `python src/xbank/training/train_thp.py`.
#
# This host has 2x RTX A5000 (24GB each) shared with other users' containers.
# Override GPUS to grab just one, e.g.:  GPUS='"device=0"' ./drun.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="/mnt/storage/d.tanyushkina/transactions"
HF_CACHE_DIR="/mnt/storage/d.tanyushkina/hf_cache"
IMAGE_NAME="${IMAGE_NAME:-xbank-transfer:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"
GPUS="${GPUS:-all}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' is already running."
    exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' exists but is stopped -- starting it."
    docker start "${CONTAINER_NAME}"
    exit 0
fi

mkdir -p "${HF_CACHE_DIR}"

docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus "${GPUS}" \
    --shm-size=16g \
    -v "${REPO_DIR}:/app" \
    -v "${DATA_DIR}:/app/data:ro" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/app/src \
    -w /app \
    "${IMAGE_NAME}" \
    tail -f /dev/null

echo "Started '${CONTAINER_NAME}'. Use ./environments/dexec.sh for a shell."
