#!/usr/bin/env bash
# Run from the server host. LightGBM stays on CPU while GPUs infer xbank.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"
SESSION=mbd-lightgbm-hpo

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' is not running." >&2
    exit 1
fi
if docker exec "${CONTAINER_NAME}" tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' already exists; refusing duplicate HPO." >&2
    exit 1
fi
docker exec "${CONTAINER_NAME}" test -f /app/data/mbd_data/raw_adapted/targets.parquet
docker exec "${CONTAINER_NAME}" mkdir -p /app/data/logs
docker exec "${CONTAINER_NAME}" tmux new-session -d -s "${SESSION}" \
    '/bin/bash /app/environments/mbd_lightgbm_hpo_worker.sh > /app/data/logs/mbd_raw_lightgbm_hpo_controller.log 2>&1'
echo "Started ${SESSION} on CPU. Check /app/data/logs/mbd_raw_lightgbm_hpo_*.log"
