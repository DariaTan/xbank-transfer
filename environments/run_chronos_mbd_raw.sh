#!/usr/bin/env bash
# Start resumable MBD-raw Chronos-2 inference on both GPUs from the Docker host.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"
SESSION=chronos-mbd-raw

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' is not running." >&2
    exit 1
fi
if docker exec "${CONTAINER_NAME}" tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' already exists; refusing a duplicate run." >&2
    exit 1
fi
docker exec "${CONTAINER_NAME}" test -f /app/data/mbd_data/raw_adapted/transactions.parquet
docker exec "${CONTAINER_NAME}" test -f /app/data/mbd_data/raw_adapted/targets.parquet
docker exec "${CONTAINER_NAME}" nvidia-smi --query-gpu=index --format=csv,noheader
docker exec "${CONTAINER_NAME}" tmux new-session -d -s "${SESSION}" \
    /app/environments/chronos_mbd_raw_worker.sh
echo "Started ${SESSION}. Check docker exec ${CONTAINER_NAME} tmux ls and /app/data/logs/chronos_mbd_raw_*.log"
