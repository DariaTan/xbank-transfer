#!/usr/bin/env bash
# Start an unattended Chronos-2 xbank queue on the Docker host.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"
SESSION=chronos-xbank

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' is not running." >&2
    exit 1
fi
if docker exec "${CONTAINER_NAME}" tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' already exists; refusing a duplicate run." >&2
    exit 1
fi
docker exec "${CONTAINER_NAME}" test -f /app/configs/mappings/xbank_to_mbd_raw.json
docker exec "${CONTAINER_NAME}" test -f /app/data/xbank_data/trans_any_pos_anonym_encoded.parquet
docker exec "${CONTAINER_NAME}" test -f /app/data/xbank_data/targets_anonym_encoded.parquet
docker exec "${CONTAINER_NAME}" mkdir -p /app/data/logs
docker exec "${CONTAINER_NAME}" tmux new-session -d -s "${SESSION}" \
    '/bin/bash /app/environments/chronos_xbank_worker.sh > /app/data/logs/chronos_xbank_controller.log 2>&1'
echo "Started ${SESSION}; it waits for encoder queues, then uses both GPUs."
echo "Check /app/data/logs/chronos_xbank_controller.log"
