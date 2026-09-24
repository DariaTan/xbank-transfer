#!/usr/bin/env bash
# Launch the semirelaxed-FGW xbank variant without touching legacy embeddings.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"
EVAL=xbank_fgw_v2
CONFIG=/app/configs/data/xbank_fgw_v2.yaml

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' is not running." >&2
    exit 1
fi
for model in coles cotic thp nep mlm; do
    docker exec "${CONTAINER_NAME}" test -d "/app/data/checkpoints/mbd_source/${model}"
done
docker exec "${CONTAINER_NAME}" test -f "${CONFIG}"
docker exec "${CONTAINER_NAME}" test -f /app/configs/mappings/xbank_to_mbd_raw_fgw_v2.json
docker exec "${CONTAINER_NAME}" mkdir -p /app/data/logs

for gpu in 0 1; do
    session="infer-${EVAL}-gpu${gpu}"
    if docker exec "${CONTAINER_NAME}" tmux has-session -t "${session}" 2>/dev/null; then
        echo "Session '${session}' already exists; refusing duplicate inference." >&2
        exit 1
    fi
done
for gpu in 0 1; do
    session="infer-${EVAL}-gpu${gpu}"
    docker exec "${CONTAINER_NAME}" tmux new-session -d -s "${session}" \
        "/bin/bash /app/environments/infer_xbank_queue_worker.sh ${gpu} ${CONFIG} ${EVAL}"
    echo "Started ${session} on GPU ${gpu}"
done
echo "New embeddings: /app/data/embeds/${EVAL}/mbd_source/<model>/"
