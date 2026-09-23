#!/usr/bin/env bash
# Run from the server host; workers execute inside the persistent container.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' is not running." >&2
    exit 1
fi

for model in coles cotic thp nep mlm; do
    checkpoint="/app/data/checkpoints/mbd_source/${model}"
    if ! docker exec "${CONTAINER_NAME}" test -d "${checkpoint}"; then
        echo "Missing checkpoint directory: ${checkpoint}" >&2
        exit 1
    fi
done

docker exec "${CONTAINER_NAME}" test -f /app/configs/mappings/xbank_to_mbd_raw.json
docker exec "${CONTAINER_NAME}" mkdir -p /app/data/logs

for session in infer-xbank-gpu0 infer-xbank-gpu1; do
    if docker exec "${CONTAINER_NAME}" tmux has-session -t "${session}" 2>/dev/null; then
        echo "Session '${session}' already exists; refusing to duplicate inference." >&2
        exit 1
    fi
done

for gpu in 0 1; do
    session="infer-xbank-gpu${gpu}"
    docker exec "${CONTAINER_NAME}" tmux new-session -d -s "${session}" \
        "/bin/bash /app/environments/infer_xbank_queue_worker.sh ${gpu}"
    echo "Started ${session} on GPU ${gpu}"
done

echo "Inspect with: docker exec ${CONTAINER_NAME} tmux ls"
echo "Logs: /app/data/logs/infer_xbank_mbd_source_*.log"
