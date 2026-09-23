#!/usr/bin/env bash
# Run all MBD-raw and MBD-daily inference jobs unattended: one sequential
# queue per GPU, with at most two heavyweight processes active at once.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' is not running." >&2
    exit 1
fi

docker exec "${CONTAINER_NAME}" mkdir -p /app/data/logs

for model in coles cotic thp nep mlm; do
    checkpoint="/app/data/checkpoints/mbd_source/${model}"
    if ! docker exec "${CONTAINER_NAME}" test -d "${checkpoint}"; then
        echo "Missing checkpoint directory: ${checkpoint}" >&2
        exit 1
    fi
done

start_queue() {
    local gpu="$1"
    local session="$2"

    if docker exec "${CONTAINER_NAME}" tmux has-session -t "${session}" 2>/dev/null; then
        echo "Session '${session}' already exists; refusing to duplicate the queue." >&2
        exit 1
    fi

    docker exec "${CONTAINER_NAME}" tmux new-session -d -s "${session}" \
        "/app/environments/infer_mbd_queue_worker.sh ${gpu}"
    echo "Started ${session} on GPU ${gpu}"
}

start_queue 0 infer-mbd-gpu0
start_queue 1 infer-mbd-gpu1

echo "Queues started. Inspect with: docker exec ${CONTAINER_NAME} tmux ls"
