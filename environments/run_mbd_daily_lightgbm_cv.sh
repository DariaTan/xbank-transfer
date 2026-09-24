#!/usr/bin/env bash
# Run from the Docker host. --wait queues a safe launch after MBD raw completes.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"
SESSION=mbd-daily-lightgbm-cv
RAW_SESSION=mbd-lightgbm-cv
RAW_SUMMARY=/app/data/downstream/mbd_raw/mbd_source/lightgbm_hpo_cv_summary/results_aggregated.csv

if [[ "${1:-}" == --wait ]]; then
    QUEUE_SESSION=mbd-daily-lightgbm-queue
    if tmux has-session -t "${QUEUE_SESSION}" 2>/dev/null; then
        echo "Session '${QUEUE_SESSION}' already exists; refusing duplicate queue." >&2
        exit 1
    fi
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    tmux new-session -d -s "${QUEUE_SESSION}" \
        "/bin/bash '${script_dir}/run_mbd_daily_lightgbm_cv.sh' --wait-worker > /mnt/storage/d.tanyushkina/transactions/logs/mbd_daily_lightgbm_queue.log 2>&1"
    echo "Queued ${SESSION} after MBD raw in host tmux session ${QUEUE_SESSION}."
    exit 0
fi

if [[ "${1:-}" == --wait-worker ]]; then
    while docker exec "${CONTAINER_NAME}" tmux has-session -t "${RAW_SESSION}" 2>/dev/null; do
        sleep 120
    done
    if ! docker exec "${CONTAINER_NAME}" test -f "${RAW_SUMMARY}"; then
        echo "MBD raw exited without a five-fold summary; daily HPO will not start." >&2
        exit 1
    fi
fi

if [[ $# -gt 0 && "${1}" != --wait-worker ]]; then
    echo "Usage: $0 [--wait]" >&2
    exit 2
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' is not running." >&2
    exit 1
fi
if docker exec "${CONTAINER_NAME}" tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' already exists; refusing duplicate HPO." >&2
    exit 1
fi
if docker exec "${CONTAINER_NAME}" tmux has-session -t "${RAW_SESSION}" 2>/dev/null; then
    echo "MBD raw LightGBM is still running; wait for its full result." >&2
    exit 1
fi
if ! docker exec "${CONTAINER_NAME}" test -f "${RAW_SUMMARY}"; then
    echo "MBD raw five-fold summary is missing: ${RAW_SUMMARY}" >&2
    exit 1
fi
docker exec "${CONTAINER_NAME}" test -f /app/data/mbd_data/daily_adapted/targets.parquet
for model in coles cotic thp nep mlm; do
    embeds_dir="/app/data/embeds/mbd_daily/mbd_source/${model}"
    docker exec "${CONTAINER_NAME}" test -f "${embeds_dir}/2022-02-01.parquet"
    docker exec "${CONTAINER_NAME}" test -f "${embeds_dir}/2023-01-01.parquet"
    n_files=$(docker exec "${CONTAINER_NAME}" find "${embeds_dir}" -maxdepth 1 -type f -name '*.parquet' | wc -l)
    if (( n_files != 12 )); then
        echo "Expected 12 MBD daily embedding dates for ${model}, found ${n_files}." >&2
        exit 1
    fi
done
docker exec "${CONTAINER_NAME}" mkdir -p /app/data/logs
docker exec "${CONTAINER_NAME}" tmux new-session -d -s "${SESSION}" \
    '/bin/bash /app/environments/mbd_daily_lightgbm_cv_worker.sh > /app/data/logs/mbd_daily_lightgbm_cv_controller.log 2>&1'
echo "Started ${SESSION} on CPU. Logs: /app/data/logs/mbd_daily_lightgbm_cv_*.log"
