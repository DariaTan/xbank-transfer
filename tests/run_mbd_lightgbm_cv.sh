#!/usr/bin/env bash
# Run from the Docker host; one resumable CPU queue completes five-fold HPO.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"
SESSION=mbd-lightgbm-cv

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' is not running." >&2
    exit 1
fi
if docker exec "${CONTAINER_NAME}" tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' already exists; refusing duplicate HPO." >&2
    exit 1
fi
docker exec "${CONTAINER_NAME}" test -f /app/data/mbd_data/raw_adapted/targets.parquet
docker exec "${CONTAINER_NAME}" test -f /app/src/training/summarize_lightgbm_mbd_cv.py
for model in coles cotic thp nep mlm; do
    for target in col_2 col_3 col_4 col_5; do
        previous="/app/data/downstream/mbd_raw/mbd_source/${model}/lightgbm_hpo_holdout"
        docker exec "${CONTAINER_NAME}" test -f "${previous}/${target}_metrics.json"
        docker exec "${CONTAINER_NAME}" test -f "${previous}/${target}_model.txt"
    done
done
docker exec "${CONTAINER_NAME}" mkdir -p /app/data/logs
docker exec "${CONTAINER_NAME}" tmux new-session -d -s "${SESSION}" \
    '/bin/bash /app/environments/mbd_lightgbm_cv_worker.sh > /app/data/logs/mbd_raw_lightgbm_cv_controller.log 2>&1'
echo "Started ${SESSION} on CPU; fold 4 is reused from the previous run."
echo "Logs: /app/data/logs/mbd_raw_lightgbm_cv_*.log"
