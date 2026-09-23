#!/usr/bin/env bash
# Runs inside the container in one tmux session. Both GPU workers own disjoint
# client shards, so completed shard files survive a disconnect or restart.
set -euo pipefail

SCRIPT=/app/src/training/infer_chronos_mbd_raw.py
LOG_DIR=/app/data/logs
mkdir -p "${LOG_DIR}"

echo "=== Chronos MBD-raw prepare $(date --iso-8601=seconds) ==="
python -u "${SCRIPT}" prepare 2>&1 | tee -a "${LOG_DIR}/chronos_mbd_raw_prepare.log"

echo "=== Chronos GPU workers $(date --iso-8601=seconds) ==="
CUDA_VISIBLE_DEVICES=0 python -u "${SCRIPT}" worker --worker-index 0 --workers 2 \
    2>&1 | tee -a "${LOG_DIR}/chronos_mbd_raw_gpu0.log" &
gpu0_pid=$!
CUDA_VISIBLE_DEVICES=1 python -u "${SCRIPT}" worker --worker-index 1 --workers 2 \
    2>&1 | tee -a "${LOG_DIR}/chronos_mbd_raw_gpu1.log" &
gpu1_pid=$!

set +e
wait "${gpu0_pid}"
gpu0_status=$?
wait "${gpu1_pid}"
gpu1_status=$?
set -e
if (( gpu0_status != 0 || gpu1_status != 0 )); then
    echo "Chronos workers failed: gpu0=${gpu0_status}, gpu1=${gpu1_status}" >&2
    exit 1
fi

python -u "${SCRIPT}" finalize 2>&1 | tee -a "${LOG_DIR}/chronos_mbd_raw_finalize.log"
echo "=== Chronos MBD-raw DONE $(date --iso-8601=seconds) ==="
