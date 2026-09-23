#!/usr/bin/env bash
# Runs inside the container after the five MBD-source xbank encoders finish.
set -euo pipefail

SCRIPT=/app/src/training/infer_chronos_mbd_raw.py
DATA_CONFIG=/app/configs/data/xbank.yaml
DOWNSTREAM_CONFIG=/app/configs/models/downstream.yaml
LOG_DIR=/app/data/logs
mkdir -p "${LOG_DIR}"

echo "Waiting for xbank encoder queues to finish before using both GPUs..."
while tmux has-session -t infer-xbank-gpu0 2>/dev/null || tmux has-session -t infer-xbank-gpu1 2>/dev/null; do
    sleep 60
done
for model in coles cotic thp nep mlm; do
    if [[ ! -f "/app/data/embeds/xbank/mbd_source/${model}/2024-02-01.parquet" ]]; then
        echo "Encoder ${model} did not publish the final target date; refusing to start Chronos." >&2
        exit 1
    fi
done

echo "=== Chronos xbank prepare $(date --iso-8601=seconds) ==="
python -u "${SCRIPT}" prepare --data-config "${DATA_CONFIG}" \
    --downstream-config "${DOWNSTREAM_CONFIG}" \
    2>&1 | tee -a "${LOG_DIR}/chronos_xbank_prepare.log"

echo "=== Chronos xbank GPU workers $(date --iso-8601=seconds) ==="
CUDA_VISIBLE_DEVICES=0 python -u "${SCRIPT}" worker --data-config "${DATA_CONFIG}" \
    --downstream-config "${DOWNSTREAM_CONFIG}" --worker-index 0 --workers 2 \
    2>&1 | tee -a "${LOG_DIR}/chronos_xbank_gpu0.log" &
gpu0_pid=$!
CUDA_VISIBLE_DEVICES=1 python -u "${SCRIPT}" worker --data-config "${DATA_CONFIG}" \
    --downstream-config "${DOWNSTREAM_CONFIG}" --worker-index 1 --workers 2 \
    2>&1 | tee -a "${LOG_DIR}/chronos_xbank_gpu1.log" &
gpu1_pid=$!

set +e
wait "${gpu0_pid}"
gpu0_status=$?
wait "${gpu1_pid}"
gpu1_status=$?
set -e
if (( gpu0_status != 0 || gpu1_status != 0 )); then
    echo "Chronos xbank workers failed: gpu0=${gpu0_status}, gpu1=${gpu1_status}" >&2
    exit 1
fi

python -u "${SCRIPT}" finalize --data-config "${DATA_CONFIG}" \
    --downstream-config "${DOWNSTREAM_CONFIG}" \
    2>&1 | tee -a "${LOG_DIR}/chronos_xbank_finalize.log"
echo "=== Chronos xbank DONE $(date --iso-8601=seconds) ==="
