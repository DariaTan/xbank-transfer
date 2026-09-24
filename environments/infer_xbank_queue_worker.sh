#!/usr/bin/env bash
# One resumable xbank inference queue inside the container, pinned to one GPU.
set -euo pipefail

GPU="${1:?Usage: infer_xbank_queue_worker.sh GPU_ID}"
DATA_CONFIG="${2:-/app/configs/data/xbank.yaml}"
EVALUATION_NAME="${3:-xbank}"

run_job() {
    local model="$1"
    local log="/app/data/logs/infer_${EVALUATION_NAME}_mbd_source_${model}.log"

    echo "=== START xbank ${model} $(date --iso-8601=seconds) GPU=${GPU} ===" | tee -a "${log}"
    if ! CUDA_VISIBLE_DEVICES="${GPU}" python -u /app/src/training/infer_mbd.py \
        --model "${model}" \
        --data-config "${DATA_CONFIG}" \
        --downstream-config /app/configs/models/downstream.yaml \
        --checkpoint-source mbd 2>&1 | tee -a "${log}"; then
        echo "=== FAILED xbank ${model} $(date --iso-8601=seconds) ===" | tee -a "${log}"
        return 1
    fi
    echo "=== DONE xbank ${model} $(date --iso-8601=seconds) ===" | tee -a "${log}"
}

case "${GPU}" in
    0)
        run_job coles
        run_job nep
        run_job mlm
        ;;
    1)
        run_job cotic
        run_job thp
        ;;
    *)
        echo "Unsupported GPU id: ${GPU}; expected 0 or 1." >&2
        exit 2
        ;;
esac
