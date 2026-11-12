#!/usr/bin/env bash
# Runs inside the container. A failed inference stops this GPU queue while the
# other tmux queue remains independent.
set -euo pipefail

GPU="${1:?Usage: infer_mbd_queue_worker.sh GPU_ID}"

run_job() {
    local evaluation_name="$1"
    local model="$2"
    local data_config="$3"
    local log="/app/data/logs/infer_${evaluation_name}_${model}.log"

    echo "=== START ${evaluation_name} ${model} $(date --iso-8601=seconds) ===" | tee -a "${log}"
    CUDA_VISIBLE_DEVICES="${GPU}" python /app/src/training/infer_mbd.py \
        --model "${model}" \
        --data-config "/app/configs/data/${data_config}.yaml" \
        --downstream-config /app/configs/models/downstream_mbd.yaml \
        --checkpoint-source mbd 2>&1 | tee -a "${log}"
    echo "=== DONE ${evaluation_name} ${model} $(date --iso-8601=seconds) ===" | tee -a "${log}"
}

case "${GPU}" in
    0)
        run_job mbd_raw coles mbd
        run_job mbd_raw thp mbd
        run_job mbd_raw mlm mbd
        run_job mbd_daily coles mbd_daily
        run_job mbd_daily thp mbd_daily
        run_job mbd_daily mlm mbd_daily
        ;;
    1)
        run_job mbd_raw cotic mbd
        run_job mbd_raw nep mbd
        run_job mbd_daily cotic mbd_daily
        run_job mbd_daily nep mbd_daily
        ;;
    *)
        echo "Unsupported GPU id: ${GPU}; expected 0 or 1." >&2
        exit 2
        ;;
esac
