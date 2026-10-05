#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU id required}"
case "${GPU}" in 0) models='coles nep mlm' ;; 1) models='cotic thp' ;; *) exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
IMAGE_NAME="${IMAGE_NAME:-xbank-transfer:latest}"
NAME="daily-infer-gpu${GPU}"

wait_resources() {
    local available used utilization reserve name percent remaining
    while true; do
        exec 9>"${DATA_DIR}/logs/daily_infer_resources.lock"
        flock 9
        available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        reserve=0
        for name in daily-infer-gpu0 daily-infer-gpu1; do
            if [[ "$(docker inspect --format '{{.State.Running}}' "${name}" 2>/dev/null || true)" == true ]]; then
                percent=$(docker stats --no-stream --format '{{.MemPerc}}' "${name}" | tr -d '%')
                remaining=$(awk -v p="${percent}" 'BEGIN {printf "%.0f", 32*1024*1024*(100-p)/100+2048}')
                reserve=$((reserve + remaining))
            fi
        done
        read -r used utilization <<< "$(nvidia-smi -i "${GPU}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')"
        if (( available >= 40*1024*1024+reserve && used < 1024 && utilization < 10 )); then return; fi
        flock -u 9; exec 9>&-
        echo "WAIT GPU${GPU} available=$((available/1024/1024))GiB reserved=$((reserve/1024/1024))GiB GPU=${used}MiB/${utilization}%"
        sleep 30
    done
}

for model in ${models}; do
    for evaluation in mbd_daily xbank_fgw_v2 xbank; do
        wait_resources
        log="${DATA_DIR}/logs/infer_${evaluation}_mbd_daily_source_${model}.log"
        echo "START ${evaluation}/${model} GPU=${GPU} $(date --iso-8601=seconds)" | tee -a "${log}"
        docker run --rm --name "${NAME}" --user "$(id -u):$(id -g)" \
            --gpus "device=${GPU}" --cpus=6 --memory=32g --memory-swap=32g --shm-size=1g \
            -v "${REPO_DIR}:/app:ro" -v "${DATA_DIR}:/app/data" \
            -e HOME=/tmp -e PYTHONPATH=/app/src -e PYTHONDONTWRITEBYTECODE=1 \
            -e OMP_NUM_THREADS=6 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
            -w /app "${IMAGE_NAME}" python -u -m training.infer_mbd \
            --model "${model}" --checkpoint-source mbd_daily --require-cuda \
            --data-config "/app/configs/data/${evaluation}.yaml" \
            --downstream-config /app/configs/models/inference_daily_source.yaml \
            >> "${log}" 2>&1 9>&- &
        pid=$!
        until docker inspect "${NAME}" >/dev/null 2>&1; do
            if ! kill -0 "${pid}" 2>/dev/null; then break; fi
            sleep 1
        done
        flock -u 9; exec 9>&-
        if wait "${pid}"; then
            echo "DONE ${evaluation}/${model} $(date --iso-8601=seconds)" | tee -a "${log}"
        else
            echo "FAILED ${evaluation}/${model}; inspect ${log}. Other GPU queue is independent." | tee -a "${log}"
            exit 1
        fi
    done
done
echo "COMPLETE daily-source inference GPU${GPU} $(date --iso-8601=seconds)"
