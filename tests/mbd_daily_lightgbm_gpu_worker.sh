#!/usr/bin/env bash
# Host-side, single-GPU MBD daily continuation. Existing fold results are reused.
set -euo pipefail

GPU="${1:?Usage: mbd_daily_lightgbm_gpu_worker.sh GPU_ID}"
case "${GPU}" in 0|1) ;; *) echo "GPU must be 0 or 1" >&2; exit 2 ;; esac

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
IMAGE_NAME="${IMAGE_NAME:-xbank-transfer:latest}"
CONTAINER_NAME="mbd-daily-lgb-gpu${GPU}"
mkdir -p "${DATA_DIR}/logs"

cleanup() {
    docker stop --time 10 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for model in cotic thp nep mlm coles; do
    for fold in 0 1 2 3 4; do
        log="${DATA_DIR}/logs/mbd_daily_lightgbm_gpu_${model}_fold${fold}.log"
        while true; do
            while true; do
                available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
                if (( available_kib >= 30 * 1024 * 1024 )); then break; fi
                echo "Waiting for host RAM: $((available_kib / 1024 / 1024)) GiB available; need 30 GiB" | tee -a "${log}"
                sleep 60
            done
            echo "=== START ${model} fold=${fold} GPU=${GPU} $(date --iso-8601=seconds) ===" | tee -a "${log}"
            docker run --rm \
                --name "${CONTAINER_NAME}" \
                --user "$(id -u):$(id -g)" \
                --gpus "device=${GPU}" \
                --cpus=6 --memory=22g --memory-swap=22g --shm-size=2g \
                -v "${REPO_DIR}:/app:ro" \
                -v "${DATA_DIR}:/app/data" \
                -v /etc/OpenCL/vendors:/etc/OpenCL/vendors:ro \
                -e HOME=/tmp -e PYTHONPATH=/app/src \
                -e OMP_NUM_THREADS=6 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
                -w /app "${IMAGE_NAME}" \
                python -u /app/src/training/tune_lightgbm_mbd.py \
                    --model "${model}" --data-config /app/configs/data/mbd_daily.yaml \
                    --test-fold "${fold}" --trials 4 --threads 6 \
                    --tune-client-cap 50000 --max-rounds 400 --device-type gpu \
                    >> "${log}" 2>&1 &
            worker_pid=$!
            low_memory=0
            while kill -0 "${worker_pid}" 2>/dev/null; do
                sleep 20
                available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
                if (( available_kib < 8 * 1024 * 1024 )); then
                    echo "Pausing ${model} fold=${fold}: host RAM below 8 GiB" | tee -a "${log}"
                    docker stop --time 10 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
                    low_memory=1
                    break
                fi
            done
            if wait "${worker_pid}"; then status=0; else status=$?; fi
            if (( low_memory )); then
                echo "Will retry ${model} fold=${fold} when RAM recovers" | tee -a "${log}"
                continue
            fi
            if (( status != 0 )); then
                echo "FAILED ${model} fold=${fold} exit=${status}; see ${log}" >&2
                exit "${status}"
            fi
            echo "=== DONE ${model} fold=${fold} $(date --iso-8601=seconds) ===" | tee -a "${log}"
            break
        done
    done
done

docker run --rm --user "$(id -u):$(id -g)" --cpus=2 --memory=4g \
    -v "${REPO_DIR}:/app:ro" -v "${DATA_DIR}:/app/data" \
    -e HOME=/tmp -e PYTHONPATH=/app/src -w /app "${IMAGE_NAME}" \
    python -u /app/src/training/summarize_lightgbm_mbd_cv.py --evaluation-name mbd_daily \
    > "${DATA_DIR}/logs/mbd_daily_lightgbm_gpu_summary.log" 2>&1
echo "Five-fold MBD daily summary is ready."
