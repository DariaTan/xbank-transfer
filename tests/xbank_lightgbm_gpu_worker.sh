#!/usr/bin/env bash
# Host-side GPU queue. Each model compares both frozen xbank mappings.
set -euo pipefail

GPU="${1:?Usage: xbank_lightgbm_gpu_worker.sh GPU_ID}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
IMAGE_NAME="${IMAGE_NAME:-xbank-transfer:latest}"

case "${GPU}" in
    0) MODELS=(coles nep mlm) ;;
    1) MODELS=(cotic thp) ;;
    *) echo "GPU must be 0 or 1" >&2; exit 2 ;;
esac

for model in "${MODELS[@]}"; do
    log="${DATA_DIR}/logs/xbank_lightgbm_paired_${model}.log"
    while true; do
        available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        if (( available_kib >= 75 * 1024 * 1024 )); then
            break
        fi
        echo "Waiting for host memory: $((available_kib / 1024 / 1024)) GiB available; need 75 GiB" | tee -a "${log}"
        sleep 60
    done
    echo "=== START ${model} GPU=${GPU} $(date --iso-8601=seconds) ===" | tee -a "${log}"
    docker run --rm \
        --name "xbank-lgb-gpu${GPU}" \
        --user "$(id -u):$(id -g)" \
        --gpus "device=${GPU}" \
        --cpus=6 --memory=24g --shm-size=8g \
        -v "${REPO_DIR}:/app:ro" \
        -v "${DATA_DIR}:/app/data" \
        -v /etc/OpenCL/vendors:/etc/OpenCL/vendors:ro \
        -e HOME=/tmp -e PYTHONPATH=/app/src \
        -w /app "${IMAGE_NAME}" \
        python -u /app/src/training/tune_lightgbm_xbank_paired.py \
            --model "${model}" --device gpu --trials 3 \
            --threads 6 --max-rounds 300 >> "${log}" 2>&1
    echo "=== DONE ${model} GPU=${GPU} $(date --iso-8601=seconds) ===" | tee -a "${log}"
done
