#!/usr/bin/env bash
# One memory-capped worker; never competes for the other user's GPU.
set -euo pipefail

GPU="${1:?Usage: chronos_lightgbm_worker.sh GPU_ID}"
case "${GPU}" in 0|1) ;; *) exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
IMAGE_NAME="${IMAGE_NAME:-xbank-transfer:latest}"
CONTAINER_NAME="chronos-probe-gpu${GPU}"
PROBE=/app/src/training/tune_lightgbm_chronos.py
INFER=/app/src/training/infer_chronos_mbd_raw.py

cleanup() {
    docker stop --time 15 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

run_job() {
    local name="$1"
    shift
    local log="${DATA_DIR}/logs/chronos_${name}.log"
    while true; do
        local available_kib
        available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        if (( available_kib >= 32 * 1024 * 1024 )); then break; fi
        echo "Waiting before ${name}: $((available_kib / 1024 / 1024)) GiB RAM available; need 32 GiB"
        sleep 60
    done
    echo "START ${name} $(date --iso-8601=seconds)" | tee -a "${log}"
    docker run --rm --name "${CONTAINER_NAME}" \
        --user "$(id -u):$(id -g)" --gpus "device=${GPU}" \
        --cpus=6 --memory=24g --memory-swap=24g --shm-size=1g \
        -v "${REPO_DIR}:/app:ro" -v "${DATA_DIR}:/app/data" \
        -v /mnt/storage/d.tanyushkina/hf_cache:/hf_cache \
        -v /etc/OpenCL/vendors:/etc/OpenCL/vendors:ro \
        -e HOME=/tmp -e HF_HOME=/hf_cache -e PYTHONPATH=/app/src \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -e OMP_NUM_THREADS=6 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
        -w /app "${IMAGE_NAME}" python -u "$@" >> "${log}" 2>&1
    echo "DONE ${name} $(date --iso-8601=seconds)" | tee -a "${log}"
}

# Existing xbank embeddings give the first useful numbers fastest.
run_job lightgbm_xbank "${PROBE}" \
    --data-config /app/configs/data/xbank.yaml \
    --downstream-config /app/configs/models/downstream.yaml \
    --device gpu --trials 3 --max-rounds 300 --max-bin 63
run_job lightgbm_xbank_summary "${PROBE}" \
    --data-config /app/configs/data/xbank.yaml --summarize --cleanup-cache

for fold in 0 1 2 3 4; do
    run_job "lightgbm_mbd_raw_fold${fold}" "${PROBE}" \
        --data-config /app/configs/data/mbd.yaml --test-fold "${fold}" \
        --device gpu --trials 4 --tune-client-cap 50000 --max-rounds 400 --max-bin 255
done
run_job lightgbm_mbd_raw_summary "${PROBE}" \
    --data-config /app/configs/data/mbd.yaml --summarize --cleanup-cache

# Daily normalization differs from raw normalization, so its Chronos input
# cannot be relabeled as an existing raw embedding. Prepare it once.
for phase in prepare worker finalize; do
    run_job "mbd_daily_${phase}" "${INFER}" "${phase}" \
        --data-config /app/configs/data/mbd_daily.yaml \
        --downstream-config /app/configs/models/downstream_mbd.yaml \
        --workers 1 --worker-index 0 --prepare-memory-gb 8 --prepare-threads 6
done
for fold in 0 1 2 3 4; do
    run_job "lightgbm_mbd_daily_fold${fold}" "${PROBE}" \
        --data-config /app/configs/data/mbd_daily.yaml --test-fold "${fold}" \
        --device gpu --trials 4 --tune-client-cap 50000 --max-rounds 400 --max-bin 255
done
run_job lightgbm_mbd_daily_summary "${PROBE}" \
    --data-config /app/configs/data/mbd_daily.yaml --summarize --cleanup-cache
echo "Chronos LightGBM complete: xbank and five-fold MBD raw/daily."
