#!/usr/bin/env bash
# Host-side queue: daily-pretrained probes only, one physical GPU, no inference changes.
set -euo pipefail
GPU="${1:-1}"
case "${GPU}" in 0|1) ;; *) exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
CONTAINER="daily-source-lightgbm-gpu${GPU}"
IMAGE="${IMAGE_NAME:-xbank-transfer:latest}"
exec 9>"/tmp/${CONTAINER}.lock"
flock -n 9 || { echo "Worker already holds ${CONTAINER} lock" >&2; exit 1; }
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "Container ${CONTAINER} already exists; refusing duplicate" >&2
    exit 1
fi
UUID=$(nvidia-smi -i "${GPU}" --query-gpu=uuid --format=csv,noheader)
cleanup() { docker stop --time 20 "${CONTAINER}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_resources() {
    while true; do
        local ram_kib
        ram_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        if (( ram_kib >= 40 * 1024 * 1024 )) &&
           ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "${UUID}"; then
            break
        fi
        echo "WAIT resources GPU=${GPU} RAM=$((ram_kib / 1024 / 1024))GiB $(date --iso-8601=seconds)"
        sleep 30
    done
}

container_run() {
    docker run --rm --name "${CONTAINER}" --user "$(id -u):$(id -g)" \
        --gpus "device=${GPU}" --cpus=6 --memory=24g --memory-swap=24g --shm-size=1g \
        --read-only --tmpfs /tmp:rw,size=512m \
        -v "${REPO_DIR}:/app:ro" -v "${DATA_DIR}:/app/data" \
        -v /etc/OpenCL/vendors:/etc/OpenCL/vendors:ro \
        -e HOME=/tmp -e PYTHONPATH=/app/src -e PYTHONDONTWRITEBYTECODE=1 \
        -e OMP_NUM_THREADS=6 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
        -w /app "${IMAGE}" python -u -m training.tune_lightgbm_daily_source "$@"
}

run_job() {
    local job="$1"; shift
    wait_resources
    local log="${DATA_DIR}/logs/daily_source_lightgbm_${job}.log"
    echo "START ${job} GPU=${GPU} $(date --iso-8601=seconds)" | tee -a "${log}"
    if ! container_run "$@" >>"${log}" 2>&1; then
        echo "FAILED ${job}; inspect ${log}" >&2
        return 1
    fi
    echo "DONE ${job} $(date --iso-8601=seconds)" | tee -a "${log}"
}

# One outstanding job per (model, corpus). Do not let pending MLM block ready encoders.
declare -A finished
remaining=10
while (( remaining > 0 )); do
    progressed=0
    for model in cotic thp coles nep mlm; do
        for corpus in xbank mbd_daily; do
            key="${corpus}_${model}"
            [[ "${finished[$key]:-0}" == 1 ]] && continue
            wait_resources
            status=0
            container_run --model "${model}" --corpus "${corpus}" --ready || status=$?
            if (( status == 75 )); then continue; fi
            if (( status != 0 )); then echo "Invalid inference inputs: ${key}" >&2; exit "${status}"; fi
            if [[ "${corpus}" == mbd_daily ]]; then
                for fold in 0 1 2 3 4; do
                    run_job "${key}_fold${fold}" --model "${model}" --corpus "${corpus}" --test-fold "${fold}"
                done
            else
                run_job "${key}_paired" --model "${model}" --corpus "${corpus}"
            fi
            run_job "${key}_summary" --model "${model}" --corpus "${corpus}" --summarize --cleanup-cache
            finished[$key]=1
            remaining=$((remaining - 1))
            progressed=1
        done
    done
    if (( progressed == 0 )); then
        echo "WAIT published embeddings: ${remaining} jobs remain $(date --iso-8601=seconds)"
        sleep 60
    fi
done
echo "COMPLETE: five daily-source encoders, MBD five folds and both paired xbank mappings $(date --iso-8601=seconds)"
