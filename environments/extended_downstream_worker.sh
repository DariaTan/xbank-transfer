#!/usr/bin/env bash
# HOST-side worker. Each representation/cache has exactly one owner.
set -euo pipefail
GPU="${1:?physical GPU 0 or 1 required}"
case "${GPU}" in 0|1) ;; *) exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
CONTAINER="extended-downstream-gpu${GPU}"
IMAGE="${IMAGE_NAME:-xbank-transfer:latest}"
exec 9>"/tmp/${CONTAINER}.lock"
flock -n 9 || { echo "Duplicate worker ${CONTAINER}" >&2; exit 1; }
UUID=$(nvidia-smi -i "${GPU}" --query-gpu=uuid --format=csv,noheader)
owned=0
cleanup() {
    if [[ "${owned}" == 1 ]]; then
        docker stop --time 20 "${CONTAINER}" >/dev/null 2>&1 || true
        docker rm "${CONTAINER}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_when_ready() {
    # Serialize admission, including the not-yet-allocated RAM reservation
    # of the other worker. GPU/device ownership is rechecked for every job.
    while true; do
        exec 8>/tmp/xbank-extended-downstream-admission.lock
        flock 8
        ram_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        workers=$(docker ps --format '{{.Names}}' | awk '/^extended-downstream-gpu[01]$/ {n++} END {print n+0}')
        gpu_mem=$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits)
        if (( ram_kib >= ((workers + 1) * 24 + 8) * 1024 * 1024 && gpu_mem < 512 )) &&
           ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "${UUID}"; then
            if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER}"; then
                flock -u 8; echo "Existing container ${CONTAINER}; refusing overwrite" >&2; return 1
            fi
            docker run -d --name "${CONTAINER}" --user "$(id -u):$(id -g)" \
                --gpus "device=${GPU}" --cpus=6 --memory=24g --memory-swap=24g --shm-size=1g \
                --read-only --tmpfs /tmp:rw,size=1g \
                -v "${REPO_DIR}:/app:ro" -v "${DATA_DIR}:/app/data" \
                -v /etc/OpenCL/vendors:/etc/OpenCL/vendors:ro \
                -e HOME=/tmp -e PYTHONPATH=/app/src -e PYTHONDONTWRITEBYTECODE=1 \
                -e OMP_NUM_THREADS=6 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
                -w /app "${IMAGE}" python -u -m "$@"
            owned=1; flock -u 8; exec 8>&-; return
        fi
        flock -u 8; exec 8>&-
        echo "WAIT GPU=${GPU} RAM=$((ram_kib/1024/1024))GiB reserved_workers=${workers} $(date --iso-8601=seconds)"
        sleep 30
    done
}

run_job() {
    local key="$1"; shift
    local log="${DATA_DIR}/logs/extended_downstream_${key}.log"
    echo "START ${key} GPU=${GPU} $(date --iso-8601=seconds)" | tee -a "${log}"
    start_when_ready "$@" >>"${log}" 2>&1
    docker logs -f "${CONTAINER}" >>"${log}" 2>&1 || true
    local status
    status=$(docker wait "${CONTAINER}")
    docker inspect --format 'exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "${CONTAINER}" >>"${log}"
    docker rm "${CONTAINER}" >/dev/null; owned=0
    if [[ "${status}" != 0 ]]; then
        echo "FAILED ${key}, exit=${status}; inspect ${log}. Completed targets are resumable." >&2
        return 1
    fi
    echo "DONE ${key} $(date --iso-8601=seconds)" | tee -a "${log}"
}

mkdir -p "${DATA_DIR}/logs"
# Listing is read-only and requires no functioning CUDA in the service container.
jobs=$(docker exec -e PYTHONDONTWRITEBYTECODE=1 xbank-transfer python -m training.tune_mlp --list-jobs --worker-index "${GPU}")
if [[ "${GPU}" == 0 ]]; then
    run_job lightgbm_chronos_fgw training.chronos_fgw lightgbm --device gpu
fi
while read -r evaluation source model; do
    [[ -z "${evaluation}" ]] && continue
    run_job "mlp_${evaluation}_${source}_${model}" training.tune_mlp \
        --evaluation "${evaluation}" --checkpoint-source "${source}" --model "${model}" --device gpu
done <<<"${jobs}"
echo "COMPLETE worker GPU=${GPU} $(date --iso-8601=seconds)"
