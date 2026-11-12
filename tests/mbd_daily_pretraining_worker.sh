#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
IMAGE_NAME="${IMAGE_NAME:-xbank-transfer:latest}"
MODE="${1:-all}"
case "${MODE}" in all|--gpu0-only) ;; *) exit 2 ;; esac

run_container() {
    local name="$1" gpu="$2"
    shift 2
    local gpu_args=()
    if [[ "${gpu}" != cpu ]]; then gpu_args=(--gpus "device=${gpu}"); fi
    docker run --rm --name "${name}" --user "$(id -u):$(id -g)" \
        "${gpu_args[@]}" --cpus=6 --memory=24g --memory-swap=24g --shm-size=1g \
        -v "${REPO_DIR}:/app:ro" -v "${DATA_DIR}:/app/data" \
        -e HOME=/tmp -e PYTHONPATH=/app/src -e PYTHONDONTWRITEBYTECODE=1 \
        -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e OMP_NUM_THREADS=6 \
        -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
        -w /app "${IMAGE_NAME}" python -u "$@"
}

wait_resources() {
    local gpu="$1" available memory utilization
    while true; do
        # Serialize the launch decisions. Existing memory-capped containers
        # reserve their FULL remaining headroom, not just their current RSS.
        exec 9>"${DATA_DIR}/logs/mbd_daily_pretrain_resources.lock"
        flock 9
        available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        local reserve=0 name percent remaining
        for name in daily-pretrain-prepare daily-pretrain-gpu0 daily-pretrain-gpu1; do
            if [[ "$(docker inspect --format '{{.State.Running}}' "${name}" 2>/dev/null || true)" == true ]]; then
                percent=$(docker stats --no-stream --format '{{.MemPerc}}' "${name}" | tr -d '%')
                remaining=$(awk -v p="${percent}" 'BEGIN {printf "%.0f", 24*1024*1024*(100-p)/100+2048}')
                reserve=$((reserve + remaining))
            fi
        done
        memory=0; utilization=0
        if [[ "${gpu}" != cpu ]]; then
            read -r memory utilization <<< "$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')"
        fi
        if (( available >= 32 * 1024 * 1024 + reserve && memory < 1024 && utilization < 10 )); then
            # Keep fd9 locked until this container exists; caller releases it
            # after startup. This closes the simultaneous launch race.
            return
        fi
        flock -u 9
        exec 9>&-
        echo "Waiting for GPU=${gpu}: available=$((available / 1024 / 1024)) GiB, reserve=$((reserve / 1024 / 1024)) GiB, GPU=${memory} MiB/${utilization}%"
        sleep 30
    done
}

run_job() {
    local gpu="$1" model="$2" name="daily-pretrain-gpu${1}"
    local log="${DATA_DIR}/logs/${model}_mbd_daily_v2.log"
    wait_resources "${gpu}"
    echo "START ${model} GPU=${gpu} $(date --iso-8601=seconds)" | tee -a "${log}"
    # Full-vocabulary shape preflight is a separate process, so neither its
    # optimizer step nor its RNG consumption can affect actual training.
    if ! run_container "${name}" "${gpu}" -m training.daily_cli train --model "${model}" --preflight >> "${log}" 2>&1; then
        echo "FAILED ${model} preflight; inspect ${log}" | tee -a "${log}"
        return 1
    fi
    run_container "${name}" "${gpu}" -m training.daily_cli train --model "${model}" >> "${log}" 2>&1 9>&- &
    local job_pid=$!
    # Wait until Docker registered the reservation, or startup failed.
    until docker inspect "${name}" >/dev/null 2>&1; do
        if ! kill -0 "${job_pid}" 2>/dev/null; then
            flock -u 9; exec 9>&-
            wait "${job_pid}"
            echo "DONE ${model} $(date --iso-8601=seconds)" | tee -a "${log}"
            return
        fi
        sleep 1
    done
    flock -u 9; exec 9>&-
    if wait "${job_pid}"; then
        echo "DONE ${model} $(date --iso-8601=seconds)" | tee -a "${log}"
    else
        echo "FAILED ${model}; inspect ${log}; rerun launcher to resume." | tee -a "${log}"
        return 1
    fi
}

if [[ "${MODE}" == --gpu0-only ]]; then
    # Reuse frozen preparation; never relaunch the live GPU1 queue.
    echo "RECOVERY MLM -> CoLES, GPU0 only $(date --iso-8601=seconds)"
    run_job 0 mlm
    run_job 0 coles
    echo "COMPLETE GPU0 recovery $(date --iso-8601=seconds)"
    exit 0
fi

wait_resources cpu
echo "START shared preparation $(date --iso-8601=seconds)"
# Preparation is alone; the reservation lock can remain held until it exits.
run_container daily-pretrain-prepare cpu -m training.daily_cli prepare \
    >> "${DATA_DIR}/logs/mbd_daily_pretrain_prepare.log" 2>&1
flock -u 9; exec 9>&-
echo "DONE shared preparation $(date --iso-8601=seconds)"

# A model failure does not terminate the other GPU's queue.
(run_job 0 mlm; run_job 0 coles) & first_pid=$!
(run_job 1 thp; run_job 1 cotic) & second_pid=$!
first_status=0; second_status=0
wait "${first_pid}" || first_status=$?
wait "${second_pid}" || second_status=$?
if (( first_status || second_status )); then
    echo "Incomplete queues: GPU0=${first_status}, GPU1=${second_status}. Check per-model logs."
    exit 1
fi
echo "COMPLETE four daily encoders $(date --iso-8601=seconds); existing NEP untouched."
