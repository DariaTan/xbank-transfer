#!/usr/bin/env bash
# CPU-only sequential HPO queue for the MBD-raw in-domain benchmark.
set -euo pipefail

export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /app/data/logs

wait_for_memory() {
    # The host is shared and has no swap. A concurrent user's large job can
    # leave too little headroom for the pandas join + LightGBM Dataset.
    local required_kib=$((70 * 1024 * 1024))
    local available_kib
    while true; do
        available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        if (( available_kib >= required_kib )); then
            return
        fi
        echo "Waiting for host memory: available $((available_kib / 1024 / 1024)) GiB, need 70 GiB"
        sleep 120
    done
}

for model in coles cotic thp nep mlm; do
    wait_for_memory
    log="/app/data/logs/mbd_raw_lightgbm_hpo_${model}.log"
    echo "=== START ${model} $(date --iso-8601=seconds) ===" | tee -a "${log}"
    if ! python -u /app/src/training/tune_lightgbm_mbd.py \
        --model "${model}" --trials 4 --threads 6 \
        --tune-client-cap 50000 --max-rounds 400 \
        2>&1 | tee -a "${log}"; then
        echo "=== FAILED ${model} $(date --iso-8601=seconds) ===" | tee -a "${log}"
        exit 1
    fi
    echo "=== DONE ${model} $(date --iso-8601=seconds) ===" | tee -a "${log}"
done
