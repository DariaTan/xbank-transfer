#!/usr/bin/env bash
# Five-fold HPO on MBD daily embeddings after the MBD raw queue completes.
set -euo pipefail

export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /app/data/logs

raw_summary=/app/data/downstream/mbd_raw/mbd_source/lightgbm_hpo_cv_summary/results_aggregated.csv
if [[ ! -f "${raw_summary}" ]]; then
    echo "MBD raw five-fold summary is not complete: ${raw_summary}" >&2
    exit 1
fi

wait_for_memory() {
    local available_kib
    while true; do
        available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        if (( available_kib >= 85 * 1024 * 1024 )); then
            return
        fi
        echo "Waiting for host memory: available $((available_kib / 1024 / 1024)) GiB, need 85 GiB"
        sleep 120
    done
}

for model in coles cotic thp nep mlm; do
    for fold in 0 1 2 3 4; do
        log="/app/data/logs/mbd_daily_lightgbm_cv_${model}_fold${fold}.log"
        while true; do
            wait_for_memory
            echo "=== START ${model} fold=${fold} $(date --iso-8601=seconds) ===" | tee -a "${log}"
            python -u /app/src/training/tune_lightgbm_mbd.py \
                --model "${model}" --data-config /app/configs/data/mbd_daily.yaml \
                --test-fold "${fold}" --trials 4 --threads 6 \
                --tune-client-cap 50000 --max-rounds 400 >> "${log}" 2>&1 &
            worker_pid=$!
            low_memory=0
            while kill -0 "${worker_pid}" 2>/dev/null; do
                sleep 20
                available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
                if (( available_kib < 40 * 1024 * 1024 )); then
                    echo "Pausing ${model} fold=${fold}: available RAM $((available_kib / 1024 / 1024)) GiB < 40 GiB" | tee -a "${log}"
                    kill -TERM "${worker_pid}" 2>/dev/null || true
                    low_memory=1
                    break
                fi
            done
            if wait "${worker_pid}"; then
                status=0
            else
                status=$?
            fi
            if (( low_memory )); then
                echo "Will resume ${model} fold=${fold} when RAM recovers" | tee -a "${log}"
                continue
            fi
            if (( status != 0 )); then
                echo "=== FAILED ${model} fold=${fold} exit=${status} $(date --iso-8601=seconds) ===" | tee -a "${log}"
                exit "${status}"
            fi
            echo "=== DONE ${model} fold=${fold} $(date --iso-8601=seconds) ===" | tee -a "${log}"
            break
        done
    done
done

python -u /app/src/training/summarize_lightgbm_mbd_cv.py --evaluation-name mbd_daily \
    2>&1 | tee -a /app/data/logs/mbd_daily_lightgbm_cv_summary.log
