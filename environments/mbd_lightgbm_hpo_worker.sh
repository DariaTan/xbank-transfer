#!/usr/bin/env bash
# CPU-only sequential HPO queue for the MBD-raw in-domain benchmark.
set -euo pipefail

export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /app/data/logs

for model in coles cotic thp nep mlm; do
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
