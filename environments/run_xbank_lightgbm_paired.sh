#!/usr/bin/env bash
# Run from the Docker host; does not modify the active xbank-transfer container.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions

if [[ ! -f /etc/OpenCL/vendors/nvidia.icd ]]; then
    echo "NVIDIA OpenCL ICD is unavailable on the host." >&2
    exit 1
fi
for model in coles cotic thp nep mlm; do
    for variant in xbank xbank_fgw_v2; do
        dir="${DATA_DIR}/embeds/${variant}/mbd_source/${model}"
        n_files=$(find "${dir}" -maxdepth 1 -type f -name '*.parquet' | wc -l)
        if (( n_files != 12 )); then
            echo "${variant}/${model}: expected 12 embedding files; found ${n_files}" >&2
            exit 1
        fi
    done
done
mkdir -p "${DATA_DIR}/logs"
for gpu in 0 1; do
    session="xbank-lgb-gpu${gpu}"
    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "Session ${session} already exists; refusing duplicate." >&2
        exit 1
    fi
done
for gpu in 0 1; do
    session="xbank-lgb-gpu${gpu}"
    tmux new-session -d -s "${session}" \
        "/bin/bash '${REPO_DIR}/environments/xbank_lightgbm_gpu_worker.sh' ${gpu} > '${DATA_DIR}/logs/xbank_lightgbm_gpu${gpu}_controller.log' 2>&1"
    echo "Started ${session}; logs: ${DATA_DIR}/logs/xbank_lightgbm_paired_*.log"
done
