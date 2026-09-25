#!/usr/bin/env bash
# Run from the Docker host. One GPU is deliberate under current RAM pressure.
set -euo pipefail

GPU="${1:-0}"
case "${GPU}" in 0|1) ;; *) echo "Usage: $0 [0|1]" >&2; exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
SESSION=mbd-daily-lightgbm-gpu

if [[ ! -f /etc/OpenCL/vendors/nvidia.icd ]]; then
    echo "NVIDIA OpenCL ICD is unavailable on the host." >&2
    exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session ${SESSION} already exists; refusing duplicate." >&2
    exit 1
fi
if docker exec xbank-transfer tmux has-session -t mbd-daily-lightgbm-cv 2>/dev/null; then
    echo "Stop the old CPU daily queue before starting this GPU queue." >&2
    exit 1
fi
mkdir -p "${DATA_DIR}/logs"
tmux new-session -d -s "${SESSION}" \
    "/bin/bash '${REPO_DIR}/environments/mbd_daily_lightgbm_gpu_worker.sh' '${GPU}' > '${DATA_DIR}/logs/mbd_daily_lightgbm_gpu_controller.log' 2>&1"
echo "Started ${SESSION} on GPU ${GPU}. Controller: ${DATA_DIR}/logs/mbd_daily_lightgbm_gpu_controller.log"
