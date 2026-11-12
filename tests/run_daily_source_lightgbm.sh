#!/usr/bin/env bash
# Launch from the server host, not from inside the container.
set -euo pipefail
GPU="${1:-1}"
case "${GPU}" in 0|1) ;; *) echo "Usage: $0 [0|1]" >&2; exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
SESSION="daily-source-lightgbm-gpu${GPU}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session ${SESSION} already exists" >&2; exit 1
fi
test -f /etc/OpenCL/vendors/nvidia.icd
docker image inspect "${IMAGE_NAME:-xbank-transfer:latest}" >/dev/null
mkdir -p "${DATA_DIR}/logs"
tmux new-session -d -s "${SESSION}" \
    "/bin/bash '${REPO_DIR}/environments/daily_source_lightgbm_worker.sh' '${GPU}' >> '${DATA_DIR}/logs/daily_source_lightgbm_gpu${GPU}_controller.log' 2>&1"
echo "Started ${SESSION}; controller: ${DATA_DIR}/logs/daily_source_lightgbm_gpu${GPU}_controller.log"
