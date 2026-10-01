#!/usr/bin/env bash
# Start the complete Chronos probe queue from the Docker host.
set -euo pipefail

GPU="${1:-0}"
case "${GPU}" in 0|1) ;; *) echo "Usage: $0 [0|1]" >&2; exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
SESSION="chronos-lightgbm-gpu${GPU}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session ${SESSION} already exists; refusing duplicate." >&2
    exit 1
fi
if docker ps --format '{{.Names}}' | grep -qx "chronos-probe-gpu${GPU}"; then
    echo "Chronos worker already exists on GPU ${GPU}." >&2
    exit 1
fi
test -f /etc/OpenCL/vendors/nvidia.icd
docker image inspect "${IMAGE_NAME:-xbank-transfer:latest}" >/dev/null
for evaluation in mbd_raw xbank; do
    count=$(find "${DATA_DIR}/embeds/${evaluation}/zero_shot/chronos2" -maxdepth 1 -name '*.parquet' -type f | wc -l)
    if (( count != 12 )); then
        echo "${evaluation}: need 12 published Chronos dates, found ${count}" >&2
        exit 1
    fi
done
mkdir -p "${DATA_DIR}/logs"
tmux new-session -d -s "${SESSION}" \
    "/bin/bash '${REPO_DIR}/environments/chronos_lightgbm_worker.sh' '${GPU}' > '${DATA_DIR}/logs/chronos_lightgbm_controller.log' 2>&1"
echo "Started ${SESSION}; controller: ${DATA_DIR}/logs/chronos_lightgbm_controller.log"
