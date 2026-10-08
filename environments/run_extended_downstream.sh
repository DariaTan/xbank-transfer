#!/usr/bin/env bash
# Run on the HOST after smoke tests; no existing container/process is restarted.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
for gpu in 0 1; do
    session="extended-downstream-gpu${gpu}"
    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "Existing tmux session ${session}; refusing duplicate launch" >&2; exit 1
    fi
done
mkdir -p "${DATA_DIR}/logs"
# This validates provenance before publishing any FGW hardlinks.
docker exec -e PYTHONDONTWRITEBYTECODE=1 xbank-transfer python -m training.chronos_fgw publish
for gpu in 0 1; do
    session="extended-downstream-gpu${gpu}"
    printf -v command 'bash %q %q >> %q 2>&1' \
        "${REPO_DIR}/environments/extended_downstream_worker.sh" "${gpu}" \
        "${DATA_DIR}/logs/extended_downstream_gpu${gpu}_queue.log"
    tmux new-session -d -s "${session}" "${command}"
    echo "STARTED ${session}; log ${DATA_DIR}/logs/extended_downstream_gpu${gpu}_queue.log"
done
