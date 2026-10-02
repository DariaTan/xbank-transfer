#!/usr/bin/env bash
# Host entry point: one shared preparation, two GPU queues, never NEP.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
SESSION=mbd-daily-pretrain
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session ${SESSION} already exists; refusing duplicate." >&2
    exit 1
fi
if docker ps --format '{{.Names}}' | grep -Eq '^daily-pretrain-(prepare|gpu[01])$'; then
    echo 'A daily pretraining worker already exists; refusing duplicate.' >&2
    exit 1
fi
docker image inspect "${IMAGE_NAME:-xbank-transfer:latest}" >/dev/null
test -f "${DATA_DIR}/mbd_data/daily_adapted/transactions.parquet"
mkdir -p "${DATA_DIR}/logs"
tmux new-session -d -s "${SESSION}" \
    "/bin/bash '${REPO_DIR}/environments/mbd_daily_pretraining_worker.sh' > '${DATA_DIR}/logs/mbd_daily_pretrain_controller.log' 2>&1"
echo "Started ${SESSION}. Log: ${DATA_DIR}/logs/mbd_daily_pretrain_controller.log"
