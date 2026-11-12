#!/usr/bin/env bash
# Host entry point: one shared preparation, two GPU queues, never NEP.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
SESSION=mbd-daily-pretrain
MODE="${1:-all}"
case "${MODE}" in
    all) names='^daily-pretrain-(prepare|gpu[01])$' ;;
    --gpu0-only)
        SESSION=mbd-daily-pretrain-gpu0
        names='^daily-pretrain-(prepare|gpu0)$'
        test -f "${DATA_DIR}/training_cache/mbd_daily/v2/ready.json"
        ;;
    *) echo "Usage: bash $0 [--gpu0-only]" >&2; exit 2 ;;
esac
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session ${SESSION} already exists; refusing duplicate." >&2
    exit 1
fi
if docker ps --format '{{.Names}}' | grep -Eq "${names}"; then
    echo 'A daily pretraining worker already exists; refusing duplicate.' >&2
    exit 1
fi
docker image inspect "${IMAGE_NAME:-xbank-transfer:latest}" >/dev/null
test -f "${DATA_DIR}/mbd_data/daily_adapted/transactions.parquet"
mkdir -p "${DATA_DIR}/logs"
tmux new-session -d -s "${SESSION}" \
    "/bin/bash '${REPO_DIR}/environments/mbd_daily_pretraining_worker.sh' '${MODE}' >> '${DATA_DIR}/logs/${SESSION}_controller.log' 2>&1"
echo "Started ${SESSION}. Log: ${DATA_DIR}/logs/${SESSION}_controller.log"
