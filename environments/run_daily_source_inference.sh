#!/usr/bin/env bash
# Docker HOST: two independent queues using frozen daily-source checkpoints.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/mnt/storage/d.tanyushkina/transactions
IMAGE_NAME="${IMAGE_NAME:-xbank-transfer:latest}"
docker image inspect "${IMAGE_NAME}" >/dev/null
if docker ps --format '{{.Names}}' | grep -Eq '^daily-(pretrain|infer)-(prepare|gpu[01])$'; then
    echo 'A daily training/inference worker already exists; refusing duplicate.' >&2
    exit 1
fi
for gpu in 0 1; do
    if tmux has-session -t "daily-source-infer-gpu${gpu}" 2>/dev/null; then
        echo "GPU${gpu} queue session already exists." >&2; exit 1
    fi
done
for model in coles cotic thp mlm nep; do
    checkpoint="${DATA_DIR}/checkpoints/mbd_daily_source/${model}"
    case "${model}" in coles|cotic) test -f "${checkpoint}/best.ckpt" ;; *) test -f "${checkpoint}/best.pt" ;; esac
    if [[ "${model}" != nep ]]; then test -f "${checkpoint}/complete.json"; fi
done
test -f "${DATA_DIR}/mbd_data/daily_adapted/transactions.parquet"
test -f "${DATA_DIR}/mbd_data/daily_adapted/targets.parquet"
test -f "${DATA_DIR}/xbank_data/trans_any_pos_anonym_encoded.parquet"
test -f "${DATA_DIR}/xbank_data/targets_anonym_encoded.parquet"
mkdir -p "${DATA_DIR}/logs"
for gpu in 0 1; do
    session="daily-source-infer-gpu${gpu}"
    tmux new-session -d -s "${session}" \
        "/bin/bash '${REPO_DIR}/environments/daily_source_inference_worker.sh' '${gpu}' >> '${DATA_DIR}/logs/${session}_controller.log' 2>&1"
    echo "Started ${session}. Logs: ${DATA_DIR}/logs/${session}_controller.log"
done
echo 'Only embeds/{mbd_daily,xbank,xbank_fgw_v2}/mbd_daily_source/ will be written.'
