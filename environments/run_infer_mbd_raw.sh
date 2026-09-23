#!/usr/bin/env bash
# Launch resumable MBD-raw inference sessions from the Docker host.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-xbank-transfer}"
if (( $# == 0 )); then
    echo "Pass one or two models, e.g. $0 coles cotic" >&2
    echo "Run at most two concurrently; launch the next pair after they finish." >&2
    exit 2
fi
if (( $# > 2 )); then
    echo "Refusing to start more than two full-scale inference jobs at once." >&2
    exit 2
fi
MODELS=("$@")

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' is not running." >&2
    exit 1
fi

docker exec "${CONTAINER_NAME}" mkdir -p /app/data/logs

slot=0
for model in "${MODELS[@]}"; do
        case "${model}" in
            coles|cotic|thp|nep|mlm) ;;
            *) echo "Unsupported model: ${model}" >&2; exit 2 ;;
        esac

        gpu=$((slot % 2))
        slot=$((slot + 1))
        session="infer-mbd-raw-${model}"
        log="/app/data/logs/infer_mbd_raw_${model}.log"

        if docker exec "${CONTAINER_NAME}" tmux has-session -t "${session}" 2>/dev/null; then
            echo "${session}: already exists, skipping"
            continue
        fi

        command="cd /app && CUDA_VISIBLE_DEVICES=${gpu} python src/training/infer_mbd.py --model ${model} --data-config /app/configs/data/mbd.yaml --downstream-config /app/configs/models/downstream_mbd.yaml --checkpoint-source mbd 2>&1 | tee -a ${log}"
        docker exec "${CONTAINER_NAME}" tmux new-session -d -s "${session}" "${command}"
        echo "Started ${session} on GPU ${gpu}; log: ${log}"
done

echo "Inspect sessions: docker exec ${CONTAINER_NAME} tmux ls"
