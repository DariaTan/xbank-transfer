#!/usr/bin/env bash
# Temporary HOST shim for workers whose functions were loaded before the RAM fix.
# Does not start/stop jobs, change training code, or touch other users' containers.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_DIR}/environments/extended_downstream_resources.sh"
exec 9>/tmp/xbank-chronos-ram-supervisor.lock
flock -n 9 || { echo 'RAM supervisor already running' >&2; exit 1; }
echo "START RAM supervisor $(date --iso-8601=seconds)"
while true; do
    exec 8>/tmp/xbank-extended-downstream-admission.lock
    flock 8
    names=$(docker ps --format '{{.Names}}')
    own_workers=0
    while IFS= read -r name; do
        case "${name}" in extended-downstream-gpu0|extended-downstream-gpu1) ;; *) continue ;; esac
        own_workers=$((own_workers + 1))
        id=$(docker inspect --format '{{.Id}}' "${name}") || continue
        status=0
        upgrade_chronos_container "${id}" || status=$?
        if (( status != 0 && status != 75 )); then
            flock -u 8; echo "ERROR updating ${name}; training workers are untouched" >&2; exit "${status}"
        fi
    done <<<"${names}"
    flock -u 8; exec 8>&-
    queues=0
    for gpu in 0 1; do
        if tmux has-session -t "extended-downstream-gpu${gpu}" 2>/dev/null; then queues=$((queues + 1)); fi
    done
    if (( own_workers == 0 && queues == 0 )); then
        echo "COMPLETE RAM supervisor: both queues ended $(date --iso-8601=seconds)"; exit 0
    fi
    sleep 15
done
