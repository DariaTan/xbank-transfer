#!/usr/bin/env bash
# Shared HOST resource policy. Only MBD Chronos' ~32-GiB feature panel needs 40 GiB.

task_memory_gib() {
    local module="$1"; shift
    local previous='' model='' evaluation='' arg
    for arg in "$@"; do
        case "${previous}" in
            --model) model="${arg}" ;;
            --evaluation) evaluation="${arg}" ;;
        esac
        previous="${arg}"
    done
    if [[ "${module}" == training.tune_mlp && "${model}" == chronos2 &&
          ( "${evaluation}" == mbd_raw || "${evaluation}" == mbd_daily ) ]]; then
        echo 40
    else
        echo 24
    fi
}

available_ram_kib() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }

reserved_worker_bytes() {
    local names name limit remaining total=0
    names=$(docker ps --format '{{.Names}}') || return 1
    while IFS= read -r name; do
        case "${name}" in extended-downstream-gpu0|extended-downstream-gpu1) ;; *) continue ;; esac
        limit=$(docker inspect --format '{{.HostConfig.Memory}}' "${name}" 2>/dev/null) || {
            # An old job may finish/remove its container while admission is
            # locked. New starts use this lock, so a vanished name is benign.
            remaining=$(docker ps --format '{{.Names}}') || return 1
            if [[ $'\n'"${remaining}"$'\n' == *$'\n'"${name}"$'\n'* ]]; then return 1; fi
            continue
        }
        if [[ ! "${limit}" =~ ^[0-9]+$ ]] || (( limit < 1 )); then
            echo "Refusing admission: own worker ${name} has no known RAM bound" >&2; return 1
        fi
        total=$((total + limit))
    done <<<"${names}"
    echo "${total}"
}

required_available_kib() {
    # Reserve complete worker caps plus 8 GiB for the host/other users.
    # MemAvailable includes reclaimable page cache, so do not subtract it.
    local new_limit_gib="$1" existing_bytes="$2"
    echo "$(((new_limit_gib + 8) * 1024 * 1024 + (existing_bytes + 1023) / 1024))"
}

upgrade_chronos_container() {
    # CALL UNDER the shared admission lock. Use immutable Docker ID to avoid
    # updating a different job if the name gets reused while inspecting it.
    local id="$1" state name running current pid command_lines arg wanted reserved required ram
    local args=()
    state=$(docker inspect --format '{{.Name}} {{.State.Running}} {{.HostConfig.Memory}} {{.State.Pid}}' "${id}") || return 0
    read -r name running current pid <<<"${state}"
    case "${name}" in /extended-downstream-gpu0|/extended-downstream-gpu1) ;; *) return 0 ;; esac
    [[ "${running}" == true ]] || return 0
    command_lines=$(docker inspect --format '{{range .Config.Cmd}}{{println .}}{{end}}' "${id}") || return 0
    while IFS= read -r arg; do args+=("${arg}"); done <<<"${command_lines}"
    if (( ${#args[@]} < 4 )) || [[ "${args[0]}" != python || "${args[1]}" != -u || "${args[2]}" != -m ]]; then return 0; fi
    wanted=$(task_memory_gib "${args[@]:3}")
    if (( wanted != 40 || current >= wanted * 1024 * 1024 * 1024 )); then return 0; fi
    reserved=$(reserved_worker_bytes) || return 1
    required=$(required_available_kib "${wanted}" "$((reserved - current))")
    ram=$(available_ram_kib)
    if (( ram < required )); then
        echo "WAIT RAM upgrade ${name}: available=$((ram/1024/1024))GiB required=$((required/1024/1024))GiB"
        return 75
    fi
    docker update --memory="${wanted}g" --memory-swap="${wanted}g" "${id}" >/dev/null || return 1
    echo "UPGRADED ${name} id=${id} PID=${pid} RAM=$((current/1024/1024/1024))->${wanted}GiB; no restart $(date --iso-8601=seconds)"
}
